# Extract candidate and anchor representations for data selection.

import argparse
import gc
import json
from pathlib import Path

import torch

from main.tools.cache_extraction import extract_source
from main.tools.extraction import load_model_for_extraction
from main.tools.extract_dual_last_layer import _load_model, _sha256, _write_json


def main():
    parser = argparse.ArgumentParser(description='Extract candidate and anchor representations for data selection.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--task', choices=('gsm8k', 'truthfulqa', 'mmlu', 'bbh', 'tydiqa'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=28)
    parser.add_argument('--token-budget', type=int, default=4096)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.token_budget < 1 or args.limit < 0:
        parser.error('Invalid extraction batch size or limit')
    config = json.loads((args.model / 'config.json').read_text())
    if not 0 <= args.layer < config['num_hidden_layers']:
        parser.error('Layer index outside model')
    tokenizer = next((args.model / name for name in ('tokenizer.model', 'tokenizer.json')
                      if (args.model / name).is_file()), None)
    weights = next((args.model / name for name in ('model.safetensors.index.json', 'model.safetensors',
                    'pytorch_model.bin.index.json', 'pytorch_model.bin') if (args.model / name).is_file()), None)
    if tokenizer is None or weights is None:
        raise FileNotFoundError('Model weights identity or tokenizer file is missing')
    identity = {'model': str(args.model.resolve()), 'hidden_size': config['hidden_size'],
                'config_sha256': _sha256(args.model / 'config.json'),
                'weight_index_sha256': _sha256(weights), 'tokenizer_sha256': _sha256(tokenizer),
                'tokenizer_config_sha256': _sha256(args.model / 'tokenizer_config.json')}
    sources = {'pool': args.pool, 'anchors': args.anchors}
    source_specs = {}
    for name, path in sources.items():
        count = 0
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError('Blank JSONL rows would break pool index alignment')
                json.loads(line)
                count += 1
        if not count:
            raise ValueError(f'Empty source: {path}')
        source_specs[name] = {'path': str(path.resolve()), 'count': min(count, args.limit) if args.limit else count,
                              'sha256': _sha256(path)}
    spec = {'task': args.task, 'identity': identity, 'sources': source_specs, 'smoke_limit': args.limit,
            't': {'layer': args.layer, 'layer_ordinal': args.layer+1, 'max_tokens': 5120,
                  'text_view': 'canonical', 'pooling': 'position_weighted_mean', 'precision': 'float16'},
            'm': {'layer': 'final_post_norm', 'max_tokens': 2048, 'text_view': 'instruction_context_response',
                  'pooling': 'last_token', 'precision': 'bfloat16'},
            'separate_forward_views': True, 'storage_dtype': 'float16', 'truncation': 'head_tail'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.output_dir / f'spec_{args.task}.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('Task cache directory belongs to a different configuration')
    _write_json(spec_path, spec)
    for view in ('t', 'm'):
        model, tok = load_model_for_extraction(args.model) if view == 't' else _load_model(args.model, 'cuda:0')
        try:
            fingerprint = tok('Question: 2 + 2?\n\nAnswer: 4', add_special_tokens=True)['input_ids']
            if view == 't':
                first_fingerprint = fingerprint
            elif fingerprint != first_fingerprint:
                raise ValueError('T and M tokenizers disagree on the same text')
            for name, source in sources.items():
                filename = 'pool.npy' if name == 'pool' else f'{args.task}_anchors.npy'
                extract_source(source, args.output_dir / view / filename,
                               {**identity, **source_specs[name], **spec[view], 'view': view}, model, tok,
                               view=view, chunk_size=args.chunk_size, token_budget=args.token_budget)
        finally:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
    _write_json(args.output_dir / f'manifest_{args.task}.json', {'spec': spec, 'status': 'complete'})


if __name__ == '__main__':
    main()

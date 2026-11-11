# Evaluate TruthfulQA with an explicit answer cue.
import argparse
from functools import partial
import os
from pathlib import Path
import runpy
import sys


def format_with_cue(format_prompt, *args, **kwargs):
    return format_prompt(*args, **kwargs) + '\nA:'


def main():
    parser = argparse.ArgumentParser(
        description='TruthfulQA answer-cue evaluation; accepts backend evaluator arguments')
    parser.add_argument('--metrics', nargs='+', choices=('truth', 'info'), default=['truth', 'info'])
    args, remaining = parser.parse_known_args()
    if '--use_chat_format' in remaining or '--openai_engine' in remaining:
        parser.error('Answer-cue evaluation uses the HF question-answer prompt')
    from eval.truthfulqa import utilities
    original_format = utilities.format_prompt
    original_argv = sys.argv
    utilities.format_prompt = partial(format_with_cue, original_format)
    sys.argv = [original_argv[0], *remaining, '--metrics', *args.metrics]
    try:
        if os.environ.get('TRUTHFULQA_LOW_MEMORY') == '1':
            runpy.run_path(str(Path(__file__).with_name('evaluate_truthfulqa_low_memory.py')),
                           run_name='__main__')
        else:
            runpy.run_module('eval.truthfulqa.run_eval', run_name='__main__')
    finally:
        utilities.format_prompt = original_format
        sys.argv = original_argv


if __name__ == '__main__':
    main()

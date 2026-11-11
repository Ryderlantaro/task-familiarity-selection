# Run the TruthfulQA evaluator with lower judge GPU usage.

from __future__ import annotations

import os
import runpy

import torch
from eval.truthfulqa import metrics


judge_batch_size = int(os.environ.get("TRUTHFULQA_JUDGE_BATCH_SIZE", "2"))
original_score_completions = metrics.score_completions
original_run_hf_classifier_eval = metrics.run_hf_classifier_eval


def score_completions(model, tokenizer, examples, *, batch_size=10):
    return original_score_completions(model, tokenizer, examples, batch_size=judge_batch_size)


def run_hf_classifier_eval(model_key, tag, model, tokenizer, frame, info=False):
    result = original_run_hf_classifier_eval(model_key, tag, model, tokenizer, frame, info=info)
    model.cpu()
    torch.cuda.empty_cache()
    return result


metrics.score_completions = score_completions
metrics.run_hf_classifier_eval = run_hf_classifier_eval
runpy.run_module("eval.truthfulqa.run_eval", run_name="__main__")

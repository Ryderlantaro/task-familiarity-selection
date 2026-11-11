# Train each planned adapter, then evaluate its task.
import argparse
import os
from pathlib import Path
import subprocess

from scripts.experiment_utils import TASKS, digest, read_json, save_json


def main():
    parser = argparse.ArgumentParser(description='Run a prepared experiment plan')
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--condition', action='append')
    parser.add_argument('--task', choices=TASKS, action='append')
    args = parser.parse_args()
    plan = read_json(args.plan)
    root = args.plan.resolve().parent
    project = Path(__file__).resolve().parents[1]
    jobs = [job for job in plan['jobs']
            if (not args.condition or job['condition'] in args.condition)
            and (not args.task or job['task'] in args.task)]
    if not jobs:
        raise ValueError('No matching jobs')
    for job in jobs:
        if digest(job['train_file']) != job['train_sha256']:
            raise ValueError('Training data changed')
        print(f"{job['run_name']}: train -> {job['evaluation_repeats']} task evaluation(s)", flush=True)
    if not args.execute:
        return
    if plan['budget'] != 10000:
        raise ValueError('The training entry point requires a 10,000-example budget')
    run_root = root / 'runs'
    run_root.mkdir(exist_ok=True)
    for job in jobs:
        folder = run_root / job['run_name']
        folder.mkdir(exist_ok=False)
        env = dict(os.environ, PROJECT=str(project), BASE_MODEL=job['base_model'],
                   TRAIN_FILE=job['train_file'], SEED=str(job['seed']),
                   RUN_NAME=job['run_name'], RUN_ROOT=str(folder), MODE='train',
                   OUTPUT_DIR=str(folder / 'training/lora'), LOG_DIR=str(folder / 'logs'),
                   MANIFEST=str(folder / 'train_manifest.json'))
        save_json(folder / 'job.json', job)
        subprocess.run(['bash', str(project / 'scripts/train_lora.sh')], env=env, check=True)
        for repeat in range(1, job['evaluation_repeats'] + 1):
            eval_env = dict(env, TASKS='gsm' if job['task'] == 'gsm8k' else job['task'],
                ADAPTER_DIR=str(folder / 'training/lora'),
                RESULTS_ROOT=str(folder / f'evaluation_{repeat}'),
                MERGED_DIR=str(folder / f'merged_{repeat}'), DATA_PROTOCOL='heldout',
                TRUTHFULQA_PROMPT='original', TRUTHFULQA_RUNNER='',
                MODEL_DIR='', OVERWRITE='0')
            subprocess.run(['bash', str(project / 'scripts/evaluate_tasks.sh')], env=eval_env, check=True)
        save_json(folder / 'complete.json', dict(state='complete',
            plan_sha256=digest(args.plan), train_sha256=job['train_sha256']))


if __name__ == '__main__':
    main()

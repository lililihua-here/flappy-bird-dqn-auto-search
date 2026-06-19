"""History persistence, summary generation, and top-K recheck logic."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from train_eval import run_trial
from version_utils import get_git_hash, infer_reward_scheme_version


def _make_serializable(obj):
    """Convert numpy / torch objects to JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


class HistoryManager:
    """Append-only JSONL history for trial results."""

    def __init__(self, history_path='search_history.jsonl'):
        self.path = Path(history_path)

    def append(self, result):
        with open(self.path, 'a', encoding='utf-8') as f:
            serializable = dict(_make_serializable(result))
            serializable.setdefault('record_type', 'trial')
            f.write(json.dumps(serializable, ensure_ascii=False) + '\n')
            f.flush()

    def load(self):
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def success_count(self):
        return sum(
            1 for r in self.load()
            if r.get('record_type', 'trial') == 'trial' and r.get('status') == 'success'
        )

    def failure_count(self):
        return sum(
            1 for r in self.load()
            if r.get('record_type', 'trial') == 'trial' and r.get('status') == 'failure'
        )

    def best_trial(self):
        rows = [r for r in self.load() if r.get('record_type', 'trial') == 'trial']
        successes = [r for r in rows if r.get('status') == 'success']
        if successes:
            return min(successes, key=lambda r: r.get('objective', float('inf')))
        if rows:
            return min(rows, key=lambda r: r.get('objective', float('inf')))
        return None

    def top_k(self, k=5):
        """P1-3: Sort by Section 16.2 priority rules. Filter trial records only."""
        all_rows = self.load()
        trial_rows = [
            r for r in all_rows
            if r.get('record_type', 'trial') == 'trial'
            and r.get('config')
        ]
        latest_recheck_by_trial = {}
        for row in all_rows:
            if row.get('record_type') == 'recheck' and 'trial_id' in row:
                latest_recheck_by_trial[row['trial_id']] = row

        rows = []
        for trial_row in trial_rows:
            merged = dict(trial_row)
            recheck_row = latest_recheck_by_trial.get(trial_row.get('trial_id'))
            if recheck_row:
                for key, value in recheck_row.items():
                    if key not in ('record_type', 'trial_id'):
                        merged[key] = value
            rows.append(merged)

        def ranking_key(r):
            return (
                not r.get('recheck_passed', False),
                r.get('median_train_raw_env_frames_to_stable_1000',
                      r.get('objective', float('inf'))),
                -r.get('success_rate_1000', 0),
                -r.get('median_score', 0),
                r.get('total_raw_env_frames', float('inf')),
            )

        sorted_rows = sorted(rows, key=ranking_key)
        return sorted_rows[:k]


def generate_summary(history, top_k=5):
    """Print a compact summary and return it as a dict for tests."""
    rows = history.load()
    trial_rows = [r for r in rows if r.get('record_type', 'trial') == 'trial']

    if not trial_rows:
        summary = {
            'trial_count': 0, 'success_count': 0, 'failure_count': 0,
            'best_trial_id': None, 'best_objective': None,
            'top_k': [], 'failure_reasons': {},
        }
        print('[SUMMARY] No trial records found.')
        return summary

    failure_reasons = {}
    for row in trial_rows:
        if row.get('status') == 'failure':
            reason = row.get('failure_reason') or 'unknown'
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    best = history.best_trial()
    top_trials = history.top_k(top_k)
    summary = {
        'trial_count': len(trial_rows),
        'success_count': history.success_count(),
        'failure_count': history.failure_count(),
        'best_trial_id': best.get('trial_id') if best else None,
        'best_objective': best.get('objective') if best else None,
        'top_k': [
            {
                'trial_id': row.get('trial_id'),
                'source': row.get('source'),
                'objective': row.get('objective'),
                'median_score': row.get('median_score'),
                'success_rate_1000': row.get('success_rate_1000'),
            }
            for row in top_trials
        ],
        'failure_reasons': failure_reasons,
    }

    print('\n[SUMMARY]')
    print(f"Trials: {summary['trial_count']}  "
          f"Success: {summary['success_count']}  Failure: {summary['failure_count']}")
    print(f"Best trial: {summary['best_trial_id']}  Objective: {summary['best_objective']}")
    for idx, row in enumerate(summary['top_k'], start=1):
        print(f"Top {idx}: trial={row['trial_id']}  obj={row['objective']}  "
              f"median={row['median_score']}  sr={row['success_rate_1000']}")
    return summary


def recheck_top_k(history, k=5, recheck_seeds=(101, 202, 303),
                  max_trial_frames=1_000_000, eval_episodes=20):
    """Re-evaluate top K configs with multiple independent seeds."""
    top_configs = history.top_k(k)
    results = []

    for rank, trial in enumerate(top_configs):
        config = trial.get('config', {})
        if not config:
            continue
        seed_scores = []
        for seed in recheck_seeds:
            result = run_trial(
                config=config, trial_id=-1, seed=seed, source='recheck',
                max_trial_frames=max_trial_frames,
                eval_episodes=eval_episodes,
            )
            seed_scores.append({
                'seed': seed,
                'status': result['status'],
                'train_raw_env_frames': result['train_raw_env_frames'],
                'median_score': result['median_score'],
                'success_rate_1000': result['success_rate_1000'],
            })

        all_seed_scores = [s['median_score'] for s in seed_scores]
        successful_train_frames = [
            s['train_raw_env_frames'] for s in seed_scores
            if s['status'] == 'success'
        ]

        recheck_summary = {
            'rank': rank + 1,
            'original_trial_id': trial.get('trial_id'),
            'config': config,
            'seeds': seed_scores,
            'recheck_median': float(np.median(all_seed_scores)),
            'recheck_mean': float(np.mean(all_seed_scores)),
            'recheck_success_rate': float(np.mean([s['success_rate_1000'] for s in seed_scores])),
            'p10_score': float(np.percentile(all_seed_scores, 10)),
            'p90_score': float(np.percentile(all_seed_scores, 90)),
            'score_std': float(np.std(all_seed_scores)),
            'median_train_raw_env_frames_to_stable_1000': (
                float(np.median(successful_train_frames)) if successful_train_frames else None
            ),
            'recheck_passed': all(s['status'] == 'success' for s in seed_scores),
            'failed_seeds': [s for s in seed_scores if s['status'] != 'success'],
        }
        results.append(recheck_summary)

        recheck_record = {
            'record_type': 'recheck',
            'trial_id': trial.get('trial_id'),
            **recheck_summary,
            'recheck_seeds_used': list(recheck_seeds),
        }
        history.append(recheck_record)

    return results

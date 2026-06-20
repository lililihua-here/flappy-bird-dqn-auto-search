"""History persistence, checkpoint assets, recheck logic, and report generation."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_eval import run_trial
from version_utils import get_git_hash


CHECKPOINT_FORMAT_VERSION = 'v2_checkpoint_1'
REPORT_VERSION = 'v2.0'
DEFAULT_ENVIRONMENT_VERSION = 'fixed_env_v1'
DEFAULT_STATE_REPRESENTATION_VERSION = 'low_dim_v1'


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


def normalize_legacy_record(record):
    """Normalize legacy JSONL records to the V2 schema."""
    normalized = dict(record)
    normalized.setdefault('record_type', 'trial')
    normalized.setdefault('n_step', 1)
    normalized.setdefault('priority', False)
    if 'per_beta_train_updates' not in normalized and 'per_beta_frames' in normalized:
        normalized['per_beta_train_updates'] = normalized['per_beta_frames']
    if 'environment_version' not in normalized:
        normalized['environment_version'] = normalized.get(
            'env_version', 'unknown_env_version'
        )
    normalized.setdefault('state_representation_version', 'unknown_state_version')
    normalized.setdefault('reward_scheme_version', 'mvp_reward_v1')

    config = dict(normalized.get('config', {}))
    if 'reward_pipe' in config and 'pipe_reward' not in config:
        config['pipe_reward'] = config['reward_pipe']
    if 'reward_death' in config and 'death_ratio' not in config:
        config['death_ratio'] = abs(config['reward_death'])
    if 'reward_alive' in config and 'alive_ratio' not in config:
        config['alive_ratio'] = config['reward_alive']
    if config:
        normalized['config'] = config

    return normalized


def build_checkpoint_payload(q_net, target_net, config, trial_id, seed, source,
                             train_raw_env_frames, decision_steps, *,
                             state_dim=7, n_actions=2,
                             environment_version=DEFAULT_ENVIRONMENT_VERSION,
                             state_representation_version=DEFAULT_STATE_REPRESENTATION_VERSION):
    """Build the full V2 checkpoint payload."""
    reward_config = {
        key: config.get(key)
        for key in ('pipe_reward', 'death_ratio', 'alive_ratio', 'reward_scale', 'reward_clip')
    }
    return {
        'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
        'code_version': get_git_hash(),
        'environment_version': environment_version,
        'state_representation_version': state_representation_version,
        'reward_scheme_version': config.get('reward_scheme_version', 'mvp_reward_v1'),
        'reward_config': reward_config,
        'trial_id': trial_id,
        'seed': seed,
        'source': source,
        'config': dict(config),
        'n_step': config.get('n_step', 1),
        'priority': config.get('priority', False),
        'train_raw_env_frames': int(train_raw_env_frames),
        'decision_steps': int(decision_steps),
        'state_dim': int(state_dim),
        'n_actions': int(n_actions),
        'q_net_state_dict': q_net.state_dict(),
        'target_net_state_dict': (
            target_net.state_dict() if target_net is not None else q_net.state_dict()
        ),
    }


def save_checkpoint(state_dict_or_payload, checkpoint_dir, subdir='', prefix='checkpoint'):
    """Save checkpoint and return (path, sha256_hex)."""
    save_dir = Path(checkpoint_dir) / subdir if subdir else Path(checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = prefix if str(prefix).endswith('.pt') else f'{prefix}.pt'
    path = (save_dir / filename).resolve()
    torch.save(state_dict_or_payload, path)
    sha256_hex = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path), sha256_hex


def is_checkpoint_compatible(record, current_env, current_reward, current_state):
    """Check whether a record/checkpoint matches the current evaluation protocol."""
    return (
        record.get('environment_version') == current_env
        and record.get('reward_scheme_version') == current_reward
        and record.get('state_representation_version') == current_state
    )


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
                if not line:
                    continue
                try:
                    rows.append(normalize_legacy_record(json.loads(line)))
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
        """Sort by Section 16.2 priority rules. Filter trial records only."""
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


def _append_group_comparison(md_lines, trial_rows, key, label):
    groups = {}
    for row in trial_rows:
        value = row.get('config', {}).get(key, row.get(key, 'unknown'))
        groups.setdefault(str(value), []).append(row)
    for value, group in sorted(groups.items()):
        successes = [r for r in group if r.get('status') == 'success']
        best_obj = min((r.get('objective', float('inf')) for r in successes), default=None)
        suffix = f', best_obj={best_obj:.0f}' if best_obj is not None else ''
        md_lines.append(
            f'- {label}={value}: {len(group)} trials, {len(successes)} successes{suffix}'
        )


def generate_experiment_manifest(history, study_db_path, output_dir='.'):
    """Generate experiment_manifest.json."""
    best = history.best_trial()
    manifest = {
        'history_path': str(history.path),
        'study_db_path': str(study_db_path),
        'code_version': get_git_hash(),
        'environment_version': (
            best.get('environment_version') if best else DEFAULT_ENVIRONMENT_VERSION
        ),
        'reward_scheme_version': best.get('reward_scheme_version') if best else 'mvp_reward_v1',
        'state_representation_version': (
            best.get('state_representation_version')
            if best else DEFAULT_STATE_REPRESENTATION_VERSION
        ),
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'total_trials': len([r for r in history.load() if r.get('record_type') == 'trial']),
        'success_trials': history.success_count(),
        'best_trial_id': best.get('trial_id') if best else None,
        'report_version': REPORT_VERSION,
    }
    path = Path(output_dir) / 'experiment_manifest.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(path)


def generate_summary_report_md(history, output_dir='.'):
    """Generate a human-readable markdown experiment summary."""
    rows = history.load()
    trial_rows = [r for r in rows if r.get('record_type', 'trial') == 'trial']
    best = history.best_trial()
    top5 = history.top_k(5)

    md = ['# Flappy Bird DQN V2 实验报告', '']
    md.append(f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    md.append(f'代码版本: {get_git_hash()}')
    md.append('')

    md.append('## 1. 实验概览')
    md.append(f'- 总 Trial 数: {len(trial_rows)}')
    md.append(f'- 成功: {history.success_count()}')
    md.append(f'- 失败: {history.failure_count()}')
    md.append('')

    md.append('## 2. 最优 Trial')
    if best:
        md.append(f"- Trial #{best.get('trial_id')}")
        md.append(f"- Objective: {best.get('objective', 'N/A')}")
        md.append(f"- Median Score: {best.get('median_score', 'N/A')}")
        md.append(f"- Success Rate: {best.get('success_rate_1000', 'N/A')}")
    else:
        md.append('- 无可用 Trial')
    md.append('')

    md.append('## 3. Top-K 排名')
    for idx, trial in enumerate(top5, start=1):
        md.append(
            f"{idx}. Trial #{trial.get('trial_id')} — obj={trial.get('objective', 'N/A')}, "
            f"median={trial.get('median_score', 'N/A')}, "
            f"sr={trial.get('success_rate_1000', 'N/A')}"
        )
    md.append('')

    statuses = {}
    for row in trial_rows:
        status = row.get('status', 'unknown')
        statuses[status] = statuses.get(status, 0) + 1
    md.append('## 4. 状态分布')
    for status, count in sorted(statuses.items()):
        md.append(f'- {status}: {count}')
    md.append('')

    md.append('## 5. n-step 对比')
    _append_group_comparison(md, trial_rows, 'n_step', 'n_step')
    md.append('## 6. PER vs Uniform Replay')
    _append_group_comparison(md, trial_rows, 'priority', 'PER/replay')
    md.append('## 7. 奖励方案对比')
    _append_group_comparison(md, trial_rows, 'reward_scheme_version', 'reward')
    md.append('')

    with_checkpoint = sum(1 for row in trial_rows if row.get('checkpoint_path'))
    md.append('## 8. Checkpoint 状态')
    md.append(f'- 有 Checkpoint 的 Trial: {with_checkpoint}')
    md.append('')

    md.append('## 9. 版本信息')
    md.append(
        f"- environment_version: {best.get('environment_version', DEFAULT_ENVIRONMENT_VERSION) if best else DEFAULT_ENVIRONMENT_VERSION}"
    )
    md.append(
        f"- reward_scheme_version: {best.get('reward_scheme_version', 'mvp_reward_v1') if best else 'mvp_reward_v1'}"
    )
    md.append(
        f"- state_representation_version: {best.get('state_representation_version', DEFAULT_STATE_REPRESENTATION_VERSION) if best else DEFAULT_STATE_REPRESENTATION_VERSION}"
    )
    md.append(f'- report_version: {REPORT_VERSION}')
    md.append('')

    md.append('## 10. 建议')
    md.append('（如有 LLM 离线分析，可在此处补充下一阶段搜索建议）')
    md.append('')

    path = Path(output_dir) / 'summary_report.md'
    path.write_text('\n'.join(md), encoding='utf-8')
    return str(path)


def generate_topk_summary_json(history, output_dir='.'):
    """Generate topk_summary.json."""
    summary = []
    for rank, trial in enumerate(history.top_k(5), start=1):
        summary.append({
            'rank': rank,
            'trial_id': trial.get('trial_id'),
            'objective': trial.get('objective'),
            'median_score': trial.get('median_score'),
            'success_rate_1000': trial.get('success_rate_1000'),
            'config': trial.get('config'),
        })
    path = Path(output_dir) / 'topk_summary.json'
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(path)


def generate_recheck_summary_json(history, output_dir='.'):
    """Generate recheck_summary.json from the latest recheck rows."""
    latest = {}
    for row in history.load():
        if row.get('record_type') != 'recheck':
            continue
        trial_id = row.get('trial_id')
        prev = latest.get(trial_id)
        if prev is None or row.get('timestamp', '') > prev.get('timestamp', ''):
            latest[trial_id] = row
    summary = list(latest.values())
    path = Path(output_dir) / 'recheck_summary.json'
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(path)


def generate_all_reports(history, study_db_path, output_dir='.'):
    """Generate all V2 report artifacts and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return {
        'experiment_manifest': generate_experiment_manifest(history, study_db_path, output_dir),
        'summary_report': generate_summary_report_md(history, output_dir),
        'topk_summary': generate_topk_summary_json(history, output_dir),
        'recheck_summary': generate_recheck_summary_json(history, output_dir),
    }

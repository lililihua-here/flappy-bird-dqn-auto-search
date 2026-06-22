"""Standard experiment matrices for V3 ablation studies."""
import numpy as np
import json
from pathlib import Path
from search_driver import BASELINE_CONFIG

DEFAULT_BEST_CONFIG_PATH = "best_config.json"
PROTOCOL_REWARD_DEFAULTS = {
    "reward_v1_sparse": {},
    "reward_v2_ratio": {
        "death_ratio": 20,
        "alive_ratio": 0.001,
        "reward_scale": 0.1,
        "reward_clip": 10,
    },
    "reward_v3_gap_shaping": {
        "death_ratio": 20,
        "alive_ratio": 0.001,
        "reward_scale": 0.1,
        "reward_clip": 10,
        "gap_shaping_coef": 0.05,
    },
}

BASELINE_MATRIX = [
    {"name": "V2_baseline", "config": dict(BASELINE_CONFIG), "seeds": 3},
    {"name": "V2_best_known_config", "config_path": "best_config.json", "seeds": 3},
    {"name": "V3_fresh_baseline", "config": dict(BASELINE_CONFIG), "seeds": 3},
]

STRUCTURE_ABLATION = [
    {"name": "mlp_eps", "base_config_path": DEFAULT_BEST_CONFIG_PATH, "network_backbone": "mlp", "exploration_head": "epsilon_greedy"},
    {"name": "dueling_eps", "base_config_path": DEFAULT_BEST_CONFIG_PATH, "network_backbone": "dueling_mlp", "exploration_head": "epsilon_greedy"},
    {"name": "mlp_noisy", "base_config_path": DEFAULT_BEST_CONFIG_PATH, "network_backbone": "mlp", "exploration_head": "noisy_net"},
    {"name": "dueling_noisy", "base_config_path": DEFAULT_BEST_CONFIG_PATH, "network_backbone": "dueling_mlp", "exploration_head": "noisy_net"},
]

PROTOCOL_ABLATION = [
    {"name": "v1_state_v1_reward", "base_config_path": DEFAULT_BEST_CONFIG_PATH, "state_representation_version": "low_dim_v1", "reward_scheme_version": "reward_v1_sparse"},
    {"name": "v2_state_v1_reward", "base_config_path": DEFAULT_BEST_CONFIG_PATH, "state_representation_version": "low_dim_v2", "reward_scheme_version": "reward_v1_sparse"},
    {"name": "v3_state_v1_reward", "base_config_path": DEFAULT_BEST_CONFIG_PATH, "state_representation_version": "low_dim_v3", "reward_scheme_version": "reward_v1_sparse"},
    {
        "name": "v1_state_v2_reward",
        "base_config_path": DEFAULT_BEST_CONFIG_PATH,
        "state_representation_version": "low_dim_v1",
        "reward_scheme_version": "reward_v2_ratio",
        "config": {
            "death_ratio": 20,
            "alive_ratio": 0.001,
            "reward_scale": 0.1,
            "reward_clip": 10,
        },
    },
    {
        "name": "v1_state_v3_reward",
        "base_config_path": DEFAULT_BEST_CONFIG_PATH,
        "state_representation_version": "low_dim_v1",
        "reward_scheme_version": "reward_v3_gap_shaping",
        "config": {
            "death_ratio": 20,
            "alive_ratio": 0.001,
            "reward_scale": 0.1,
            "reward_clip": 10,
            "gap_shaping_coef": 0.05,
        },
    },
]

SEARCHER_COMPARISON = [
    {"name": "tpe_fresh", "search_strategy": "tpe_fresh", "max_trials": 30},
    {"name": "warmstart_tpe", "search_strategy": "warmstart_tpe", "max_trials": 30},
    {"name": "population_async", "search_strategy": "population_async", "max_total_frames": 3_000_000},
]

def allocate_matrix_trial_id(entry_name, seed):
    """Generate stable trial_id from entry name + seed."""
    import hashlib
    h = hashlib.md5(f"{entry_name}_{seed}".encode()).hexdigest()[:8]
    return int(h, 16) % 900000 + 100000


def _resolve_config_path(path_value):
    config_path = Path(path_value)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return config_path


def _load_json_config_if_exists(path_value):
    config_path = _resolve_config_path(path_value)
    if not config_path.exists():
        return None, config_path
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f), config_path


def _build_matrix_config(entry):
    if "config_path" in entry:
        loaded_config, config_path = _load_json_config_if_exists(entry["config_path"])
        if loaded_config is None:
            return None, {
                'record_type': 'matrix_entry',
                'name': entry.get('name'),
                'status': 'skipped',
                'reason': f'missing config_path: {config_path}',
            }
        base_config = loaded_config
        base_config_source = str(config_path)
    else:
        base_config = {}
        base_config_source = 'baseline_default'
        if "base_config_path" in entry:
            loaded_config, config_path = _load_json_config_if_exists(entry["base_config_path"])
            if loaded_config is not None:
                base_config = loaded_config
                base_config_source = str(config_path)

    reward_scheme_version = entry.get("reward_scheme_version")
    protocol_defaults = PROTOCOL_REWARD_DEFAULTS.get(reward_scheme_version, {})
    config = {**BASELINE_CONFIG, **base_config, **protocol_defaults, **entry.get("config", {})}
    config.update({k: v for k, v in entry.items() if k in (
        'network_backbone', 'exploration_head',
        'state_representation_version', 'reward_scheme_version',
        'death_ratio', 'alive_ratio', 'reward_scale', 'reward_clip',
        'pipe_reward', 'gap_shaping_coef',
    )})
    return config, {
        'name': entry.get('name'),
        'base_config_source': base_config_source,
    }


def summarize_matrix_results(matrix_name, results):
    entry_groups = {}
    for row in results:
        if row.get('record_type') == 'matrix_entry':
            entry_groups[row['name']] = {
                'name': row['name'],
                'status': row.get('status', 'unknown'),
                'reason': row.get('reason', ''),
            }
            continue
        if row.get('record_type', 'trial') != 'trial':
            continue
        name = row.get('matrix_entry_name')
        if not name:
            source = row.get('source', '')
            if not source.startswith('matrix_'):
                continue
            name = source[len('matrix_'):]
        group = entry_groups.setdefault(name, {
            'name': name,
            'status': 'completed',
            'trial_count': 0,
            'success_count': 0,
            'failure_count': 0,
            'best_objective': None,
            'best_median_score': None,
            'best_success_rate_1000': None,
            'failure_reasons': {},
            'base_config_source': row.get('base_config_source', 'unknown'),
        })
        group['trial_count'] += 1
        if row.get('status') == 'success':
            group['success_count'] += 1
        else:
            group['failure_count'] += 1
            reason = row.get('failure_reason') or 'unknown'
            group['failure_reasons'][reason] = group['failure_reasons'].get(reason, 0) + 1
        objective = row.get('objective')
        median_score = row.get('median_score')
        success_rate = row.get('success_rate_1000')
        if group['best_objective'] is None or (
            objective is not None and objective < group['best_objective']
        ):
            group['best_objective'] = objective
        if group['best_median_score'] is None or (
            median_score is not None and median_score > group['best_median_score']
        ):
            group['best_median_score'] = median_score
        if group['best_success_rate_1000'] is None or (
            success_rate is not None and success_rate > group['best_success_rate_1000']
        ):
            group['best_success_rate_1000'] = success_rate

    entries = sorted(
        entry_groups.values(),
        key=lambda item: (
            item.get('status') != 'completed',
            -(item.get('success_count', 0)),
            -(item.get('best_median_score') or 0.0),
            item.get('best_objective') if item.get('best_objective') is not None else float('inf'),
            item.get('name', ''),
        ),
    )
    return {
        'matrix': matrix_name,
        'result_count': len(results),
        'entries': entries,
    }


def run_search_strategy(entry, mode="debug", budget="debug_matrix"):
    from search_driver import SearchDriver, get_mode_presets

    presets = get_mode_presets(mode)
    if budget == "debug_matrix":
        max_trials = min(entry.get("max_trials", 5), 5)
        max_trial_frames = min(entry.get("max_total_frames", presets['max_trial_frames']),
                               presets['max_trial_frames'])
    elif budget == "normal_matrix":
        max_trials = entry.get("max_trials", 30)
        max_trial_frames = entry.get("max_total_frames", presets['max_trial_frames'])
    else:
        raise ValueError(f"Unknown budget: {budget}")

    driver = SearchDriver(
        max_trials=max_trials,
        max_trial_frames=max_trial_frames,
        eval_interval_frames=presets['eval_interval_frames'],
        eval_episodes=presets['eval_episodes'],
        history_path=entry.get('history_path', f"{entry['name']}.jsonl"),
        study_db=entry.get('study_db', f"{entry['name']}.db"),
    )
    strategy = entry["search_strategy"]
    if strategy == "tpe_fresh":
        driver.run()
    elif strategy == "warmstart_tpe":
        driver.run_warmstart_tpe()
    elif strategy == "population_async":
        driver.run_population_async(total_frame_budget=entry.get("max_total_frames", max_trial_frames))
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")
    rows = []
    for row in driver.history.load():
        row_copy = dict(row)
        row_copy['matrix_entry_name'] = entry['name']
        row_copy['base_config_source'] = 'search_strategy'
        rows.append(row_copy)
    return rows

def run_matrix(matrix, mode="debug", budget="debug_matrix", *, output_dir=None, source='matrix'):
    if budget == "debug_matrix":
        seeds_per = [42]; max_trial_frames = 100_000
    elif budget == "normal_matrix":
        seeds_per = [42, 43, 44]; max_trial_frames = 1_000_000
    else:
        raise ValueError(f"Unknown budget: {budget}")

    from train_eval import run_trial
    results = []
    for entry in matrix:
        if "search_strategy" in entry:
            results.extend(run_search_strategy(entry, mode=mode, budget=budget))
            continue
        config, config_meta = _build_matrix_config(entry)
        if config is None:
            results.append(config_meta)
            continue
        for seed in seeds_per:
            trial_id = allocate_matrix_trial_id(entry["name"], seed)
            result = run_trial(config, trial_id=trial_id, seed=seed,
                               source=f"{source}_{entry['name']}",
                               max_trial_frames=max_trial_frames)
            result['base_config_source'] = config_meta['base_config_source']
            results.append(result)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        history_path = output_path / 'history.jsonl'
        summary_path = output_path / 'summary.json'
        with open(history_path, 'w', encoding='utf-8') as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        summary = summarize_matrix_results('custom_matrix', results)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return results

def final_confirm(config, seeds=(1001, 1002, 1003, 1004, 1005),
                  max_trial_frames=1_000_000, checkpoint_dir='checkpoints',
                  source='final_confirm', output_path=None, output_dir=None):
    import numpy as np
    from train_eval import run_trial
    all_scores = []
    per_seed_results = []
    for seed in seeds:
        result = run_trial(config,
                           trial_id=int(f"9999{seed}"),
                           seed=seed, source=source,
                           max_trial_frames=max_trial_frames,
                           candidate_verify_episodes=20,
                           force_final_eval=True,
                           checkpoint_dir=checkpoint_dir)
        per_seed_results.append(result)
        all_scores.extend(result.get('final_eval_scores', []))
    expected_count = len(seeds) * 20
    if len(all_scores) != expected_count:
        return {'config': config, 'seeds_used': list(seeds),
                'status': 'invalid',
                'reason': f"expected {expected_count} scores, got {len(all_scores)}"}
    overall_success_rate = sum(s >= 1000 for s in all_scores) / max(len(all_scores), 1)
    overall_median = float(np.median(all_scores)) if all_scores else 0.0
    confirmed = overall_success_rate >= 0.80 and overall_median >= 1000
    result = {'config': config, 'seeds_used': list(seeds),
              'all_eval_scores': all_scores, 'per_seed': per_seed_results,
              'overall_success_rate_1000': overall_success_rate,
              'overall_median_score': overall_median,
              'status': 'confirmed' if confirmed else 'not_confirmed'}
    if output_dir is not None and output_path is None:
        output_path = Path(output_dir) / 'final_confirm_summary.json'
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result

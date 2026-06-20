"""Standard experiment matrices for V3 ablation studies."""
import numpy as np
import json
from search_driver import BASELINE_CONFIG

BASELINE_MATRIX = [
    {"name": "V2_baseline", "config": dict(BASELINE_CONFIG), "seeds": 3},
    {"name": "V2_best_known_config", "config_path": "best_config.json", "seeds": 3},
    {"name": "V3_fresh_baseline", "config": dict(BASELINE_CONFIG), "seeds": 3},
]

STRUCTURE_ABLATION = [
    {"name": "mlp_eps", "network_backbone": "mlp", "exploration_head": "epsilon_greedy"},
    {"name": "dueling_eps", "network_backbone": "dueling_mlp", "exploration_head": "epsilon_greedy"},
    {"name": "mlp_noisy", "network_backbone": "mlp", "exploration_head": "noisy_net"},
    {"name": "dueling_noisy", "network_backbone": "dueling_mlp", "exploration_head": "noisy_net"},
]

PROTOCOL_ABLATION = [
    {"name": "v1_state_v1_reward", "state_representation_version": "low_dim_v1", "reward_scheme_version": "reward_v1_sparse"},
    {"name": "v2_state_v1_reward", "state_representation_version": "low_dim_v2", "reward_scheme_version": "reward_v1_sparse"},
    {"name": "v3_state_v1_reward", "state_representation_version": "low_dim_v3", "reward_scheme_version": "reward_v1_sparse"},
    {"name": "v1_state_v2_reward", "state_representation_version": "low_dim_v1", "reward_scheme_version": "reward_v2_ratio"},
    {"name": "v1_state_v3_reward", "state_representation_version": "low_dim_v1", "reward_scheme_version": "reward_v3_gap_shaping"},
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

def run_matrix(matrix, mode="debug", budget="debug_matrix"):
    if budget == "debug_matrix":
        seeds_per = [42]; max_trial_frames = 100_000
    elif budget == "normal_matrix":
        seeds_per = [42, 43, 44]; max_trial_frames = 1_000_000
    else:
        raise ValueError(f"Unknown budget: {budget}")

    from train_eval import run_trial
    results = []
    for entry in matrix:
        if "config_path" in entry:
            with open(entry["config_path"], "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
        else:
            loaded_config = entry.get("config", {})
        config = {**BASELINE_CONFIG, **loaded_config}
        config.update({k: v for k, v in entry.items() if k in (
            'network_backbone', 'exploration_head',
            'state_representation_version', 'reward_scheme_version',
        )})
        for seed in seeds_per:
            trial_id = allocate_matrix_trial_id(entry["name"], seed)
            if "search_strategy" in entry:
                result = run_trial(config, trial_id=trial_id, seed=seed,
                                   source=f"matrix_{entry['name']}",
                                   trial_type="fresh",
                                   max_trial_frames=max_trial_frames)
            else:
                result = run_trial(config, trial_id=trial_id, seed=seed,
                                   source=f"matrix_{entry['name']}",
                                   max_trial_frames=max_trial_frames)
            results.append(result)
    return results

def final_confirm(config, seeds=(1001, 1002, 1003, 1004, 1005)):
    import numpy as np
    from train_eval import run_trial
    all_scores = []
    per_seed_results = []
    for seed in seeds:
        result = run_trial(config,
                           trial_id=int(f"9999{seed}"),
                           seed=seed, source='final_confirm',
                           max_trial_frames=1_000_000,
                           candidate_verify_episodes=20,
                           force_final_eval=True)
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
    return {'config': config, 'seeds_used': list(seeds),
            'all_eval_scores': all_scores, 'per_seed': per_seed_results,
            'overall_success_rate_1000': overall_success_rate,
            'overall_median_score': overall_median,
            'status': 'confirmed' if confirmed else 'not_confirmed'}

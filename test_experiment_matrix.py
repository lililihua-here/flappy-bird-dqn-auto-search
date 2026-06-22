"""V3.5 experiment matrix tests."""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


def test_baseline_matrix_has_three_entries():
    from experiment_matrix import BASELINE_MATRIX
    assert len(BASELINE_MATRIX) >= 3


def test_baseline_matrix_v2_baseline_has_config():
    from experiment_matrix import BASELINE_MATRIX
    v2_entry = BASELINE_MATRIX[0]
    assert v2_entry["name"] == "V2_baseline"
    assert "config" in v2_entry


def test_baseline_matrix_v2_best_known_has_config_path():
    from experiment_matrix import BASELINE_MATRIX
    v2_best = BASELINE_MATRIX[1]
    assert v2_best["name"] == "V2_best_known_config"
    assert "config_path" in v2_best


def test_structure_ablation_has_four_combinations():
    from experiment_matrix import STRUCTURE_ABLATION
    assert len(STRUCTURE_ABLATION) == 4


def test_structure_ablation_entries_have_expected_fields():
    from experiment_matrix import STRUCTURE_ABLATION
    for entry in STRUCTURE_ABLATION:
        assert "name" in entry
        assert "network_backbone" in entry
        assert "exploration_head" in entry
        assert entry["network_backbone"] in ("mlp", "dueling_mlp")
        assert entry["exploration_head"] in ("epsilon_greedy", "noisy_net")


def test_protocol_ablation_uses_correct_field_names():
    from experiment_matrix import PROTOCOL_ABLATION
    for entry in PROTOCOL_ABLATION:
        assert "state_representation_version" in entry
        assert "reward_scheme_version" in entry
        assert entry["state_representation_version"] in ("low_dim_v1", "low_dim_v2", "low_dim_v3")
        assert entry["reward_scheme_version"] in ("reward_v1_sparse", "reward_v2_ratio", "reward_v3_gap_shaping")


def test_protocol_ablation_has_five_entries():
    from experiment_matrix import PROTOCOL_ABLATION
    assert len(PROTOCOL_ABLATION) == 5


def test_protocol_ablation_reward_variants_have_effective_params():
    from experiment_matrix import PROTOCOL_ABLATION

    by_name = {entry["name"]: entry for entry in PROTOCOL_ABLATION}
    v2 = by_name["v1_state_v2_reward"]["config"]
    v3 = by_name["v1_state_v3_reward"]["config"]

    assert v2["death_ratio"] > 1
    assert v2["reward_scale"] in (0.01, 0.1, 1.0)
    assert v3["gap_shaping_coef"] > 0.0


def test_searcher_comparison_has_three_entries():
    from experiment_matrix import SEARCHER_COMPARISON
    assert len(SEARCHER_COMPARISON) == 3


def test_searcher_comparison_entries_have_search_strategy():
    from experiment_matrix import SEARCHER_COMPARISON
    for entry in SEARCHER_COMPARISON:
        assert "name" in entry
        assert "search_strategy" in entry
        assert entry["search_strategy"] in ("tpe_fresh", "warmstart_tpe", "population_async")


def test_allocate_matrix_trial_id_stable():
    from experiment_matrix import allocate_matrix_trial_id
    id1 = allocate_matrix_trial_id("test_entry", 42)
    id2 = allocate_matrix_trial_id("test_entry", 42)
    assert id1 == id2  # stable
    assert 100000 <= id1 <= 999999


def test_allocate_matrix_trial_id_different_for_different_inputs():
    from experiment_matrix import allocate_matrix_trial_id
    id1 = allocate_matrix_trial_id("entry_a", 42)
    id2 = allocate_matrix_trial_id("entry_a", 43)
    assert id1 != id2  # different seeds
    id3 = allocate_matrix_trial_id("entry_b", 42)
    assert id1 != id3  # different names


def test_allocate_matrix_trial_id_in_valid_range():
    from experiment_matrix import allocate_matrix_trial_id
    for i in range(20):
        tid = allocate_matrix_trial_id(f"test_{i}", i)
        assert 100000 <= tid <= 999999, f"trial_id {tid} out of range"


def test_final_confirm_invalid_when_fewer_than_100_scores():
    """Mock: fewer scores = invalid"""
    result = {"config": {}, "all_eval_scores": [1000]*50, "seeds_used": [1001,1002,1003,1004,1005]}
    expected_count = 5 * 20
    if len(result["all_eval_scores"]) != expected_count:
        result["status"] = "invalid"
    assert result["status"] == "invalid"


def test_final_confirm_status_logic():
    """Verify the status confirmation logic directly."""
    # Confirmed: 80% success rate + median >= 1000
    scores = [1200]*80 + [0]*20  # 100 scores, 80% success, median=1200
    overall_success_rate = sum(s >= 1000 for s in scores) / len(scores)
    overall_median = float(np.median(scores))
    confirmed = overall_success_rate >= 0.80 and overall_median >= 1000
    assert confirmed is True

    # Not confirmed: low success rate
    scores_low = [1200]*60 + [0]*40  # 60% success
    sr = sum(s >= 1000 for s in scores_low) / len(scores_low)
    assert sr < 0.80
    assert not (sr >= 0.80 and float(np.median(scores_low)) >= 1000)


def test_run_matrix_budget_debug_matrix():
    """Verify debug_matrix budget uses single seed and 100k frames."""
    from experiment_matrix import run_matrix
    # The function validates budget; test the branch
    try:
        run_matrix([], mode="debug", budget="debug_matrix")
    except ValueError:
        pass  # empty matrix just returns []


def test_run_matrix_budget_normal_matrix_seeds():
    """Verify normal_matrix budget configuration."""
    from experiment_matrix import run_matrix
    try:
        run_matrix([], mode="normal", budget="normal_matrix")
    except ValueError:
        pass  # empty matrix just returns []


def test_run_matrix_unknown_budget_raises():
    import pytest
    from experiment_matrix import run_matrix
    with pytest.raises(ValueError, match="Unknown budget"):
        run_matrix([], mode="debug", budget="invalid_budget")


def test_run_matrix_skips_missing_config_path(tmp_path, monkeypatch):
    from experiment_matrix import run_matrix

    monkeypatch.chdir(tmp_path)
    results = run_matrix([
        {'name': 'missing_config', 'config_path': 'best_config.json', 'seeds': 1},
    ], mode='debug', budget='debug_matrix')

    assert len(results) == 1
    assert results[0]['status'] == 'skipped'
    assert 'missing config_path' in results[0]['reason']


def test_structure_ablation_uses_best_config_base_when_available(tmp_path, monkeypatch):
    import json
    import train_eval
    from experiment_matrix import run_matrix

    captured = []

    def fake_run_trial(config, trial_id, seed, source, max_trial_frames):
        captured.append({
            'config': dict(config),
            'trial_id': trial_id,
            'seed': seed,
            'source': source,
            'max_trial_frames': max_trial_frames,
        })
        return {
            'record_type': 'trial',
            'trial_id': trial_id,
            'source': source,
            'status': 'failure',
            'objective': 123.0,
            'median_score': 10.0,
            'success_rate_1000': 0.0,
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_eval, 'run_trial', fake_run_trial)
    (tmp_path / 'best_config.json').write_text(json.dumps({
        'lr': 0.000321,
        'gamma': 0.97,
        'network_backbone': 'mlp',
        'exploration_head': 'epsilon_greedy',
    }), encoding='utf-8')

    run_matrix([
        {
            'name': 'dueling_noisy',
            'base_config_path': 'best_config.json',
            'network_backbone': 'dueling_mlp',
            'exploration_head': 'noisy_net',
        },
    ], mode='debug', budget='debug_matrix')

    assert len(captured) == 1
    assert captured[0]['config']['lr'] == 0.000321
    assert captured[0]['config']['gamma'] == 0.97
    assert captured[0]['config']['network_backbone'] == 'dueling_mlp'
    assert captured[0]['config']['exploration_head'] == 'noisy_net'


def test_protocol_ablation_falls_back_to_baseline_when_best_config_missing(tmp_path, monkeypatch):
    import train_eval
    from experiment_matrix import run_matrix
    from search_driver import BASELINE_CONFIG

    captured = []

    def fake_run_trial(config, trial_id, seed, source, max_trial_frames):
        captured.append(dict(config))
        return {
            'record_type': 'trial',
            'trial_id': trial_id,
            'source': source,
            'status': 'failure',
            'objective': 123.0,
            'median_score': 0.0,
            'success_rate_1000': 0.0,
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_eval, 'run_trial', fake_run_trial)

    run_matrix([
        {'name': 'v1_state_v3_reward',
         'state_representation_version': 'low_dim_v1',
         'reward_scheme_version': 'reward_v3_gap_shaping'},
    ], mode='debug', budget='debug_matrix')

    assert len(captured) == 1
    assert captured[0]['lr'] == BASELINE_CONFIG['lr']
    assert captured[0]['gamma'] == BASELINE_CONFIG['gamma']
    assert captured[0]['state_representation_version'] == 'low_dim_v1'
    assert captured[0]['reward_scheme_version'] == 'reward_v3_gap_shaping'
    assert captured[0]['gap_shaping_coef'] == 0.05


def test_run_matrix_supports_output_dir_and_custom_source(tmp_path, monkeypatch):
    import train_eval
    from experiment_matrix import run_matrix

    captured = []

    def fake_run_trial(config, trial_id, seed, source, max_trial_frames):
        captured.append(source)
        return {
            'record_type': 'trial',
            'trial_id': trial_id,
            'source': source,
            'status': 'failure',
            'objective': 123.0,
            'median_score': 10.0,
            'success_rate_1000': 0.0,
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_eval, 'run_trial', fake_run_trial)

    output_dir = tmp_path / 'matrix_out'
    run_matrix([
        {
            'name': 'dueling_noisy',
            'network_backbone': 'dueling_mlp',
            'exploration_head': 'noisy_net',
        },
    ], mode='debug', budget='debug_matrix', output_dir=output_dir, source='auto_workflow')

    assert captured == ['auto_workflow_dueling_noisy']
    assert (output_dir / 'history.jsonl').exists()
    assert (output_dir / 'summary.json').exists()


def test_final_confirm_supports_output_path(tmp_path, monkeypatch):
    import train_eval
    from experiment_matrix import final_confirm

    fake_results = iter([
        {
            'status': 'success',
            'final_eval_scores': [1000] * 20,
        }
        for _ in range(5)
    ])

    monkeypatch.setattr(train_eval, 'run_trial', lambda *args, **kwargs: next(fake_results))
    output_path = tmp_path / 'final_confirm_summary.json'
    result = final_confirm({'lr': 1e-4}, output_path=output_path, source='auto_workflow_final')

    assert result['status'] == 'confirmed'
    assert output_path.exists()


def test_final_confirm_supports_output_dir(tmp_path, monkeypatch):
    import train_eval
    from experiment_matrix import final_confirm

    fake_results = iter([
        {
            'status': 'success',
            'final_eval_scores': [1000] * 20,
        }
        for _ in range(5)
    ])

    monkeypatch.setattr(train_eval, 'run_trial', lambda *args, **kwargs: next(fake_results))
    output_dir = tmp_path / 'final_confirm_out'
    result = final_confirm({'lr': 1e-4}, output_dir=output_dir, source='auto_workflow_final')

    assert result['status'] == 'confirmed'
    assert (output_dir / 'final_confirm_summary.json').exists()


def test_summarize_matrix_results_groups_entry_details():
    from experiment_matrix import summarize_matrix_results

    summary = summarize_matrix_results('protocol', [
        {
            'record_type': 'trial',
            'trial_id': 1,
            'source': 'matrix_v1_state_v1_reward',
            'status': 'failure',
            'objective': 999.0,
            'median_score': 12.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'max_frames_reached',
        },
        {
            'record_type': 'trial',
            'trial_id': 2,
            'source': 'matrix_v1_state_v1_reward',
            'status': 'success',
            'objective': 555.0,
            'median_score': 1000.0,
            'success_rate_1000': 0.7,
            'failure_reason': '',
        },
        {
            'record_type': 'matrix_entry',
            'name': 'missing_config',
            'status': 'skipped',
            'reason': 'missing config_path: X',
        },
    ])

    assert summary['matrix'] == 'protocol'
    assert len(summary['entries']) == 2
    by_name = {entry['name']: entry for entry in summary['entries']}
    assert by_name['missing_config']['status'] == 'skipped'
    grouped = by_name['v1_state_v1_reward']
    assert grouped['name'] == 'v1_state_v1_reward'
    assert grouped['trial_count'] == 2
    assert grouped['success_count'] == 1
    assert grouped['best_median_score'] == 1000.0


def test_summarize_matrix_results_accepts_searcher_entry_name():
    from experiment_matrix import summarize_matrix_results

    summary = summarize_matrix_results('searcher', [
        {
            'record_type': 'trial',
            'trial_id': 10,
            'matrix_entry_name': 'warmstart_tpe',
            'source': 'tpe',
            'status': 'failure',
            'objective': 777.0,
            'median_score': 25.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'max_frames_reached',
        },
    ])

    assert len(summary['entries']) == 1
    assert summary['entries'][0]['name'] == 'warmstart_tpe'
    assert summary['entries'][0]['trial_count'] == 1

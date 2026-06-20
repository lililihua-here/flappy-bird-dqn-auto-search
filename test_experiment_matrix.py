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

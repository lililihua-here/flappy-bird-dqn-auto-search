"""Contract tests for reward protocol versions (V3.2)."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


def test_v1_sparse_pipe_reward():
    from reward_protocols import compute_reward_v1_sparse
    assert compute_reward_v1_sparse({'hit': False, 'passed_pipe': True}) == 1.0
    assert compute_reward_v1_sparse({'hit': False, 'passed_pipe': False}) == 0.0
    assert compute_reward_v1_sparse({'hit': True, 'passed_pipe': False}) == -1.0


def test_v1_sparse_hit_overrides_pipe():
    """Death reward should take precedence over pipe reward."""
    from reward_protocols import compute_reward_v1_sparse
    assert compute_reward_v1_sparse({'hit': True, 'passed_pipe': True}) == -1.0


def test_v2_ratio_death():
    from reward_protocols import compute_reward_v2_ratio
    r = compute_reward_v2_ratio({'hit': True, 'passed_pipe': False}, death_ratio=10)
    assert r == -10.0


def test_v2_ratio_alive():
    from reward_protocols import compute_reward_v2_ratio
    r = compute_reward_v2_ratio({'hit': False, 'passed_pipe': False}, alive_ratio=0.001)
    assert r == 0.001


def test_v2_ratio_pipe():
    from reward_protocols import compute_reward_v2_ratio
    r = compute_reward_v2_ratio({'hit': False, 'passed_pipe': True})
    assert r == 1.0


def test_v2_ratio_scale():
    from reward_protocols import compute_reward_v2_ratio
    r = compute_reward_v2_ratio({'hit': False, 'passed_pipe': True}, scale=0.01)
    assert r == 0.01


def test_v2_ratio_clip():
    from reward_protocols import compute_reward_v2_ratio
    r = compute_reward_v2_ratio({'hit': True, 'passed_pipe': False},
                                 death_ratio=100, clip=10)
    assert r == -10.0


def test_v3_gap_shaping_center():
    """Bird at gap center should get positive bonus."""
    from reward_protocols import compute_reward_v3_gap_shaping
    r = compute_reward_v3_gap_shaping(
        {'hit': False, 'passed_pipe': False},
        bird_y=400, gap_center=400, gap_shaping_coef=0.05)
    assert r > 0  # aligned at center = bonus


def test_v3_gap_shaping_edge():
    """Bird at edge of gap should get less bonus."""
    from reward_protocols import compute_reward_v3_gap_shaping
    r_center = compute_reward_v3_gap_shaping(
        {'hit': False, 'passed_pipe': False},
        bird_y=400, gap_center=400, pipe_gap=200, gap_shaping_coef=0.05)
    r_edge = compute_reward_v3_gap_shaping(
        {'hit': False, 'passed_pipe': False},
        bird_y=500, gap_center=400, pipe_gap=200, gap_shaping_coef=0.05)
    assert r_center >= r_edge


def test_v3_gap_shaping_death_overrides():
    """Death should still give negative reward even with gap shaping."""
    from reward_protocols import compute_reward_v3_gap_shaping
    r = compute_reward_v3_gap_shaping(
        {'hit': True, 'passed_pipe': False},
        bird_y=400, gap_center=400, death_ratio=5, gap_shaping_coef=0.1)
    assert r < 0


def test_v3_gap_shaping_zero_coef():
    """With gap_shaping_coef=0, should behave like v2_ratio."""
    from reward_protocols import compute_reward_v3_gap_shaping, compute_reward_v2_ratio
    r_v3 = compute_reward_v3_gap_shaping(
        {'hit': False, 'passed_pipe': False},
        bird_y=400, gap_center=300, alive_ratio=0.01, gap_shaping_coef=0.0)
    r_v2 = compute_reward_v2_ratio(
        {'hit': False, 'passed_pipe': False}, alive_ratio=0.01)
    assert r_v3 == r_v2


def test_v3_gap_shaping_clip_with_shaping():
    from reward_protocols import compute_reward_v3_gap_shaping
    r = compute_reward_v3_gap_shaping(
        {'hit': False, 'passed_pipe': False},
        bird_y=400, gap_center=400, gap_shaping_coef=0.05, clip=0.03)
    assert r <= 0.03

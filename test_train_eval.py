"""Contract tests for train_eval module."""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# run_trial baseline smoke test
# ============================================================================
def test_run_trial_baseline_smoke():
    import tempfile
    from train_eval import run_trial
    # Use a minimal config that matches the baseline pattern
    test_config = {
        'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64, 'buffer_sz': 50000,
        'hidden': [128, 64],
        'double_q': True, 'n_step': 1, 'frame_skip': 1,
        'eps_start': 0.05, 'eps_end': 0.005, 'eps_decay_decision_steps': 50000,
        'replay_start_size': 500, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'torch_optimizer': 'Adam', 'loss_type': 'Huber', 'grad_clip_norm': 5,
        'reward_pipe': 1.0, 'reward_death': -1.0, 'reward_alive': 0.0,
        'reward_clip': None, 'reward_scale': 1.0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_trial(
            config=test_config,
            trial_id=0,
            seed=42,
            source='baseline',
            max_trial_frames=3000,
            eval_interval_frames=1000,
            eval_episodes=2,
            candidate_verify_episodes=3,
            checkpoint_dir=tmpdir,
        )
        required = [
            'trial_id', 'config', 'seed', 'source', 'status', 'objective',
            'train_raw_env_frames', 'total_raw_env_frames', 'eval_raw_env_frames',
            'decision_steps', 'episodes', 'best_train_score',
            'best_eval_score', 'duration_sec', 'checkpoint_path',
            'checkpoint_sha256', 'checkpoint_format_version',
            'environment_version', 'state_representation_version',
        ]
        for k in required:
            assert k in result, f"Missing: {k}"
        assert result['status'] in ('success', 'failure', 'pruned')
        assert result['objective'] > 0
        assert result['train_raw_env_frames'] > 0
        assert result['total_raw_env_frames'] == result['train_raw_env_frames'] + result['eval_raw_env_frames']
        assert os.path.exists(result['checkpoint_path'])
        assert result['environment_version'] == 'fixed_env_v1'
        assert result['state_representation_version'] == 'low_dim_v1'


# ============================================================================
# greedy_eval tests
# ============================================================================
def test_greedy_eval_returns_expected_keys_with_independent_env():
    """P0-2: eval creates its own env and returns correct keys."""
    import torch
    from dqn_agent import DQNAgent
    from flappy_bird_env import FlappyBirdEnv
    from replay_buffer import StateEncoder
    from train_eval import greedy_eval

    encoder = StateEncoder()
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.0, 'eps_end': 0.0,
        'eps_decay_decision_steps': 10000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5, 'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    result = greedy_eval(
        agent=agent, env_factory=FlappyBirdEnv, encoder=encoder,
        n_episodes=3, eval_seed=42, max_raw_frames_per_ep=120000,
    )
    required = ['scores', 'mean', 'median', 'max', 'min', 'success_rate_1000', 'raw_env_frames']
    for k in required:
        assert k in result, f"Missing: {k}"
    assert len(result['scores']) == 3


def test_greedy_eval_score_from_env_score():
    """P1-9: eval scores should come from env.score, not reward inference."""
    import torch
    from dqn_agent import DQNAgent
    from flappy_bird_env import FlappyBirdEnv
    from replay_buffer import StateEncoder
    from train_eval import greedy_eval

    encoder = StateEncoder()
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.0, 'eps_end': 0.0,
        'eps_decay_decision_steps': 10000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5, 'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    result = greedy_eval(
        agent=agent, env_factory=FlappyBirdEnv, encoder=encoder,
        n_episodes=3, eval_seed=42, max_raw_frames_per_ep=5000,
    )
    for s in result['scores']:
        assert isinstance(s, int) and s >= 0


# ============================================================================
# is_stable_success tests
# ============================================================================
def test_is_stable_success_requires_rate_and_median():
    """Stable success: >=70% of episodes >=1000 AND median >=1000."""
    from train_eval import is_stable_success

    # 14/20 >= 1000, median=1000 -> pass
    assert is_stable_success({
        'scores': [1000]*14 + [0]*6,
        'success_rate_1000': 0.70,
        'median': 1000,
    }) is True

    # 13/20 >= 1000 -> fail (rate < 0.70)
    assert is_stable_success({
        'scores': [1000]*13 + [0]*7,
        'success_rate_1000': 0.65,
        'median': 1000,
    }) is False

    # 14/20 but median=999 -> fail
    assert is_stable_success({
        'scores': [1000]*14 + [998]*6,
        'success_rate_1000': 0.70,
        'median': 999,
    }) is False

    # All pass -> pass
    assert is_stable_success({
        'scores': [1200]*20,
        'success_rate_1000': 1.0,
        'median': 1200,
    }) is True


def test_is_stable_success_honors_custom_threshold():
    """Custom threshold must be computed from scores, not hardcoded 1000-rate."""
    from train_eval import is_stable_success

    result = {
        'scores': [1100] * 20,
        'success_rate_1000': 1.0,
        'median': 1100,
    }
    assert is_stable_success(result, threshold=1200, min_rate=0.70, min_median=1000) is False


# ============================================================================
# run_trial n-step smoke test (Stage B)
# ============================================================================
def test_run_trial_with_nstep_3():
    from train_eval import run_trial
    from search_driver import BASELINE_CONFIG
    import tempfile
    config = dict(BASELINE_CONFIG)
    config.update({'n_step': 3, 'replay_start_size': 200})
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_trial(config, trial_id=99, seed=42, source='test',
                           max_trial_frames=2000, eval_interval_frames=500,
                           eval_episodes=2, candidate_verify_episodes=3,
                           checkpoint_dir=tmpdir)
        assert result['config']['n_step'] == 3
        assert result['n_step'] == 3
        assert result['status'] in ('success', 'failure')


# ============================================================================
# compute_objective tests
# ============================================================================
def test_compute_objective_success_equals_train_frames():
    from train_eval import compute_objective
    obj = compute_objective(
        success=True,
        train_raw_env_frames=123456,
        max_trial_frames=1_000_000,
        best_eval_score=1000,
    )
    assert obj == 123456


def test_compute_objective_failure_is_worse_than_any_success():
    from train_eval import compute_objective
    failed = compute_objective(
        success=False,
        train_raw_env_frames=1_000_000,
        max_trial_frames=1_000_000,
        best_eval_score=500,
    )
    successful = compute_objective(
        success=True,
        train_raw_env_frames=900_000,
        max_trial_frames=1_000_000,
        best_eval_score=1000,
    )
    assert failed > successful


def test_compute_objective_failure_prefers_better_eval_score():
    from train_eval import compute_objective
    bad = compute_objective(
        success=False,
        train_raw_env_frames=1_000_000,
        max_trial_frames=1_000_000,
        best_eval_score=0,
    )
    good = compute_objective(
        success=False,
        train_raw_env_frames=1_000_000,
        max_trial_frames=1_000_000,
        best_eval_score=500,
    )
    assert bad > good


def test_compute_objective_failure_clamped_at_1000():
    from train_eval import compute_objective
    above = compute_objective(
        success=False,
        train_raw_env_frames=1_000_000,
        max_trial_frames=1_000_000,
        best_eval_score=1200,
    )
    at_limit = compute_objective(
        success=False,
        train_raw_env_frames=1_000_000,
        max_trial_frames=1_000_000,
        best_eval_score=1000,
    )
    # Higher eval score -> better (lower) objective
    assert above < at_limit


def test_compute_objective_clamps_invalid_failure_score():
    from train_eval import compute_objective
    nan_obj = compute_objective(False, 1_000_000, 1_000_000, best_eval_score=float('nan'))
    neg_obj = compute_objective(False, 1_000_000, 1_000_000, best_eval_score=-10)
    zero_obj = compute_objective(False, 1_000_000, 1_000_000, best_eval_score=0)
    assert nan_obj == zero_obj
    assert neg_obj == zero_obj


# ============================================================================
# P0-2: Eval isolation contract
# ============================================================================
def test_contract_greedy_eval_does_not_mutate_training_env():
    """greedy_eval() must create its own env, not mutate the training env."""
    import torch
    from dqn_agent import DQNAgent
    from flappy_bird_env import FlappyBirdEnv
    from replay_buffer import StateEncoder
    from train_eval import greedy_eval

    train_env = FlappyBirdEnv(seed=42)
    train_env.reset()
    train_state_before = train_env._get_state()
    train_total_before = train_env.total_raw_env_frames

    encoder = StateEncoder()
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.0, 'eps_end': 0.0,
        'eps_decay_decision_steps': 10000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1, 'frame_skip': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')

    result = greedy_eval(
        agent=agent, env_factory=FlappyBirdEnv, encoder=encoder,
        n_episodes=3, eval_seed=999, max_raw_frames_per_ep=5000,
    )

    # Training env must be untouched
    assert train_env.total_raw_env_frames == train_total_before
    train_state_after = train_env._get_state()
    for k in train_state_before:
        assert train_state_after[k] == train_state_before[k], f"train env key '{k}' mutated by eval"

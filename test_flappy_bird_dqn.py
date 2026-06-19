"""Contract tests for Flappy Bird DQN Auto-Search System"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# P0-1: Frame counter contract — total frames must survive reset()
# ============================================================================
def test_env_total_frames_not_reset_by_episode_reset():
    """reset() must not clear the trial-level total_raw_env_frames counter."""
    from flappy_bird_dqn_auto_search import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    for _ in range(10):
        env.step(0)
    before_total = env.total_raw_env_frames
    before_ep = env.episode_raw_env_frames
    assert before_total == 10
    assert before_ep == 10
    env.reset()
    assert env.total_raw_env_frames == before_total, "total_raw_env_frames must survive reset()"
    assert env.episode_raw_env_frames == 0, "episode_raw_env_frames must be 0 after reset()"


# ============================================================================
# P0-2: Eval isolation contract
# ============================================================================
def test_contract_greedy_eval_does_not_mutate_training_env():
    """greedy_eval() must create its own env, not mutate the training env."""
    from flappy_bird_dqn_auto_search import FlappyBirdEnv, StateEncoder, DQNAgent, greedy_eval
    import torch

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


# ============================================================================
# P1-4: Stable success semantics
# ============================================================================
def test_is_stable_success_requires_rate_and_median():
    """Stable success: >=70% of episodes >=1000 AND median >=1000."""
    from flappy_bird_dqn_auto_search import is_stable_success

    # 14/20 >= 1000, median=1000 → pass
    assert is_stable_success({
        'scores': [1000]*14 + [0]*6,
        'success_rate_1000': 0.70,
        'median': 1000,
    }) is True

    # 13/20 >= 1000 → fail (rate < 0.70)
    assert is_stable_success({
        'scores': [1000]*13 + [0]*7,
        'success_rate_1000': 0.65,
        'median': 1000,
    }) is False

    # 14/20 but median=999 → fail
    assert is_stable_success({
        'scores': [1000]*14 + [998]*6,
        'success_rate_1000': 0.70,
        'median': 999,
    }) is False

    # All pass → pass
    assert is_stable_success({
        'scores': [1200]*20,
        'success_rate_1000': 1.0,
        'median': 1200,
    }) is True

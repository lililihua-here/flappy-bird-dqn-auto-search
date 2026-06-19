"""Contract tests for Flappy Bird DQN Auto-Search System"""
import numpy as np
import random
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


# ============================================================================
# Task 2: Environment tests
# ============================================================================
def test_env_reset_returns_valid_state():
    from flappy_bird_dqn_auto_search import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    state = env.reset()
    required_keys = [
        'bird_y', 'bird_velocity', 'pipe_x',
        'pipe_gap_top', 'pipe_gap_bottom', 'pipe_gap_center',
    ]
    for k in required_keys:
        assert k in state, f"Missing key: {k}"
    assert 0 <= state['bird_y'] <= 800
    assert state['pipe_x'] > 0


def test_env_step_advances_both_counters():
    from flappy_bird_dqn_auto_search import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    env.step(0)
    assert env.total_raw_env_frames == 1
    assert env.episode_raw_env_frames == 1
    env.step(0)
    assert env.total_raw_env_frames == 2
    assert env.episode_raw_env_frames == 2


def test_env_step_returns_tuple():
    from flappy_bird_dqn_auto_search import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    result = env.step(0)
    assert isinstance(result, tuple) and len(result) == 3
    state, reward, done = result
    assert isinstance(state, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)


def test_env_pipe_reward_on_pass():
    from flappy_bird_dqn_auto_search import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    env.bird_y = env.pipe_gap_center
    env.bird_velocity = 0.0
    env._scored_current_pipe = False
    env.pipe_x = env.BIRD_X - env.PIPE_WIDTH - 1
    _, reward, _ = env.step(0)
    assert reward == 1.0, f"Expected pipe_reward=1.0, got {reward}"


def test_env_death_reward_on_collision():
    from flappy_bird_dqn_auto_search import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    env.bird_y = 5
    env.bird_velocity = 0.0
    for _ in range(10):
        _, reward, done = env.step(0)
        if done:
            break
    assert reward == -1.0
    assert done is True


# ============================================================================
# Task 3: StateEncoder tests
# ============================================================================
def test_state_encoder_output_shape():
    """encode() returns a 1D array of length state_dim."""
    from flappy_bird_dqn_auto_search import StateEncoder
    encoder = StateEncoder()
    state = {
        'bird_y': 400.0, 'bird_velocity': 2.0, 'pipe_x': 300.0,
        'pipe_gap_top': 100.0, 'pipe_gap_bottom': 500.0, 'pipe_gap_center': 300.0,
    }
    vec = encoder.encode(state)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (7,)
    assert encoder.state_dim == 7


def test_state_encoder_normalization_range():
    """All features should be roughly in [-1, 1] range after normalization."""
    from flappy_bird_dqn_auto_search import StateEncoder
    encoder = StateEncoder()
    test_states = [
        {'bird_y': 400.0, 'bird_velocity': 0.0, 'pipe_x': 300.0,
         'pipe_gap_top': 100.0, 'pipe_gap_bottom': 500.0, 'pipe_gap_center': 300.0},
        {'bird_y': 100.0, 'bird_velocity': -10.0, 'pipe_x': 600.0,
         'pipe_gap_top': 50.0, 'pipe_gap_bottom': 450.0, 'pipe_gap_center': 250.0},
        {'bird_y': 700.0, 'bird_velocity': 10.0, 'pipe_x': 50.0,
         'pipe_gap_top': 300.0, 'pipe_gap_bottom': 700.0, 'pipe_gap_center': 500.0},
    ]
    for state in test_states:
        vec = encoder.encode(state)
        assert np.all(np.abs(vec) < 2.0), f"Values out of range: {vec}"


def test_state_encoder_deterministic():
    """Same state → same encoding."""
    from flappy_bird_dqn_auto_search import StateEncoder
    encoder = StateEncoder()
    state = {
        'bird_y': 333.0, 'bird_velocity': -3.0, 'pipe_x': 444.0,
        'pipe_gap_top': 150.0, 'pipe_gap_bottom': 550.0, 'pipe_gap_center': 350.0,
    }
    v1 = encoder.encode(state)
    v2 = encoder.encode(state)
    assert np.array_equal(v1, v2)


# ============================================================================
# Task 4: ReplayBuffer tests
# ============================================================================
def test_replay_buffer_add_and_len():
    from flappy_bird_dqn_auto_search import ReplayBuffer
    rb = ReplayBuffer(capacity=3)
    s = np.zeros(7, dtype=np.float32)
    rb.add(s, 0, 0.0, s, False)
    rb.add(s, 1, 1.0, s, True)
    assert len(rb) == 2


def test_replay_buffer_capacity_evicts_oldest():
    from flappy_bird_dqn_auto_search import ReplayBuffer
    rb = ReplayBuffer(capacity=2)
    for i in range(3):
        s = np.full(7, i, dtype=np.float32)
        rb.add(s, i % 2, float(i), s, False)
    assert len(rb) == 2
    states, actions, rewards, next_states, dones = rb.sample(2)
    assert 0.0 not in rewards


def test_replay_buffer_can_sample():
    from flappy_bird_dqn_auto_search import ReplayBuffer
    rb = ReplayBuffer(capacity=10)
    s = np.zeros(7, dtype=np.float32)
    for _ in range(3):
        rb.add(s, 0, 0.0, s, False)
    assert rb.can_sample(2) is True
    assert rb.can_sample(4) is False


def test_replay_buffer_sample_shapes():
    from flappy_bird_dqn_auto_search import ReplayBuffer
    rb = ReplayBuffer(capacity=10)
    for i in range(5):
        s = np.full(7, i, dtype=np.float32)
        ns = np.full(7, i + 1, dtype=np.float32)
        rb.add(s, i % 2, float(i), ns, i == 4)
    states, actions, rewards, next_states, dones = rb.sample(4)
    assert states.shape == (4, 7)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 7)
    assert dones.shape == (4,)


# ============================================================================
# Task 5: DQN network tests
# ============================================================================
def test_dqn_forward_shape():
    import torch
    from flappy_bird_dqn_auto_search import DQN
    net = DQN(state_dim=7, hidden=[64, 32], n_actions=2)
    x = torch.randn(4, 7)
    y = net(x)
    assert tuple(y.shape) == (4, 2)


def test_dqn_uses_requested_hidden_layers():
    import torch.nn as nn
    from flappy_bird_dqn_auto_search import DQN
    net = DQN(state_dim=7, hidden=[128, 64], n_actions=2)
    linears = [m for m in net.net if isinstance(m, nn.Linear)]
    assert [layer.in_features for layer in linears] == [7, 128, 64]
    assert [layer.out_features for layer in linears] == [128, 64, 2]


# ============================================================================
# Task 6: DQNAgent tests
# ============================================================================
def test_dqn_agent_act_greedy():
    import torch
    from flappy_bird_dqn_auto_search import DQNAgent
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.0, 'eps_end': 0.0,
        'eps_decay_decision_steps': 10000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    state = np.array([0.5, 0.0, 0.8, 0.2, 0.6, 0.0, 0.3], dtype=np.float32)
    a1 = agent.act(state, training=False)
    a2 = agent.act(state, training=False)
    assert a1 == a2


def test_dqn_agent_epsilon_decay():
    import torch
    from flappy_bird_dqn_auto_search import DQNAgent
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.10, 'eps_end': 0.01,
        'eps_decay_decision_steps': 1000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    assert abs(agent.epsilon - 0.10) < 1e-6
    agent.decision_steps = 500
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.055) < 1e-3
    agent.decision_steps = 1000
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.01) < 1e-6


def test_dqn_agent_eps_frames_backward_compat():
    """P1-5: old 'eps_frames' key still works as fallback."""
    import torch
    from flappy_bird_dqn_auto_search import DQNAgent
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.10, 'eps_end': 0.01,
        'eps_frames': 500,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    agent.decision_steps = 500
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.01) < 1e-6


def test_dqn_agent_config_assertions():
    """P1-8: MVP fixed params are asserted on init."""
    import torch
    from flappy_bird_dqn_auto_search import DQNAgent
    import pytest
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.05, 'eps_end': 0.005,
        'eps_decay_decision_steps': 10000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    bad = dict(config, target_update_mode='hard')
    with pytest.raises(AssertionError):
        DQNAgent(bad, state_dim=7, n_actions=2, device='cpu')


def test_dqn_agent_train_returns_loss():
    import torch
    from flappy_bird_dqn_auto_search import DQNAgent
    config = {
        'hidden': [64, 32], 'lr': 1e-3, 'gamma': 0.99, 'batch_sz': 16,
        'buffer_sz': 1000, 'eps_start': 0.05, 'eps_end': 0.005,
        'eps_decay_decision_steps': 50000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    for _ in range(200):
        s = np.random.randn(7).astype(np.float32)
        ns = np.random.randn(7).astype(np.float32)
        agent.buffer.add(s, random.randint(0, 1), random.random(), ns, random.random() < 0.1)
    loss = agent.train()
    assert isinstance(loss, float) and loss > 0


# ============================================================================
# Task 7: Greedy Evaluation + is_stable_success
# ============================================================================
def test_greedy_eval_returns_expected_keys_with_independent_env():
    """P0-2: eval creates its own env and returns correct keys."""
    import torch
    from flappy_bird_dqn_auto_search import FlappyBirdEnv, StateEncoder, DQNAgent, greedy_eval

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
    from flappy_bird_dqn_auto_search import FlappyBirdEnv, StateEncoder, DQNAgent, greedy_eval

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
# Task 8: Training Loop tests
# ============================================================================
def test_run_trial_baseline_smoke():
    import torch
    from flappy_bird_dqn_auto_search import run_trial, BASELINE_CONFIG

    test_config = dict(BASELINE_CONFIG)
    test_config['replay_start_size'] = 500

    result = run_trial(
        config=test_config,
        trial_id=0,
        seed=42,
        source='baseline',
        max_trial_frames=3000,
        eval_interval_frames=1000,
        eval_episodes=2,
        candidate_verify_episodes=3,
    )
    required = [
        'trial_id', 'config', 'seed', 'source', 'status', 'objective',
        'train_raw_env_frames', 'total_raw_env_frames', 'eval_raw_env_frames',
        'decision_steps', 'episodes', 'best_train_score',
        'best_eval_score', 'duration_sec',
    ]
    for k in required:
        assert k in result, f"Missing: {k}"
    assert result['status'] in ('success', 'failure', 'pruned')
    assert result['train_raw_env_frames'] > 0
    assert result['total_raw_env_frames'] == result['train_raw_env_frames'] + result['eval_raw_env_frames']


# ============================================================================
# Task 9: Objective tests
# ============================================================================
def test_compute_objective_success_equals_train_frames():
    from flappy_bird_dqn_auto_search import compute_objective
    obj = compute_objective(
        success=True,
        train_raw_env_frames=123456,
        max_trial_frames=1_000_000,
        best_eval_score=1000,
    )
    assert obj == 123456


def test_compute_objective_failure_is_worse_than_any_success():
    from flappy_bird_dqn_auto_search import compute_objective
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
    from flappy_bird_dqn_auto_search import compute_objective
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
    from flappy_bird_dqn_auto_search import compute_objective
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
    assert above == at_limit

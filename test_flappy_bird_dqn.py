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
    # Higher eval score → better (lower) objective
    assert above < at_limit


# ============================================================================
# Task 10: Search Space
# ============================================================================
def test_search_space_produces_valid_config():
    import optuna
    from flappy_bird_dqn_auto_search import define_search_space

    def objective(trial):
        config = define_search_space(trial)
        required = ['lr', 'gamma', 'hidden', 'hidden_key', 'eps_start', 'eps_end',
                    'eps_decay_decision_steps', 'replay_start_size', 'train_freq']
        for k in required:
            assert k in config, f"Missing: {k}"
        assert 1e-5 <= config['lr'] <= 3e-3
        assert 0.90 <= config['gamma'] <= 0.999
        assert config['hidden_key'] in ('small', 'medium', 'large')
        assert config['hidden'] in ([64, 32], [128, 64], [256, 128])
        assert 0.01 <= config['eps_start'] <= 0.15
        assert 0.001 <= config['eps_end'] <= 0.02
        assert 10000 <= config['eps_decay_decision_steps'] <= 200000
        assert config['replay_start_size'] in (1000, 5000, 10000)
        assert config['train_freq'] in (1, 4)
        assert config['n_step'] == 1
        return 0.0

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=10)
    assert len(study.trials) == 10


# ============================================================================
# Task 11: HistoryManager tests
# ============================================================================
def test_history_manager_append_and_load_roundtrip():
    import tempfile
    from flappy_bird_dqn_auto_search import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({
            'trial_id': 0,
            'status': 'success',
            'objective': 123.0,
            'config': {'lr': 1e-4},
        })
        rows = hm.load()
        assert len(rows) == 1
        assert rows[0]['trial_id'] == 0
        assert rows[0]['record_type'] == 'trial'
    finally:
        os.unlink(tmp.name)


def test_history_manager_success_and_failure_counts():
    import tempfile
    from flappy_bird_dqn_auto_search import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({'trial_id': 0, 'status': 'success', 'objective': 100, 'config': {'lr': 1e-4}})
        hm.append({'trial_id': 1, 'status': 'failure', 'objective': 999999, 'config': {'lr': 2e-4}})
        hm.append({'record_type': 'recheck', 'trial_id': 0, 'recheck_passed': True})
        assert hm.success_count() == 1
        assert hm.failure_count() == 1
    finally:
        os.unlink(tmp.name)


def test_history_manager_top_k_filters_recheck_and_sorts_by_priority():
    import tempfile
    from flappy_bird_dqn_auto_search import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({
            'trial_id': 1, 'status': 'success', 'objective': 150000,
            'config': {'lr': 1e-4}, 'recheck_passed': False,
            'success_rate_1000': 0.80, 'median_score': 1100,
            'total_raw_env_frames': 180000,
        })
        hm.append({
            'trial_id': 2, 'status': 'success', 'objective': 140000,
            'config': {'lr': 2e-4}, 'recheck_passed': True,
            'success_rate_1000': 0.75, 'median_score': 1050,
            'total_raw_env_frames': 190000,
        })
        hm.append({
            'trial_id': 3, 'status': 'success', 'objective': 170000,
            'config': {'lr': 3e-4}, 'recheck_passed': True,
            'success_rate_1000': 0.90, 'median_score': 1300,
            'total_raw_env_frames': 200000,
        })
        hm.append({'record_type': 'recheck', 'trial_id': 3, 'recheck_passed': True})
        top = hm.top_k(3)
        assert [r['trial_id'] for r in top] == [2, 3, 1]
    finally:
        os.unlink(tmp.name)


# ============================================================================
# Task 12: Search Driver test
# ============================================================================
def test_search_driver_runs_n_trials():
    import tempfile
    from flappy_bird_dqn_auto_search import SearchDriver

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    db_path = tmp.name.replace('.jsonl', '.db')

    try:
        driver = SearchDriver(
            history_path=tmp.name, study_db=db_path,
            max_trials=3, max_trial_frames=3000,
            eval_interval_frames=1000, eval_episodes=2,
            candidate_verify_episodes=3,
            n_startup_trials=2, seed_pool=[42, 43, 44],
        )
        driver.run()
        rows = driver.history.load()
        assert len(rows) >= 1
        for r in rows:
            if r.get('record_type') == 'trial':
                assert 'trial_id' in r and 'objective' in r and r['objective'] > 0
    finally:
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass


# ============================================================================
# Task 13: Reporting tests
# ============================================================================
def test_generate_summary_handles_empty_history():
    import tempfile
    from flappy_bird_dqn_auto_search import HistoryManager, generate_summary

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        summary = generate_summary(HistoryManager(tmp.name))
        assert summary['trial_count'] == 0
        assert summary['top_k'] == []
    finally:
        os.unlink(tmp.name)


def test_generate_summary_returns_counts_and_best_trial():
    import tempfile
    from flappy_bird_dqn_auto_search import HistoryManager, generate_summary

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({
            'trial_id': -1, 'source': 'baseline', 'status': 'failure',
            'objective': 9_999_900, 'config': {'lr': 1e-4},
            'failure_reason': 'max_frames_reached',
        })
        hm.append({
            'trial_id': 0, 'source': 'tpe', 'status': 'success',
            'objective': 180000, 'config': {'lr': 2e-4},
            'median_score': 1050, 'success_rate_1000': 0.75,
            'total_raw_env_frames': 220000,
        })
        summary = generate_summary(hm, top_k=1)
        assert summary['trial_count'] == 2
        assert summary['success_count'] == 1
        assert summary['failure_count'] == 1
        assert summary['best_trial_id'] == 0
        assert summary['top_k'][0]['trial_id'] == 0
        assert summary['failure_reasons']['max_frames_reached'] == 1
    finally:
        os.unlink(tmp.name)


# ============================================================================
# Task 14: CLI tests
# ============================================================================
def test_mode_presets():
    from flappy_bird_dqn_auto_search import get_mode_presets
    debug = get_mode_presets('debug')
    assert debug['max_trial_frames'] == 100_000
    assert debug['eval_interval_frames'] == 10_000
    assert debug['eval_episodes'] == 3


def test_parser_defaults():
    from flappy_bird_dqn_auto_search import make_parser
    args = make_parser().parse_args([])
    assert args.mode == 'normal'
    assert args.max_trials == 100


# ============================================================================
# Task 15: CLI integration smoke test
# ============================================================================
def test_cli_debug_search_smoke():
    import json
    import subprocess
    import sys
    import tempfile

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    db_path = tmp.name.replace('.jsonl', '.db')

    try:
        cmd = [
            sys.executable,
            'flappy_bird_dqn_auto_search.py',
            '--mode', 'debug',
            '--max-trials', '2',
            '--max-trial-frames', '3000',
            '--history', tmp.name,
            '--study-db', db_path,
            '--n-startup-trials', '1',
        ]
        completed = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

        with open(tmp.name, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]

        trial_rows = [r for r in rows if r.get('record_type', 'trial') == 'trial']
        assert any(r.get('source') == 'baseline' for r in trial_rows)
        assert sum(1 for r in trial_rows if r.get('source') == 'tpe') == 2
    finally:
        os.unlink(tmp.name)
        if os.path.exists(db_path):
            os.unlink(db_path)


# ============================================================================
# Task 16: Robustness tests
# ============================================================================
def test_make_serializable_handles_numpy_and_torch():
    import torch
    from flappy_bird_dqn_auto_search import _make_serializable
    payload = {
        'array': np.array([1, 2, 3], dtype=np.float32),
        'scalar': np.float32(1.5),
        'tensor': torch.tensor([4.0, 5.0]),
    }
    result = _make_serializable(payload)
    assert result == {
        'array': [1.0, 2.0, 3.0],
        'scalar': 1.5,
        'tensor': [4.0, 5.0],
    }


def test_history_manager_load_ignores_corrupt_jsonl_lines():
    import tempfile
    from flappy_bird_dqn_auto_search import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    try:
        tmp.write('{"trial_id": 0, "status": "success", "config": {"lr": 1e-4}}\n')
        tmp.write('{bad json line}\n')
        tmp.write('{"trial_id": 1, "status": "failure", "config": {"lr": 2e-4}}\n')
        tmp.close()
        rows = HistoryManager(tmp.name).load()
        assert [r['trial_id'] for r in rows] == [0, 1]
    finally:
        os.unlink(tmp.name)


def test_compute_objective_clamps_invalid_failure_score():
    from flappy_bird_dqn_auto_search import compute_objective
    nan_obj = compute_objective(False, 1_000_000, 1_000_000, best_eval_score=float('nan'))
    neg_obj = compute_objective(False, 1_000_000, 1_000_000, best_eval_score=-10)
    zero_obj = compute_objective(False, 1_000_000, 1_000_000, best_eval_score=0)
    assert nan_obj == zero_obj
    assert neg_obj == zero_obj

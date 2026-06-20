"""Contract tests for StateEncoder and ReplayBuffer."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from replay_buffer import StateEncoder, ReplayBuffer, NStepReplayBuffer


# ============================================================================
# StateEncoder tests
# ============================================================================
def test_state_encoder_output_shape():
    """encode() returns a 1D array of length state_dim."""
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


def test_state_encoder_feature_contract():
    """Feature positions must match the documented 7-D contract."""
    from flappy_bird_env import FlappyBirdEnv
    encoder = StateEncoder()
    state = {
        'bird_y': 500.0,
        'bird_velocity': -2.0,
        'pipe_x': 300.0,
        'pipe_gap_top': 150.0,
        'pipe_gap_bottom': 550.0,
        'pipe_gap_center': 350.0,
    }
    vec = encoder.encode(state)
    expected = np.array([
        500.0 / FlappyBirdEnv.SCREEN_HEIGHT,
        -2.0 / FlappyBirdEnv.MAX_FALL_SPEED,
        (300.0 - FlappyBirdEnv.BIRD_X) / FlappyBirdEnv.SCREEN_WIDTH,
        150.0 / FlappyBirdEnv.SCREEN_HEIGHT,
        550.0 / FlappyBirdEnv.SCREEN_HEIGHT,
        350.0 / FlappyBirdEnv.SCREEN_HEIGHT,
        (500.0 - 350.0) / FlappyBirdEnv.SCREEN_HEIGHT,
    ], dtype=np.float32)
    assert np.allclose(vec, expected)


def test_state_encoder_deterministic():
    """Same state → same encoding."""
    encoder = StateEncoder()
    state = {
        'bird_y': 333.0, 'bird_velocity': -3.0, 'pipe_x': 444.0,
        'pipe_gap_top': 150.0, 'pipe_gap_bottom': 550.0, 'pipe_gap_center': 350.0,
    }
    v1 = encoder.encode(state)
    v2 = encoder.encode(state)
    assert np.array_equal(v1, v2)


# ============================================================================
# ReplayBuffer tests
# ============================================================================
def test_replay_buffer_add_and_len():
    rb = ReplayBuffer(capacity=3)
    s = np.zeros(7, dtype=np.float32)
    rb.add(s, 0, 0.0, s, False)
    rb.add(s, 1, 1.0, s, True)
    assert len(rb) == 2


def test_replay_buffer_capacity_evicts_oldest():
    rb = ReplayBuffer(capacity=2)
    for i in range(3):
        s = np.full(7, i, dtype=np.float32)
        rb.add(s, i % 2, float(i), s, False)
    assert len(rb) == 2
    states, actions, rewards, next_states, dones = rb.sample(2)
    assert 0.0 not in rewards


def test_replay_buffer_can_sample():
    rb = ReplayBuffer(capacity=10)
    s = np.zeros(7, dtype=np.float32)
    for _ in range(3):
        rb.add(s, 0, 0.0, s, False)
    assert rb.can_sample(2) is True
    assert rb.can_sample(4) is False


def test_replay_buffer_sample_shapes():
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
# NStepReplayBuffer tests (Stage B)
# ============================================================================
def test_nstep_buffer_stores_n_step_transitions():
    buf = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.99)
    s = np.zeros(7, dtype=np.float32)
    for _ in range(20):
        buf.add(s, 0, 1.0, s, False)
    assert len(buf) >= 1


def test_nstep_buffer_truncates_on_done():
    buf = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.99)
    s = np.zeros(7, dtype=np.float32)
    buf.add(s, 0, 0.0, s, False)
    buf.add(s, 0, 0.0, s, False)
    buf.add(s, 0, 1.0, s, True)
    assert len(buf) >= 1


def test_nstep_buffer_does_not_cross_episode():
    buf = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.99)
    s0 = np.zeros(7, dtype=np.float32)
    s1 = np.ones(7, dtype=np.float32)
    buf.add(s0, 0, 1.0, s0, True)
    # Add more steps — they should NOT mix with the done episode
    for _ in range(6):
        buf.add(s1, 0, 0.0, s1, False)
    assert buf.can_sample(1)


def test_nstep_buffer_sample_shapes():
    buf = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.99)
    s = np.zeros(7, dtype=np.float32)
    for _ in range(30):
        buf.add(s, 0, 1.0, s, False)
    states, actions, n_returns, next_states, dones, gamma_powers, actual_ns = buf.sample(16)
    assert states.shape == (16, 7)
    assert n_returns.shape == (16,)
    assert gamma_powers.shape == (16,)
    assert actual_ns.shape == (16,)

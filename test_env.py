"""Contract tests for FlappyBirdEnv — extracted from test_flappy_bird_dqn.py."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# P0-1: Frame counter contract — total frames must survive reset()
# ============================================================================
def test_env_total_frames_not_reset_by_episode_reset():
    """reset() must not clear the trial-level total_raw_env_frames counter."""
    from flappy_bird_env import FlappyBirdEnv
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
# Task 2: Environment tests
# ============================================================================
def test_env_reset_returns_valid_state():
    from flappy_bird_env import FlappyBirdEnv
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
    from flappy_bird_env import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    env.step(0)
    assert env.total_raw_env_frames == 1
    assert env.episode_raw_env_frames == 1
    env.step(0)
    assert env.total_raw_env_frames == 2
    assert env.episode_raw_env_frames == 2


def test_env_step_returns_tuple():
    from flappy_bird_env import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    result = env.step(0)
    assert isinstance(result, tuple) and len(result) == 3
    state, reward, done = result
    assert isinstance(state, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)


def test_env_pipe_reward_on_pass():
    from flappy_bird_env import FlappyBirdEnv
    env = FlappyBirdEnv(seed=42)
    env.reset()
    env.bird_y = env.pipe_gap_center
    env.bird_velocity = 0.0
    env._scored_current_pipe = False
    env.pipe_x = env.BIRD_X - env.PIPE_WIDTH - 1
    _, reward, _ = env.step(0)
    assert reward == 1.0, f"Expected pipe_reward=1.0, got {reward}"


def test_env_death_reward_on_collision():
    from flappy_bird_env import FlappyBirdEnv
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

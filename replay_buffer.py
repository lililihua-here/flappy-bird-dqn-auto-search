"""Replay buffer — uniform sampling, with extension points for n-step and PER."""
from collections import deque
import random
import numpy as np
from flappy_bird_env import FlappyBirdEnv


# ============================================================================
# State Encoder — 低维特征归一化 (Section 10.2)
# ============================================================================
class StateEncoder:
    """Encodes env state dict into normalized 7-dim feature vector."""

    def __init__(self, screen_width=600, screen_height=800, max_fall_speed=10):
        self.screen_w = float(screen_width)
        self.screen_h = float(screen_height)
        self.max_speed = float(max_fall_speed)

    def encode(self, state):
        """state dict → np.array of shape (7,) dtype float32."""
        bird_y = state['bird_y']
        bird_vy = state['bird_velocity']
        pipe_x = state['pipe_x']
        gap_top = state['pipe_gap_top']
        gap_bot = state['pipe_gap_bottom']
        gap_ctr = state['pipe_gap_center']

        features = [
            bird_y / self.screen_h,                                          # 0: normalized bird y
            bird_vy / self.max_speed,                                         # 1: normalized velocity
            (pipe_x - FlappyBirdEnv.BIRD_X) / self.screen_w,                  # 2: horizontal dist to pipe
            gap_top / self.screen_h,                                          # 3: normalized gap top
            gap_bot / self.screen_h,                                          # 4: normalized gap bottom
            gap_ctr / self.screen_h,                                          # 5: normalized gap center
            (bird_y - gap_ctr) / self.screen_h,                               # 6: bird to gap-center offset
        ]
        return np.array(features, dtype=np.float32)

    @property
    def state_dim(self):
        return 7


# ============================================================================
# Replay Buffer — uniform sampling
# ============================================================================
class ReplayBuffer:
    """Simple FIFO replay buffer with uniform random sampling."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# NStepReplayBuffer -- n-step TD returns with done flush (Stage B)
# ============================================================================
class NStepReplayBuffer:
    """Replay buffer that stores n-step transitions.

    Uses an internal deque to accumulate n consecutive single steps,
    then computes the n-step return and stores the n-step transition.
    On done, flushes all remaining partial transitions and clears
    the queue (P0-3 fix from V2 plan reviews).
    """

    def __init__(self, capacity, n_step, gamma):
        self.capacity = int(capacity)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.buffer = deque(maxlen=self.capacity)
        self._n_step_queue = deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        self._n_step_queue.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))
        if done:
            self._flush_on_done()
            return
        if len(self._n_step_queue) < self.n_step:
            return
        self._store_from_queue(0)
        self._n_step_queue.popleft()

    def _store_from_queue(self, start_idx):
        s0, a0, _, _, _ = self._n_step_queue[start_idx]
        n_return = 0.0
        actual_n = 0
        for k in range(start_idx, min(start_idx + self.n_step, len(self._n_step_queue))):
            _, _, r_k, _, d_k = self._n_step_queue[k]
            n_return += (self.gamma ** actual_n) * r_k
            actual_n += 1
            if d_k:
                break
        last_idx = start_idx + actual_n - 1
        next_s = self._n_step_queue[last_idx][3]
        done_flag = self._n_step_queue[last_idx][4]
        gamma_power = self.gamma ** actual_n
        self.buffer.append((
            s0, a0, float(n_return), next_s, bool(done_flag),
            float(gamma_power), int(actual_n),
        ))

    def _flush_on_done(self):
        """Flush all pending partial n-step transitions on done, then clear queue."""
        while len(self._n_step_queue) > 0:
            self._store_from_queue(0)
            self._n_step_queue.popleft()

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, n_returns, next_states, dones, gamma_powers, actual_ns = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(n_returns, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(gamma_powers, dtype=np.float32),
            np.asarray(actual_ns, dtype=np.int64),
        )

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size

    def __len__(self):
        return len(self.buffer)

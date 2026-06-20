"""Replay buffer — uniform sampling, with extension points for n-step and PER."""
from collections import deque
import random
import numpy as np
from flappy_bird_env import FlappyBirdEnv


# ============================================================================
# SumTree — binary tree for proportional PER sampling (Stage C)
# ============================================================================
class SumTree:
    """Binary sum-tree for proportional PER sampling."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = [None] * self.capacity
        self._ptr = 0
        self._size = 0

    def add(self, priority, data):
        idx = self._ptr + self.capacity - 1
        self.data[self._ptr] = data
        self.update(idx, float(priority))
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return idx

    def update(self, idx, priority):
        delta = float(priority) - self.tree[idx]
        self.tree[idx] = float(priority)
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self._size


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
# PERBuffer — Proportional Prioritized Experience Replay (Stage C)
# ============================================================================
class PERBuffer:
    """Proportional Prioritized Experience Replay buffer using SumTree.

    Stores single-step transitions (n_step=1). For n-step+PER, use NStepPERBuffer.
    Returns 7-tuple: (states, actions, rewards, next_states, dones, weights, indices)
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_train_updates=50000, priority_eps=1e-6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_start = float(beta)
        self.beta_train_updates = int(beta_train_updates)
        self.priority_eps = float(priority_eps)
        self.tree = SumTree(self.capacity)
        self._max_raw_priority = 1.0
        self._train_updates = 0

    def add(self, state, action, reward, next_state, done):
        data = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self.tree.add(self._max_raw_priority ** self.alpha, data)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.tree))
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices, weights = [], []
        total = self.tree.total()
        segment = total / batch_size
        self._train_updates += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) *
                   self._train_updates / self.beta_train_updates)
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            prob = priority / total
            weight = (len(self.tree) * prob) ** (-beta)
            weights.append(weight)
            indices.append(idx)
            st, a, r, ns, d = data
            states.append(st); actions.append(a); rewards.append(r)
            next_states.append(ns); dones.append(d)
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()
        return (
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
            weights.reshape(-1, 1),
            np.asarray(indices, dtype=np.int64),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, np.abs(td_errors) + self.priority_eps):
            raw_priority = float(td_err)
            tree_priority = raw_priority ** self.alpha
            self.tree.update(int(idx), tree_priority)
            self._max_raw_priority = max(self._max_raw_priority, raw_priority)

    def total_priority(self): return self.tree.total()
    def can_sample(self, batch_size): return len(self.tree) >= batch_size
    def __len__(self): return len(self.tree)


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


# ============================================================================
# NStepPERBuffer -- n-step TD returns + PER (Stage C)
# ============================================================================
class NStepPERBuffer:
    """PER buffer that stores n-step transitions in SumTree.

    Combines NStepReplayBuffer's n-step return computation with
    PERBuffer's prioritized sampling. Returns 9-tuple.
    """

    def __init__(self, capacity, n_step, gamma, alpha=0.6, beta=0.4,
                 beta_train_updates=50000, priority_eps=1e-6):
        self.capacity = int(capacity)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_start = float(beta)
        self.beta_train_updates = int(beta_train_updates)
        self.priority_eps = float(priority_eps)
        self.tree = SumTree(self.capacity)
        self._max_raw_priority = 1.0
        self._train_updates = 0
        self._n_step_queue = deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        self._n_step_queue.append((
            np.asarray(state, dtype=np.float32), int(action),
            float(reward), np.asarray(next_state, dtype=np.float32), bool(done),
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
        data = (
            s0, a0, float(n_return), next_s, bool(done_flag),
            float(gamma_power), int(actual_n),
        )
        self.tree.add(self._max_raw_priority ** self.alpha, data)

    def _flush_on_done(self):
        while len(self._n_step_queue) > 0:
            self._store_from_queue(0)
            self._n_step_queue.popleft()

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.tree))
        states, actions, n_returns, next_states, dones = [], [], [], [], []
        gamma_powers_list, actual_ns_list = [], []
        indices, weights = [], []
        total = self.tree.total()
        segment = total / batch_size
        self._train_updates += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) *
                   self._train_updates / self.beta_train_updates)
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            prob = priority / total
            weight = (len(self.tree) * prob) ** (-beta)
            weights.append(weight); indices.append(idx)
            st, a, r, ns, d, gp, an = data
            states.append(st); actions.append(a); n_returns.append(r)
            next_states.append(ns); dones.append(d)
            gamma_powers_list.append(gp); actual_ns_list.append(an)
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()
        return (
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(n_returns, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(gamma_powers_list, dtype=np.float32),
            np.asarray(actual_ns_list, dtype=np.int64),
            weights.reshape(-1, 1),
            np.asarray(indices, dtype=np.int64),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, np.abs(td_errors) + self.priority_eps):
            raw_priority = float(td_err)
            tree_priority = raw_priority ** self.alpha
            self.tree.update(int(idx), tree_priority)
            self._max_raw_priority = max(self._max_raw_priority, raw_priority)

    def total_priority(self): return self.tree.total()
    def can_sample(self, batch_size): return len(self.tree) >= batch_size
    def __len__(self): return len(self.tree)

"""
Flappy Bird DQN — 自进化超参优化系统 (MVP v1.3)
=================================================
单文件 MVP：标准环境 + 低维状态 DQN + Optuna TPE 自动搜索
"""
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import optuna
except ImportError:
    optuna = None


# ============================================================================
# 2. Flappy Bird Environment — 固定标准环境 (Section 5)
# ============================================================================
class FlappyBirdEnv:
    """Standard fixed-physics Flappy Bird environment.

    P0-1 FIX: total_raw_env_frames survives reset().
    episode_raw_env_frames resets per episode.

    MVP uses single-pipe recycle mode. When a pipe scrolls off-screen, it is
    immediately recycled to the right edge with a new random gap position.
    """

    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 800
    GRAVITY = 0.5
    FLAP_STRENGTH = -3
    PIPE_GAP = 400
    PIPE_VELOCITY = -2.5
    PIPE_WIDTH = 80
    BIRD_X = 100
    BIRD_SIZE = 20
    MAX_FALL_SPEED = 10
    PIPE_SPAWN_X = SCREEN_WIDTH

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.total_raw_env_frames = 0    # P0-1: never reset
        self.episode_raw_env_frames = 0  # P0-1: reset per episode
        self.reset()

    def reset(self):
        self.bird_y = float(self.SCREEN_HEIGHT // 2)
        self.bird_velocity = 0.0
        self.pipe_x = float(self.PIPE_SPAWN_X)
        self.pipe_gap_center = float(self.rng.randint(200, self.SCREEN_HEIGHT - 200))
        self.score = 0
        self.done = False
        self._scored_current_pipe = False
        self.episode_raw_env_frames = 0  # P0-1: only episode counter resets
        return self._get_state()

    def step(self, action):
        self.total_raw_env_frames += 1   # P0-1: always increment
        self.episode_raw_env_frames += 1

        if action == 1:
            self.bird_velocity = float(self.FLAP_STRENGTH)
        self.bird_velocity += self.GRAVITY
        self.bird_velocity = max(-self.MAX_FALL_SPEED, min(self.MAX_FALL_SPEED, self.bird_velocity))
        self.bird_y += self.bird_velocity
        self.pipe_x += self.PIPE_VELOCITY

        bird_top = self.bird_y - self.BIRD_SIZE // 2
        bird_bottom = self.bird_y + self.BIRD_SIZE // 2
        pipe_top = self.pipe_gap_center - self.PIPE_GAP // 2
        pipe_bottom = self.pipe_gap_center + self.PIPE_GAP // 2

        hit_pipe = (
            self.BIRD_X + self.BIRD_SIZE // 2 > self.pipe_x
            and self.BIRD_X - self.BIRD_SIZE // 2 < self.pipe_x + self.PIPE_WIDTH
            and (bird_top < pipe_top or bird_bottom > pipe_bottom)
        )
        hit_boundary = bird_top <= 0 or bird_bottom >= self.SCREEN_HEIGHT

        reward = 0.0
        if hit_pipe or hit_boundary:
            reward = -1.0
            self.done = True

        if self.pipe_x + self.PIPE_WIDTH < self.BIRD_X and not self._scored_current_pipe:
            self.score += 1
            self._scored_current_pipe = True
            if not self.done:
                reward = 1.0

        if self.pipe_x < -self.PIPE_WIDTH:
            self.pipe_x = float(self.PIPE_SPAWN_X)
            self.pipe_gap_center = float(self.rng.randint(200, self.SCREEN_HEIGHT - 200))
            self._scored_current_pipe = False

        return self._get_state(), reward, self.done

    def _get_state(self):
        pipe_top = self.pipe_gap_center - self.PIPE_GAP // 2
        pipe_bottom = self.pipe_gap_center + self.PIPE_GAP // 2
        return {
            'bird_y': self.bird_y,
            'bird_velocity': self.bird_velocity,
            'pipe_x': self.pipe_x,
            'pipe_gap_top': pipe_top,
            'pipe_gap_bottom': pipe_bottom,
            'pipe_gap_center': self.pipe_gap_center,
        }


# ============================================================================
# 3. State Encoder — 低维特征归一化 (Section 10.2)
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
            pipe_x / self.screen_w,                                           # 2: normalized pipe x
            gap_top / self.screen_h,                                          # 3: normalized gap top
            gap_bot / self.screen_h,                                          # 4: normalized gap bottom
            (gap_ctr - bird_y) / self.screen_h,                               # 5: vertical dist to gap center
            (pipe_x - FlappyBirdEnv.BIRD_X) / self.screen_w,                  # 6: horizontal dist to pipe
        ]
        return np.array(features, dtype=np.float32)

    @property
    def state_dim(self):
        return 7


# ============================================================================
# 4. Replay Buffer — uniform sampling
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
# 5. DQN network — configurable MLP
# ============================================================================
class DQN(nn.Module):
    """Small MLP Q-network for 7-D low-dimensional state."""

    def __init__(self, state_dim, hidden, n_actions):
        super().__init__()
        dims = [state_dim] + list(hidden) + [n_actions]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


# ============================================================================
# 6. DQN Agent — P0-3/P1-5/P1-8 fixes applied
# ============================================================================
class DQNAgent:
    """DQN agent (1-step Double DQN, Adam, Huber loss, soft target update)."""

    def __init__(self, config, state_dim, n_actions, device):
        # P1-8: Assert MVP fixed parameters
        assert config.get('target_update_mode', 'soft') == 'soft', \
            "MVP fixed: target_update_mode must be 'soft'"
        assert config.get('n_step', 1) == 1, \
            "MVP fixed: n_step must be 1 (n-step deferred to enhancement)"
        # P1-3 (v1.3): Assert all MVP-fixed params that env/agent hardcode
        assert config.get('frame_skip', 1) == 1, \
            "MVP fixed: frame_skip must be 1"
        assert config.get('torch_optimizer', 'Adam') == 'Adam', \
            "MVP fixed: torch_optimizer must be 'Adam'"
        assert config.get('loss_type', 'Huber') == 'Huber', \
            "MVP fixed: loss_type must be 'Huber'"
        assert config.get('reward_pipe', 1.0) == 1.0, \
            "MVP fixed: reward_pipe must be 1.0 (env hardcoded)"
        assert config.get('reward_death', -1.0) == -1.0, \
            "MVP fixed: reward_death must be -1.0 (env hardcoded)"
        assert config.get('reward_alive', 0.0) == 0.0, \
            "MVP fixed: reward_alive must be 0.0 (env hardcoded)"

        self.config = config
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        self.q_net = DQN(state_dim, config['hidden'], n_actions).to(device)
        self.target_net = DQN(state_dim, config['hidden'], n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.buffer = ReplayBuffer(config['buffer_sz'])
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config['lr'])
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        self.epsilon = float(config['eps_start'])
        self.decision_steps = 0

    def act(self, state, training=True):
        if training:
            self.decision_steps += 1
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            return int(self.q_net(state_t).argmax(dim=1).item())

    def train(self):
        if not self.buffer.can_sample(self.config['batch_sz']):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config['batch_sz'])
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.config.get('double_q', True):
                next_actions = self.q_net(next_states_t).argmax(1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            # P0-3: 1-step TD target
            target = rewards_t + (1.0 - dones_t) * self.config['gamma'] * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config['grad_clip_norm'])
        self.optimizer.step()

        # Soft target update (MVP fixed)
        tau = self.config['tau']
        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        return float(loss.item())

    def decay_epsilon(self):
        # P1-5: use eps_decay_decision_steps, fallback to eps_frames
        decay_steps = self.config.get('eps_decay_decision_steps',
                                      self.config.get('eps_frames', 50000))
        if self.decision_steps >= decay_steps:
            self.epsilon = float(self.config['eps_end'])
        else:
            progress = self.decision_steps / decay_steps
            self.epsilon = float(self.config['eps_start']
                + progress * (self.config['eps_end'] - self.config['eps_start']))

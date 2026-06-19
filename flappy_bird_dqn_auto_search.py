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

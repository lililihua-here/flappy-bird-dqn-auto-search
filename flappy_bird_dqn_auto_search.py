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


# ============================================================================
# 8. Evaluation — P0-2: independent eval env
# ============================================================================
def greedy_eval(agent, env_factory, encoder, n_episodes=5,
                eval_seed=0, max_raw_frames_per_ep=120000):
    """P0-2: Creates independent env. Returns statistics dict."""
    eval_env = env_factory(seed=eval_seed)
    scores = []
    frames_before = eval_env.total_raw_env_frames

    for _ in range(n_episodes):
        state_dict = eval_env.reset()
        ep_frames = 0
        done = False
        while not done and ep_frames < max_raw_frames_per_ep:
            state_vec = encoder.encode(state_dict)
            action = agent.act(state_vec, training=False)
            state_dict, _reward, done = eval_env.step(action)
            ep_frames += 1
        scores.append(eval_env.score)

    total_raw_frames = eval_env.total_raw_env_frames - frames_before
    scores_arr = np.array(scores, dtype=np.float64)

    return {
        'scores': [int(s) for s in scores],
        'mean': float(np.mean(scores_arr)),
        'median': float(np.median(scores_arr)),
        'max': int(np.max(scores_arr)),
        'min': int(np.min(scores_arr)),
        'success_rate_1000': float(np.mean(scores_arr >= 1000)),
        'raw_env_frames': total_raw_frames,
    }


# ============================================================================
# P1-4: Stable success pure function
# ============================================================================
def is_stable_success(eval_result, threshold=1000, min_rate=0.70, min_median=1000):
    """Check if eval result meets the stable success criteria (Section 3.3)."""
    return (
        eval_result['success_rate_1000'] >= min_rate
        and eval_result['median'] >= min_median
    )


# ============================================================================
# Baseline default config (Section 10.5) — P0-3: n_step=1
# ============================================================================
BASELINE_CONFIG = {
    'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64, 'buffer_sz': 50000,
    'hidden': [128, 64],
    'double_q': True, 'n_step': 1, 'frame_skip': 1,
    'eps_start': 0.05, 'eps_end': 0.005, 'eps_decay_decision_steps': 50000,
    'replay_start_size': 5000, 'train_freq': 1,
    'target_update_mode': 'soft', 'tau': 0.005,
    'torch_optimizer': 'Adam', 'loss_type': 'Huber', 'grad_clip_norm': 5,
    'reward_pipe': 1.0, 'reward_death': -1.0, 'reward_alive': 0.0,
    'reward_clip': None, 'reward_scale': 1.0,
}


# ============================================================================
# P0-6 (v1.3): Global seed helper
# ============================================================================
def set_global_seed(seed):
    """Set seed for random, numpy, and torch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 7. Training Loop — P0-1/P0-2/P1-1/P1-9 fixes
# ============================================================================
def run_trial(config, trial_id, seed, source='tpe',
              max_trial_frames=1_000_000,
              eval_interval_frames=20_000,
              eval_episodes=5,
              candidate_verify_episodes=20,
              candidate_threshold=1000,
              candidate_min_rate=0.70,
              candidate_min_median=1000,
              eval_max_frames_per_ep=120_000):
    """Run one trial from scratch. Returns result dict (Section 14.3)."""
    # P0-6 (v1.3): Set global seeds BEFORE any random operations
    set_global_seed(seed)

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = FlappyBirdEnv(seed=seed)
    encoder = StateEncoder()

    agent = DQNAgent(
        config={
            **config,
            'hidden': config.get('hidden', [128, 64]),
            'lr': config.get('lr', 1e-4),
            'gamma': config.get('gamma', 0.99),
            'batch_sz': config.get('batch_sz', 64),
            'buffer_sz': config.get('buffer_sz', 50000),
            'eps_start': config.get('eps_start', 0.05),
            'eps_end': config.get('eps_end', 0.005),
            'eps_decay_decision_steps': config.get('eps_decay_decision_steps',
                                                    config.get('eps_frames', 50000)),
            'replay_start_size': config.get('replay_start_size', 5000),
            'train_freq': config.get('train_freq', 1),
            'target_update_mode': config.get('target_update_mode', 'soft'),
            'tau': config.get('tau', 0.005),
            'double_q': config.get('double_q', True),
            'grad_clip_norm': config.get('grad_clip_norm', 5),
            'n_step': config.get('n_step', 1),
        },
        state_dim=encoder.state_dim,
        n_actions=2,
        device=device,
    )

    train_raw_env_frames = 0
    eval_raw_env_frames = 0

    # Warmup (Section 8.2)
    state_dict = env.reset()
    for _ in range(config.get('replay_start_size', 5000)):
        action = random.randint(0, 1)
        next_dict, reward, done = env.step(action)
        agent.buffer.add(
            encoder.encode(state_dict), action, reward,
            encoder.encode(next_dict), done,
        )
        if done:
            state_dict = env.reset()
        else:
            state_dict = next_dict
    train_raw_env_frames = env.total_raw_env_frames

    # Training loop
    best_train_score = 0
    best_eval_median = 0.0
    last_improvement_frame = 0
    total_episodes = 0
    state_dict = env.reset()
    candidate_verified = False
    candidate_result = None
    recent_losses = deque(maxlen=100)
    status = 'failure'
    failure_reason = 'max_frames_reached'
    eval_call_count = 0

    while train_raw_env_frames < max_trial_frames:
        state_vec = encoder.encode(state_dict)
        action = agent.act(state_vec, training=True)
        next_dict, reward, done = env.step(action)

        agent.buffer.add(state_vec, action, reward, encoder.encode(next_dict), done)

        if agent.decision_steps % config.get('train_freq', 1) == 0:
            loss = agent.train()
            if loss is not None:
                recent_losses.append(loss)

        agent.decay_epsilon()

        if env.score > best_train_score:
            best_train_score = env.score

        if done:
            total_episodes += 1
            state_dict = env.reset()
        else:
            state_dict = next_dict

        # P0-1: eval uses independent env, training env total IS train-only
        train_raw_env_frames = env.total_raw_env_frames

        # Periodic eval (Section 8.3)
        if train_raw_env_frames > 0 and train_raw_env_frames % eval_interval_frames == 0:
            eval_call_count += 1
            eval_seed = seed + 100000 + eval_call_count
            eval_result = greedy_eval(
                agent=agent, env_factory=FlappyBirdEnv, encoder=encoder,
                n_episodes=eval_episodes, eval_seed=eval_seed,
                max_raw_frames_per_ep=eval_max_frames_per_ep,
            )
            eval_raw_env_frames += eval_result['raw_env_frames']

            if eval_result['median'] > best_eval_median:
                best_eval_median = eval_result['median']
                last_improvement_frame = train_raw_env_frames
            if last_improvement_frame == 0:
                last_improvement_frame = train_raw_env_frames

            # Candidate success trigger (Section 8.4)
            if not candidate_verified and (
                eval_result['median'] >= candidate_threshold
                or eval_result['max'] >= 1200
            ):
                eval_call_count += 1
                verify_seed = seed + 200000 + eval_call_count
                verify_result = greedy_eval(
                    agent=agent, env_factory=FlappyBirdEnv, encoder=encoder,
                    n_episodes=candidate_verify_episodes, eval_seed=verify_seed,
                    max_raw_frames_per_ep=eval_max_frames_per_ep,
                )
                eval_raw_env_frames += verify_result['raw_env_frames']

                if is_stable_success(verify_result, candidate_threshold,
                                     candidate_min_rate, candidate_min_median):
                    candidate_verified = True
                    candidate_result = verify_result
                    status = 'success'
                    break

            # Early stop
            should_stop, stop_reason = check_early_stop(
                train_frames=train_raw_env_frames,
                best_eval_score=best_eval_median,
                best_train_score=best_train_score,
                last_improvement_frame=last_improvement_frame,
                recent_losses=recent_losses,
                max_trial_frames=max_trial_frames,
            )
            if should_stop:
                status = 'failure'
                failure_reason = stop_reason
                break

    # Build result
    final_eval_scores = None
    final_median = 0.0
    final_mean = 0.0
    final_success_rate = 0.0

    if candidate_result is not None:
        final_eval_scores = candidate_result['scores']
        final_median = candidate_result['median']
        final_mean = candidate_result['mean']
        final_success_rate = candidate_result['success_rate_1000']
    else:
        eval_call_count += 1
        final_eval_seed = seed + 300000 + eval_call_count
        final_eval = greedy_eval(
            agent=agent, env_factory=FlappyBirdEnv, encoder=encoder,
            n_episodes=20, eval_seed=final_eval_seed,
            max_raw_frames_per_ep=eval_max_frames_per_ep,
        )
        eval_raw_env_frames += final_eval['raw_env_frames']
        final_eval_scores = final_eval['scores']
        final_median = final_eval['median']
        final_mean = final_eval['mean']
        final_success_rate = final_eval['success_rate_1000']

    total_raw_env_frames = train_raw_env_frames + eval_raw_env_frames
    duration = time.time() - t_start
    code_version = _get_git_hash()

    return {
        'trial_id': trial_id,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': config,
        'source': source,
        'seed': seed,
        'status': status,
        'objective': 0.0,
        'train_raw_env_frames': train_raw_env_frames,
        'total_raw_env_frames': total_raw_env_frames,
        'eval_raw_env_frames': eval_raw_env_frames,
        'decision_steps': agent.decision_steps,
        'episodes': total_episodes,
        'record_type': 'trial',
        'best_train_score': best_train_score,
        'best_eval_score': float(best_eval_median),
        'best_eval_median_score': float(best_eval_median),
        'final_eval_scores': final_eval_scores,
        'success_rate_1000': final_success_rate,
        'median_score': final_median,
        'mean_score': final_mean,
        'failure_reason': failure_reason if status != 'success' else '',
        'early_stop_reason': failure_reason if status != 'success' else '',
        'duration_sec': duration,
        'init_strategy': 'random_init',
        'env_version': 'fixed_env_v1',
        'reward_scheme_version': 'mvp_reward_v1',
        'code_version': code_version,
        'implementation_version': 'mvp_v0.2',
    }


def _get_git_hash():
    """P1-6: Get current git hash or 'unknown'."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


# ============================================================================
# 9. Early Stopping (Section 9)
# ============================================================================
def check_early_stop(train_frames, best_eval_score, best_train_score,
                     last_improvement_frame, recent_losses, max_trial_frames):
    """Returns (should_stop: bool, reason: str)."""
    if recent_losses:
        recent = list(recent_losses)
        if any(math.isnan(l) or math.isinf(l) for l in recent):
            return True, 'loss_nan_inf'

    if train_frames > 50000 and best_eval_score < 20:
        return True, 'no_learning_50k'
    if train_frames > 150000 and best_eval_score < 100:
        return True, 'slow_learning_150k'
    if train_frames > 300000 and best_eval_score < 300:
        return True, 'cannot_reach_target_300k'

    if train_frames - last_improvement_frame > 100000 and train_frames > 100000:
        return True, 'plateau_100k'

    return False, ''


# ============================================================================
# 10. Failure Penalty Objective (Section 9.4)
# ============================================================================
def compute_objective(success, train_raw_env_frames, max_trial_frames, best_eval_score):
    """Compute optimization objective that incorporates failure information."""
    if success:
        return float(train_raw_env_frames)

    progress_score = min(best_eval_score / 1000.0, 1.0)
    penalty_factor = 2.0 - 0.9 * progress_score
    return float(max_trial_frames) * penalty_factor

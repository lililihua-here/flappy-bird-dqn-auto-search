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

from flappy_bird_env import FlappyBirdEnv
from replay_buffer import StateEncoder, ReplayBuffer
from dqn_agent import DQN, DQNAgent
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
    scores = eval_result.get('scores')
    if scores is not None:
        scores_arr = np.asarray(scores, dtype=np.float64)
        success_rate = float(np.mean(scores_arr >= threshold))
    elif threshold == 1000:
        success_rate = float(eval_result.get('success_rate_1000', 0.0))
    else:
        success_rate = 0.0
    return (
        success_rate >= min_rate
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


def _build_checkpoint_path(checkpoint_dir, source, trial_id, seed):
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_trial_{trial_id}_seed_{seed}.pt"
    return (checkpoint_root / filename).resolve()


def _save_agent_checkpoint(agent, config, trial_id, seed, source, checkpoint_dir):
    checkpoint_path = _build_checkpoint_path(checkpoint_dir, source, trial_id, seed)
    payload = {
        'trial_id': trial_id,
        'seed': seed,
        'source': source,
        'config': config,
        'state_dim': agent.state_dim,
        'n_actions': agent.n_actions,
        'q_net_state_dict': agent.q_net.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path)


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
              eval_max_frames_per_ep=120_000,
              checkpoint_dir='checkpoints'):
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
    checkpoint_path = _save_agent_checkpoint(
        agent=agent,
        config=agent.config,
        trial_id=trial_id,
        seed=seed,
        source=source,
        checkpoint_dir=checkpoint_dir,
    )

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
        'checkpoint_path': checkpoint_path,
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
    """Optuna minimizes this objective.

    Successful trials are ranked by training frames to stable 1000 score.
    Failed trials receive a large penalty, but better near-misses get a
    slightly smaller penalty so TPE can still learn from them.
    """
    if success:
        return float(train_raw_env_frames)

    try:
        best_eval_score = float(best_eval_score)
    except (TypeError, ValueError):
        best_eval_score = 0.0
    if math.isnan(best_eval_score) or best_eval_score < 0:
        best_eval_score = 0.0

    return float(max_trial_frames * 10 - best_eval_score)


# ============================================================================
# 11. Search Space — Optuna parameter definition (Section 10.6)
# ============================================================================
def define_search_space(trial):
    """Define MVP 8-parameter search space.
    Uses scalar categorical choices for Optuna SQLite persistence,
    then maps to actual layer size lists.
    """
    hidden_map = {
        'small': [64, 32],
        'medium': [128, 64],
        'large': [256, 128],
    }
    hidden_key = trial.suggest_categorical('hidden_key', ['small', 'medium', 'large'])
    return {
        # Searchable (Section 10.6)
        'lr': trial.suggest_float('lr', 1e-5, 3e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.90, 0.999),
        'hidden_key': hidden_key,
        'hidden': hidden_map[hidden_key],
        'eps_start': trial.suggest_float('eps_start', 0.01, 0.15),
        'eps_end': trial.suggest_float('eps_end', 0.001, 0.02),
        'eps_decay_decision_steps': trial.suggest_int('eps_decay_decision_steps', 10000, 200000),
        'replay_start_size': trial.suggest_categorical('replay_start_size', [1000, 5000, 10000]),
        'train_freq': trial.suggest_categorical('train_freq', [1, 4]),
        # MVP fixed (Section 10.3, 10.4)
        'double_q': True,
        'n_step': 1,
        'frame_skip': 1,
        'target_update_mode': 'soft',
        'tau': 0.005,
        'torch_optimizer': 'Adam',
        'loss_type': 'Huber',
        'grad_clip_norm': 5,
        'batch_sz': 64,
        'buffer_sz': 50000,
        'reward_pipe': 1.0,
        'reward_death': -1.0,
        'reward_alive': 0.0,
        'reward_clip': None,
        'reward_scale': 1.0,
    }


# ============================================================================
# 12. History Manager — JSONL persistence (Section 14)
# ============================================================================
def _make_serializable(obj):
    """Convert numpy / torch objects to JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


class HistoryManager:
    """Append-only JSONL history for trial results."""

    def __init__(self, history_path='search_history.jsonl'):
        self.path = Path(history_path)

    def append(self, result):
        with open(self.path, 'a', encoding='utf-8') as f:
            serializable = dict(_make_serializable(result))
            serializable.setdefault('record_type', 'trial')
            f.write(json.dumps(serializable, ensure_ascii=False) + '\n')
            f.flush()

    def load(self):
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def success_count(self):
        return sum(
            1 for r in self.load()
            if r.get('record_type', 'trial') == 'trial' and r.get('status') == 'success'
        )

    def failure_count(self):
        return sum(
            1 for r in self.load()
            if r.get('record_type', 'trial') == 'trial' and r.get('status') == 'failure'
        )

    def best_trial(self):
        rows = [r for r in self.load() if r.get('record_type', 'trial') == 'trial']
        successes = [r for r in rows if r.get('status') == 'success']
        if successes:
            return min(successes, key=lambda r: r.get('objective', float('inf')))
        if rows:
            return min(rows, key=lambda r: r.get('objective', float('inf')))
        return None

    def top_k(self, k=5):
        """P1-3: Sort by Section 16.2 priority rules. Filter trial records only."""
        all_rows = self.load()
        trial_rows = [
            r for r in all_rows
            if r.get('record_type', 'trial') == 'trial'
            and r.get('config')
        ]
        latest_recheck_by_trial = {}
        for row in all_rows:
            if row.get('record_type') == 'recheck' and 'trial_id' in row:
                latest_recheck_by_trial[row['trial_id']] = row

        rows = []
        for trial_row in trial_rows:
            merged = dict(trial_row)
            recheck_row = latest_recheck_by_trial.get(trial_row.get('trial_id'))
            if recheck_row:
                for key, value in recheck_row.items():
                    if key not in ('record_type', 'trial_id'):
                        merged[key] = value
            rows.append(merged)

        def ranking_key(r):
            return (
                not r.get('recheck_passed', False),
                r.get('median_train_raw_env_frames_to_stable_1000',
                      r.get('objective', float('inf'))),
                -r.get('success_rate_1000', 0),
                -r.get('median_score', 0),
                r.get('total_raw_env_frames', float('inf')),
            )

        sorted_rows = sorted(rows, key=ranking_key)
        return sorted_rows[:k]


# ============================================================================
# 13. Search Driver — P0-4/P0-5 fixes
# ============================================================================
def get_best_render_record(history):
    """Return the best renderable trial record or raise a clear error."""
    best = history.best_trial()
    if best is None:
        raise ValueError('No trial records found in history.')
    checkpoint_path = best.get('checkpoint_path')
    if not checkpoint_path:
        raise ValueError('Best trial has no checkpoint_path.')
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_file}')
    return best


def load_agent_from_checkpoint(checkpoint_path, device=None):
    """Load an agent and encoder from a saved checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    encoder = StateEncoder()
    agent = DQNAgent(
        config=config,
        state_dim=checkpoint.get('state_dim', encoder.state_dim),
        n_actions=checkpoint.get('n_actions', 2),
        device=device,
    )
    agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['q_net_state_dict'])
    return agent, encoder, checkpoint


def _draw_render_frame(screen, font, env, best_record, episode_idx, total_episodes):
    try:
        import pygame
    except ImportError as exc:
        raise ImportError('pygame is required for render mode.') from exc

    sky = (135, 206, 235)
    green = (46, 160, 67)
    yellow = (255, 220, 0)
    white = (255, 255, 255)
    black = (20, 20, 20)

    screen.fill(sky)

    pipe_top = int(env.pipe_gap_center - env.PIPE_GAP // 2)
    pipe_bottom = int(env.pipe_gap_center + env.PIPE_GAP // 2)
    pipe_x = int(env.pipe_x)

    pygame.draw.rect(screen, green, pygame.Rect(pipe_x, 0, env.PIPE_WIDTH, pipe_top))
    pygame.draw.rect(
        screen,
        green,
        pygame.Rect(pipe_x, pipe_bottom, env.PIPE_WIDTH, env.SCREEN_HEIGHT - pipe_bottom),
    )
    pygame.draw.circle(screen, yellow, (env.BIRD_X, int(env.bird_y)), env.BIRD_SIZE // 2)

    lines = [
        f"Trial: {best_record.get('trial_id')}  Source: {best_record.get('source')}",
        f"Episode: {episode_idx}/{total_episodes}  Score: {env.score}",
        f"Best objective: {best_record.get('objective')}",
        f"Median: {best_record.get('median_score')}  SR1000: {best_record.get('success_rate_1000')}",
        "Esc / close window to exit",
    ]
    y = 12
    for line in lines:
        text = font.render(line, True, black, white)
        screen.blit(text, (12, y))
        y += 28


def render_best_demo(history_path='search_history.jsonl', episodes=1, fps=60,
                     max_raw_frames_per_ep=120_000):
    """Render a greedy demo using the best trial checkpoint from history."""
    try:
        import pygame
    except ImportError as exc:
        raise ImportError('pygame is required for --render mode. Install it via requirements.txt.') from exc

    history = HistoryManager(history_path)
    best_record = get_best_render_record(history)
    agent, encoder, _checkpoint = load_agent_from_checkpoint(best_record['checkpoint_path'])
    env = FlappyBirdEnv(seed=best_record.get('seed', 0))

    pygame.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption('Flappy Bird DQN Render Demo')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    running = True
    episode_idx = 0
    try:
        while running and episode_idx < episodes:
            state_dict = env.reset()
            done = False
            ep_frames = 0
            episode_idx += 1

            while running and not done and ep_frames < max_raw_frames_per_ep:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                if not running:
                    break

                state_vec = encoder.encode(state_dict)
                action = agent.act(state_vec, training=False)
                state_dict, _reward, done = env.step(action)
                ep_frames += 1

                _draw_render_frame(screen, font, env, best_record, episode_idx, episodes)
                pygame.display.flip()
                clock.tick(fps)
    finally:
        pygame.quit()


class SearchDriver:
    """Orchestrates the full hyperparameter search with Optuna TPE."""

    def __init__(self, history_path='search_history.jsonl', study_db='optuna_study.db',
                 max_trials=100, max_trial_frames=1_000_000,
                 eval_interval_frames=20_000, eval_episodes=5,
                 candidate_verify_episodes=20, n_startup_trials=30,
                 seed_pool=(11, 22, 33), checkpoint_dir='checkpoints'):
        self.history = HistoryManager(history_path)
        self.study_db = study_db
        self.max_trials = max_trials
        self.max_trial_frames = max_trial_frames
        self.eval_interval_frames = eval_interval_frames
        self.eval_episodes = eval_episodes
        self.candidate_verify_episodes = candidate_verify_episodes
        self.n_startup_trials = n_startup_trials
        self.seed_pool = list(seed_pool)
        self.checkpoint_dir = checkpoint_dir
        self._interrupted = False

    def _objective(self, trial):
        """Optuna objective. P0-5: trial_id = trial.number."""
        config = define_search_space(trial)
        trial_id = trial.number
        seed = self.seed_pool[trial_id % len(self.seed_pool)]

        print(f"\n{'=' * 50}")
        print(f"Trial #{trial_id}  |  Source: TPE  |  Seed: {seed}")
        print(f"Config: lr={config['lr']:.2e}, gamma={config['gamma']:.4f}, "
              f"hidden={config['hidden']}, eps={config['eps_start']:.3f}->{config['eps_end']:.3f} "
              f"over {config['eps_decay_decision_steps']} steps")
        print(f"{'=' * 50}")

        result = run_trial(
            config=config, trial_id=trial_id, seed=seed, source='tpe',
            max_trial_frames=self.max_trial_frames,
            eval_interval_frames=self.eval_interval_frames,
            eval_episodes=self.eval_episodes,
            candidate_verify_episodes=self.candidate_verify_episodes,
            checkpoint_dir=self.checkpoint_dir,
        )

        obj = compute_objective(
            success=(result['status'] == 'success'),
            train_raw_env_frames=result['train_raw_env_frames'],
            max_trial_frames=self.max_trial_frames,
            best_eval_score=result['best_eval_score'],
        )
        result['objective'] = obj

        self.history.append(result)

        if result['status'] == 'success':
            print(f"SUCCESS Trial #{trial_id}  train_frames={result['train_raw_env_frames']}  "
                  f"median={result['median_score']:.0f}  sr={result['success_rate_1000']:.0%}")
        else:
            print(f"FAILED Trial #{trial_id}  reason={result['failure_reason']}  "
                  f"best_eval={result['best_eval_score']:.0f}  objective={obj:.0f}")

        if self._interrupted:
            trial.study.stop()

        return obj

    def run(self):
        if optuna is None:
            raise ImportError('optuna required. Install: pip install optuna')

        study = optuna.create_study(
            study_name='flappy_bird_dqn_search',
            storage=f'sqlite:///{self.study_db}',
            direction='minimize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials, seed=42),
            load_if_exists=True,
        )

        existing = len(study.trials)
        remaining = max(0, self.max_trials - existing)

        # Baseline as independent sanity check
        history_rows = self.history.load()
        has_baseline = any(r.get('source') == 'baseline' for r in history_rows)

        if existing == 0 and not has_baseline:
            print(f"\n[STAGE 0] Baseline verification (independent, not counted in max_trials)...")
            result = run_trial(
                config=dict(BASELINE_CONFIG), trial_id=-1, seed=11, source='baseline',
                max_trial_frames=self.max_trial_frames,
                eval_interval_frames=self.eval_interval_frames,
                eval_episodes=self.eval_episodes,
                candidate_verify_episodes=self.candidate_verify_episodes,
                checkpoint_dir=self.checkpoint_dir,
            )
            obj = compute_objective(
                success=(result['status'] == 'success'),
                train_raw_env_frames=result['train_raw_env_frames'],
                max_trial_frames=self.max_trial_frames,
                best_eval_score=result['best_eval_score'],
            )
            result['objective'] = obj
            self.history.append(result)
            print(f"[STAGE 0] Baseline complete. status={result['status']}\n")

        print(f"[INFO] Study: {existing} Optuna trials completed, {remaining} remaining")
        print(f"[INFO] Max trial frames: {self.max_trial_frames}")

        original_handler = signal.signal(signal.SIGINT, self._sigint_handler)

        try:
            study.optimize(self._objective, n_trials=remaining)
        finally:
            signal.signal(signal.SIGINT, original_handler)

        if self._interrupted:
            print("\n[Ctrl+C] Search stopped by user. History saved.")

        generate_summary(self.history)

    def _sigint_handler(self, signum, frame):
        """P0-4: Set flag only. Let current trial finish, then stop."""
        print("\n[Ctrl+C] Will stop after current trial completes. Press again to force-quit.")
        if self._interrupted:
            print("[Ctrl+C] Force quitting...")
            sys.exit(1)
        self._interrupted = True


# ============================================================================
# 14. Reporting — console summary + structured return value
# ============================================================================
def generate_summary(history, top_k=5):
    """Print a compact summary and return it as a dict for tests."""
    rows = history.load()
    trial_rows = [r for r in rows if r.get('record_type', 'trial') == 'trial']

    if not trial_rows:
        summary = {
            'trial_count': 0, 'success_count': 0, 'failure_count': 0,
            'best_trial_id': None, 'best_objective': None,
            'top_k': [], 'failure_reasons': {},
        }
        print('[SUMMARY] No trial records found.')
        return summary

    failure_reasons = {}
    for row in trial_rows:
        if row.get('status') == 'failure':
            reason = row.get('failure_reason') or 'unknown'
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    best = history.best_trial()
    top_trials = history.top_k(top_k)
    summary = {
        'trial_count': len(trial_rows),
        'success_count': history.success_count(),
        'failure_count': history.failure_count(),
        'best_trial_id': best.get('trial_id') if best else None,
        'best_objective': best.get('objective') if best else None,
        'top_k': [
            {
                'trial_id': row.get('trial_id'),
                'source': row.get('source'),
                'objective': row.get('objective'),
                'median_score': row.get('median_score'),
                'success_rate_1000': row.get('success_rate_1000'),
            }
            for row in top_trials
        ],
        'failure_reasons': failure_reasons,
    }

    print('\n[SUMMARY]')
    print(f"Trials: {summary['trial_count']}  "
          f"Success: {summary['success_count']}  Failure: {summary['failure_count']}")
    print(f"Best trial: {summary['best_trial_id']}  Objective: {summary['best_objective']}")
    for idx, row in enumerate(summary['top_k'], start=1):
        print(f"Top {idx}: trial={row['trial_id']}  obj={row['objective']}  "
              f"median={row['median_score']}  sr={row['success_rate_1000']}")
    return summary


# ============================================================================
# Top-K Recheck (P1-2: Minimal implementation)
# ============================================================================
def recheck_top_k(history, k=5, recheck_seeds=(101, 202, 303),
                  max_trial_frames=1_000_000, eval_episodes=20):
    """Re-evaluate top K configs with multiple independent seeds."""
    top_configs = history.top_k(k)
    results = []

    for rank, trial in enumerate(top_configs):
        config = trial.get('config', {})
        if not config:
            continue
        seed_scores = []
        for seed in recheck_seeds:
            result = run_trial(
                config=config, trial_id=-1, seed=seed, source='recheck',
                max_trial_frames=max_trial_frames,
                eval_episodes=eval_episodes,
            )
            seed_scores.append({
                'seed': seed,
                'status': result['status'],
                'train_raw_env_frames': result['train_raw_env_frames'],
                'median_score': result['median_score'],
                'success_rate_1000': result['success_rate_1000'],
            })

        all_seed_scores = [s['median_score'] for s in seed_scores]
        successful_train_frames = [
            s['train_raw_env_frames'] for s in seed_scores
            if s['status'] == 'success'
        ]

        recheck_summary = {
            'rank': rank + 1,
            'original_trial_id': trial.get('trial_id'),
            'config': config,
            'seeds': seed_scores,
            'recheck_median': float(np.median(all_seed_scores)),
            'recheck_mean': float(np.mean(all_seed_scores)),
            'recheck_success_rate': float(np.mean([s['success_rate_1000'] for s in seed_scores])),
            'p10_score': float(np.percentile(all_seed_scores, 10)),
            'p90_score': float(np.percentile(all_seed_scores, 90)),
            'score_std': float(np.std(all_seed_scores)),
            'median_train_raw_env_frames_to_stable_1000': (
                float(np.median(successful_train_frames)) if successful_train_frames else None
            ),
            'recheck_passed': all(s['status'] == 'success' for s in seed_scores),
            'failed_seeds': [s for s in seed_scores if s['status'] != 'success'],
        }
        results.append(recheck_summary)

        recheck_record = {
            'record_type': 'recheck',
            'trial_id': trial.get('trial_id'),
            **recheck_summary,
            'recheck_seeds_used': list(recheck_seeds),
        }
        history.append(recheck_record)

    return results


# ============================================================================
# 15. CLI Entrypoint
# ============================================================================
def get_mode_presets(mode):
    presets = {
        'debug':    {'max_trial_frames': 100_000, 'eval_interval_frames': 10_000, 'eval_episodes': 3},
        'normal':   {'max_trial_frames': 1_000_000, 'eval_interval_frames': 20_000, 'eval_episodes': 5},
        'deep':     {'max_trial_frames': 5_000_000, 'eval_interval_frames': 50_000, 'eval_episodes': 20},
    }
    if mode not in presets:
        raise ValueError(f"Unknown mode: {mode}")
    return presets[mode]


def make_parser():
    p = argparse.ArgumentParser(description='Flappy Bird DQN Auto-Search System')
    p.add_argument('--mode', choices=['debug', 'normal', 'deep'], default='normal')
    p.add_argument('--max-trials', type=int, default=100)
    p.add_argument('--max-trial-frames', type=int, default=None)
    p.add_argument('--history', default='search_history.jsonl')
    p.add_argument('--study-db', default='optuna_study.db')
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--n-startup-trials', type=int, default=30)
    p.add_argument('--baseline-only', action='store_true',
                   help='Run a single baseline trial and exit (no search)')
    p.add_argument('--render', action='store_true',
                   help='Render the best checkpoint from history using pygame')
    p.add_argument('--render-episodes', type=int, default=1)
    p.add_argument('--render-fps', type=int, default=60)
    return p


def main():
    args = make_parser().parse_args()
    presets = get_mode_presets(args.mode)
    max_trial_frames = args.max_trial_frames or presets['max_trial_frames']

    print(f"[MODE] {args.mode}  |  max_trial_frames={max_trial_frames}")

    if args.render:
        print(f"[RENDER] Loading best trial from {args.history}")
        render_best_demo(
            history_path=args.history,
            episodes=args.render_episodes,
            fps=args.render_fps,
        )
        return

    if args.baseline_only:
        print("[BASELINE-ONLY] Running single baseline trial...")
        result = run_trial(
            config=dict(BASELINE_CONFIG), trial_id=-1, seed=11, source='baseline',
            max_trial_frames=max_trial_frames,
            eval_interval_frames=presets['eval_interval_frames'],
            eval_episodes=presets['eval_episodes'],
            checkpoint_dir=args.checkpoint_dir,
        )
        obj = compute_objective(
            success=(result['status'] == 'success'),
            train_raw_env_frames=result['train_raw_env_frames'],
            max_trial_frames=max_trial_frames,
            best_eval_score=result['best_eval_score'],
        )
        result['objective'] = obj
        hm = HistoryManager(args.history)
        hm.append(result)
        generate_summary(hm)
        return

    driver = SearchDriver(
        history_path=args.history, study_db=args.study_db,
        max_trials=args.max_trials, max_trial_frames=max_trial_frames,
        eval_interval_frames=presets['eval_interval_frames'],
        eval_episodes=presets['eval_episodes'],
        n_startup_trials=args.n_startup_trials,
        checkpoint_dir=args.checkpoint_dir,
    )

    try:
        driver.run()
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

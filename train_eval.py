"""Training loop, evaluation, early stopping, and objective functions."""
import math
import random
import time
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import torch
from dqn_agent import DQNAgent
from flappy_bird_env import FlappyBirdEnv
from replay_buffer import StateEncoder, ReplayBuffer
from version_utils import get_git_hash, infer_reward_scheme_version


def set_global_seed(seed):
    """Set seed for random, numpy, and torch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    reward_config = {
        'pipe_reward': config.get('pipe_reward', 1.0),
        'death_ratio': config.get('death_ratio', 1),
        'alive_ratio': config.get('alive_ratio', 0.0),
        'reward_scale': config.get('reward_scale', 1.0),
        'reward_clip': config.get('reward_clip', None),
    }
    env = FlappyBirdEnv(seed=seed, reward_config=reward_config)
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

    # Stage C: Buffer selection (4 branches: standard / n-step / PER / n-step+PER)
    n_step = config.get('n_step', 1)
    priority = config.get('priority', False)

    if priority and n_step > 1:
        from replay_buffer import NStepPERBuffer
        buffer = NStepPERBuffer(
            capacity=config['buffer_sz'], n_step=n_step, gamma=config['gamma'],
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta_start', 0.4),
            beta_train_updates=config.get('per_beta_train_updates', 50000),
            priority_eps=config.get('per_priority_eps', 1e-6),
        )
    elif priority:
        from replay_buffer import PERBuffer
        buffer = PERBuffer(
            capacity=config['buffer_sz'],
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta_start', 0.4),
            beta_train_updates=config.get('per_beta_train_updates', 50000),
            priority_eps=config.get('per_priority_eps', 1e-6),
        )
    elif n_step > 1:
        from replay_buffer import NStepReplayBuffer
        buffer = NStepReplayBuffer(
            capacity=config['buffer_sz'], n_step=n_step, gamma=config['gamma'],
        )
    else:
        buffer = None

    if buffer is not None:
        agent.buffer = buffer

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
    code_version = get_git_hash()
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
        'n_step': config.get('n_step', 1),
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
        'reward_scheme_version': infer_reward_scheme_version(config),
        'code_version': code_version,
        'implementation_version': 'mvp_v0.2',
        'checkpoint_path': checkpoint_path,
        'death_ratio': config.get('death_ratio', 1),
        'alive_ratio': config.get('alive_ratio', 0.0),
        'reward_scale': config.get('reward_scale', 1.0),
        'reward_clip': config.get('reward_clip', None),
        'priority': config.get('priority', False),
    }


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

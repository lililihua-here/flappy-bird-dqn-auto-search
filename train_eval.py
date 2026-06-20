"""Training loop, evaluation, early stopping, and objective functions."""
import math
import random
import sys
import time
from collections import deque

import numpy as np
import torch
from dqn_agent import DQNAgent
from flappy_bird_env import FlappyBirdEnv
from replay_buffer import StateEncoder, ReplayBuffer
from version_utils import get_git_hash, infer_reward_scheme_version


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


def _format_progress_line(stage, current, total, elapsed_sec, detail=''):
    stage_label = stage.upper()
    bar_width = 24

    if total and total > 0:
        ratio = max(0.0, min(1.0, current / total))
        filled = int(ratio * bar_width)
        bar = '#' * filled + '-' * (bar_width - filled)
        percent_text = f'{ratio * 100:3.0f}%'
        eta_text = '--:--'
        if current > 0 and current < total:
            eta_sec = elapsed_sec * (total - current) / current
            eta_text = _format_duration(eta_sec)
    else:
        bar = '-' * bar_width
        percent_text = '--%'
        eta_text = '--:--'

    line = f'[{stage_label:<10}] |{bar}| {percent_text} ETA {eta_text}'
    if detail:
        line += f'  {detail}'
    return line


class _ProgressDisplay:
    def __init__(self):
        self._last_line = ''

    def update(self, stage, current, total, elapsed_sec, detail=''):
        line = _format_progress_line(stage, current, total, elapsed_sec, detail)
        padding = ' ' * max(0, len(self._last_line) - len(line))
        sys.stdout.write('\r' + line + padding)
        sys.stdout.flush()
        self._last_line = line

    def finish(self):
        if self._last_line:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self._last_line = ''


def set_global_seed(seed):
    """Set seed for random, numpy, and torch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def greedy_eval(agent, env_factory, encoder, n_episodes=5,
                eval_seed=0, max_raw_frames_per_ep=120000,
                progress_callback=None, progress_stage='eval'):
    """P0-2: Creates independent env. Returns statistics dict."""
    eval_env = env_factory(seed=eval_seed)
    scores = []
    frames_before = eval_env.total_raw_env_frames
    stage_t0 = time.time()
    progress_interval = max(1, n_episodes // 10)

    for idx in range(n_episodes):
        state_dict = eval_env.reset()
        ep_frames = 0
        done = False
        while not done and ep_frames < max_raw_frames_per_ep:
            state_vec = encoder.encode(state_dict)
            action = agent.act(state_vec, training=False)
            state_dict, _reward, done = eval_env.step(action)
            ep_frames += 1
        scores.append(eval_env.score)
        if progress_callback is not None and (
            idx == 0 or (idx + 1) % progress_interval == 0 or idx + 1 == n_episodes
        ):
            progress_callback(
                progress_stage,
                idx + 1,
                n_episodes,
                time.time() - stage_t0,
                f'episodes={idx + 1}/{n_episodes} score={eval_env.score}',
            )

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
    progress_interval = max(1000, min(10000, eval_interval_frames // 2))
    progress = _ProgressDisplay()

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
    warmup_target = config.get('replay_start_size', 5000)
    warmup_interval = max(1, warmup_target // 20)
    warmup_t0 = time.time()
    state_dict = env.reset()
    for idx in range(warmup_target):
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
        if idx == 0 or (idx + 1) % warmup_interval == 0 or idx + 1 == warmup_target:
            progress.update(
                'warmup',
                idx + 1,
                warmup_target,
                time.time() - warmup_t0,
                f'frames={idx + 1}/{warmup_target}',
            )
    train_raw_env_frames = env.total_raw_env_frames
    next_progress_frame = (
        ((train_raw_env_frames // progress_interval) + 1) * progress_interval
    )
    train_t0 = time.time()

    # Training loop
    best_train_score = 0
    best_eval_score = 0.0
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

        if train_raw_env_frames >= next_progress_frame:
            progress.update(
                'train',
                train_raw_env_frames,
                max_trial_frames,
                time.time() - train_t0,
                (
                    f'frames={train_raw_env_frames}/{max_trial_frames} '
                    f'episodes={total_episodes} '
                    f'best_train={best_train_score} '
                    f'best_eval={best_eval_median:.0f}'
                ),
            )
            next_progress_frame += progress_interval

        # Periodic eval (Section 8.3)
        if train_raw_env_frames > 0 and train_raw_env_frames % eval_interval_frames == 0:
            eval_call_count += 1
            eval_seed = seed + 100000 + eval_call_count
            eval_result = greedy_eval(
                agent=agent, env_factory=FlappyBirdEnv, encoder=encoder,
                n_episodes=eval_episodes, eval_seed=eval_seed,
                max_raw_frames_per_ep=eval_max_frames_per_ep,
                progress_callback=progress.update,
                progress_stage='eval',
            )
            eval_raw_env_frames += eval_result['raw_env_frames']
            progress.update(
                'eval',
                eval_episodes,
                eval_episodes,
                0.0,
                (
                    f'median={eval_result["median"]:.0f} '
                    f'max={eval_result["max"]} '
                    f'sr={eval_result["success_rate_1000"]:.0%}'
                ),
            )

            if eval_result['median'] > best_eval_median:
                best_eval_median = eval_result['median']
                last_improvement_frame = train_raw_env_frames
            if eval_result['max'] > best_eval_score:
                best_eval_score = float(eval_result['max'])
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
                    progress_callback=progress.update,
                    progress_stage='verify',
                )
                eval_raw_env_frames += verify_result['raw_env_frames']
                progress.update(
                    'verify',
                    candidate_verify_episodes,
                    candidate_verify_episodes,
                    0.0,
                    (
                        f'median={verify_result["median"]:.0f} '
                        f'max={verify_result["max"]} '
                        f'sr={verify_result["success_rate_1000"]:.0%}'
                    ),
                )

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
            progress_callback=progress.update,
            progress_stage='final-eval',
        )
        eval_raw_env_frames += final_eval['raw_env_frames']
        progress.update(
            'final-eval',
            20,
            20,
            0.0,
            (
                f'median={final_eval["median"]:.0f} '
                f'max={final_eval["max"]} '
                f'sr={final_eval["success_rate_1000"]:.0%}'
            ),
        )
        final_eval_scores = final_eval['scores']
        final_median = final_eval['median']
        final_mean = final_eval['mean']
        final_success_rate = final_eval['success_rate_1000']

    if final_eval_scores:
        best_eval_score = max(best_eval_score, float(max(final_eval_scores)))
    best_eval_median = max(best_eval_median, float(final_median))
    total_raw_env_frames = train_raw_env_frames + eval_raw_env_frames
    duration = time.time() - t_start
    code_version = get_git_hash()
    objective = compute_objective(
        success=(status == 'success'),
        train_raw_env_frames=train_raw_env_frames,
        max_trial_frames=max_trial_frames,
        best_eval_score=best_eval_score,
    )
    from history_reporting import build_checkpoint_payload, save_checkpoint
    payload = build_checkpoint_payload(
        q_net=agent.q_net,
        target_net=agent.target_net,
        config={
            **agent.config,
            'reward_scheme_version': infer_reward_scheme_version(config),
        },
        trial_id=trial_id,
        seed=seed,
        source=source,
        train_raw_env_frames=train_raw_env_frames,
        decision_steps=agent.decision_steps,
        state_dim=agent.state_dim,
        n_actions=agent.n_actions,
    )
    checkpoint_prefix = f'{source}_trial_{trial_id}_seed_{seed}'
    checkpoint_path, checkpoint_sha256 = save_checkpoint(
        payload, checkpoint_dir, prefix=checkpoint_prefix
    )
    progress.finish()

    return {
        'trial_id': trial_id,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': config,
        'n_step': config.get('n_step', 1),
        'source': source,
        'seed': seed,
        'status': status,
        'objective': objective,
        'train_raw_env_frames': train_raw_env_frames,
        'total_raw_env_frames': total_raw_env_frames,
        'eval_raw_env_frames': eval_raw_env_frames,
        'decision_steps': agent.decision_steps,
        'episodes': total_episodes,
        'record_type': 'trial',
        'best_train_score': best_train_score,
        'best_eval_score': float(best_eval_score),
        'best_eval_median_score': float(best_eval_median),
        'final_eval_scores': final_eval_scores,
        'success_rate_1000': final_success_rate,
        'median_score': final_median,
        'mean_score': final_mean,
        'failure_reason': failure_reason if status != 'success' else '',
        'early_stop_reason': failure_reason if status != 'success' else '',
        'duration_sec': duration,
        'init_strategy': 'random_init',
        'environment_version': 'fixed_env_v1',
        'state_representation_version': 'low_dim_v1',
        'reward_scheme_version': infer_reward_scheme_version(config),
        'code_version': code_version,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_format_version': payload['checkpoint_format_version'],
        'death_ratio': config.get('death_ratio', 1),
        'alive_ratio': config.get('alive_ratio', 0.0),
        'reward_scale': config.get('reward_scale', 1.0),
        'reward_clip': config.get('reward_clip', None),
        'priority': config.get('priority', False),
        'per_alpha': config.get('per_alpha'),
        'per_beta_start': config.get('per_beta_start'),
        'per_beta_train_updates': config.get('per_beta_train_updates'),
        'per_priority_eps': config.get('per_priority_eps'),
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

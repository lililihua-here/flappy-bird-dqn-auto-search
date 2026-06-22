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
from lineage import LineageTracker
from replay_buffer import ReplayBuffer
from reward_protocols import (
    compute_reward_v1_sparse,
    compute_reward_v2_ratio,
    compute_reward_v3_gap_shaping,
)
from snapshot import (
    SNAPSHOT_INTERVAL,
    capture_snapshot,
    load_snapshot,
    restore_lineage_from_snapshot,
    restore_replay_buffer,
    restore_rng_state,
    restore_snapshot_to_agent,
    save_snapshot,
    verify_snapshot_compatible,
)
from state_encoder_variants import get_encoder
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


def _make_env_config():
    return {
        'SCREEN_WIDTH': FlappyBirdEnv.SCREEN_WIDTH,
        'SCREEN_HEIGHT': FlappyBirdEnv.SCREEN_HEIGHT,
        'MAX_FALL_SPEED': FlappyBirdEnv.MAX_FALL_SPEED,
        'BIRD_X': FlappyBirdEnv.BIRD_X,
    }


def _resolve_reward_scheme(config):
    reward_scheme_version = infer_reward_scheme_version(config)
    reward_scale = config.get('reward_scale', 1.0)
    reward_clip = config.get('reward_clip')
    reward_kwargs = {
        'death_ratio': config.get('death_ratio', 1),
        'alive_ratio': config.get('alive_ratio', 0.0),
        'scale': reward_scale,
        'clip': reward_clip,
    }
    if reward_scheme_version == 'reward_v1_sparse':
        runtime_reward = {
            'pipe_reward': 1.0,
            'death_ratio': 1,
            'alive_ratio': 0.0,
            'reward_scale': 1.0,
            'reward_clip': None,
        }
    else:
        runtime_reward = {
            'pipe_reward': config.get('pipe_reward', 1.0),
            'death_ratio': reward_kwargs['death_ratio'],
            'alive_ratio': reward_kwargs['alive_ratio'],
            'reward_scale': reward_scale,
            'reward_clip': reward_clip,
        }
    return reward_scheme_version, runtime_reward, reward_kwargs


def _apply_reward_protocol(env, raw_reward, reward_scheme_version, reward_kwargs, config):
    events = getattr(env, 'last_events', {})
    if reward_scheme_version == 'reward_v1_sparse':
        return compute_reward_v1_sparse(events)
    if reward_scheme_version == 'reward_v3_gap_shaping':
        return compute_reward_v3_gap_shaping(
            events,
            bird_y=env.bird_y,
            gap_center=env.pipe_gap_center,
            pipe_gap=config.get('pipe_gap', env.PIPE_GAP),
            gap_shaping_coef=config.get('gap_shaping_coef', 0.0),
            **reward_kwargs,
        )
    if reward_scheme_version == 'reward_v2_ratio':
        return compute_reward_v2_ratio(events, **reward_kwargs)
    return raw_reward


def _build_agent_config(config, reward_scheme_version):
    return {
        **config,
        'hidden': config.get('hidden', [128, 64]),
        'lr': config.get('lr', 1e-4),
        'gamma': config.get('gamma', 0.99),
        'batch_sz': config.get('batch_sz', 64),
        'buffer_sz': config.get('buffer_sz', 50000),
        'eps_start': config.get('eps_start', 0.05),
        'eps_end': config.get('eps_end', 0.005),
        'eps_decay_decision_steps': config.get(
            'eps_decay_decision_steps',
            config.get('eps_frames', 50000),
        ),
        'replay_start_size': config.get('replay_start_size', 5000),
        'train_freq': config.get('train_freq', 1),
        'target_update_mode': config.get('target_update_mode', 'soft'),
        'tau': config.get('tau', 0.005),
        'double_q': config.get('double_q', True),
        'grad_clip_norm': config.get('grad_clip_norm', 5),
        'n_step': config.get('n_step', 1),
        'torch_optimizer': config.get('torch_optimizer', 'Adam'),
        'loss_type': config.get('loss_type', 'Huber'),
        'network_backbone': config.get('network_backbone', 'mlp'),
        'exploration_head': config.get('exploration_head', 'epsilon_greedy'),
        'hard_update_interval_decision_steps': config.get(
            'hard_update_interval_decision_steps',
            config.get('hard_update_freq', 1000),
        ),
        'state_representation_version': config.get('state_representation_version', 'low_dim_v1'),
        'reward_scheme_version': reward_scheme_version,
        'environment_version': config.get('environment_version', 'fixed_env_v1'),
        'code_version': get_git_hash(),
    }


def run_trial(config, trial_id, seed, source='tpe',
              max_trial_frames=1_000_000,
              eval_interval_frames=20_000,
              eval_episodes=5,
              candidate_verify_episodes=20,
              candidate_threshold=1000,
              candidate_min_rate=0.70,
              candidate_min_median=1000,
              eval_max_frames_per_ep=120_000,
              checkpoint_dir='checkpoints',
              trial_type='fresh',
              resume_snapshot_path=None,
              parent_trial_id=None,
              parent_snapshot_ref=None,
              snapshot_interval=SNAPSHOT_INTERVAL,
              force_final_eval=False,
              max_additional_train_raw_env_frames=None):
    """Run one trial. Supports fresh, resume, warm_start and population_inherited."""
    if trial_type == 'fresh' and resume_snapshot_path is not None:
        raise ValueError("resume_snapshot_path requires trial_type not 'fresh'")
    if trial_type == 'resume' and not resume_snapshot_path:
        raise ValueError("trial_type='resume' requires resume_snapshot_path")
    if trial_type in ('warm_start', 'population_inherited') and not resume_snapshot_path:
        raise ValueError(f"trial_type='{trial_type}' requires resume_snapshot_path")

    set_global_seed(seed)

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_config = _make_env_config()
    reward_scheme_version, runtime_reward_config, reward_kwargs = _resolve_reward_scheme(config)
    full_config = _build_agent_config(config, reward_scheme_version)
    env = FlappyBirdEnv(seed=seed, reward_config=runtime_reward_config)
    encoder = get_encoder(full_config['state_representation_version'], env_config=env_config)
    progress_interval = max(1000, min(10000, eval_interval_frames // 2))
    progress = _ProgressDisplay()

    agent = DQNAgent(
        config=full_config,
        state_dim=encoder.state_dim,
        n_actions=2,
        device=device,
    )

    n_step = full_config.get('n_step', 1)
    priority = full_config.get('priority', False)
    if priority and n_step > 1:
        from replay_buffer import NStepPERBuffer
        agent.buffer = NStepPERBuffer(
            capacity=full_config['buffer_sz'], n_step=n_step, gamma=full_config['gamma'],
            alpha=full_config.get('per_alpha', 0.6),
            beta=full_config.get('per_beta_start', 0.4),
            beta_train_updates=full_config.get('per_beta_train_updates', 50000),
            priority_eps=full_config.get('per_priority_eps', 1e-6),
        )
    elif priority:
        from replay_buffer import PERBuffer
        agent.buffer = PERBuffer(
            capacity=full_config['buffer_sz'],
            alpha=full_config.get('per_alpha', 0.6),
            beta=full_config.get('per_beta_start', 0.4),
            beta_train_updates=full_config.get('per_beta_train_updates', 50000),
            priority_eps=full_config.get('per_priority_eps', 1e-6),
        )
    elif n_step > 1:
        from replay_buffer import NStepReplayBuffer
        agent.buffer = NStepReplayBuffer(
            capacity=full_config['buffer_sz'], n_step=n_step, gamma=full_config['gamma'],
        )
    else:
        agent.buffer = ReplayBuffer(full_config['buffer_sz'])

    train_raw_env_frames = 0
    eval_raw_env_frames = 0
    best_train_score = 0
    best_eval_score = 0.0
    best_eval_median = 0.0
    last_improvement_frame = 0
    total_episodes = 0
    candidate_verified = False
    candidate_result = None
    recent_losses = deque(maxlen=100)
    status = 'failure'
    failure_reason = 'max_frames_reached'
    eval_call_count = 0
    last_snapshot_path = ''
    lineage_tracker = None
    state_dict = None

    if trial_type in ('resume', 'warm_start', 'population_inherited'):
        snapshot = load_snapshot(resume_snapshot_path)
        if not verify_snapshot_compatible(snapshot, full_config):
            raise ValueError('resume snapshot is not compatible with current config')
        restore_snapshot_to_agent(snapshot, agent)
        agent.buffer = restore_replay_buffer(snapshot, full_config)
        restore_rng_state(snapshot)
        env.restore_runtime_state(snapshot.env_runtime_state)
        state_dict = env._get_state()
        last_snapshot_path = str(resume_snapshot_path)

        if trial_type == 'resume':
            lineage_tracker = restore_lineage_from_snapshot(snapshot)
            restored_meta = dict(snapshot.training_meta or {})
            saved_eval_interval = restored_meta.get('eval_interval_frames')
            eval_metrics_compatible = saved_eval_interval == eval_interval_frames
            eval_raw_env_frames = int(restored_meta.get('eval_raw_env_frames', 0)) if eval_metrics_compatible else 0
            best_train_score = int(restored_meta.get('best_train_score', env.score))
            best_eval_score = float(restored_meta.get('best_eval_score', 0.0)) if eval_metrics_compatible else 0.0
            best_eval_median = float(restored_meta.get('best_eval_median', 0.0)) if eval_metrics_compatible else 0.0
            last_improvement_frame = int(
                restored_meta.get('last_improvement_frame', snapshot.local_train_raw_env_frames)
            ) if eval_metrics_compatible else env.total_raw_env_frames
            total_episodes = int(restored_meta.get('total_episodes', 0))
            eval_call_count = int(restored_meta.get('eval_call_count', 0)) if eval_metrics_compatible else 0
            if eval_metrics_compatible:
                for loss in restored_meta.get('recent_losses', []):
                    recent_losses.append(float(loss))
        else:
            parent_lineage = restore_lineage_from_snapshot(snapshot)
            lineage_tracker = LineageTracker(
                trial_type=trial_type,
                trial_id=trial_id,
                parent_lineage=parent_lineage,
                parent_trial_id=(
                    parent_trial_id if parent_trial_id is not None else snapshot.trial_id
                ),
                parent_snapshot_ref=(
                    parent_snapshot_ref if parent_snapshot_ref is not None else str(resume_snapshot_path)
                ),
            )
            best_train_score = env.score
    else:
        lineage_tracker = LineageTracker(trial_type='fresh', trial_id=trial_id)
        warmup_target = full_config.get('replay_start_size', 5000)
        warmup_interval = max(1, warmup_target // 20)
        warmup_t0 = time.time()
        state_dict = env.reset()
        for idx in range(warmup_target):
            action = random.randint(0, 1)
            next_dict, raw_reward, done = env.step(action)
            reward = _apply_reward_protocol(
                env, raw_reward, reward_scheme_version, reward_kwargs, full_config,
            )
            agent.buffer.add(
                encoder.encode(state_dict), action, reward,
                encoder.encode(next_dict), done,
            )
            lineage_tracker.add_frames(1)
            if done:
                total_episodes += 1
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
        best_train_score = env.score

    train_raw_env_frames = env.total_raw_env_frames
    start_train_raw_env_frames = train_raw_env_frames
    if max_additional_train_raw_env_frames is not None:
        target_train_raw_env_frames = train_raw_env_frames + max_additional_train_raw_env_frames
    else:
        target_train_raw_env_frames = max_trial_frames
    next_progress_frame = (
        ((train_raw_env_frames // progress_interval) + 1) * progress_interval
    )
    train_t0 = time.time()

    while train_raw_env_frames < target_train_raw_env_frames:
        state_vec = encoder.encode(state_dict)
        action = agent.act(state_vec, training=True)
        next_dict, raw_reward, done = env.step(action)
        reward = _apply_reward_protocol(
            env, raw_reward, reward_scheme_version, reward_kwargs, full_config,
        )

        agent.buffer.add(state_vec, action, reward, encoder.encode(next_dict), done)
        lineage_tracker.add_frames(1)

        if agent.decision_steps % full_config.get('train_freq', 1) == 0:
            loss = agent.train()
            if loss is not None:
                recent_losses.append(loss)
                lineage_tracker.add_update()

        if full_config.get('exploration_head', 'epsilon_greedy') != 'noisy_net':
            agent.decay_epsilon()

        if env.score > best_train_score:
            best_train_score = env.score

        if done:
            total_episodes += 1
            state_dict = env.reset()
        else:
            state_dict = next_dict

        train_raw_env_frames = env.total_raw_env_frames

        if (
            snapshot_interval
            and train_raw_env_frames > 0
            and train_raw_env_frames % snapshot_interval == 0
        ):
            snapshot = capture_snapshot(
                agent=agent,
                env=env,
                config=full_config,
                trial_id=trial_id,
                seed=seed,
                source=source,
                lineage_tracker=lineage_tracker,
            )
            snapshot.training_meta = {
                'eval_raw_env_frames': eval_raw_env_frames,
                'best_train_score': best_train_score,
                'best_eval_score': best_eval_score,
                'best_eval_median': best_eval_median,
                'last_improvement_frame': last_improvement_frame,
                'total_episodes': total_episodes,
                'eval_call_count': eval_call_count,
                'recent_losses': list(recent_losses),
                'eval_interval_frames': eval_interval_frames,
            }
            last_snapshot_path, _ = save_snapshot(snapshot, checkpoint_dir)

        if train_raw_env_frames >= next_progress_frame:
            progress.update(
                'train',
                train_raw_env_frames,
                target_train_raw_env_frames,
                time.time() - train_t0,
                (
                    f'frames={train_raw_env_frames}/{target_train_raw_env_frames} '
                    f'episodes={total_episodes} best_train={best_train_score} '
                    f'best_eval={best_eval_median:.0f}'
                ),
            )
            next_progress_frame += progress_interval

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
                best_eval_median = float(eval_result['median'])
                last_improvement_frame = train_raw_env_frames
            if eval_result['max'] > best_eval_score:
                best_eval_score = float(eval_result['max'])
            if last_improvement_frame == 0:
                last_improvement_frame = train_raw_env_frames

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
                if is_stable_success(
                    verify_result,
                    candidate_threshold,
                    candidate_min_rate,
                    candidate_min_median,
                ):
                    candidate_verified = True
                    candidate_result = verify_result
                    status = 'success'
                    if not force_final_eval:
                        break

            should_stop, stop_reason = check_early_stop(
                train_frames=train_raw_env_frames,
                best_eval_score=best_eval_median,
                best_train_score=best_train_score,
                last_improvement_frame=last_improvement_frame,
                recent_losses=recent_losses,
                max_trial_frames=target_train_raw_env_frames,
            )
            if should_stop:
                status = 'failure'
                failure_reason = stop_reason
                break

    final_eval_scores = None
    final_median = 0.0
    final_mean = 0.0
    final_success_rate = 0.0

    if candidate_result is not None and not force_final_eval:
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

    if status != 'success' and is_stable_success(
        {'scores': final_eval_scores, 'median': final_median},
        candidate_threshold,
        candidate_min_rate,
        candidate_min_median,
    ):
        status = 'success'
        failure_reason = ''

    if final_eval_scores:
        best_eval_score = max(best_eval_score, float(max(final_eval_scores)))
    best_eval_median = max(best_eval_median, float(final_median))
    total_raw_env_frames = train_raw_env_frames + eval_raw_env_frames
    duration = time.time() - t_start
    objective = compute_objective(
        success=(status == 'success'),
        train_raw_env_frames=train_raw_env_frames,
        max_trial_frames=target_train_raw_env_frames,
        best_eval_score=best_eval_score,
    )

    final_snapshot = capture_snapshot(
        agent=agent,
        env=env,
        config=full_config,
        trial_id=trial_id,
        seed=seed,
        source=source,
        lineage_tracker=lineage_tracker,
    )
    final_snapshot.training_meta = {
        'eval_raw_env_frames': eval_raw_env_frames,
        'best_train_score': best_train_score,
        'best_eval_score': best_eval_score,
        'best_eval_median': best_eval_median,
        'last_improvement_frame': last_improvement_frame,
        'total_episodes': total_episodes,
        'eval_call_count': eval_call_count,
        'recent_losses': list(recent_losses),
        'eval_interval_frames': eval_interval_frames,
    }
    last_snapshot_path, _ = save_snapshot(final_snapshot, checkpoint_dir)

    from history_reporting import build_checkpoint_payload, save_checkpoint as save_inference_checkpoint
    payload = build_checkpoint_payload(
        q_net=agent.q_net,
        target_net=agent.target_net,
        config=full_config,
        trial_id=trial_id,
        seed=seed,
        source=source,
        train_raw_env_frames=train_raw_env_frames,
        decision_steps=agent.decision_steps,
        state_dim=agent.state_dim,
        n_actions=agent.n_actions,
        environment_version=full_config['environment_version'],
        state_representation_version=full_config['state_representation_version'],
    )
    checkpoint_prefix = f'{source}_trial_{trial_id}_seed_{seed}'
    checkpoint_path, checkpoint_sha256 = save_inference_checkpoint(
        payload, checkpoint_dir, prefix=checkpoint_prefix
    )
    progress.finish()

    return {
        'trial_id': trial_id,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': full_config,
        'n_step': full_config.get('n_step', 1),
        'source': source,
        'seed': seed,
        'status': status,
        'objective': objective,
        'train_raw_env_frames': train_raw_env_frames,
        'local_train_raw_env_frames': lineage_tracker.local_train_raw_env_frames,
        'lineage_train_raw_env_frames': lineage_tracker.lineage_train_raw_env_frames,
        'block_raw_env_frames': train_raw_env_frames - start_train_raw_env_frames,
        'total_raw_env_frames': total_raw_env_frames,
        'eval_raw_env_frames': eval_raw_env_frames,
        'decision_steps': agent.decision_steps,
        'train_updates': lineage_tracker.train_updates,
        'episodes': total_episodes,
        'record_type': 'trial',
        'trial_type': trial_type,
        'lineage_chain_id': lineage_tracker.lineage_chain_id,
        'lineage_node_id': lineage_tracker.lineage_node_id,
        'lineage_root_trial_id': lineage_tracker.lineage_root_trial_id,
        'parent_trial_id': lineage_tracker.parent_trial_id,
        'parent_snapshot_ref': lineage_tracker.parent_snapshot_ref,
        'inheritance_depth': lineage_tracker.inheritance_depth,
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
        'init_strategy': trial_type,
        'environment_version': full_config['environment_version'],
        'state_representation_version': full_config['state_representation_version'],
        'reward_scheme_version': reward_scheme_version,
        'code_version': full_config['code_version'],
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_format_version': payload['checkpoint_format_version'],
        'checkpoint_capability': 'inference_only',
        'last_snapshot_path': last_snapshot_path,
        'snapshot_interval': snapshot_interval,
        'replay_buffer_size': len(agent.buffer),
        'n_step_queue_length': len(getattr(agent.buffer, '_n_step_queue', [])),
        'epsilon': agent.epsilon,
        'death_ratio': full_config.get('death_ratio', 1),
        'alive_ratio': full_config.get('alive_ratio', 0.0),
        'reward_scale': full_config.get('reward_scale', 1.0),
        'reward_clip': full_config.get('reward_clip', None),
        'gap_shaping_coef': full_config.get('gap_shaping_coef', 0.0),
        'priority': full_config.get('priority', False),
        'per_alpha': full_config.get('per_alpha'),
        'per_beta_start': full_config.get('per_beta_start'),
        'per_beta_train_updates': full_config.get('per_beta_train_updates'),
        'per_priority_eps': full_config.get('per_priority_eps'),
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
    Failed trials receive a progress-aware penalty so TPE can still learn
    from near-misses instead of treating all failures as nearly identical.
    """
    if success:
        return float(train_raw_env_frames)

    try:
        best_eval_score = float(best_eval_score)
    except (TypeError, ValueError):
        best_eval_score = 0.0
    if math.isnan(best_eval_score) or best_eval_score < 0:
        best_eval_score = 0.0
    progress_score = min(best_eval_score / 1000.0, 1.0)
    penalty_factor = 2.0 - 0.9 * progress_score
    return float(max_trial_frames * penalty_factor)

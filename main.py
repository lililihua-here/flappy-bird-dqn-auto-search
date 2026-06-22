"""Flappy Bird DQN V2 — CLI entrypoint and render demo."""
import argparse
import json
import sys
from pathlib import Path


# ============================================================================
# Render helpers
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
    import torch
    from replay_buffer import StateEncoder
    from dqn_agent import DQNAgent

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
    agent.target_net.load_state_dict(
        checkpoint.get('target_net_state_dict', checkpoint['q_net_state_dict'])
    )
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
    from flappy_bird_env import FlappyBirdEnv
    from history_reporting import HistoryManager

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


# ============================================================================
# CLI Entrypoint
# ============================================================================
MATRIX_CHOICES = {
    'baseline': 'BASELINE_MATRIX',
    'structure': 'STRUCTURE_ABLATION',
    'protocol': 'PROTOCOL_ABLATION',
    'searcher': 'SEARCHER_COMPARISON',
}


def get_matrix_by_name(name):
    if name not in MATRIX_CHOICES:
        raise ValueError(f'Unknown matrix: {name}')
    from experiment_matrix import (
        BASELINE_MATRIX,
        STRUCTURE_ABLATION,
        PROTOCOL_ABLATION,
        SEARCHER_COMPARISON,
    )

    matrices = {
        'baseline': BASELINE_MATRIX,
        'structure': STRUCTURE_ABLATION,
        'protocol': PROTOCOL_ABLATION,
        'searcher': SEARCHER_COMPARISON,
    }
    return matrices[name]


def load_json_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    p.add_argument('--report', action='store_true',
                   help='Generate experiment reports after baseline/search')
    p.add_argument('--search-strategy',
                   choices=['tpe_fresh', 'warmstart_tpe', 'population_async'],
                   default='tpe_fresh',
                   help='Select the V3 search strategy')
    p.add_argument('--resume', type=str,
                   help='Resume from a V3 snapshot path (same-trial continue)')
    p.add_argument('--recheck-topk', type=int,
                   help='Recheck the top K configs from history')
    p.add_argument('--final-confirm', type=str,
                   help='Run 5-seed final confirm for a config JSON file')
    p.add_argument('--matrix', choices=sorted(MATRIX_CHOICES.keys()),
                   help='Run a predefined experiment matrix')
    p.add_argument('--matrix-budget', choices=['debug_matrix', 'normal_matrix'],
                   default='debug_matrix',
                   help='Budget profile for --matrix')
    p.add_argument('--population-size', type=int, default=4,
                   help='Population size for population_async search')
    p.add_argument('--population-eval-interval', type=int, default=20_000,
                   help='Frames per worker block in population_async search')
    p.add_argument('--population-exploit-interval', type=int, default=50_000,
                   help='Global frames between exploit/explore steps')
    p.add_argument('--population-total-frame-budget', type=int, default=None,
                   help='Total frame budget for population_async search')
    p.add_argument('--n-step', type=int, choices=[1, 3, 5], default=None,
                   help='Override n_step for baseline/search')
    p.add_argument('--priority', action='store_true',
                   help='Force-enable prioritized replay (PER)')
    p.add_argument('--per-alpha', type=float, default=None,
                   help='Override PER alpha')
    p.add_argument('--per-beta-start', type=float, default=None,
                   help='Override PER beta start')
    p.add_argument('--per-beta-train-updates', type=int, default=None,
                   help='Override PER beta schedule length in train_updates')
    p.add_argument('--death-ratio', type=int, default=None,
                   help='Override death reward ratio')
    p.add_argument('--alive-ratio', type=float, default=None,
                   help='Override alive reward ratio')
    p.add_argument('--reward-scale', type=float, default=None,
                   help='Override reward scale')
    p.add_argument('--reward-clip', type=float, default=None,
                   help='Override reward clip threshold')
    return p


def _collect_config_overrides(args):
    return {
        'n_step': args.n_step,
        'priority': True if args.priority else None,
        'per_alpha': args.per_alpha,
        'per_beta_start': args.per_beta_start,
        'per_beta_train_updates': args.per_beta_train_updates,
        'death_ratio': args.death_ratio,
        'alive_ratio': args.alive_ratio,
        'reward_scale': args.reward_scale,
        'reward_clip': args.reward_clip,
    }


def main():
    args = make_parser().parse_args()
    print('[BOOT] Loading runtime...', flush=True)

    if args.render:
        print(f"[RENDER] Loading best trial from {args.history}")
        render_best_demo(
            history_path=args.history,
            episodes=args.render_episodes,
            fps=args.render_fps,
        )
        return

    from experiment_matrix import final_confirm, run_matrix, summarize_matrix_results
    from snapshot import load_snapshot
    from train_eval import run_trial, compute_objective
    from search_driver import SearchDriver, BASELINE_CONFIG, get_mode_presets
    from history_reporting import (
        HistoryManager, export_best_config, generate_summary, generate_all_reports, recheck_top_k,
    )

    presets = get_mode_presets(args.mode)
    max_trial_frames = args.max_trial_frames or presets['max_trial_frames']
    config_overrides = _collect_config_overrides(args)

    print(f"[MODE] {args.mode}  |  max_trial_frames={max_trial_frames}", flush=True)

    if args.resume:
        snapshot = load_snapshot(args.resume)
        resume_config = dict(snapshot.config)
        for key, value in config_overrides.items():
            if value is not None:
                resume_config[key] = value
        result = run_trial(
            config=resume_config,
            trial_id=snapshot.trial_id,
            seed=snapshot.seed,
            source='resume',
            trial_type='resume',
            resume_snapshot_path=args.resume,
            max_trial_frames=max_trial_frames,
            max_additional_train_raw_env_frames=max_trial_frames,
            eval_interval_frames=presets['eval_interval_frames'],
            eval_episodes=presets['eval_episodes'],
            checkpoint_dir=args.checkpoint_dir,
        )
        hm = HistoryManager(args.history)
        hm.append(result)
        export_best_config(hm)
        generate_summary(hm)
        if args.report:
            generate_all_reports(hm, args.study_db)
        return

    if args.recheck_topk:
        hm = HistoryManager(args.history)
        recheck_results = recheck_top_k(
            hm,
            k=args.recheck_topk,
            max_trial_frames=max_trial_frames,
            eval_episodes=20,
        )
        print(f"[RECHECK] Rechecked top {len(recheck_results)} configs.")
        if args.report:
            generate_all_reports(hm, args.study_db)
        return

    if args.final_confirm:
        config = load_json_config(args.final_confirm)
        result = final_confirm(
            config,
            max_trial_frames=max_trial_frames,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.matrix:
        matrix = get_matrix_by_name(args.matrix)
        results = run_matrix(matrix, mode=args.mode, budget=args.matrix_budget)
        summary = summarize_matrix_results(args.matrix, results)
        summary['budget'] = args.matrix_budget
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.baseline_only:
        print("[BASELINE-ONLY] Running single baseline trial...")
        baseline_config = dict(BASELINE_CONFIG)
        for key, value in config_overrides.items():
            if value is not None:
                baseline_config[key] = value
        result = run_trial(
            config=baseline_config, trial_id=-1, seed=11, source='baseline',
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
        export_best_config(hm)
        generate_summary(hm)
        if args.report:
            generate_all_reports(hm, args.study_db)
        return

    driver = SearchDriver(
        history_path=args.history, study_db=args.study_db,
        max_trials=args.max_trials, max_trial_frames=max_trial_frames,
        eval_interval_frames=presets['eval_interval_frames'],
        eval_episodes=presets['eval_episodes'],
        n_startup_trials=args.n_startup_trials,
        checkpoint_dir=args.checkpoint_dir,
        config_overrides=config_overrides,
    )

    try:
        if args.search_strategy == 'warmstart_tpe':
            driver.run_warmstart_tpe()
        elif args.search_strategy == 'population_async':
            driver.run_population_async(
                total_frame_budget=args.population_total_frame_budget,
                population_size=args.population_size,
                eval_interval=args.population_eval_interval,
                exploit_interval=args.population_exploit_interval,
            )
        else:
            driver.run()
        if args.report:
            generate_all_reports(driver.history, args.study_db)
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

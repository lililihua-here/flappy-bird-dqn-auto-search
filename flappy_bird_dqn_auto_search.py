"""Backward-compatible entry point. Re-exports all public names."""
from flappy_bird_env import FlappyBirdEnv
from replay_buffer import StateEncoder, ReplayBuffer
from dqn_agent import DQN, DQNAgent
from train_eval import (
    run_trial, greedy_eval, is_stable_success,
    check_early_stop, compute_objective, set_global_seed,
)
from search_driver import SearchDriver, BASELINE_CONFIG, define_search_space, get_mode_presets
from history_reporting import (
    HistoryManager, generate_summary, recheck_top_k,
    _make_serializable,
)
from main import (
    main, make_parser,
    get_best_render_record, load_agent_from_checkpoint,
    render_best_demo, _draw_render_frame,
)

if __name__ == '__main__':
    main()

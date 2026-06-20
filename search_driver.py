"""Search driver — Optuna TPE orchestration, search space, and CLI presets."""
import signal
import sys
import time

from dqn_agent import DQNAgent
from flappy_bird_env import FlappyBirdEnv
from history_reporting import HistoryManager, generate_summary
from replay_buffer import StateEncoder
from train_eval import compute_objective, greedy_eval, run_trial

try:
    import optuna
except ImportError:
    optuna = None


# ============================================================================
# Baseline default config (Section 10.5) -- P0-3: n_step=1
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
    # Stage C + D: V2 fields with V1-compatible defaults
    'priority': False,
    'per_alpha': 0.6, 'per_beta_start': 0.4, 'per_beta_train_updates': 50000,
    'per_priority_eps': 1e-6,
    'death_ratio': 1, 'alive_ratio': 0.0,
    'pipe_reward': 1.0,
    # V3.3: network/exploration family defaults
    'network_backbone': 'mlp',
    'exploration_head': 'epsilon_greedy',
    'hard_update_freq': 1000,
}


# ============================================================================
# Search Space -- Optuna parameter definition (Section 10.6)
# ============================================================================
def define_search_space(trial):
    """Define V2 search space with PER and reward ratio parameters.
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
        # Searchable (Section 10.6 — V1 baseline)
        'lr': trial.suggest_float('lr', 1e-5, 3e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.90, 0.999),
        'hidden_key': hidden_key,
        'hidden': hidden_map[hidden_key],
        'eps_start': trial.suggest_float('eps_start', 0.01, 0.15),
        'eps_end': trial.suggest_float('eps_end', 0.001, 0.02),
        'eps_decay_decision_steps': trial.suggest_int('eps_decay_decision_steps', 10000, 200000),
        'replay_start_size': trial.suggest_categorical('replay_start_size', [1000, 5000, 10000]),
        'train_freq': trial.suggest_categorical('train_freq', [1, 4]),
        'n_step': trial.suggest_categorical('n_step', [1, 3, 5]),
        # Stage C: PER parameters
        'priority': trial.suggest_categorical('priority', [False, True]),
        'per_alpha': trial.suggest_float('per_alpha', 0.3, 0.8),
        'per_beta_start': trial.suggest_float('per_beta_start', 0.3, 0.7),
        'per_beta_train_updates': trial.suggest_int('per_beta_train_updates', 50000, 500000),
        # Stage D: Reward ratio search
        'death_ratio': trial.suggest_int('death_ratio', 5, 100),
        'alive_ratio': trial.suggest_float('alive_ratio', 0.0, 0.01),
        'reward_scale': trial.suggest_categorical('reward_scale', [0.01, 0.1, 1.0]),
        'reward_clip': trial.suggest_categorical('reward_clip', [None, 10, 100]),
        'pipe_reward': 1.0,  # fixed anchor
        # V3.3: network/exploration/optimizer/target-update family
        'network_backbone': trial.suggest_categorical('network_backbone', ['mlp', 'dueling_mlp']),
        'exploration_head': trial.suggest_categorical('exploration_head', ['epsilon_greedy', 'noisy_net']),
        'torch_optimizer': trial.suggest_categorical('torch_optimizer', ['Adam', 'AdamW', 'RMSprop']),
        'target_update_mode': trial.suggest_categorical('target_update_mode', ['soft', 'hard']),
        # MVP fixed (Section 10.3, 10.4)
        'double_q': True,
        'frame_skip': 1,
        'tau': 0.005,
        'loss_type': 'Huber',
        'grad_clip_norm': 5,
        'batch_sz': 64,
        'buffer_sz': 50000,
        'reward_pipe': 1.0,
        'reward_death': -1.0,
        'reward_alive': 0.0,
        # V3.2: State/reward protocol versioning
        'state_representation_version': trial.suggest_categorical(
            'state_representation_version', ['low_dim_v1', 'low_dim_v2', 'low_dim_v3']),
        'reward_scheme_version': trial.suggest_categorical(
            'reward_scheme_version', ['reward_v1_sparse', 'reward_v2_ratio', 'reward_v3_gap_shaping']),
    }


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

    def run_population_async(self, total_frame_budget=None, population_size=4,
                              eval_interval=20_000, exploit_interval=50_000):
        """Run population-based asynchronous self-evolution search.

        Warm-starts workers from top K history configs, then runs the
        PopulationController exploit/explore loop for the given frame budget.
        """
        from population import PopulationController

        if total_frame_budget is None:
            total_frame_budget = self.max_trial_frames * 5

        pc = PopulationController(
            population_size=population_size,
            history=self.history,
            eval_interval=eval_interval,
            exploit_interval=exploit_interval,
        )

        # Warm-start: seed workers from top K history configs
        top_configs = self.history.top_k(population_size)
        for i, row in enumerate(top_configs):
            config = dict(row.get('config', {}))
            if not config:
                continue
            seed = self.seed_pool[i % len(self.seed_pool)]
            trial_id = pc._next_trial_id()
            pc.add_worker(trial_id, config, seed)

        # Fill remaining slots with fresh baseline-config workers
        while len(pc.workers) < population_size:
            config = dict(BASELINE_CONFIG)
            seed = self.seed_pool[len(pc.workers) % len(self.seed_pool)]
            trial_id = pc._next_trial_id()
            pc.add_worker(trial_id, config, seed)

        print(f"\n[POPULATION] {len(pc.workers)} workers, "
              f"budget={total_frame_budget:,} frames")
        pc.run(total_frame_budget)
        generate_summary(self.history)

    def _sigint_handler(self, signum, frame):
        """P0-4: Set flag only. Let current trial finish, then stop."""
        print("\n[Ctrl+C] Will stop after current trial completes. Press again to force-quit.")
        if self._interrupted:
            print("[Ctrl+C] Force quitting...")
            sys.exit(1)
        self._interrupted = True


def get_mode_presets(mode):
    presets = {
        'debug':    {'max_trial_frames': 100_000, 'eval_interval_frames': 10_000, 'eval_episodes': 3},
        'normal':   {'max_trial_frames': 1_000_000, 'eval_interval_frames': 20_000, 'eval_episodes': 5},
        'deep':     {'max_trial_frames': 5_000_000, 'eval_interval_frames': 50_000, 'eval_episodes': 20},
    }
    if mode not in presets:
        raise ValueError(f"Unknown mode: {mode}")
    return presets[mode]

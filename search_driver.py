"""Search driver — Optuna TPE orchestration, search space, and CLI presets."""
import signal
import sys
import time

from dqn_agent import DQNAgent
from flappy_bird_env import FlappyBirdEnv
from history_reporting import HistoryManager, export_best_config, generate_summary
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
    # V3.2 protocol defaults
    'state_representation_version': 'low_dim_v1',
    'reward_scheme_version': 'reward_v1_sparse',
    'gap_shaping_coef': 0.0,
}


HIDDEN_KEY_TO_LAYERS = {
    'small': [64, 32],
    'medium': [128, 64],
    'large': [256, 128],
}


# ============================================================================
# Search Space -- Optuna parameter definition (Section 10.6)
# ============================================================================
def _categorical_choices(name, default_choices, focused_search_space):
    if not focused_search_space:
        return default_choices
    return focused_search_space.get('categorical', {}).get(name, default_choices)


def _float_bounds(name, default_low, default_high, default_log, focused_search_space):
    if not focused_search_space:
        return default_low, default_high, default_log
    spec = focused_search_space.get('continuous', {}).get(name)
    if not spec:
        return default_low, default_high, default_log
    return spec.get('low', default_low), spec.get('high', default_high), spec.get('log', default_log)


def _int_bounds(name, default_low, default_high, focused_search_space):
    if not focused_search_space:
        return default_low, default_high
    spec = focused_search_space.get('continuous', {}).get(name)
    if not spec:
        return default_low, default_high
    return spec.get('low', default_low), spec.get('high', default_high)


def define_search_space(trial, overrides=None, focused_search_space=None):
    """Define the V3 search space with protocol and structure variants."""
    hidden_key = trial.suggest_categorical(
        'hidden_key',
        _categorical_choices('hidden_key', ['small', 'medium', 'large'], focused_search_space),
    )
    lr_low, lr_high, lr_log = _float_bounds('lr', 1e-5, 3e-3, True, focused_search_space)
    gamma_low, gamma_high, gamma_log = _float_bounds('gamma', 0.90, 0.999, False, focused_search_space)
    eps_decay_low, eps_decay_high = _int_bounds(
        'eps_decay_decision_steps', 10000, 200000, focused_search_space
    )
    per_beta_updates_low, per_beta_updates_high = _int_bounds(
        'per_beta_train_updates', 50000, 500000, focused_search_space
    )
    death_ratio_low, death_ratio_high = _int_bounds('death_ratio', 5, 100, focused_search_space)
    config = {
        # Core hyperparameters
        'lr': trial.suggest_float('lr', lr_low, lr_high, log=lr_log),
        'gamma': trial.suggest_float('gamma', gamma_low, gamma_high, log=gamma_log),
        'hidden_key': hidden_key,
        'hidden': HIDDEN_KEY_TO_LAYERS[hidden_key],
        'eps_start': trial.suggest_float('eps_start', 0.01, 0.15),
        'eps_end': trial.suggest_float('eps_end', 0.001, 0.02),
        'eps_decay_decision_steps': trial.suggest_int(
            'eps_decay_decision_steps', eps_decay_low, eps_decay_high
        ),
        'replay_start_size': trial.suggest_categorical(
            'replay_start_size',
            _categorical_choices('replay_start_size', [1000, 5000, 10000], focused_search_space),
        ),
        'train_freq': trial.suggest_categorical(
            'train_freq',
            _categorical_choices('train_freq', [1, 4], focused_search_space),
        ),
        'n_step': trial.suggest_categorical(
            'n_step',
            _categorical_choices('n_step', [1, 3, 5], focused_search_space),
        ),
        # Stage C: PER parameters
        'priority': trial.suggest_categorical(
            'priority',
            _categorical_choices('priority', [False, True], focused_search_space),
        ),
        'per_alpha': trial.suggest_float('per_alpha', 0.3, 0.8),
        'per_beta_start': trial.suggest_float('per_beta_start', 0.3, 0.7),
        'per_beta_train_updates': trial.suggest_int(
            'per_beta_train_updates', per_beta_updates_low, per_beta_updates_high
        ),
        # Reward / protocol family
        'death_ratio': trial.suggest_int('death_ratio', death_ratio_low, death_ratio_high),
        'alive_ratio': trial.suggest_float('alive_ratio', 0.0, 0.01),
        'reward_scale': trial.suggest_categorical(
            'reward_scale',
            _categorical_choices('reward_scale', [0.01, 0.1, 1.0], focused_search_space),
        ),
        'reward_clip': trial.suggest_categorical(
            'reward_clip',
            _categorical_choices('reward_clip', [None, 10, 100], focused_search_space),
        ),
        'reward_scheme_version': trial.suggest_categorical(
            'reward_scheme_version',
            _categorical_choices(
                'reward_scheme_version',
                ['reward_v1_sparse', 'reward_v2_ratio', 'reward_v3_gap_shaping'],
                focused_search_space,
            ),
        ),
        'gap_shaping_coef': trial.suggest_categorical(
            'gap_shaping_coef',
            _categorical_choices('gap_shaping_coef', [0.0, 0.01, 0.05, 0.1], focused_search_space),
        ),
        'state_representation_version': trial.suggest_categorical(
            'state_representation_version',
            _categorical_choices(
                'state_representation_version',
                ['low_dim_v1', 'low_dim_v2', 'low_dim_v3'],
                focused_search_space,
            ),
        ),
        'pipe_reward': 1.0,  # fixed anchor
        # V3 structure family
        'network_backbone': trial.suggest_categorical(
            'network_backbone',
            _categorical_choices('network_backbone', ['mlp', 'dueling_mlp'], focused_search_space),
        ),
        'exploration_head': trial.suggest_categorical(
            'exploration_head',
            _categorical_choices('exploration_head', ['epsilon_greedy', 'noisy_net'], focused_search_space),
        ),
        # Fixed defaults
        'double_q': True,
        'frame_skip': 1,
        'torch_optimizer': 'Adam',
        'target_update_mode': 'soft',
        'tau': 0.005,
        'loss_type': 'Huber',
        'grad_clip_norm': 5,
        'batch_sz': 64,
        'buffer_sz': 50000,
        'reward_pipe': 1.0,
        'reward_death': -1.0,
        'reward_alive': 0.0,
    }
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    return config


def _hidden_layers_to_key(hidden):
    hidden_list = list(hidden or [])
    for key, value in HIDDEN_KEY_TO_LAYERS.items():
        if list(value) == hidden_list:
            return key
    return 'medium'


def config_to_search_params(config):
    """Map a full config dict back to the Optuna search-space parameter dict."""
    return {
        'lr': config.get('lr', BASELINE_CONFIG['lr']),
        'gamma': config.get('gamma', BASELINE_CONFIG['gamma']),
        'hidden_key': config.get('hidden_key', _hidden_layers_to_key(config.get('hidden'))),
        'eps_start': config.get('eps_start', BASELINE_CONFIG['eps_start']),
        'eps_end': config.get('eps_end', BASELINE_CONFIG['eps_end']),
        'eps_decay_decision_steps': config.get(
            'eps_decay_decision_steps', BASELINE_CONFIG['eps_decay_decision_steps']
        ),
        'replay_start_size': config.get(
            'replay_start_size', BASELINE_CONFIG['replay_start_size']
        ),
        'train_freq': config.get('train_freq', BASELINE_CONFIG['train_freq']),
        'n_step': config.get('n_step', BASELINE_CONFIG['n_step']),
        'priority': config.get('priority', BASELINE_CONFIG['priority']),
        'per_alpha': config.get('per_alpha', BASELINE_CONFIG['per_alpha']),
        'per_beta_start': config.get('per_beta_start', BASELINE_CONFIG['per_beta_start']),
        'per_beta_train_updates': config.get(
            'per_beta_train_updates', BASELINE_CONFIG['per_beta_train_updates']
        ),
        'death_ratio': config.get('death_ratio', BASELINE_CONFIG['death_ratio']),
        'alive_ratio': config.get('alive_ratio', BASELINE_CONFIG['alive_ratio']),
        'reward_scale': config.get('reward_scale', BASELINE_CONFIG['reward_scale']),
        'reward_clip': config.get('reward_clip', BASELINE_CONFIG['reward_clip']),
        'reward_scheme_version': config.get(
            'reward_scheme_version', BASELINE_CONFIG['reward_scheme_version']
        ),
        'gap_shaping_coef': config.get(
            'gap_shaping_coef', BASELINE_CONFIG['gap_shaping_coef']
        ),
        'state_representation_version': config.get(
            'state_representation_version', BASELINE_CONFIG['state_representation_version']
        ),
        'network_backbone': config.get(
            'network_backbone', BASELINE_CONFIG['network_backbone']
        ),
        'exploration_head': config.get(
            'exploration_head', BASELINE_CONFIG['exploration_head']
        ),
    }


def _completed_trial_count(study):
    return sum(1 for trial in study.trials if trial.state.is_finished())


class SearchDriver:
    """Orchestrates the full hyperparameter search with Optuna TPE."""

    def __init__(self, history_path='search_history.jsonl', study_db='optuna_study.db',
                 max_trials=100, max_trial_frames=1_000_000,
                 eval_interval_frames=20_000, eval_episodes=5,
                 candidate_verify_episodes=20, n_startup_trials=30,
                 seed_pool=(11, 22, 33), checkpoint_dir='checkpoints',
                 config_overrides=None, focused_search_space=None,
                 run_baseline_first=True):
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
        self.config_overrides = dict(config_overrides or {})
        self.focused_search_space = focused_search_space
        self.run_baseline_first = run_baseline_first
        self._interrupted = False

    def _create_study(self):
        return optuna.create_study(
            study_name='flappy_bird_dqn_search',
            storage=f'sqlite:///{self.study_db}',
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=42,
            ),
            load_if_exists=True,
        )

    def _run_baseline_if_needed(self, existing_trials):
        history_rows = self.history.load()
        has_baseline = any(r.get('source') == 'baseline' for r in history_rows)
        if (not self.run_baseline_first) or existing_trials != 0 or has_baseline:
            return

        print("\n[STAGE 0] Baseline verification (independent, not counted in max_trials)...")
        baseline_config = dict(BASELINE_CONFIG)
        for key, value in self.config_overrides.items():
            if value is not None:
                baseline_config[key] = value
        result = run_trial(
            config=baseline_config, trial_id=-1, seed=11, source='baseline',
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
        export_best_config(self.history)
        print(f"[STAGE 0] Baseline complete. status={result['status']}\n")

    def _optimize_study(self, study, remaining):
        print(f"[INFO] Study: {_completed_trial_count(study)} Optuna trials completed, {remaining} remaining")
        print(f"[INFO] Max trial frames: {self.max_trial_frames}")

        original_handler = signal.signal(signal.SIGINT, self._sigint_handler)
        try:
            study.optimize(self._objective, n_trials=remaining)
        finally:
            signal.signal(signal.SIGINT, original_handler)

        if self._interrupted:
            print("\n[Ctrl+C] Search stopped by user. History saved.")
        generate_summary(self.history)

    def _objective(self, trial):
        """Optuna objective. P0-5: trial_id = trial.number."""
        config = define_search_space(trial, self.config_overrides, self.focused_search_space)
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
        export_best_config(self.history)

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

        study = self._create_study()

        existing = _completed_trial_count(study)
        remaining = max(0, self.max_trials - existing)
        self._run_baseline_if_needed(existing)
        self._optimize_study(study, remaining)

    def run_warmstart_tpe(self, warmstart_top_k=5):
        """Run TPE with historical top configs enqueued as warm-start seeds."""
        if optuna is None:
            raise ImportError('optuna required. Install: pip install optuna')

        study = self._create_study()
        existing = _completed_trial_count(study)
        self._run_baseline_if_needed(existing)

        if existing == 0:
            seen = set()
            for row in self.history.top_k(warmstart_top_k):
                config = row.get('config') or {}
                if not config:
                    continue
                params = config_to_search_params(config)
                params_key = tuple(sorted(params.items(), key=lambda item: item[0]))
                if params_key in seen:
                    continue
                study.enqueue_trial(params)
                seen.add(params_key)

        remaining = max(0, self.max_trials - _completed_trial_count(study))
        self._optimize_study(study, remaining)

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
            checkpoint_dir=self.checkpoint_dir,
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
            pc.workers[-1]['snapshot_path'] = row.get('last_snapshot_path', '')
            pc.workers[-1]['resume_snapshot_path'] = row.get('last_snapshot_path', None)
            pc.workers[-1]['parent_trial_id'] = row.get('trial_id')
            pc.workers[-1]['parent_snapshot_ref'] = row.get('last_snapshot_path', '')

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
        export_best_config(self.history)

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

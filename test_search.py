"""Contract tests for search_driver module."""
import os
import subprocess
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Search space test
# ============================================================================
def test_search_space_produces_valid_config():
    import optuna
    from search_driver import define_search_space

    def objective(trial):
        config = define_search_space(trial)
        required = ['lr', 'gamma', 'hidden', 'hidden_key', 'eps_start', 'eps_end',
                    'eps_decay_decision_steps', 'replay_start_size', 'train_freq',
                    'n_step', 'priority', 'per_alpha', 'per_beta_start',
                    'per_beta_train_updates', 'death_ratio', 'alive_ratio',
                    'reward_scale', 'reward_clip', 'pipe_reward',
                    'reward_scheme_version', 'gap_shaping_coef',
                    'state_representation_version', 'network_backbone',
                    'exploration_head']
        for k in required:
            assert k in config, f"Missing: {k}"
        assert 1e-5 <= config['lr'] <= 3e-3
        assert 0.90 <= config['gamma'] <= 0.999
        assert config['hidden_key'] in ('small', 'medium', 'large')
        assert config['hidden'] in ([64, 32], [128, 64], [256, 128])
        assert 0.01 <= config['eps_start'] <= 0.15
        assert 0.001 <= config['eps_end'] <= 0.02
        assert 10000 <= config['eps_decay_decision_steps'] <= 200000
        assert config['replay_start_size'] in (1000, 5000, 10000)
        assert config['train_freq'] in (1, 4)
        assert config['n_step'] in (1, 3, 5)
        assert config['priority'] in (False, True)
        assert 0.3 <= config['per_alpha'] <= 0.8
        assert 0.3 <= config['per_beta_start'] <= 0.7
        assert 50000 <= config['per_beta_train_updates'] <= 500000
        assert 5 <= config['death_ratio'] <= 100
        assert 0.0 <= config['alive_ratio'] <= 0.01
        assert config['reward_scale'] in (0.01, 0.1, 1.0)
        assert config['reward_clip'] in (None, 10, 100)
        assert config['pipe_reward'] == 1.0
        assert config['reward_scheme_version'] in (
            'reward_v1_sparse', 'reward_v2_ratio', 'reward_v3_gap_shaping'
        )
        assert config['gap_shaping_coef'] in (0.0, 0.01, 0.05, 0.1)
        assert config['state_representation_version'] in (
            'low_dim_v1', 'low_dim_v2', 'low_dim_v3'
        )
        assert config['network_backbone'] in ('mlp', 'dueling_mlp')
        assert config['exploration_head'] in ('epsilon_greedy', 'noisy_net')
        return 0.0

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=10)
    assert len(study.trials) == 10


def test_search_space_applies_cli_overrides():
    from search_driver import define_search_space

    class DummyTrial:
        def suggest_float(self, name, *args, **kwargs):
            if name == 'lr':
                return 1e-4
            if name == 'gamma':
                return 0.99
            if name == 'eps_start':
                return 0.05
            if name == 'eps_end':
                return 0.005
            if name == 'per_alpha':
                return 0.5
            if name == 'per_beta_start':
                return 0.35
            if name == 'alive_ratio':
                return 0.001
            raise AssertionError(f'unexpected float param: {name}')

        def suggest_int(self, name, *args, **kwargs):
            if name == 'eps_decay_decision_steps':
                return 50000
            if name == 'per_beta_train_updates':
                return 200000
            if name == 'death_ratio':
                return 20
            raise AssertionError(f'unexpected int param: {name}')

        def suggest_categorical(self, name, choices):
            mapping = {
                'hidden_key': 'medium',
                'replay_start_size': 5000,
                'train_freq': 1,
                'n_step': 1,
                'priority': False,
                'reward_scale': 1.0,
                'reward_clip': None,
                'reward_scheme_version': 'reward_v1_sparse',
                'gap_shaping_coef': 0.0,
                'state_representation_version': 'low_dim_v1',
                'network_backbone': 'mlp',
                'exploration_head': 'epsilon_greedy',
            }
            if name not in mapping:
                raise AssertionError(f'unexpected categorical param: {name}')
            return mapping[name]

    config = define_search_space(
        DummyTrial(),
        overrides={
            'n_step': 3,
            'priority': True,
            'per_alpha': 0.6,
            'per_beta_start': 0.4,
            'reward_scale': 0.1,
            'reward_clip': 10,
        },
    )

    assert config['n_step'] == 3
    assert config['priority'] is True
    assert config['per_alpha'] == 0.6
    assert config['per_beta_start'] == 0.4
    assert config['reward_scale'] == 0.1
    assert config['reward_clip'] == 10


def test_search_space_respects_focused_search_space():
    from search_driver import define_search_space

    class DummyTrial:
        def suggest_float(self, name, low, high, **kwargs):
            if name == 'lr':
                assert low == 5e-05
                assert high == 0.0002
                assert kwargs.get('log') is True
                return 1e-4
            if name == 'gamma':
                assert low == 0.97
                assert high == 0.995
                return 0.99
            if name == 'eps_start':
                return 0.05
            if name == 'eps_end':
                return 0.005
            if name == 'per_alpha':
                return 0.6
            if name == 'per_beta_start':
                return 0.4
            if name == 'alive_ratio':
                return 0.001
            raise AssertionError(f'unexpected float param: {name}')

        def suggest_int(self, name, low, high, **kwargs):
            if name == 'eps_decay_decision_steps':
                return 50000
            if name == 'per_beta_train_updates':
                return 200000
            if name == 'death_ratio':
                return 20
            raise AssertionError(f'unexpected int param: {name}')

        def suggest_categorical(self, name, choices):
            expected = {
                'hidden_key': ['medium'],
                'replay_start_size': [5000],
                'train_freq': [1],
                'n_step': [1],
                'priority': [True],
                'reward_scale': [0.1],
                'reward_clip': [10],
                'reward_scheme_version': ['reward_v2_ratio'],
                'gap_shaping_coef': [0.05],
                'state_representation_version': ['low_dim_v1'],
                'network_backbone': ['dueling_mlp'],
                'exploration_head': ['noisy_net'],
            }
            assert choices == expected[name]
            return choices[0]

    focused = {
        'categorical': {
            'hidden_key': ['medium'],
            'replay_start_size': [5000],
            'train_freq': [1],
            'n_step': [1],
            'priority': [True],
            'reward_scale': [0.1],
            'reward_clip': [10],
            'reward_scheme_version': ['reward_v2_ratio'],
            'gap_shaping_coef': [0.05],
            'state_representation_version': ['low_dim_v1'],
            'network_backbone': ['dueling_mlp'],
            'exploration_head': ['noisy_net'],
        },
        'continuous': {
            'lr': {'low': 5e-05, 'high': 0.0002, 'log': True},
            'gamma': {'low': 0.97, 'high': 0.995},
        },
    }

    config = define_search_space(DummyTrial(), focused_search_space=focused)
    assert config['reward_scheme_version'] == 'reward_v2_ratio'
    assert config['network_backbone'] == 'dueling_mlp'
    assert config['exploration_head'] == 'noisy_net'
    assert config['hidden'] == [128, 64]


# ============================================================================
# Search driver test
# ============================================================================
def test_search_driver_runs_n_trials():
    import tempfile
    from search_driver import SearchDriver

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    db_path = tmp.name.replace('.jsonl', '.db')

    try:
        driver = SearchDriver(
            history_path=tmp.name, study_db=db_path,
            max_trials=3, max_trial_frames=3000,
            eval_interval_frames=1000, eval_episodes=2,
            candidate_verify_episodes=3,
            n_startup_trials=2, seed_pool=[42, 43, 44],
        )
        driver.run()
        rows = driver.history.load()
        assert len(rows) >= 1
        for r in rows:
            if r.get('record_type') == 'trial':
                assert 'trial_id' in r and 'objective' in r and r['objective'] > 0
    finally:
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass


# ============================================================================
# Mode presets test
# ============================================================================
def test_mode_presets():
    from search_driver import get_mode_presets
    debug = get_mode_presets('debug')
    assert debug['max_trial_frames'] == 100_000
    assert debug['eval_interval_frames'] == 10_000
    assert debug['eval_episodes'] == 3


# ============================================================================
# Parser tests (make_parser still lives in flappy_bird_dqn_auto_search)
# ============================================================================
def test_parser_defaults():
    from main import make_parser
    args = make_parser().parse_args([])
    assert args.mode == 'normal'
    assert args.max_trials == 100
    assert args.search_strategy == 'tpe_fresh'
    assert args.matrix_budget == 'debug_matrix'


def test_parser_render_flags():
    from main import make_parser
    args = make_parser().parse_args([
        '--render',
        '--render-episodes', '2',
        '--render-fps', '30',
        '--checkpoint-dir', 'my_ckpts',
        '--report',
        '--n-step', '3',
        '--priority',
        '--per-alpha', '0.6',
        '--per-beta-start', '0.4',
        '--per-beta-train-updates', '12345',
        '--death-ratio', '10',
        '--alive-ratio', '0.001',
        '--reward-scale', '0.1',
        '--reward-clip', '10',
        '--search-strategy', 'population_async',
        '--population-size', '3',
        '--population-eval-interval', '1234',
        '--population-exploit-interval', '5678',
        '--population-total-frame-budget', '9999',
    ])
    assert args.render is True
    assert args.render_episodes == 2
    assert args.render_fps == 30
    assert args.checkpoint_dir == 'my_ckpts'
    assert args.report is True
    assert args.n_step == 3
    assert args.priority is True
    assert args.per_alpha == 0.6
    assert args.per_beta_start == 0.4
    assert args.per_beta_train_updates == 12345
    assert args.death_ratio == 10
    assert args.alive_ratio == 0.001
    assert args.reward_scale == 0.1
    assert args.reward_clip == 10.0
    assert args.search_strategy == 'population_async'
    assert args.population_size == 3
    assert args.population_eval_interval == 1234
    assert args.population_exploit_interval == 5678
    assert args.population_total_frame_budget == 9999


def test_parser_v3_cli_flags():
    from main import make_parser
    args = make_parser().parse_args([
        '--resume', 'snap.pt',
        '--recheck-topk', '5',
        '--final-confirm', 'config.json',
        '--matrix', 'protocol',
        '--matrix-budget', 'normal_matrix',
        '--search-strategy', 'warmstart_tpe',
    ])
    assert args.resume == 'snap.pt'
    assert args.recheck_topk == 5
    assert args.final_confirm == 'config.json'
    assert args.matrix == 'protocol'
    assert args.matrix_budget == 'normal_matrix'
    assert args.search_strategy == 'warmstart_tpe'


def test_get_matrix_by_name():
    from experiment_matrix import BASELINE_MATRIX, PROTOCOL_ABLATION
    from main import get_matrix_by_name

    assert get_matrix_by_name('baseline') is BASELINE_MATRIX
    assert get_matrix_by_name('protocol') is PROTOCOL_ABLATION


def test_load_json_config(tmp_path):
    from main import load_json_config

    path = tmp_path / 'config.json'
    path.write_text('{"lr": 0.0001, "n_step": 3}', encoding='utf-8')
    assert load_json_config(str(path)) == {'lr': 0.0001, 'n_step': 3}


def test_import_main_is_lightweight():
    script = (
        'import sys; '
        'import main; '
        'print("torch" in sys.modules); '
        'print("optuna" in sys.modules)'
    )
    proc = subprocess.run(
        [sys.executable, '-c', script],
        cwd=os.path.dirname(__file__),
        text=True,
        capture_output=True,
        check=True,
    )
    assert proc.stdout.splitlines() == ['False', 'False']

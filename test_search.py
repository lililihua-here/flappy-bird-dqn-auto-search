"""Contract tests for search_driver module."""
import os
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
                    'reward_scale', 'reward_clip', 'pipe_reward']
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
        return 0.0

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=10)
    assert len(study.trials) == 10


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
    from flappy_bird_dqn_auto_search import make_parser
    args = make_parser().parse_args([])
    assert args.mode == 'normal'
    assert args.max_trials == 100


def test_parser_render_flags():
    from flappy_bird_dqn_auto_search import make_parser
    args = make_parser().parse_args([
        '--render',
        '--render-episodes', '2',
        '--render-fps', '30',
        '--checkpoint-dir', 'my_ckpts',
        '--report',
    ])
    assert args.render is True
    assert args.render_episodes == 2
    assert args.render_fps == 30
    assert args.checkpoint_dir == 'my_ckpts'
    assert args.report is True

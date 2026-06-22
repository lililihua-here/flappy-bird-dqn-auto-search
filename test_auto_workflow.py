"""Contract tests for the V0.1a auto workflow entrypoint."""
import json

import pytest


def test_workflow_state_roundtrip(tmp_path):
    from workflow_state import WorkflowState, load_workflow_state

    state = WorkflowState.new(
        goal='stable_1000',
        profile='cpu_quick',
        run_dir=tmp_path,
    )
    state.current_stage = 'baseline_check'
    state.completed_stages.append('init')
    state.save()

    loaded = load_workflow_state(tmp_path / 'workflow_state.json')
    assert loaded.goal == 'stable_1000'
    assert loaded.profile == 'cpu_quick'
    assert loaded.current_stage == 'baseline_check'
    assert loaded.completed_stages == ['init']
    assert loaded.stage_status_path == str(tmp_path / 'stage_status.json')


def test_auto_workflow_baseline_check_writes_state_and_stage_status(tmp_path, monkeypatch):
    import auto_workflow as mod

    calls = {'count': 0}

    def fake_run_trial(**kwargs):
        calls['count'] += 1
        return {
            'trial_id': -1,
            'seed': kwargs['seed'],
            'source': kwargs['source'],
            'status': 'failure',
            'failure_reason': 'max_frames_reached',
            'train_raw_env_frames': 1234,
            'best_eval_score': 17,
            'median_score': 3.0,
            'success_rate_1000': 0.0,
            'config': dict(kwargs['config']),
        }

    monkeypatch.setattr(mod, 'run_trial', fake_run_trial)
    monkeypatch.setattr(mod, 'compute_objective', lambda **kwargs: 9999.0)

    state = mod.WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'baseline_check'
    state.save()
    result = mod.run_baseline_stage(state)

    assert calls['count'] == 1
    assert result.current_stage == 'protocol_matrix'
    assert 'baseline_check' in result.completed_stages

    workflow_state_path = tmp_path / 'workflow_state.json'
    stage_status_path = tmp_path / 'stage_status.json'
    history_path = tmp_path / 'baseline' / 'history.jsonl'

    assert workflow_state_path.exists()
    assert stage_status_path.exists()
    assert history_path.exists()

    workflow_payload = json.loads(workflow_state_path.read_text(encoding='utf-8'))
    assert workflow_payload['current_stage'] == 'protocol_matrix'

    stage_payload = json.loads(stage_status_path.read_text(encoding='utf-8'))
    baseline_stage = stage_payload['baseline_check']
    assert baseline_stage['status'] == 'completed'
    assert baseline_stage['history_path'].endswith('baseline/history.jsonl')
    assert history_path.as_posix() in [p.replace('\\', '/') for p in baseline_stage['output_files']]


def test_auto_workflow_baseline_check_is_idempotent(tmp_path, monkeypatch):
    import auto_workflow as mod

    calls = {'count': 0}

    def fake_run_trial(**kwargs):
        calls['count'] += 1
        return {
            'trial_id': -1,
            'seed': kwargs['seed'],
            'source': kwargs['source'],
            'status': 'failure',
            'failure_reason': 'max_frames_reached',
            'train_raw_env_frames': 1234,
            'best_eval_score': 17,
            'median_score': 3.0,
            'success_rate_1000': 0.0,
            'config': dict(kwargs['config']),
        }

    monkeypatch.setattr(mod, 'run_trial', fake_run_trial)
    monkeypatch.setattr(mod, 'compute_objective', lambda **kwargs: 9999.0)
    monkeypatch.setattr(mod, 'run_matrix', lambda matrix, mode, budget: [])
    monkeypatch.setattr(mod, 'summarize_matrix_results', lambda matrix_name, results: {
        'matrix': matrix_name,
        'entries': [],
    })

    state = mod.WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'baseline_check'
    state.save()
    mod.run_baseline_stage(state)
    resumed = mod.load_workflow_state(tmp_path / 'workflow_state.json')
    resumed.current_stage = 'baseline_check'
    resumed.save()
    mod.run_baseline_stage(resumed)

    assert calls['count'] == 1


def test_auto_workflow_stage_with_invalid_summary_is_not_skipped(tmp_path, monkeypatch):
    import auto_workflow as mod

    calls = {'count': 0}

    def fake_run_trial(**kwargs):
        calls['count'] += 1
        return {
            'trial_id': -1,
            'seed': kwargs['seed'],
            'source': kwargs['source'],
            'status': 'failure',
            'failure_reason': 'max_frames_reached',
            'train_raw_env_frames': 1234,
            'best_eval_score': 17,
            'median_score': 3.0,
            'success_rate_1000': 0.0,
            'config': dict(kwargs['config']),
        }

    monkeypatch.setattr(mod, 'run_trial', fake_run_trial)
    monkeypatch.setattr(mod, 'compute_objective', lambda **kwargs: 9999.0)

    state = mod.WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'baseline_check'
    state.save()
    mod.run_baseline_stage(state)

    summary_path = tmp_path / 'baseline' / 'baseline_summary.json'
    summary_path.write_text('{broken json', encoding='utf-8')

    resumed = mod.load_workflow_state(tmp_path / 'workflow_state.json')
    resumed.current_stage = 'baseline_check'
    resumed.save()
    mod.run_baseline_stage(resumed)

    assert calls['count'] == 2


def test_auto_workflow_parser_defaults():
    from auto_workflow import make_parser

    args = make_parser().parse_args([])
    assert args.goal == 'stable_1000'
    assert args.profile == 'cpu_normal'
    assert args.run_dir == 'runs/auto_run1'


def test_auto_workflow_protocol_matrix_writes_history_and_summary(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'protocol_matrix'
    state.save()

    monkeypatch.setattr(mod, 'run_matrix', lambda matrix, mode, budget: [
        {
            'record_type': 'trial',
            'trial_id': 101,
            'source': 'matrix_v1_state_v1_reward',
            'status': 'failure',
            'objective': 1234.0,
            'median_score': 10.0,
            'success_rate_1000': 0.0,
            'best_eval_score': 439.0,
            'failure_reason': 'plateau_100k',
        },
    ])
    monkeypatch.setattr(mod, 'summarize_matrix_results', lambda matrix_name, results: {
        'matrix': matrix_name,
        'entries': [{'name': 'v1_state_v1_reward', 'trial_count': 1}],
    })

    result = mod.run_matrix_stage(
        state=state,
        stage_name='protocol_matrix',
        matrix_name='protocol',
        matrix=[],
        next_stage='structure_matrix',
    )

    assert result.current_stage == 'structure_matrix'
    assert 'protocol_matrix' in result.completed_stages
    assert (tmp_path / 'protocol_matrix' / 'history.jsonl').exists()
    assert (tmp_path / 'protocol_matrix' / 'summary.json').exists()


def test_auto_workflow_structure_matrix_advances_to_candidate_derivation(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'structure_matrix'
    state.save()

    monkeypatch.setattr(mod, 'run_matrix', lambda matrix, mode, budget: [
        {
            'record_type': 'trial',
            'trial_id': 201,
            'source': 'matrix_dueling_noisy',
            'status': 'failure',
            'objective': 1100.0,
            'median_score': 25.0,
            'success_rate_1000': 0.0,
            'best_eval_score': 300.0,
            'failure_reason': 'no_learning_50k',
        },
    ])
    monkeypatch.setattr(mod, 'summarize_matrix_results', lambda matrix_name, results: {
        'matrix': matrix_name,
        'entries': [{'name': 'dueling_noisy', 'trial_count': 1}],
    })

    result = mod.run_matrix_stage(
        state=state,
        stage_name='structure_matrix',
        matrix_name='structure',
        matrix=[],
        next_stage='derive_candidates_and_search_space',
    )

    assert result.current_stage == 'derive_candidates_and_search_space'
    assert 'structure_matrix' in result.completed_stages
    assert (tmp_path / 'structure_matrix' / 'history.jsonl').exists()
    assert (tmp_path / 'structure_matrix' / 'summary.json').exists()


def test_auto_workflow_derives_candidates_and_writes_focused_space(tmp_path):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    protocol_dir = tmp_path / 'protocol_matrix'
    structure_dir = tmp_path / 'structure_matrix'
    protocol_dir.mkdir(parents=True)
    structure_dir.mkdir(parents=True)

    protocol_rows = [
        {
            'record_type': 'trial',
            'trial_id': 11,
            'source': 'matrix_v1_state_v2_reward',
            'status': 'failure',
            'objective': 1200.0,
            'best_eval_score': 439.0,
            'median_score': 90.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'plateau_100k',
            'config': {
                'state_representation_version': 'low_dim_v1',
                'reward_scheme_version': 'reward_v2_ratio',
                'reward_scale': 0.1,
                'reward_clip': 10,
                'death_ratio': 20,
                'alive_ratio': 0.001,
                'lr': 1e-4,
                'gamma': 0.99,
                'n_step': 1,
                'priority': True,
            },
        },
        {
            'record_type': 'trial',
            'trial_id': 12,
            'source': 'matrix_v1_state_v3_reward',
            'status': 'failure',
            'objective': 1300.0,
            'best_eval_score': 300.0,
            'median_score': 60.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'no_learning_50k',
            'config': {
                'state_representation_version': 'low_dim_v1',
                'reward_scheme_version': 'reward_v3_gap_shaping',
                'reward_scale': 0.1,
                'reward_clip': 10,
                'gap_shaping_coef': 0.05,
                'death_ratio': 20,
                'alive_ratio': 0.001,
                'lr': 8e-5,
                'gamma': 0.98,
                'n_step': 3,
                'priority': True,
            },
        },
    ]
    structure_rows = [
        {
            'record_type': 'trial',
            'trial_id': 21,
            'source': 'matrix_dueling_noisy',
            'status': 'failure',
            'objective': 900.0,
            'best_eval_score': 500.0,
            'median_score': 120.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'plateau_100k',
            'config': {
                'network_backbone': 'dueling_mlp',
                'exploration_head': 'noisy_net',
                'lr': 1e-4,
                'gamma': 0.99,
                'n_step': 1,
                'priority': True,
            },
        },
        {
            'record_type': 'trial',
            'trial_id': 22,
            'source': 'matrix_dueling_eps',
            'status': 'failure',
            'objective': 1100.0,
            'best_eval_score': 250.0,
            'median_score': 55.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'no_learning_50k',
            'config': {
                'network_backbone': 'dueling_mlp',
                'exploration_head': 'epsilon_greedy',
                'lr': 9e-5,
                'gamma': 0.98,
                'n_step': 3,
                'priority': False,
            },
        },
    ]
    (protocol_dir / 'history.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in protocol_rows),
        encoding='utf-8',
    )
    (structure_dir / 'history.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in structure_rows),
        encoding='utf-8',
    )

    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'derive_candidates_and_search_space'
    state.history_paths = {
        'protocol_matrix': str(protocol_dir / 'history.jsonl').replace('\\', '/'),
        'structure_matrix': str(structure_dir / 'history.jsonl').replace('\\', '/'),
    }
    state.save()

    result = mod.derive_candidates_and_search_space(state)

    assert result.current_stage == 'select_base_config'
    assert result.selected_protocol_entry == 'v1_state_v2_reward'
    assert result.selected_structure_entry == 'dueling_noisy'
    assert (tmp_path / 'selection_decision.json').exists()
    assert (tmp_path / 'focused_search_space.json').exists()

    focused = json.loads((tmp_path / 'focused_search_space.json').read_text(encoding='utf-8'))
    assert focused['categorical']['reward_scheme_version'] == [
        'reward_v2_ratio', 'reward_v3_gap_shaping'
    ]
    assert focused['categorical']['network_backbone'] == ['dueling_mlp']
    stage_status = json.loads((tmp_path / 'stage_status.json').read_text(encoding='utf-8'))
    assert stage_status['derive_candidates_and_search_space']['status'] == 'completed'
    assert result.temporarily_disabled_protocol_entries == ['v1_state_v3_reward']
    assert result.temporarily_disabled_structure_entries == ['dueling_eps']
    assert result.permanently_eliminated_protocol_entries == []
    assert result.permanently_eliminated_structure_entries == []


def test_auto_workflow_select_base_config_writes_meta(tmp_path):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    selection_decision = {
        'selected_protocol_entry': 'v1_state_v2_reward',
        'selected_structure_entry': 'dueling_noisy',
        'selected_protocol_trial': {
            'trial_id': 11,
            'source': 'matrix_v1_state_v2_reward',
            'config': {
                'state_representation_version': 'low_dim_v1',
                'reward_scheme_version': 'reward_v2_ratio',
                'reward_scale': 0.1,
                'reward_clip': 10,
                'death_ratio': 20,
                'alive_ratio': 0.001,
                'lr': 1e-4,
                'gamma': 0.99,
            },
        },
        'selected_structure_trial': {
            'trial_id': 21,
            'source': 'matrix_dueling_noisy',
            'config': {
                'network_backbone': 'dueling_mlp',
                'exploration_head': 'noisy_net',
                'lr': 9e-5,
                'gamma': 0.98,
            },
        },
        'selection_reason': {
            'stable_success_count': 0,
            'best_final_median_score': 120.0,
            'best_final_success_rate_1000': 0.0,
            'best_eval_peak_score': 500.0,
        },
    }
    (tmp_path / 'selection_decision.json').write_text(
        json.dumps(selection_decision, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'select_base_config'
    state.selected_protocol_entry = 'v1_state_v2_reward'
    state.selected_structure_entry = 'dueling_noisy'
    state.save()

    result = mod.select_base_config(state)

    assert result.current_stage == 'focused_search_round'
    assert (tmp_path / 'best_config.json').exists()
    assert (tmp_path / 'best_config_meta.json').exists()
    assert (tmp_path / 'best_trial_summary.json').exists()

    config = json.loads((tmp_path / 'best_config.json').read_text(encoding='utf-8'))
    meta = json.loads((tmp_path / 'best_config_meta.json').read_text(encoding='utf-8'))
    assert config['reward_scheme_version'] == 'reward_v2_ratio'
    assert config['network_backbone'] == 'dueling_mlp'
    assert meta['source_stage'] == 'select_base_config'
    assert meta['source_entry'] == 'v1_state_v2_reward + dueling_noisy'
    assert meta['is_confirmed'] is False
    assert meta['requires_recheck'] is True
    stage_status = json.loads((tmp_path / 'stage_status.json').read_text(encoding='utf-8'))
    assert stage_status['select_base_config']['status'] == 'completed'


def test_auto_workflow_focused_search_round_runs_driver_and_updates_best(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    focused_path = tmp_path / 'focused_search_space.json'
    focused_path.write_text(json.dumps({
        'categorical': {
            'reward_scheme_version': ['reward_v2_ratio'],
            'state_representation_version': ['low_dim_v1'],
            'network_backbone': ['dueling_mlp'],
            'exploration_head': ['noisy_net'],
            'n_step': [1],
            'priority': [True],
            'reward_scale': [0.1],
            'reward_clip': [10],
        },
        'continuous': {
            'lr': {'low': 5e-5, 'high': 2e-4, 'log': True},
            'gamma': {'low': 0.97, 'high': 0.995},
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (tmp_path / 'best_config.json').write_text(json.dumps({
        'reward_scheme_version': 'reward_v2_ratio',
        'state_representation_version': 'low_dim_v1',
        'network_backbone': 'dueling_mlp',
        'exploration_head': 'noisy_net',
        'lr': 1e-4,
        'gamma': 0.99,
        'n_step': 1,
        'priority': True,
        'reward_scale': 0.1,
        'reward_clip': 10,
    }), encoding='utf-8')

    created = {}

    class FakeDriver:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.history = mod.HistoryManager(kwargs['history_path'])

        def run(self):
            self.history.append({
                'record_type': 'trial',
                'trial_id': 301,
                'source': 'tpe',
                'status': 'success',
                'objective': 700.0,
                'median_score': 1050.0,
                'success_rate_1000': 0.8,
                'best_eval_score': 1300.0,
                'config': {
                    'reward_scheme_version': 'reward_v2_ratio',
                    'state_representation_version': 'low_dim_v1',
                    'network_backbone': 'dueling_mlp',
                    'exploration_head': 'noisy_net',
                    'lr': 8e-5,
                    'gamma': 0.99,
                },
            })

    monkeypatch.setattr(mod, 'SearchDriver', FakeDriver)

    state = WorkflowState.new(goal='stable_1000', profile='cpu_normal', run_dir=tmp_path)
    state.current_stage = 'focused_search_round'
    state.focused_search_space = str(focused_path).replace('\\', '/')
    state.best_config_path = str(tmp_path / 'best_config.json').replace('\\', '/')
    state.search_round_index = 1
    state.save()

    result = mod.run_focused_search_round(state)

    assert created['run_baseline_first'] is False
    assert created['focused_search_space'] is not None
    assert (tmp_path / 'focused_search_round_1' / 'history.jsonl').exists()
    assert (tmp_path / 'workflow_summary.json').exists()
    assert (tmp_path / 'best_trial_summary.json').exists()
    assert result.current_stage == 'recheck_topk'
    stage_status = json.loads((tmp_path / 'stage_status.json').read_text(encoding='utf-8'))
    assert stage_status['focused_search_round']['status'] == 'completed'


def test_auto_workflow_focused_search_failure_advances_to_warmstart_and_widens(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    focused_path = tmp_path / 'focused_search_space.json'
    focused_path.write_text(json.dumps({
        'categorical': {
            'reward_scheme_version': ['reward_v2_ratio'],
            'state_representation_version': ['low_dim_v1'],
            'network_backbone': ['dueling_mlp'],
            'exploration_head': ['noisy_net'],
            'n_step': [1],
            'priority': [True],
            'reward_scale': [0.1],
            'reward_clip': [10],
        },
        'continuous': {
            'lr': {'low': 5e-5, 'high': 2e-4, 'log': True},
            'gamma': {'low': 0.97, 'high': 0.995},
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (tmp_path / 'best_config.json').write_text(json.dumps({'lr': 1e-4, 'gamma': 0.99}), encoding='utf-8')

    class FakeDriver:
        def __init__(self, **kwargs):
            self.history = mod.HistoryManager(kwargs['history_path'])

        def run(self):
            self.history.append({
                'record_type': 'trial',
                'trial_id': 401,
                'source': 'tpe',
                'status': 'failure',
                'objective': 1500.0,
                'median_score': 200.0,
                'success_rate_1000': 0.0,
                'best_eval_score': 439.0,
                'failure_reason': 'plateau_100k',
                'config': {'lr': 1e-4, 'gamma': 0.99},
            })

    monkeypatch.setattr(mod, 'SearchDriver', FakeDriver)

    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'focused_search_round'
    state.focused_search_space = str(focused_path).replace('\\', '/')
    state.best_config_path = str(tmp_path / 'best_config.json').replace('\\', '/')
    state.search_round_index = 1
    state.save()

    result = mod.run_focused_search_round(state)

    assert result.current_stage == 'warmstart_search_round'
    assert result.space_widen_count == 1
    widened = json.loads(focused_path.read_text(encoding='utf-8'))
    assert widened['continuous']['lr']['low'] < 5e-5
    assert widened['continuous']['lr']['high'] > 2e-4


def test_auto_workflow_warmstart_round_advances_to_population(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    focused_path = tmp_path / 'focused_search_space.json'
    focused_path.write_text(json.dumps({
        'categorical': {'reward_scheme_version': ['reward_v2_ratio']},
        'continuous': {'lr': {'low': 5e-5, 'high': 2e-4, 'log': True}},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (tmp_path / 'best_config.json').write_text(json.dumps({'lr': 1e-4}), encoding='utf-8')

    called = {}

    class FakeDriver:
        def __init__(self, **kwargs):
            called.update(kwargs)
            self.history = mod.HistoryManager(kwargs['history_path'])

        def run_warmstart_tpe(self):
            self.history.append({
                'record_type': 'trial',
                'trial_id': 501,
                'source': 'warmstart_tpe',
                'status': 'failure',
                'objective': 1400.0,
                'median_score': 250.0,
                'success_rate_1000': 0.0,
                'best_eval_score': 500.0,
                'failure_reason': 'plateau_100k',
                'config': {'lr': 1e-4},
            })

    monkeypatch.setattr(mod, 'SearchDriver', FakeDriver)

    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'warmstart_search_round'
    state.focused_search_space = str(focused_path).replace('\\', '/')
    state.best_config_path = str(tmp_path / 'best_config.json').replace('\\', '/')
    state.search_round_index = 1
    state.save()

    result = mod.run_warmstart_search_round(state)

    assert result.current_stage == 'population_search_round'
    assert called['run_baseline_first'] is False


def test_auto_workflow_population_round_blocks_with_report(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    focused_path = tmp_path / 'focused_search_space.json'
    focused_path.write_text(json.dumps({
        'categorical': {'reward_scheme_version': ['reward_v2_ratio']},
        'continuous': {'lr': {'low': 5e-5, 'high': 2e-4, 'log': True}},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (tmp_path / 'best_config.json').write_text(json.dumps({'lr': 1e-4}), encoding='utf-8')
    (tmp_path / 'best_config_meta.json').write_text(json.dumps({
        'source_stage': 'focused_search_round',
        'source_entry': 'x',
        'source_trial_id': 1,
        'source_history_path': 'x.jsonl',
        'updated_at': '2026-06-22T00:00:00',
        'confidence_level': 'near_miss_high',
        'is_confirmed': False,
        'requires_recheck': False,
        'selection_reason': {},
    }), encoding='utf-8')

    class FakeDriver:
        def __init__(self, **kwargs):
            self.history = mod.HistoryManager(kwargs['history_path'])

        def run_population_async(self, **kwargs):
            self.history.append({
                'record_type': 'trial',
                'trial_id': 601,
                'source': 'population_async',
                'status': 'failure',
                'objective': 1600.0,
                'median_score': 180.0,
                'success_rate_1000': 0.0,
                'best_eval_score': 439.0,
                'failure_reason': 'plateau_100k',
                'config': {'lr': 1e-4},
            })

    monkeypatch.setattr(mod, 'SearchDriver', FakeDriver)

    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'population_search_round'
    state.focused_search_space = str(focused_path).replace('\\', '/')
    state.best_config_path = str(tmp_path / 'best_config.json').replace('\\', '/')
    state.best_trial_summary = str(tmp_path / 'workflow_summary.json').replace('\\', '/')
    state.history_paths['focused_search_round_1'] = str(tmp_path / 'focused_search_round_1' / 'history.jsonl').replace('\\', '/')
    state.history_paths['warmstart_round'] = str(tmp_path / 'warmstart_round' / 'history.jsonl').replace('\\', '/')
    state.retained_protocol_entries = ['v1_state_v2_reward']
    state.retained_structure_entries = ['dueling_noisy']
    state.eliminated_protocol_entries = ['v1_state_v3_reward']
    state.eliminated_structure_entries = ['dueling_eps']
    (tmp_path / 'focused_search_round_1').mkdir(parents=True)
    (tmp_path / 'focused_search_round_1' / 'history.jsonl').write_text(json.dumps({
        'record_type': 'trial',
        'trial_id': 551,
        'status': 'failure',
        'objective': 1700.0,
        'median_score': 120.0,
        'success_rate_1000': 0.0,
        'best_eval_score': 439.0,
        'failure_reason': 'plateau_100k',
        'config': {'lr': 1e-4},
    }) + '\n', encoding='utf-8')
    (tmp_path / 'warmstart_round').mkdir(parents=True)
    (tmp_path / 'warmstart_round' / 'history.jsonl').write_text(json.dumps({
        'record_type': 'trial',
        'trial_id': 552,
        'status': 'failure',
        'objective': 1650.0,
        'median_score': 150.0,
        'success_rate_1000': 0.0,
        'best_eval_score': 460.0,
        'failure_reason': 'RuntimeError: boom',
        'config': {'lr': 1e-4},
    }) + '\n', encoding='utf-8')
    state.save()

    result = mod.run_population_search_round(state)

    assert result.current_stage == 'blocked'
    blocked_report = tmp_path / 'blocked_report.json'
    assert blocked_report.exists()
    payload = json.loads(blocked_report.read_text(encoding='utf-8'))
    assert payload['blocked_reason'] == 'search_space_exhausted'
    assert 'current_best_config_path' in payload
    assert payload['current_best_config'] == {'lr': 1e-4}
    assert 'failure_reason_counter' in payload
    assert payload['failure_reason_counter']['plateau_100k'] == 1
    assert payload['recent_round_improvements']['delta']['best_final_median_score_delta'] == 30.0


def test_auto_workflow_recheck_round_writes_summary_and_advances(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    focused_history = tmp_path / 'focused_search_round_1' / 'history.jsonl'
    focused_history.parent.mkdir(parents=True)
    focused_history.write_text(json.dumps({
        'record_type': 'trial',
        'trial_id': 701,
        'status': 'success',
        'objective': 700.0,
        'median_score': 1050.0,
        'success_rate_1000': 0.8,
        'config': {'lr': 1e-4},
    }) + '\n', encoding='utf-8')
    (tmp_path / 'best_config.json').write_text(json.dumps({'lr': 1e-4}), encoding='utf-8')

    monkeypatch.setattr(mod, 'recheck_top_k', lambda history, k, **kwargs: [{
        'rank': 1,
        'original_trial_id': 701,
        'recheck_passed': True,
        'recheck_median': 1100.0,
        'recheck_success_rate': 0.9,
        'config': {'lr': 1e-4},
    }])

    state = WorkflowState.new(goal='stable_1000', profile='cpu_normal', run_dir=tmp_path)
    state.current_stage = 'recheck_topk'
    state.best_config_path = str(tmp_path / 'best_config.json').replace('\\', '/')
    state.history_paths['focused_search_round_1'] = str(focused_history).replace('\\', '/')
    state.save()

    result = mod.run_recheck_topk(state)

    assert result.current_stage == 'final_confirm'
    assert (tmp_path / 'recheck' / 'recheck_summary.json').exists()
    summary = json.loads((tmp_path / 'recheck' / 'recheck_summary.json').read_text(encoding='utf-8'))
    assert summary['recheck_passed'] is True
    assert summary['next_stage'] == 'final_confirm'
    meta = json.loads((tmp_path / 'best_config_meta.json').read_text(encoding='utf-8'))
    assert meta['source_stage'] == 'recheck_topk'
    assert meta['source_trial_id'] == 701
    stage_status = json.loads((tmp_path / 'stage_status.json').read_text(encoding='utf-8'))
    assert stage_status['recheck_topk']['status'] == 'completed'


def test_auto_workflow_final_confirm_and_report_complete(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    (tmp_path / 'best_config.json').write_text(json.dumps({'lr': 1e-4}), encoding='utf-8')
    (tmp_path / 'best_config_meta.json').write_text(json.dumps({
        'source_stage': 'recheck_topk',
        'source_entry': 'x',
        'source_trial_id': 701,
        'source_history_path': 'x.jsonl',
        'updated_at': '2026-06-22T00:00:00',
        'confidence_level': 'search_success',
        'is_confirmed': False,
        'requires_recheck': True,
        'selection_reason': {},
    }), encoding='utf-8')

    monkeypatch.setattr(mod, 'final_confirm', lambda config, **kwargs: {
        'status': 'confirmed',
        'overall_success_rate_1000': 0.85,
        'overall_median_score': 1200.0,
        'config': config,
    })

    state = WorkflowState.new(goal='stable_1000', profile='cpu_normal', run_dir=tmp_path)
    state.current_stage = 'final_confirm'
    state.best_config_path = str(tmp_path / 'best_config.json').replace('\\', '/')
    state.save()

    after_confirm = mod.run_final_confirm_stage(state)
    assert after_confirm.current_stage == 'report'
    assert (tmp_path / 'final_confirm' / 'final_confirm_summary.json').exists()

    after_report = mod.run_report_stage(after_confirm)
    assert after_report.current_stage == 'complete'
    assert (tmp_path / 'report' / 'final_report.json').exists()
    meta = json.loads((tmp_path / 'best_config_meta.json').read_text(encoding='utf-8'))
    assert meta['is_confirmed'] is True
    assert meta['requires_recheck'] is False
    assert meta['source_stage'] == 'final_confirm'
    report = json.loads((tmp_path / 'report' / 'final_report.json').read_text(encoding='utf-8'))
    assert report['best_config_explanation']['is_confirmed'] is True
    assert report['best_config_explanation']['why_current_config_is_best']['source_stage'] == 'final_confirm'


def test_auto_workflow_quick_profile_success_skips_recheck_and_final_confirm(tmp_path, monkeypatch):
    import auto_workflow as mod
    from workflow_state import WorkflowState

    focused_path = tmp_path / 'focused_search_space.json'
    focused_path.write_text(json.dumps({
        'categorical': {
            'reward_scheme_version': ['reward_v2_ratio'],
            'state_representation_version': ['low_dim_v1'],
            'network_backbone': ['dueling_mlp'],
            'exploration_head': ['noisy_net'],
            'n_step': [1],
            'priority': [True],
            'reward_scale': [0.1],
            'reward_clip': [10],
        },
        'continuous': {
            'lr': {'low': 5e-5, 'high': 2e-4, 'log': True},
            'gamma': {'low': 0.97, 'high': 0.995},
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (tmp_path / 'best_config.json').write_text(json.dumps({'lr': 1e-4}), encoding='utf-8')
    (tmp_path / 'best_config_meta.json').write_text(json.dumps({
        'source_stage': 'focused_search_round',
        'source_entry': 'x',
        'source_trial_id': 801,
        'source_history_path': '',
        'updated_at': '2026-06-22T00:00:00',
        'confidence_level': 'search_success',
        'is_confirmed': False,
        'requires_recheck': True,
        'selection_reason': {},
    }), encoding='utf-8')

    class FakeDriver:
        def __init__(self, **kwargs):
            self.history = mod.HistoryManager(kwargs['history_path'])

        def run(self):
            self.history.append({
                'record_type': 'trial',
                'trial_id': 801,
                'source': 'tpe',
                'status': 'success',
                'objective': 600.0,
                'median_score': 1100.0,
                'success_rate_1000': 0.9,
                'best_eval_score': 1300.0,
                'config': {'lr': 1e-4},
            })

    monkeypatch.setattr(mod, 'SearchDriver', FakeDriver)

    state = WorkflowState.new(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    state.current_stage = 'focused_search_round'
    state.best_config_path = str(tmp_path / 'best_config.json').replace('\\', '/')
    state.focused_search_space = str(focused_path).replace('\\', '/')
    state.search_round_index = 1
    state.save()

    result = mod.run_focused_search_round(state)

    assert result.current_stage == 'report'
    assert not (tmp_path / 'recheck' / 'recheck_summary.json').exists()
    assert not (tmp_path / 'final_confirm' / 'final_confirm_summary.json').exists()
    meta = json.loads((tmp_path / 'best_config_meta.json').read_text(encoding='utf-8'))
    assert meta['requires_recheck'] is True
    assert meta['confidence_level'] == 'search_success'


def test_auto_workflow_single_invocation_can_run_to_complete(tmp_path, monkeypatch):
    import auto_workflow as mod

    def fake_run_trial(**kwargs):
        return {
            'trial_id': -1,
            'seed': kwargs['seed'],
            'source': kwargs['source'],
            'status': 'failure',
            'failure_reason': 'max_frames_reached',
            'train_raw_env_frames': 1000,
            'best_eval_score': 10,
            'median_score': 1.0,
            'success_rate_1000': 0.0,
            'config': dict(kwargs['config']),
        }

    def fake_run_matrix(matrix, mode, budget):
        name = matrix[0]['name']
        if name.startswith('v'):
            return [{
                'record_type': 'trial',
                'trial_id': 11,
                'source': 'matrix_v1_state_v2_reward',
                'status': 'failure',
                'objective': 1000.0,
                'best_eval_score': 439.0,
                'median_score': 80.0,
                'success_rate_1000': 0.0,
                'failure_reason': 'plateau_100k',
                'config': {
                    'state_representation_version': 'low_dim_v1',
                    'reward_scheme_version': 'reward_v2_ratio',
                    'reward_scale': 0.1,
                    'reward_clip': 10,
                    'death_ratio': 20,
                    'alive_ratio': 0.001,
                    'lr': 1e-4,
                    'gamma': 0.99,
                },
            }]
        return [{
            'record_type': 'trial',
            'trial_id': 21,
            'source': 'matrix_dueling_noisy',
            'status': 'failure',
            'objective': 900.0,
            'best_eval_score': 500.0,
            'median_score': 120.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'plateau_100k',
            'config': {
                'network_backbone': 'dueling_mlp',
                'exploration_head': 'noisy_net',
                'lr': 1e-4,
                'gamma': 0.99,
                'n_step': 1,
                'priority': True,
            },
        }]

    class FakeDriver:
        def __init__(self, **kwargs):
            self.history = mod.HistoryManager(kwargs['history_path'])

        def run(self):
            self.history.append({
                'record_type': 'trial',
                'trial_id': 301,
                'source': 'tpe',
                'status': 'success',
                'objective': 700.0,
                'median_score': 1050.0,
                'success_rate_1000': 0.8,
                'best_eval_score': 1300.0,
                'config': {'lr': 8e-5, 'gamma': 0.99},
            })

        def run_warmstart_tpe(self):
            raise AssertionError('warmstart should not be needed in this path')

        def run_population_async(self, **kwargs):
            raise AssertionError('population should not be needed in this path')

    monkeypatch.setattr(mod, 'run_trial', fake_run_trial)
    monkeypatch.setattr(mod, 'compute_objective', lambda **kwargs: 9999.0)
    monkeypatch.setattr(mod, 'run_matrix', fake_run_matrix)
    monkeypatch.setattr(mod, 'summarize_matrix_results', lambda matrix_name, results: {
        'matrix': matrix_name,
        'entries': [{'name': results[0]['source'], 'trial_count': 1}],
    })
    monkeypatch.setattr(mod, 'SearchDriver', FakeDriver)
    monkeypatch.setattr(mod, 'recheck_top_k', lambda history, k, **kwargs: [{
        'rank': 1,
        'original_trial_id': 301,
        'recheck_passed': True,
        'recheck_median': 1100.0,
        'recheck_success_rate': 0.9,
        'config': {'lr': 8e-5},
    }])
    monkeypatch.setattr(mod, 'final_confirm', lambda config, **kwargs: {
        'status': 'confirmed',
        'overall_success_rate_1000': 0.85,
        'overall_median_score': 1200.0,
        'config': config,
    })

    result = mod.run_workflow(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)

    assert result.current_stage == 'complete'
    assert (tmp_path / 'report' / 'final_report.json').exists()
    assert not (tmp_path / 'final_confirm' / 'final_confirm_summary.json').exists()
    report = json.loads((tmp_path / 'report' / 'final_report.json').read_text(encoding='utf-8'))
    assert report['best_config_explanation']['unresolved_step'] == 'recheck_skipped_by_profile'


def test_auto_workflow_marks_stage_failed_on_runtime_error(tmp_path, monkeypatch):
    import auto_workflow as mod

    monkeypatch.setattr(mod, 'run_trial', lambda **kwargs: {
        'trial_id': -1,
        'seed': kwargs['seed'],
        'source': kwargs['source'],
        'status': 'failure',
        'failure_reason': 'max_frames_reached',
        'train_raw_env_frames': 1000,
        'best_eval_score': 10,
        'median_score': 1.0,
        'success_rate_1000': 0.0,
        'config': dict(kwargs['config']),
    })
    monkeypatch.setattr(mod, 'compute_objective', lambda **kwargs: 9999.0)
    monkeypatch.setattr(mod, 'run_matrix', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('matrix boom')))

    with pytest.raises(RuntimeError, match='matrix boom'):
        mod.run_workflow(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)

    stage_status = json.loads((tmp_path / 'stage_status.json').read_text(encoding='utf-8'))
    assert stage_status['protocol_matrix']['status'] == 'failed'
    assert stage_status['protocol_matrix']['attempt'] == 1

    state = json.loads((tmp_path / 'workflow_state.json').read_text(encoding='utf-8'))
    assert 'protocol_matrix' in state['failed_stages']
    assert state['last_error'] == 'matrix boom'


def test_auto_workflow_blocks_after_repeated_runtime_failure(tmp_path, monkeypatch):
    import auto_workflow as mod

    monkeypatch.setattr(mod, 'run_trial', lambda **kwargs: {
        'trial_id': -1,
        'seed': kwargs['seed'],
        'source': kwargs['source'],
        'status': 'failure',
        'failure_reason': 'max_frames_reached',
        'train_raw_env_frames': 1000,
        'best_eval_score': 10,
        'median_score': 1.0,
        'success_rate_1000': 0.0,
        'config': dict(kwargs['config']),
    })
    monkeypatch.setattr(mod, 'compute_objective', lambda **kwargs: 9999.0)
    monkeypatch.setattr(mod, 'run_matrix', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('matrix boom')))

    with pytest.raises(RuntimeError):
        mod.run_workflow(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)

    blocked = mod.run_workflow(goal='stable_1000', profile='cpu_quick', run_dir=tmp_path)
    assert blocked.current_stage == 'blocked'
    assert blocked.blocked_reason == 'repeated_runtime_failure'

    payload = json.loads((tmp_path / 'blocked_report.json').read_text(encoding='utf-8'))
    assert payload['blocked_reason'] == 'repeated_runtime_failure'

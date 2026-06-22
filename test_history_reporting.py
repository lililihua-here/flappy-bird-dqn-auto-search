"""Contract tests for history_reporting module."""
import json
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# HistoryManager tests
# ============================================================================
def test_history_manager_append_and_load_roundtrip():
    import tempfile
    from history_reporting import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({
            'trial_id': 0,
            'status': 'success',
            'objective': 123.0,
            'config': {'lr': 1e-4},
        })
        rows = hm.load()
        assert len(rows) == 1
        assert rows[0]['trial_id'] == 0
        assert rows[0]['record_type'] == 'trial'
    finally:
        os.unlink(tmp.name)


def test_history_manager_success_and_failure_counts():
    import tempfile
    from history_reporting import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({'trial_id': 0, 'status': 'success', 'objective': 100, 'config': {'lr': 1e-4}})
        hm.append({'trial_id': 1, 'status': 'failure', 'objective': 999999, 'config': {'lr': 2e-4}})
        hm.append({'record_type': 'recheck', 'trial_id': 0, 'recheck_passed': True})
        assert hm.success_count() == 1
        assert hm.failure_count() == 1
    finally:
        os.unlink(tmp.name)


def test_history_manager_top_k_filters_recheck_and_sorts_by_priority():
    import tempfile
    from history_reporting import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({
            'trial_id': 1, 'status': 'success', 'objective': 150000,
            'config': {'lr': 1e-4},
            'success_rate_1000': 0.80, 'median_score': 1100,
            'total_raw_env_frames': 180000,
        })
        hm.append({
            'trial_id': 2, 'status': 'success', 'objective': 140000,
            'config': {'lr': 2e-4},
            'success_rate_1000': 0.75, 'median_score': 1050,
            'total_raw_env_frames': 190000,
        })
        hm.append({
            'trial_id': 3, 'status': 'success', 'objective': 170000,
            'config': {'lr': 3e-4},
            'success_rate_1000': 0.90, 'median_score': 1300,
            'total_raw_env_frames': 200000,
        })
        hm.append({
            'record_type': 'recheck', 'trial_id': 2,
            'recheck_passed': True,
            'median_train_raw_env_frames_to_stable_1000': 160000,
        })
        hm.append({
            'record_type': 'recheck', 'trial_id': 3,
            'recheck_passed': True,
            'median_train_raw_env_frames_to_stable_1000': 170000,
        })
        top = hm.top_k(3)
        assert [r['trial_id'] for r in top] == [2, 3, 1]
    finally:
        os.unlink(tmp.name)


def test_history_manager_load_ignores_corrupt_jsonl_lines():
    import tempfile
    from history_reporting import HistoryManager

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    try:
        tmp.write('{"trial_id": 0, "status": "success", "config": {"lr": 1e-4}}\n')
        tmp.write('{bad json line}\n')
        tmp.write('{"trial_id": 1, "status": "failure", "config": {"lr": 2e-4}}\n')
        tmp.close()
        rows = HistoryManager(tmp.name).load()
        assert [r['trial_id'] for r in rows] == [0, 1]
    finally:
        os.unlink(tmp.name)


# ============================================================================
# generate_summary tests
# ============================================================================
def test_generate_summary_handles_empty_history():
    import tempfile
    from history_reporting import HistoryManager, generate_summary

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        summary = generate_summary(HistoryManager(tmp.name))
        assert summary['trial_count'] == 0
        assert summary['top_k'] == []
    finally:
        os.unlink(tmp.name)


def test_generate_summary_returns_counts_and_best_trial():
    import tempfile
    from history_reporting import HistoryManager, generate_summary

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({
            'trial_id': -1, 'source': 'baseline', 'status': 'failure',
            'objective': 9_999_900, 'config': {'lr': 1e-4},
            'failure_reason': 'max_frames_reached',
        })
        hm.append({
            'trial_id': 0, 'source': 'tpe', 'status': 'success',
            'objective': 180000, 'config': {'lr': 2e-4},
            'median_score': 1050, 'success_rate_1000': 0.75,
            'total_raw_env_frames': 220000,
        })
        summary = generate_summary(hm, top_k=1)
        assert summary['trial_count'] == 2
        assert summary['success_count'] == 1
        assert summary['failure_count'] == 1
        assert summary['best_trial_id'] == 0
        assert summary['top_k'][0]['trial_id'] == 0
        assert summary['failure_reasons']['max_frames_reached'] == 1
    finally:
        os.unlink(tmp.name)


def test_load_raw_trial_rows_filters_non_trial_records(tmp_path):
    from history_reporting import HistoryManager, load_raw_trial_rows

    history_path = tmp_path / 'history.jsonl'
    hm = HistoryManager(history_path)
    hm.append({
        'trial_id': 1,
        'status': 'failure',
        'objective': 999.0,
        'config': {'lr': 1e-4},
    })
    hm.append({
        'record_type': 'recheck',
        'trial_id': 1,
        'recheck_passed': True,
    })

    rows = load_raw_trial_rows(history_path)
    assert len(rows) == 1
    assert rows[0]['trial_id'] == 1
    assert rows[0]['record_type'] == 'trial'


def test_aggregate_trials_for_workflow_keeps_peak_and_final_metrics_separate():
    from history_reporting import aggregate_trials_for_workflow

    summary = aggregate_trials_for_workflow([
        {
            'record_type': 'trial',
            'trial_id': 1,
            'status': 'failure',
            'objective': 1500.0,
            'best_eval_score': 439,
            'median_score': 12.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'plateau_100k',
        },
        {
            'record_type': 'trial',
            'trial_id': 2,
            'status': 'failure',
            'objective': 1200.0,
            'best_eval_score': 200,
            'median_score': 80.0,
            'success_rate_1000': 0.0,
            'failure_reason': 'no_learning_50k',
        },
        {
            'record_type': 'trial',
            'trial_id': 3,
            'status': 'success',
            'objective': 800.0,
            'best_eval_score': 1000,
            'median_score': 1100.0,
            'success_rate_1000': 0.8,
        },
    ])

    assert summary['trial_count'] == 3
    assert summary['stable_success_count'] == 1
    assert summary['best_eval_peak_score'] == 1000
    assert summary['best_final_median_score'] == 1100.0
    assert summary['best_final_success_rate_1000'] == 0.8
    assert summary['median_of_final_medians'] == 80.0
    assert summary['best_objective'] == 800.0
    assert summary['plateau_like_count'] == 1
    assert summary['no_learning_count'] == 1
    assert summary['failure_reason_counter']['plateau_100k'] == 1
    assert summary['failure_reason_counter']['no_learning_50k'] == 1


def test_workflow_failure_reason_is_normalized():
    from workflow_metrics import aggregate_trials_for_workflow, normalize_failure_reason

    assert normalize_failure_reason({'status': 'success'}) == 'success'
    assert normalize_failure_reason({'status': 'failure', 'failure_reason': 'RuntimeError: boom'}) == 'runtime_exception'
    assert normalize_failure_reason({'status': 'failure', 'failure_reason': 'CUDA out of memory'}) == 'oom_or_resource_error'

    summary = aggregate_trials_for_workflow([
        {
            'record_type': 'trial',
            'trial_id': 1,
            'status': 'failure',
            'objective': 1500.0,
            'median_score': 10.0,
            'success_rate_1000': 0.0,
            'best_eval_score': 100.0,
            'failure_reason': 'RuntimeError: boom',
        },
        {
            'record_type': 'trial',
            'trial_id': 2,
            'status': 'failure',
            'objective': 1800.0,
            'median_score': 5.0,
            'success_rate_1000': 0.0,
            'best_eval_score': 50.0,
            'failure_reason': 'CUDA out of memory',
        },
    ])

    assert summary['failure_reason_counter']['runtime_exception'] == 1
    assert summary['failure_reason_counter']['oom_or_resource_error'] == 1


def test_export_best_config_writes_best_trial_config(tmp_path):
    from history_reporting import HistoryManager, export_best_config

    history_path = tmp_path / 'history.jsonl'
    hm = HistoryManager(history_path)
    hm.append({
        'trial_id': 1,
        'status': 'failure',
        'objective': 2_000_000,
        'config': {'lr': 1e-4},
    })
    hm.append({
        'trial_id': 2,
        'status': 'success',
        'objective': 150000,
        'config': {'lr': 2e-4, 'reward_scheme_version': 'reward_v3_gap_shaping'},
    })

    out_path = tmp_path / 'best_config.json'
    written = export_best_config(hm, out_path)

    assert written == str(out_path)
    assert out_path.exists()
    assert '"lr": 0.0002' in out_path.read_text(encoding='utf-8')


# ============================================================================
# _make_serializable tests
# ============================================================================
def test_make_serializable_handles_numpy_and_torch():
    import torch
    from history_reporting import _make_serializable
    payload = {
        'array': np.array([1, 2, 3], dtype=np.float32),
        'scalar': np.float32(1.5),
        'tensor': torch.tensor([4.0, 5.0]),
    }
    result = _make_serializable(payload)
    assert result == {
        'array': [1.0, 2.0, 3.0],
        'scalar': 1.5,
        'tensor': [4.0, 5.0],
    }


# ============================================================================
# recheck_top_k tests
# ============================================================================
def test_recheck_top_k_persists_full_summary(monkeypatch):
    import tempfile
    import history_reporting as mod
    from history_reporting import HistoryManager, recheck_top_k

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    try:
        hm = HistoryManager(tmp.name)
        hm.append({
            'trial_id': 7,
            'status': 'success',
            'objective': 123456,
            'config': {'lr': 1e-4},
            'success_rate_1000': 0.8,
            'median_score': 1100,
            'total_raw_env_frames': 150000,
        })

        fake_results = iter([
            {
                'status': 'success',
                'train_raw_env_frames': 100000,
                'median_score': 1200,
                'success_rate_1000': 0.8,
            },
            {
                'status': 'failure',
                'train_raw_env_frames': 300000,
                'median_score': 900,
                'success_rate_1000': 0.4,
            },
        ])

        def fake_run_trial(**_kwargs):
            return next(fake_results)

        monkeypatch.setattr(mod, 'run_trial', fake_run_trial)
        recheck_top_k(hm, k=1, recheck_seeds=(101, 202), max_trial_frames=5000, eval_episodes=5)

        rows = hm.load()
        recheck_rows = [r for r in rows if r.get('record_type') == 'recheck']
        assert len(recheck_rows) == 1
        assert recheck_rows[0]['trial_id'] == 7
        assert 'p10_score' in recheck_rows[0]
        assert 'p90_score' in recheck_rows[0]
        assert 'score_std' in recheck_rows[0]
        assert 'failed_seeds' in recheck_rows[0]
    finally:
        os.unlink(tmp.name)


def test_recheck_top_k_supports_output_path(monkeypatch, tmp_path):
    import history_reporting as mod
    from history_reporting import HistoryManager, recheck_top_k

    history_path = tmp_path / 'history.jsonl'
    hm = HistoryManager(history_path)
    hm.append({
        'trial_id': 9,
        'status': 'success',
        'objective': 1000,
        'config': {'lr': 1e-4},
        'success_rate_1000': 0.8,
        'median_score': 1050,
        'total_raw_env_frames': 120000,
    })

    fake_results = iter([
        {
            'status': 'success',
            'train_raw_env_frames': 100000,
            'median_score': 1200,
            'success_rate_1000': 0.8,
        },
    ])

    monkeypatch.setattr(mod, 'run_trial', lambda **kwargs: next(fake_results))
    output_path = tmp_path / 'recheck_summary.json'
    recheck_top_k(hm, k=1, recheck_seeds=(101,), output_path=output_path, source='auto_workflow_recheck')

    payload = json.loads(output_path.read_text(encoding='utf-8'))
    assert payload['k'] == 1
    assert payload['results'][0]['original_trial_id'] == 9


def test_recheck_top_k_supports_output_dir(monkeypatch, tmp_path):
    import history_reporting as mod
    from history_reporting import HistoryManager, recheck_top_k

    history_path = tmp_path / 'history.jsonl'
    hm = HistoryManager(history_path)
    hm.append({
        'trial_id': 10,
        'status': 'success',
        'objective': 900,
        'config': {'lr': 1e-4},
        'success_rate_1000': 0.8,
        'median_score': 1100,
        'total_raw_env_frames': 100000,
    })

    fake_results = iter([
        {
            'status': 'success',
            'train_raw_env_frames': 80000,
            'median_score': 1300,
            'success_rate_1000': 0.9,
        },
    ])

    monkeypatch.setattr(mod, 'run_trial', lambda **kwargs: next(fake_results))
    output_dir = tmp_path / 'recheck_out'
    recheck_top_k(hm, k=1, recheck_seeds=(101,), output_dir=output_dir, source='auto_workflow_recheck')

    payload = json.loads((output_dir / 'recheck_summary.json').read_text(encoding='utf-8'))
    assert payload['k'] == 1
    assert payload['results'][0]['original_trial_id'] == 10


# ============================================================================
# V2 schema normalization / checkpoint / reports
# ============================================================================
def test_normalize_legacy_record_maps_versions_and_per_field():
    from history_reporting import normalize_legacy_record

    normalized = normalize_legacy_record({
        'trial_id': 1,
        'env_version': 'fixed_env_v1',
        'per_beta_frames': 1234,
        'config': {'reward_pipe': 1.0, 'reward_death': -10, 'reward_alive': 0.0},
    })

    assert normalized['record_type'] == 'trial'
    assert normalized['environment_version'] == 'fixed_env_v1'
    assert normalized['per_beta_train_updates'] == 1234
    assert normalized['config']['pipe_reward'] == 1.0
    assert normalized['config']['death_ratio'] == 10
    assert normalized['config']['alive_ratio'] == 0.0


def test_checkpoint_asset_helpers_roundtrip(tmp_path):
    import torch
    from history_reporting import (
        CHECKPOINT_FORMAT_VERSION, build_checkpoint_payload, save_checkpoint,
        is_checkpoint_compatible,
    )

    net = torch.nn.Linear(2, 2)
    payload = build_checkpoint_payload(
        q_net=net,
        target_net=net,
        config={'hidden': [2], 'reward_scheme_version': 'mvp_reward_v1'},
        trial_id=0,
        seed=11,
        source='baseline',
        train_raw_env_frames=100,
        decision_steps=50,
        state_dim=2,
        n_actions=2,
    )
    path, sha = save_checkpoint(payload, tmp_path, prefix='trial_0')

    assert os.path.exists(path)
    assert len(sha) == 64
    assert payload['checkpoint_format_version'] == CHECKPOINT_FORMAT_VERSION
    assert is_checkpoint_compatible(
        payload,
        current_env='fixed_env_v1',
        current_reward='mvp_reward_v1',
        current_state='low_dim_v1',
    ) is True


def test_generate_all_reports_creates_expected_files(tmp_path):
    from history_reporting import HistoryManager, generate_all_reports

    history_path = tmp_path / 'history.jsonl'
    hm = HistoryManager(str(history_path))
    hm.append({
        'trial_id': 0,
        'status': 'success',
        'objective': 123.0,
        'config': {'lr': 1e-4, 'n_step': 1},
        'median_score': 1100,
        'success_rate_1000': 0.8,
        'environment_version': 'fixed_env_v1',
        'reward_scheme_version': 'mvp_reward_v1',
        'state_representation_version': 'low_dim_v1',
        'checkpoint_path': 'checkpoints/trial_0.pt',
    })
    hm.append({
        'record_type': 'recheck',
        'trial_id': 0,
        'timestamp': '2026-06-20T12:00:00',
        'recheck_passed': True,
    })

    outputs = generate_all_reports(hm, tmp_path / 'study.db', tmp_path)
    for key, path in outputs.items():
        assert os.path.exists(path), f'{key} not created'

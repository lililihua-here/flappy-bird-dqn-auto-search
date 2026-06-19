"""Contract tests for history_reporting module."""
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

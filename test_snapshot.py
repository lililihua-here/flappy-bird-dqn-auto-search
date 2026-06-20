"""V3 snapshot + lineage tests."""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


def test_snapshot_to_cpu_recursive():
    from snapshot import to_cpu
    import torch
    data = {"a": torch.tensor([1.0]), "b": [torch.tensor([2.0])], "c": 3}
    result = to_cpu(data)
    assert isinstance(result["a"], torch.Tensor)
    assert result["a"].device.type == "cpu"


def test_lineage_chain_id_vs_node_id():
    from lineage import LineageTracker
    a = LineageTracker(trial_type="fresh", trial_id=0)
    assert a.lineage_chain_id == a.lineage_node_id  # fresh: both newly generated

    b = LineageTracker(trial_type="warm_start", trial_id=1,
                       parent_lineage=a, parent_trial_id=0,
                       parent_snapshot_ref="snap_0.pt")
    assert b.lineage_chain_id == a.lineage_chain_id    # same chain
    assert b.lineage_node_id != a.lineage_node_id      # different node
    assert b.inheritance_depth == 1


def test_lineage_inherited_rejects_missing_parent():
    from lineage import LineageTracker
    import pytest
    with pytest.raises(ValueError):
        LineageTracker(trial_type="warm_start", trial_id=1, parent_lineage=None)


def test_lineage_to_dict():
    from lineage import LineageTracker
    lt = LineageTracker(trial_type="fresh", trial_id=42)
    d = lt.to_dict()
    assert d["trial_type"] == "fresh"
    assert d["lineage_root_trial_id"] == 42


def test_snapshot_save_load_roundtrip(tmp_path):
    import torch.nn as nn
    from snapshot import FullTrainingSnapshot, save_snapshot, load_snapshot
    import os
    s = FullTrainingSnapshot(trial_id=0, seed=42, q_net_state_dict={})
    path, sha = save_snapshot(s, str(tmp_path))
    assert os.path.exists(path)
    loaded = load_snapshot(path)
    assert loaded.trial_id == 0


def test_snapshot_missing_sha_rejected(tmp_path):
    from snapshot import FullTrainingSnapshot, save_snapshot
    import torch
    s = FullTrainingSnapshot(trial_id=0)
    tmp_pt = tmp_path / "snapshots" / "test.pt"
    tmp_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(s, str(tmp_pt))
    from snapshot import load_snapshot
    import pytest
    with pytest.raises(ValueError, match="Missing snapshot sha256"):
        load_snapshot(str(tmp_pt))


def test_snapshot_verify_compatible():
    from snapshot import FullTrainingSnapshot, verify_snapshot_compatible
    config = {
        "n_step": 1, "priority": False,
        "environment_version": "fixed_env_v1",
        "state_representation_version": "low_dim_v1",
        "reward_scheme_version": "reward_v1_sparse",
    }
    s = FullTrainingSnapshot(replay_buffer_type="ReplayBuffer", config=config)
    assert verify_snapshot_compatible(s, config) is True

    # Mismatched n_step
    config2 = dict(config, n_step=3)
    assert verify_snapshot_compatible(s, config2) is False


def test_restore_rng_state():
    """Test that RNG state is correctly saved and restored."""
    import random as py_random
    import torch
    from snapshot import FullTrainingSnapshot, restore_rng_state

    py_random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Record some RNG values
    py_random.randint(1, 100)
    np.random.randn(5)
    torch.randn(3)

    # Create a minimal snapshot with current RNG state
    s = FullTrainingSnapshot(
        trial_id=0,
        python_rng_state=py_random.getstate(),
        numpy_rng_state=np.random.get_state(),
        torch_rng_state=torch.random.get_rng_state(),
    )

    # Generate more values (advancing RNG)
    py_val_after = py_random.randint(1, 100)
    np_val_after = np.random.randn(1)
    torch_val_after = torch.randn(1)

    # Restore RNG to snapshot state
    restore_rng_state(s)

    # Regenerate — should match post-snapshot values
    py_restored = py_random.randint(1, 100)
    assert py_restored == py_val_after

    np_restored = np.random.randn(1)
    assert np_restored == np_val_after

    torch_restored = torch.randn(1)
    assert torch.equal(torch_restored, torch_val_after)


def test_restore_lineage_from_snapshot():
    from lineage import LineageTracker
    from snapshot import FullTrainingSnapshot, restore_lineage_from_snapshot

    orig = LineageTracker(trial_type="fresh", trial_id=42)
    orig.add_frames(1000)
    orig.add_update()
    orig.add_update()

    s = FullTrainingSnapshot(
        trial_id=42,
        trial_type="fresh",
        lineage_chain_id=orig.lineage_chain_id,
        lineage_node_id=orig.lineage_node_id,
        lineage_root_trial_id=orig.lineage_root_trial_id,
        parent_trial_id=orig.parent_trial_id,
        parent_snapshot_ref=orig.parent_snapshot_ref,
        inheritance_depth=orig.inheritance_depth,
        local_train_raw_env_frames=orig.local_train_raw_env_frames,
        lineage_train_raw_env_frames=orig.lineage_train_raw_env_frames,
        train_updates=orig.train_updates,
    )

    restored = restore_lineage_from_snapshot(s)
    assert restored.trial_type == "fresh"
    assert restored.lineage_chain_id == orig.lineage_chain_id
    assert restored.lineage_node_id == orig.lineage_node_id
    assert restored.trial_id == 42  # trial_id is NOT overwritten
    assert restored.local_train_raw_env_frames == 1000
    assert restored.train_updates == 2


def test_snapshot_prune_old():
    import os
    from pathlib import Path
    from snapshot import prune_old_snapshots, SNAPSHOT_KEEP_LAST_N

    base = Path(os.environ.get("TMPDIR", os.environ.get("TEMP", "/tmp")))
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_dir = Path(tmpdir) / "snapshots"
        snap_dir.mkdir()

        # Create 5 snapshots for trial 0
        for i in range(5):
            pt = snap_dir / f"snapshot_0_{i * 1000}.pt"
            sha = snap_dir / f"snapshot_0_{i * 1000}.sha256"
            pt.write_bytes(b"data")
            sha.write_text(f"sha_{i}", encoding="utf-8")

        prune_old_snapshots(tmpdir, trial_id=0, keep_last_n=SNAPSHOT_KEEP_LAST_N)
        remaining = sorted(snap_dir.glob("snapshot_0_*.pt"))
        assert len(remaining) == SNAPSHOT_KEEP_LAST_N


def test_lineage_add_frames():
    from lineage import LineageTracker

    lt = LineageTracker(trial_type="fresh", trial_id=0)
    assert lt.local_train_raw_env_frames == 0
    assert lt.lineage_train_raw_env_frames == 0

    lt.add_frames(500)
    assert lt.local_train_raw_env_frames == 500
    assert lt.lineage_train_raw_env_frames == 500

    lt.add_frames(300)
    assert lt.local_train_raw_env_frames == 800


def test_lineage_add_update():
    from lineage import LineageTracker

    lt = LineageTracker(trial_type="fresh", trial_id=0)
    assert lt.train_updates == 0
    lt.add_update()
    assert lt.train_updates == 1
    lt.add_update()
    assert lt.train_updates == 2


def test_lineage_warm_start_inherits_chain():
    from lineage import LineageTracker

    parent = LineageTracker(trial_type="fresh", trial_id=0)
    parent.add_frames(10000)
    parent.add_update()

    child = LineageTracker(trial_type="warm_start", trial_id=1,
                           parent_lineage=parent, parent_trial_id=0,
                           parent_snapshot_ref="snap_0.pt")

    assert child.trial_type == "warm_start"
    assert child.lineage_chain_id == parent.lineage_chain_id
    assert child.lineage_root_trial_id == parent.lineage_root_trial_id
    assert child.parent_trial_id == 0
    assert child.parent_snapshot_ref == "snap_0.pt"
    assert child.inheritance_depth == 1
    assert child.local_train_raw_env_frames == 0  # child starts fresh
    assert child.lineage_train_raw_env_frames == 10000  # inherits parent's total
    assert child.train_updates == 0  # child starts counting its own updates


def test_run_trial_with_legacy_params():
    """V2 behavior: run_trial with no V3.1 params must work as before."""
    import tempfile
    from train_eval import run_trial
    from search_driver import BASELINE_CONFIG

    config = dict(BASELINE_CONFIG)
    config['replay_start_size'] = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_trial(
            config=config,
            trial_id=100,
            seed=42,
            source='test',
            max_trial_frames=500,
            eval_interval_frames=250,
            eval_episodes=2,
            candidate_verify_episodes=3,
            checkpoint_dir=tmpdir,
        )
        # V2 fields must still be present
        for k in ['trial_id', 'status', 'objective', 'train_raw_env_frames']:
            assert k in result, f"Missing: {k}"
        # V3.1 fields must also be present (trial_type defaults to 'fresh')
        assert result.get('trial_type') == 'fresh'
        assert 'lineage_chain_id' in result
        assert 'block_raw_env_frames' in result


def test_run_trial_fresh_produces_snapshot():
    """Fresh trial with snapshot_interval should save snapshots."""
    import tempfile
    from train_eval import run_trial
    from search_driver import BASELINE_CONFIG

    config = dict(BASELINE_CONFIG)
    config['replay_start_size'] = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_trial(
            config=config,
            trial_id=101,
            seed=42,
            source='test',
            trial_type='fresh',
            max_trial_frames=1000,
            eval_interval_frames=500,
            eval_episodes=2,
            candidate_verify_episodes=3,
            snapshot_interval=500,
            checkpoint_dir=tmpdir,
        )
        # Snapshot should have been created
        snap_path = result.get('last_snapshot_path', '')
        assert snap_path != '', "Expected a snapshot path"
        assert os.path.exists(snap_path)


def test_resume_without_snapshot_raises():
    """trial_type='resume' without resume_snapshot_path must raise."""
    import tempfile
    from train_eval import run_trial
    from search_driver import BASELINE_CONFIG
    import pytest

    config = dict(BASELINE_CONFIG)
    config['replay_start_size'] = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="requires resume_snapshot_path"):
            run_trial(
                config=config,
                trial_id=102,
                seed=42,
                source='test',
                trial_type='resume',
                resume_snapshot_path=None,
                max_trial_frames=500,
                checkpoint_dir=tmpdir,
            )


def test_fresh_with_snapshot_path_raises():
    """trial_type='fresh' with resume_snapshot_path must raise."""
    import tempfile
    from train_eval import run_trial
    from search_driver import BASELINE_CONFIG
    import pytest

    config = dict(BASELINE_CONFIG)
    config['replay_start_size'] = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="not 'fresh'"):
            run_trial(
                config=config,
                trial_id=103,
                seed=42,
                source='test',
                trial_type='fresh',
                resume_snapshot_path='/nonexistent/path.pt',
                max_trial_frames=500,
                checkpoint_dir=tmpdir,
            )


def test_snapshot_roundtrip_and_resume_deterministic_smoke():
    """Smoke test: A/B path resume consistency (small scale)."""
    from train_eval import run_trial
    from search_driver import BASELINE_CONFIG

    config = dict(BASELINE_CONFIG)
    config['replay_start_size'] = 100

    # Path A: continuous 2000 frames
    result_a = run_trial(
        config, trial_id=0, seed=42, source='test',
        trial_type='fresh',
        max_trial_frames=2000, eval_interval_frames=1000,
        eval_episodes=2, candidate_verify_episodes=3,
        snapshot_interval=500,
    )

    # Path B: 1000 frames -> save -> resume -> continue to 2000
    result_b1 = run_trial(
        config, trial_id=1, seed=42, source='test',
        trial_type='fresh',
        max_trial_frames=1000, eval_interval_frames=500,
        eval_episodes=2, candidate_verify_episodes=3,
        snapshot_interval=500,
    )
    snapshot_path = result_b1.get('last_snapshot_path')
    assert snapshot_path, "B1 should have a snapshot"

    result_b2 = run_trial(
        config, trial_id=1, seed=42, source='test',
        trial_type='resume',
        resume_snapshot_path=snapshot_path,
        max_trial_frames=2000, eval_interval_frames=1000,
        eval_episodes=2, candidate_verify_episodes=3,
        snapshot_interval=500,
    )

    # Same-trial resume: lineage identity must be preserved
    assert result_b2.get('lineage_chain_id') == result_b1.get('lineage_chain_id')
    assert result_b2.get('lineage_node_id') == result_b1.get('lineage_node_id')

    # Decision steps and train updates should match between A and resumed B
    assert result_b2['decision_steps'] == result_a['decision_steps'], \
        f"decision_steps: {result_b2['decision_steps']} != {result_a['decision_steps']}"
    assert result_b2.get('train_updates') == result_a.get('train_updates'), \
        f"train_updates: {result_b2.get('train_updates')} != {result_a.get('train_updates')}"

    # Replay buffer size should match
    assert result_b2.get('replay_buffer_size') == result_a.get('replay_buffer_size'), \
        f"buffer size: {result_b2.get('replay_buffer_size')} != {result_a.get('replay_buffer_size')}"

    # Epsilon should be close
    assert abs(result_b2.get('epsilon', 0) - result_a.get('epsilon', 0)) < 0.01

    # Best eval score should be close (within tolerance)
    assert abs(result_b2['best_eval_score'] - result_a['best_eval_score']) < 5


def test_snapshot_roundtrip_and_resume_deterministic_acceptance():
    """Acceptance test: Path A 10,000 vs Path B 5,000 + resume + 5,000."""
    import pytest
    from train_eval import run_trial
    from search_driver import BASELINE_CONFIG

    config = dict(BASELINE_CONFIG)
    config['replay_start_size'] = 100

    result_a = run_trial(
        config, trial_id=10, seed=42, source='test_acceptance',
        trial_type='fresh',
        max_trial_frames=10_000, eval_interval_frames=5_000,
        eval_episodes=2, candidate_verify_episodes=3,
        snapshot_interval=5_000,
    )

    result_b1 = run_trial(
        config, trial_id=11, seed=42, source='test_acceptance',
        trial_type='fresh',
        max_trial_frames=5_000, eval_interval_frames=5_000,
        eval_episodes=2, candidate_verify_episodes=3,
        snapshot_interval=5_000,
    )
    snapshot_path = result_b1.get('last_snapshot_path')

    result_b2 = run_trial(
        config, trial_id=11, seed=42, source='test_acceptance',
        trial_type='resume',
        resume_snapshot_path=snapshot_path,
        max_trial_frames=10_000, eval_interval_frames=5_000,
        eval_episodes=2, candidate_verify_episodes=3,
        snapshot_interval=5_000,
    )

    assert result_b2['decision_steps'] == result_a['decision_steps']
    assert result_b2.get('train_updates') == result_a.get('train_updates')
    assert result_b2.get('replay_buffer_size') == result_a.get('replay_buffer_size')
    assert result_b2.get('n_step_queue_length') == result_a.get('n_step_queue_length')
    assert abs(result_b2.get('epsilon', 0) - result_a.get('epsilon', 0)) < 0.01

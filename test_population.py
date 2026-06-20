"""V3.4 population controller tests."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def test_population_worker_ranking():
    from population import PopulationController
    pc = PopulationController(population_size=2)
    pc.add_worker(0, {}, 42)
    pc.add_worker(1, {}, 43)
    pc.workers[0]["stable_success"] = True
    pc.workers[0]["lineage_train_raw_env_frames"] = 100
    pc.workers[1]["stable_success"] = False
    pc.workers[1]["best_eval_score"] = 50
    ranked = pc._rank_workers()
    assert ranked[0]["trial_id"] == 0  # success first


def test_mutate_online_params_clamps_lr():
    from population import _mutate_online_params
    config = {"lr": 1e-3, "tau": 0.005}
    mutated = _mutate_online_params(config)
    assert "lr" in mutated.get("_mutated_fields", [])
    assert 1e-5 <= mutated["lr"] <= 3e-3


def test_population_replace_worker_sets_child_lineage():
    from population import PopulationController
    pc = PopulationController(population_size=2)
    pc.add_worker(0, {}, 42)
    pc.add_worker(1, {}, 43)
    parent = pc.workers[0]
    parent["snapshot_path"] = "/tmp/snap.pt"
    parent["lineage"]["lineage_chain_id"] = "chain-abc"
    parent["lineage"]["lineage_train_raw_env_frames"] = 5000
    parent["lineage"]["inheritance_depth"] = 2
    bottom = pc.workers[1]
    pc._replace_worker(bottom, parent, {"lr": 2e-4, "_mutated_fields": ["lr"]})
    assert bottom["lineage"]["trial_type"] == "population_inherited"
    assert bottom["lineage"]["inheritance_depth"] > 0
    assert bottom["lineage"]["lineage_chain_id"] == "chain-abc"


def test_mutate_online_params_preserves_original():
    from population import _mutate_online_params
    config = {"lr": 1e-3, "tau": 0.005, "eps_end": 0.005, "hidden": [128, 64]}
    mutated = _mutate_online_params(config)
    # Unchanged fields should survive
    assert mutated["hidden"] == [128, 64]
    assert mutated["tau"] == 0.005 or mutated["tau"] == 0.005 * 0.8 or mutated["tau"] == 0.005 * 1.2
    assert "_mutated_fields" in mutated


def test_population_add_worker_creates_fresh_lineage():
    from population import PopulationController
    pc = PopulationController(population_size=1)
    pc.add_worker(42, {"lr": 1e-4}, 7)
    assert len(pc.workers) == 1
    w = pc.workers[0]
    assert w["trial_id"] == 42
    assert w["seed"] == 7
    assert w["lineage"]["trial_type"] == "fresh"
    assert w["lineage"]["inheritance_depth"] == 0


def test_population_next_trial_id_increments():
    from population import PopulationController
    pc = PopulationController(population_size=1)
    id1 = pc._next_trial_id()
    id2 = pc._next_trial_id()
    assert id2 > id1
    assert id1 >= 10001  # starts at counter + 10000


def test_population_rank_workers_sort_order():
    from population import PopulationController
    pc = PopulationController(population_size=3)
    pc.add_worker(10, {}, 40)
    pc.add_worker(20, {}, 41)
    pc.add_worker(30, {}, 42)
    # Worker 20: success, 200 frames
    pc.workers[1]["stable_success"] = True
    pc.workers[1]["lineage_train_raw_env_frames"] = 200
    # Worker 30: success, 100 frames (better)
    pc.workers[2]["stable_success"] = True
    pc.workers[2]["lineage_train_raw_env_frames"] = 100
    # Worker 10: failure, eval 150
    pc.workers[0]["stable_success"] = False
    pc.workers[0]["best_eval_score"] = 150
    ranked = pc._rank_workers()
    # Best should be worker 30 (success with 100 frames)
    assert ranked[0]["trial_id"] == 30
    # Second best should be worker 20 (success with 200 frames)
    assert ranked[1]["trial_id"] == 20
    # Last should be worker 10 (failure)
    assert ranked[2]["trial_id"] == 10

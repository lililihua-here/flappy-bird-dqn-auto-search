"""Population-based self-evolution search controller."""
import random
import uuid
import numpy as np
from lineage import LineageTracker


class PopulationController:
    """Small async population with exploit/explore."""

    def __init__(self, population_size=4, history=None,
                 eval_interval=20_000, exploit_interval=50_000,
                 bottom_fraction=0.25, top_fraction=0.25):
        self.population_size = population_size
        self.workers = []
        self.history = history
        self.eval_interval = eval_interval
        self.exploit_interval = exploit_interval
        self.bottom_fraction = bottom_fraction
        self.top_fraction = top_fraction
        self.total_frames = 0
        self.last_global_exploit_frame = 0
        self._trial_counter = 0

    def _next_trial_id(self):
        self._trial_counter += 1
        return self._trial_counter + 10000

    def add_worker(self, trial_id, config, seed):
        worker = {
            "trial_id": trial_id, "seed": seed, "config": config,
            "lineage": LineageTracker(trial_type="fresh", trial_id=trial_id).to_dict(),
            "snapshot_path": "", "resume_snapshot_path": None, "last_exploit_frame": 0,
            "best_eval_score": 0, "latest_eval_median_score": 0,
            "stable_success": False,
            "lineage_train_raw_env_frames": 0, "local_train_raw_env_frames": 0,
        }
        self.workers.append(worker)

    def _rank_workers(self):
        def key(w):
            success = w.get("stable_success", False)
            if success:
                return (0, w.get("lineage_train_raw_env_frames", float("inf")), 0, 0)
            return (1, -w.get("best_eval_score", 0),
                    w.get("lineage_train_raw_env_frames", float("inf")),
                    -w.get("latest_eval_median_score", 0))
        return sorted(self.workers, key=key)

    def _train_worker_block(self, worker, block_frames):
        from train_eval import run_trial
        if worker.get("resume_snapshot_path"):
            resume_path = worker["resume_snapshot_path"]
            trial_type = "population_inherited"
        elif worker.get("snapshot_path"):
            resume_path = worker["snapshot_path"]
            trial_type = "resume"
        else:
            resume_path = None
            trial_type = "fresh"
        result = run_trial(
            config=worker["config"], trial_id=worker["trial_id"], seed=worker["seed"],
            source="population_async", trial_type=trial_type,
            resume_snapshot_path=resume_path,
            parent_trial_id=worker.get("parent_trial_id"),
            parent_snapshot_ref=worker.get("parent_snapshot_ref"),
            max_trial_frames=block_frames,
        )
        # Build lineage dict from run_trial's flat result fields
        lin = {
            "trial_type": result.get("trial_type", trial_type),
            "lineage_chain_id": result.get("lineage_chain_id", ""),
            "lineage_node_id": result.get("lineage_node_id", ""),
            "lineage_root_trial_id": result.get("lineage_root_trial_id", worker["trial_id"]),
            "parent_trial_id": result.get("parent_trial_id", -1),
            "parent_snapshot_ref": result.get("parent_snapshot_ref", ""),
            "inheritance_depth": result.get("inheritance_depth", 0),
            "local_train_raw_env_frames": result.get("local_train_raw_env_frames", 0),
            "lineage_train_raw_env_frames": result.get("lineage_train_raw_env_frames", 0),
            "train_updates": result.get("train_updates", 0),
        }
        worker["lineage"] = lin
        worker["snapshot_path"] = result.get("last_snapshot_path", worker.get("snapshot_path", ""))
        worker["local_train_raw_env_frames"] = worker["lineage"].get("local_train_raw_env_frames", 0)
        worker["lineage_train_raw_env_frames"] = worker["lineage"].get("lineage_train_raw_env_frames", 0)
        worker["resume_snapshot_path"] = None
        return result.get("block_raw_env_frames", block_frames)

    def _eval_worker(self, worker):
        from snapshot import load_snapshot
        from dqn_agent import DQNAgent
        from flappy_bird_env import FlappyBirdEnv
        from state_encoder_variants import get_encoder
        import torch
        if not worker["snapshot_path"]:
            return 0
        snapshot = load_snapshot(worker["snapshot_path"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_ver = getattr(snapshot, 'state_representation_version', 'low_dim_v1')
        encoder = get_encoder(state_ver)
        state_dim = encoder.state_dim
        agent = DQNAgent(dict(snapshot.config), state_dim, 2, device)
        agent.q_net.load_state_dict(snapshot.q_net_state_dict)
        agent.q_net.eval()
        scores = []
        for ep in range(20):
            env = FlappyBirdEnv(seed=worker["seed"] + 1000 + ep)
            state = env.reset()
            done = False
            ep_frames = 0
            while not done and ep_frames < 120000:
                vec = encoder.encode(state)
                with torch.no_grad():
                    action = int(agent.q_net(torch.from_numpy(vec).unsqueeze(0).to(device)).argmax().item())
                state, _, done = env.step(action)
                ep_frames += 1
            scores.append(env.score)
        median_score = float(np.median(scores))
        success_count = sum(s >= 1000 for s in scores)
        worker["latest_eval_median_score"] = median_score
        worker["best_eval_score"] = max(worker.get("best_eval_score", 0), median_score)
        worker["stable_success"] = (success_count >= 14 and median_score >= 1000)
        worker["latest_eval_scores"] = scores
        return median_score

    def _write_population_event(self, event):
        if self.history is not None:
            self.history.append(event)

    def _replace_worker(self, bottom_worker, parent, mutated_config):
        if not parent.get("snapshot_path"):
            raise ValueError("parent snapshot_path required for exploit/explore")
        child_trial_id = self._next_trial_id()
        bottom_worker["snapshot_path"] = parent["snapshot_path"]
        bottom_worker["resume_snapshot_path"] = parent["snapshot_path"]
        bottom_worker["parent_snapshot_ref"] = parent["snapshot_path"]
        bottom_worker["trial_id"] = child_trial_id
        bottom_worker["config"] = mutated_config
        bottom_worker["parent_trial_id"] = parent["trial_id"]
        bottom_worker["lineage_train_raw_env_frames"] = parent["lineage_train_raw_env_frames"]
        bottom_worker["local_train_raw_env_frames"] = 0
        bottom_worker["stable_success"] = False
        bottom_worker["best_eval_score"] = 0
        parent_lin = parent["lineage"]
        child_lineage = {
            "trial_type": "population_inherited",
            "lineage_chain_id": parent_lin.get("lineage_chain_id", ""),
            "lineage_node_id": uuid.uuid4().hex[:12],
            "lineage_root_trial_id": parent_lin.get("lineage_root_trial_id", child_trial_id),
            "parent_trial_id": parent["trial_id"],
            "parent_snapshot_ref": parent["snapshot_path"],
            "inheritance_depth": parent_lin.get("inheritance_depth", 0) + 1,
            "local_train_raw_env_frames": 0,
            "lineage_train_raw_env_frames": parent_lin.get("lineage_train_raw_env_frames", 0),
            "train_updates": 0,
        }
        bottom_worker["lineage"] = child_lineage
        event = {
            "record_type": "population_event", "event_type": "exploit_explore",
            "parent_trial_id": parent["trial_id"], "child_trial_id": child_trial_id,
            "parent_snapshot_ref": parent["snapshot_path"],
            "mutated_fields": mutated_config.get("_mutated_fields", []),
            "parent_lineage_train_raw_env_frames": parent["lineage_train_raw_env_frames"],
            "child_local_train_raw_env_frames": 0,
            "child_lineage_train_raw_env_frames": parent["lineage_train_raw_env_frames"],
            "child_config": {k: v for k, v in mutated_config.items() if not k.startswith("_")},
        }
        self._write_population_event(event)

    def _exploit_explore(self):
        ranked = self._rank_workers()
        n_bottom = max(1, int(len(ranked) * self.bottom_fraction))
        n_top = max(1, int(len(ranked) * self.top_fraction))
        for bottom_worker in ranked[-n_bottom:]:
            parent = random.choice(ranked[:n_top])
            mutated_config = _mutate_online_params(parent['config'])
            self._replace_worker(bottom_worker, parent, mutated_config)

    def run(self, total_frame_budget):
        self.total_frames = 0
        while self.total_frames < total_frame_budget:
            for w in self.workers:
                frames = self._train_worker_block(w, block_frames=self.eval_interval)
                self.total_frames += frames
                w["latest_eval_median_score"] = self._eval_worker(w)
                w["best_eval_score"] = max(w.get("best_eval_score", 0), w["latest_eval_median_score"])
                w["lineage_train_raw_env_frames"] = w["lineage"].get("lineage_train_raw_env_frames", 0)
            if self.total_frames - self.last_global_exploit_frame >= self.exploit_interval:
                self._exploit_explore()
                self.last_global_exploit_frame = self.total_frames


def _mutate_online_params(config):
    mutations = {}
    if 'lr' in config:
        lr = config['lr'] * random.choice([0.8, 1.2])
        mutations['lr'] = max(1e-5, min(3e-3, lr))
    if 'tau' in config:
        tau = config['tau'] * random.choice([0.8, 1.2])
        mutations['tau'] = max(0.001, min(0.05, tau))
    if 'eps_end' in config:
        mutations['eps_end'] = max(0.001, min(0.02, config['eps_end'] + random.uniform(-0.002, 0.002)))
    if 'per_beta_current' in config:
        mutations['per_beta_current'] = min(1.0, config['per_beta_current'] * random.choice([1.05, 1.1]))
    new = {**config, **mutations}
    new['_mutated_fields'] = list(mutations.keys())
    return new

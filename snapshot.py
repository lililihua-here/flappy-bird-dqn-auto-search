"""Full training snapshot engine for V3 — save, load, restore, verify."""
import hashlib
import os
import pickle
import time
import numpy as np
import torch
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field

from lineage import LineageTracker

SNAPSHOT_FORMAT_VERSION = "v3_snapshot_1"
SNAPSHOT_INTERVAL = 20_000
SNAPSHOT_KEEP_LAST_N = 3


def _replace_with_retry(src, dst, retries=5, delay_sec=0.05):
    """Replace a file with small retries for transient Windows file locks."""
    src_path = Path(src)
    dst_path = Path(dst)
    last_exc = None
    for attempt in range(retries):
        try:
            os.replace(str(src_path), str(dst_path))
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt >= retries - 1:
                raise
            if dst_path.exists():
                try:
                    dst_path.unlink()
                except PermissionError:
                    pass
            time.sleep(delay_sec * (attempt + 1))
    raise last_exc


@dataclass
class FullTrainingSnapshot:
    """Complete serialisable training state for deterministic resume."""

    format_version: str = SNAPSHOT_FORMAT_VERSION
    trial_id: int = -1
    seed: int = 0
    source: str = ""
    lineage_chain_id: str = ""
    lineage_node_id: str = ""
    trial_type: str = "fresh"
    parent_trial_id: int = -1
    parent_snapshot_ref: str = ""
    lineage_root_trial_id: int = -1
    inheritance_depth: int = 0

    # Network weights
    q_net_state_dict: dict = None
    target_net_state_dict: dict = None
    optimizer_state_dict: dict = None

    # Training state
    epsilon: float = 0.0
    decision_steps: int = 0
    train_updates: int = 0
    local_train_raw_env_frames: int = 0
    lineage_train_raw_env_frames: int = 0
    env_runtime_state: dict = field(default_factory=dict)
    training_meta: dict = field(default_factory=dict)

    # Replay state
    replay_buffer_type: str = ""
    replay_buffer_meta: dict = field(default_factory=dict)
    replay_buffer_data: list = field(default_factory=list)
    per_tree_data: dict = field(default_factory=dict)
    n_step_queue: list = field(default_factory=list)

    # RNG
    python_rng_state: tuple = ()
    numpy_rng_state: tuple = ()
    torch_rng_state: object = None
    cuda_rng_state: list = field(default_factory=list)

    # Config
    config: dict = field(default_factory=dict)
    state_representation_version: str = "low_dim_v1"
    reward_scheme_version: str = "reward_v1_sparse"
    environment_version: str = "fixed_env_v1"
    code_version: str = ""

    # Metadata
    sha256: str = ""
    created_at: str = ""


# ============================================================================
# Utility: recursive CPU transfer
# ============================================================================
def to_cpu(obj):
    """Recursively move tensors to CPU for safe cross-device serialization."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


# ============================================================================
# Replay buffer type inference (used by verify)
# ============================================================================
def infer_replay_buffer_type(config):
    """Infer expected replay buffer type from config."""
    priority = config.get("priority", config.get("use_per", False))
    n_step = config.get("n_step", 1)
    if priority and n_step > 1:
        return "NStepPERBuffer"
    if priority:
        return "PERBuffer"
    if n_step > 1:
        return "NStepReplayBuffer"
    return "ReplayBuffer"


# ============================================================================
# Compatibility check
# ============================================================================
def verify_snapshot_compatible(snapshot, config):
    """Check structural compatibility between snapshot and current config."""
    expected_buffer = infer_replay_buffer_type(config)
    if snapshot.replay_buffer_type != expected_buffer:
        return False
    checks = [
        (snapshot.environment_version, config.get("environment_version", "fixed_env_v1")),
        (snapshot.state_representation_version, config.get("state_representation_version", "low_dim_v1")),
        (snapshot.reward_scheme_version, config.get("reward_scheme_version", "reward_v1_sparse")),
        (snapshot.config.get("network_backbone", "mlp"), config.get("network_backbone", "mlp")),
        (snapshot.config.get("exploration_head", "epsilon_greedy"), config.get("exploration_head", "epsilon_greedy")),
        (snapshot.config.get("n_step", 1), config.get("n_step", 1)),
        (snapshot.config.get("priority", False), config.get("priority", False)),
    ]
    return all(a == b for a, b in checks)


# ============================================================================
# Capture
# ============================================================================
def capture_snapshot(agent, env, config, trial_id, seed, source, lineage_tracker):
    """Capture full training state into a FullTrainingSnapshot."""
    import random as py_random

    s = FullTrainingSnapshot(
        trial_id=trial_id, seed=seed, source=source,
        lineage_chain_id=lineage_tracker.lineage_chain_id,
        lineage_node_id=lineage_tracker.lineage_node_id,
        trial_type=lineage_tracker.trial_type,
        parent_trial_id=lineage_tracker.parent_trial_id,
        lineage_root_trial_id=lineage_tracker.lineage_root_trial_id,
        inheritance_depth=lineage_tracker.inheritance_depth,
        parent_snapshot_ref=lineage_tracker.parent_snapshot_ref,
        q_net_state_dict={k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()},
        target_net_state_dict={k: v.cpu().clone() for k, v in agent.target_net.state_dict().items()},
        optimizer_state_dict=to_cpu(agent.optimizer.state_dict()),
        epsilon=agent.epsilon,
        decision_steps=agent.decision_steps,
        train_updates=lineage_tracker.train_updates,
        local_train_raw_env_frames=lineage_tracker.local_train_raw_env_frames,
        lineage_train_raw_env_frames=lineage_tracker.lineage_train_raw_env_frames,
        env_runtime_state=env.capture_runtime_state(),
        config=config,
        state_representation_version=config.get("state_representation_version", "low_dim_v1"),
        reward_scheme_version=config.get("reward_scheme_version", "reward_v1_sparse"),
        environment_version=config.get("environment_version", "fixed_env_v1"),
        code_version=config.get("code_version", ""),
        python_rng_state=py_random.getstate(),
        numpy_rng_state=np.random.get_state(),
        torch_rng_state=torch.random.get_rng_state(),
    )

    # Replay buffer metadata
    s.replay_buffer_type = type(agent.buffer).__name__
    s.replay_buffer_meta = {
        "capacity": getattr(agent.buffer, "capacity", None),
        "n_step": getattr(agent.buffer, "n_step", None),
        "gamma": getattr(agent.buffer, "gamma", None),
        "alpha": getattr(agent.buffer, "alpha", None),
        "beta_start": getattr(agent.buffer, "beta_start", None),
        "beta_train_updates": getattr(agent.buffer, "beta_train_updates", None),
        "priority_eps": getattr(agent.buffer, "priority_eps", None),
        "train_updates": getattr(agent.buffer, "_train_updates", None),
        "max_raw_priority": getattr(agent.buffer, "_max_raw_priority", None),
    }

    s.cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )

    # Replay buffer contents
    s.replay_buffer_data = list(agent.buffer.buffer) if hasattr(agent.buffer, 'buffer') else []

    # PER tree
    if hasattr(agent.buffer, 'tree'):
        s.per_tree_data = {
            'tree': agent.buffer.tree.tree.copy(),
            'data': agent.buffer.tree.data.copy(),
            'ptr': agent.buffer.tree._ptr,
            'size': agent.buffer.tree._size,
            'max_raw_priority': agent.buffer._max_raw_priority,
        }

    # n-step queue
    if hasattr(agent.buffer, '_n_step_queue'):
        s.n_step_queue = list(agent.buffer._n_step_queue)

    return s


# ============================================================================
# Save / Load
# ============================================================================
def save_snapshot(snapshot, checkpoint_dir):
    """Atomic write with external sha256 file. Prunes old snapshots after save."""
    save_dir = Path(checkpoint_dir) / "snapshots"
    save_dir.mkdir(parents=True, exist_ok=True)

    snapshot.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    snapshot.sha256 = ""  # do NOT embed file hash in the hashed payload

    tmp_pt = save_dir / f"snapshot_{snapshot.trial_id}_{snapshot.decision_steps}.pt.tmp"
    tmp_sha = save_dir / f"snapshot_{snapshot.trial_id}_{snapshot.decision_steps}.sha256.tmp"
    final_pt = save_dir / f"snapshot_{snapshot.trial_id}_{snapshot.decision_steps}.pt"
    final_sha = save_dir / f"snapshot_{snapshot.trial_id}_{snapshot.decision_steps}.sha256"

    torch.save(snapshot, str(tmp_pt), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    file_sha = hashlib.sha256(tmp_pt.read_bytes()).hexdigest()
    tmp_sha.write_text(file_sha, encoding="utf-8")

    _replace_with_retry(tmp_pt, final_pt)
    _replace_with_retry(tmp_sha, final_sha)

    prune_old_snapshots(checkpoint_dir, snapshot.trial_id)

    return str(final_pt), file_sha


def load_snapshot(path):
    """Load snapshot and verify sha256 integrity."""
    path = Path(path)
    sha_path = path.with_suffix(".sha256")

    if not sha_path.exists():
        raise ValueError(f"Missing snapshot sha256 file: {sha_path}")

    expected = sha_path.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Snapshot sha256 mismatch: {actual} != {expected}")

    s = torch.load(str(path), weights_only=False)
    if s.format_version != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(f"Snapshot version mismatch: {s.format_version} vs {SNAPSHOT_FORMAT_VERSION}")
    return s


def prune_old_snapshots(save_dir, trial_id, keep_last_n=SNAPSHOT_KEEP_LAST_N):
    """Remove old snapshots for a trial, keeping the most recent N."""
    save_dir = Path(save_dir) / "snapshots"
    if not save_dir.exists():
        return
    snapshots = sorted(
        [p for p in save_dir.glob(f"snapshot_{trial_id}_*.pt")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in snapshots[keep_last_n:]:
        p.unlink(missing_ok=True)
        p.with_suffix(".sha256").unlink(missing_ok=True)


# ============================================================================
# Restore functions
# ============================================================================
def restore_snapshot_to_agent(snapshot, agent):
    """Restore network weights, optimizer, epsilon, step counters."""
    agent.q_net.load_state_dict(snapshot.q_net_state_dict)
    agent.target_net.load_state_dict(snapshot.target_net_state_dict)
    agent.optimizer.load_state_dict(snapshot.optimizer_state_dict)
    agent.epsilon = snapshot.epsilon
    agent.decision_steps = snapshot.decision_steps
    agent.train_updates = snapshot.train_updates
    return agent


def _restore_per_tree(buf, tree_data):
    """Helper: restore SumTree state from snapshot data."""
    if tree_data:
        buf.tree.tree = tree_data.get("tree", buf.tree.tree).copy()
        buf.tree.data = tree_data.get("data", buf.tree.data).copy()
        buf.tree._ptr = tree_data.get("ptr", 0)
        buf.tree._size = tree_data.get("size", 0)


def restore_replay_buffer(snapshot, config):
    """Rebuild replay buffer from snapshot with correct type and contents."""
    from replay_buffer import ReplayBuffer, NStepReplayBuffer, PERBuffer, NStepPERBuffer

    buf_type = snapshot.replay_buffer_type
    meta = snapshot.replay_buffer_meta or {}

    if buf_type == "ReplayBuffer":
        buf = ReplayBuffer(capacity=meta.get("capacity", config.get("buffer_sz", 50000)))
        buf.buffer = deque(snapshot.replay_buffer_data, maxlen=buf.capacity)
    elif buf_type == "NStepReplayBuffer":
        buf = NStepReplayBuffer(
            capacity=meta.get("capacity", config.get("buffer_sz", 50000)),
            n_step=meta.get("n_step", config.get("n_step", 3)),
            gamma=meta.get("gamma", config.get("gamma", 0.99)),
        )
        buf.buffer = deque(snapshot.replay_buffer_data, maxlen=buf.capacity)
        buf._n_step_queue = deque(snapshot.n_step_queue, maxlen=buf.n_step)
    elif buf_type == "PERBuffer":
        buf = PERBuffer(
            capacity=meta.get("capacity", config.get("buffer_sz", 50000)),
            alpha=meta.get("alpha", config.get("per_alpha", 0.6)),
            beta=meta.get("beta_start", config.get("per_beta_start", 0.4)),
            beta_train_updates=meta.get("beta_train_updates", 50000),
            priority_eps=meta.get("priority_eps", 1e-6),
        )
        _restore_per_tree(buf, snapshot.per_tree_data)
        buf._train_updates = meta.get("train_updates", 0)
        buf._max_raw_priority = meta.get("max_raw_priority", 1.0)
    elif buf_type == "NStepPERBuffer":
        buf = NStepPERBuffer(
            capacity=meta.get("capacity", config.get("buffer_sz", 50000)),
            n_step=meta.get("n_step", config.get("n_step", 3)),
            gamma=meta.get("gamma", config.get("gamma", 0.99)),
            alpha=meta.get("alpha", config.get("per_alpha", 0.6)),
            beta=meta.get("beta_start", config.get("per_beta_start", 0.4)),
            beta_train_updates=meta.get("beta_train_updates", 50000),
            priority_eps=meta.get("priority_eps", 1e-6),
        )
        _restore_per_tree(buf, snapshot.per_tree_data)
        buf._n_step_queue = deque(snapshot.n_step_queue, maxlen=buf.n_step)
        buf._train_updates = meta.get("train_updates", 0)
        buf._max_raw_priority = meta.get("max_raw_priority", 1.0)
    else:
        raise ValueError(f"Unknown replay buffer type: {buf_type}")

    return buf


def restore_rng_state(snapshot):
    """Restore all RNG states from snapshot."""
    import random as py_random

    py_random.setstate(snapshot.python_rng_state)
    np.random.set_state(snapshot.numpy_rng_state)
    torch.random.set_rng_state(snapshot.torch_rng_state)
    if snapshot.cuda_rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.cuda_rng_state)


def restore_lineage_from_snapshot(snapshot):
    """Restore ALL lineage metadata from snapshot. Does NOT generate new IDs."""
    lineage = LineageTracker(trial_type="fresh", trial_id=snapshot.trial_id)
    lineage.trial_type = snapshot.trial_type
    lineage.lineage_chain_id = snapshot.lineage_chain_id
    lineage.lineage_node_id = snapshot.lineage_node_id
    lineage.parent_trial_id = snapshot.parent_trial_id
    lineage.parent_snapshot_ref = snapshot.parent_snapshot_ref
    lineage.lineage_root_trial_id = snapshot.lineage_root_trial_id
    lineage.inheritance_depth = snapshot.inheritance_depth
    lineage.local_train_raw_env_frames = snapshot.local_train_raw_env_frames
    lineage.lineage_train_raw_env_frames = snapshot.lineage_train_raw_env_frames
    lineage.train_updates = snapshot.train_updates
    return lineage


def load_agent_and_encoder_from_snapshot(path, device=None):
    """Load a V3 snapshot into an inference-ready agent and matching encoder."""
    from dqn_agent import DQNAgent
    from state_encoder_variants import get_encoder

    snapshot = load_snapshot(path)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = get_encoder(
        snapshot.state_representation_version,
        env_config={
            "SCREEN_WIDTH": snapshot.config.get("SCREEN_WIDTH", 600),
            "SCREEN_HEIGHT": snapshot.config.get("SCREEN_HEIGHT", 800),
            "MAX_FALL_SPEED": snapshot.config.get("MAX_FALL_SPEED", 10),
            "BIRD_X": snapshot.config.get("BIRD_X", 100),
        },
    )
    agent = DQNAgent(
        config=dict(snapshot.config),
        state_dim=encoder.state_dim,
        n_actions=2,
        device=device,
    )
    restore_snapshot_to_agent(snapshot, agent)
    agent.buffer = restore_replay_buffer(snapshot, snapshot.config)
    agent.q_net.eval()
    agent.target_net.eval()
    return agent, encoder, snapshot

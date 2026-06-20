"""Lineage tracking for V3 trial cost protocol."""
import uuid


class LineageTracker:
    """Tracks training lineage: local/lineage frame counts, parent-child relationships.

    lineage_chain_id: shared across all trials in the same inheritance chain.
    lineage_node_id: unique per trial (fresh gets its own chain_id == node_id).
    """

    def __init__(self, trial_type="fresh", trial_id=-1,
                 parent_lineage=None, parent_trial_id=None, parent_snapshot_ref=None):
        self.trial_type = trial_type
        self.trial_id = trial_id
        self.lineage_root_trial_id = trial_id
        self.parent_trial_id = -1
        self.parent_snapshot_ref = ""
        self.inheritance_depth = 0
        self.local_train_raw_env_frames = 0
        self.lineage_train_raw_env_frames = 0
        self.train_updates = 0

        if trial_type in ("warm_start", "population_inherited"):
            if parent_lineage is None:
                raise ValueError("parent_lineage is required for inherited trial")
            if parent_trial_id is None:
                raise ValueError("parent_trial_id is required for inherited trial")
            if parent_snapshot_ref is None:
                raise ValueError("parent_snapshot_ref is required for inherited trial")

            self.parent_trial_id = parent_trial_id
            self.parent_snapshot_ref = parent_snapshot_ref
            self.inheritance_depth = parent_lineage.inheritance_depth + 1
            self.lineage_chain_id = parent_lineage.lineage_chain_id
            self.lineage_node_id = uuid.uuid4().hex[:12]
            self.lineage_root_trial_id = parent_lineage.lineage_root_trial_id
            self.lineage_train_raw_env_frames = parent_lineage.lineage_train_raw_env_frames
        else:
            # Fresh trial: chain_id == node_id (same identity for the root)
            new_id = uuid.uuid4().hex[:12]
            self.lineage_chain_id = new_id
            self.lineage_node_id = new_id

    def add_frames(self, n):
        self.local_train_raw_env_frames += n
        self.lineage_train_raw_env_frames += n

    def add_update(self):
        self.train_updates += 1

    def to_dict(self):
        return {
            'trial_type': self.trial_type,
            'lineage_chain_id': self.lineage_chain_id,
            'lineage_node_id': self.lineage_node_id,
            'lineage_root_trial_id': self.lineage_root_trial_id,
            'parent_trial_id': self.parent_trial_id,
            'parent_snapshot_ref': self.parent_snapshot_ref,
            'inheritance_depth': self.inheritance_depth,
            'local_train_raw_env_frames': self.local_train_raw_env_frames,
            'lineage_train_raw_env_frames': self.lineage_train_raw_env_frames,
            'train_updates': self.train_updates,
        }

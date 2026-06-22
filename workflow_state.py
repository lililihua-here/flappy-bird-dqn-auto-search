"""Workflow state persistence for the auto workflow entrypoint."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


def _now_iso():
    return datetime.now().isoformat(timespec='seconds')


@dataclass
class WorkflowState:
    """Persisted state for the auto workflow."""

    workflow_version: str
    goal: str
    profile: str
    run_dir: str
    created_at: str
    updated_at: str
    current_stage: str
    stage_attempt: int
    completed_stages: list[str] = field(default_factory=list)
    failed_stages: list[str] = field(default_factory=list)
    stage_status_path: str = ''
    selected_protocol_entry: str = ''
    selected_structure_entry: str = ''
    retained_protocol_entries: list[str] = field(default_factory=list)
    retained_structure_entries: list[str] = field(default_factory=list)
    temporarily_disabled_protocol_entries: list[str] = field(default_factory=list)
    temporarily_disabled_structure_entries: list[str] = field(default_factory=list)
    permanently_eliminated_protocol_entries: list[str] = field(default_factory=list)
    permanently_eliminated_structure_entries: list[str] = field(default_factory=list)
    eliminated_protocol_entries: list[str] = field(default_factory=list)
    eliminated_structure_entries: list[str] = field(default_factory=list)
    focused_search_space: str = ''
    best_config_path: str = ''
    best_trial_summary: str = ''
    search_round_index: int = 0
    space_widen_count: int = 0
    history_paths: dict[str, str] = field(default_factory=dict)
    study_db_paths: dict[str, str] = field(default_factory=dict)
    checkpoint_dirs: dict[str, str] = field(default_factory=dict)
    blocked_reason: str = ''
    last_error: str = ''

    @classmethod
    def new(cls, goal: str, profile: str, run_dir):
        run_dir = Path(run_dir)
        now = _now_iso()
        return cls(
            workflow_version='v0.3',
            goal=goal,
            profile=profile,
            run_dir=str(run_dir),
            created_at=now,
            updated_at=now,
            current_stage='init',
            stage_attempt=0,
            completed_stages=[],
            failed_stages=[],
            stage_status_path=str(run_dir / 'stage_status.json'),
            selected_protocol_entry='',
            selected_structure_entry='',
            retained_protocol_entries=[],
            retained_structure_entries=[],
            temporarily_disabled_protocol_entries=[],
            temporarily_disabled_structure_entries=[],
            permanently_eliminated_protocol_entries=[],
            permanently_eliminated_structure_entries=[],
            eliminated_protocol_entries=[],
            eliminated_structure_entries=[],
            focused_search_space='',
            best_config_path=str(run_dir / 'best_config.json'),
            best_trial_summary='',
            search_round_index=0,
            space_widen_count=0,
            history_paths={},
            study_db_paths={},
            checkpoint_dirs={},
            blocked_reason='',
            last_error='',
        )

    def touch(self):
        self.updated_at = _now_iso()

    def save(self, path=None):
        path = Path(path or Path(self.run_dir) / 'workflow_state.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.touch()
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    @classmethod
    def from_dict(cls, payload):
        return cls(
            workflow_version=payload['workflow_version'],
            goal=payload['goal'],
            profile=payload['profile'],
            run_dir=payload['run_dir'],
            created_at=payload['created_at'],
            updated_at=payload['updated_at'],
            current_stage=payload['current_stage'],
            stage_attempt=payload.get('stage_attempt', 0),
            completed_stages=list(payload.get('completed_stages', [])),
            failed_stages=list(payload.get('failed_stages', [])),
            stage_status_path=payload.get('stage_status_path', ''),
            selected_protocol_entry=payload.get('selected_protocol_entry', ''),
            selected_structure_entry=payload.get('selected_structure_entry', ''),
            retained_protocol_entries=list(payload.get('retained_protocol_entries', [])),
            retained_structure_entries=list(payload.get('retained_structure_entries', [])),
            temporarily_disabled_protocol_entries=list(payload.get('temporarily_disabled_protocol_entries', [])),
            temporarily_disabled_structure_entries=list(payload.get('temporarily_disabled_structure_entries', [])),
            permanently_eliminated_protocol_entries=list(payload.get('permanently_eliminated_protocol_entries', [])),
            permanently_eliminated_structure_entries=list(payload.get('permanently_eliminated_structure_entries', [])),
            eliminated_protocol_entries=list(payload.get('eliminated_protocol_entries', [])),
            eliminated_structure_entries=list(payload.get('eliminated_structure_entries', [])),
            focused_search_space=payload.get('focused_search_space', ''),
            best_config_path=payload.get('best_config_path', ''),
            best_trial_summary=payload.get('best_trial_summary', ''),
            search_round_index=payload.get('search_round_index', 0),
            space_widen_count=payload.get('space_widen_count', 0),
            history_paths=dict(payload.get('history_paths', {})),
            study_db_paths=dict(payload.get('study_db_paths', {})),
            checkpoint_dirs=dict(payload.get('checkpoint_dirs', {})),
            blocked_reason=payload.get('blocked_reason', ''),
            last_error=payload.get('last_error', ''),
        )


def load_workflow_state(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return WorkflowState.from_dict(payload)

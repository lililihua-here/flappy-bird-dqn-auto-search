"""Minimal V0.1a auto-workflow entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from experiment_matrix import (
    PROTOCOL_ABLATION,
    STRUCTURE_ABLATION,
    final_confirm,
    run_matrix,
    summarize_matrix_results,
)
from history_reporting import (
    HistoryManager,
    aggregate_trials_for_workflow,
    export_best_config,
    load_raw_trial_rows,
    recheck_top_k,
)
from search_driver import BASELINE_CONFIG, SearchDriver, get_mode_presets
from train_eval import compute_objective, run_trial
from workflow_space import widen_search_space_after_stall
from workflow_state import WorkflowState, load_workflow_state


PROFILE_PRESETS = {
    'cpu_quick': {
        'baseline_mode': 'debug',
        'matrix_mode': 'debug',
        'matrix_budget': 'debug_matrix',
        'focused_search_mode': 'debug',
        'focused_search_max_trials': 8,
        'warmstart_search_max_trials': 8,
        'population_size': 2,
        'population_total_frame_budget': 120000,
        'recheck_topk': 0,
        'allow_final_confirm': False,
        'max_stage_wall_time_hours': 1,
        'max_total_wall_time_hours': 2,
    },
    'cpu_normal': {
        'baseline_mode': 'normal',
        'matrix_mode': 'normal',
        'matrix_budget': 'normal_matrix',
        'focused_search_mode': 'normal',
        'focused_search_max_trials': 24,
        'warmstart_search_max_trials': 24,
        'population_size': 2,
        'population_total_frame_budget': 600000,
        'recheck_topk': 3,
        'allow_final_confirm': True,
        'max_stage_wall_time_hours': 4,
        'max_total_wall_time_hours': 12,
    },
    'cpu_deep': {
        'baseline_mode': 'normal',
        'matrix_mode': 'normal',
        'matrix_budget': 'normal_matrix',
        'focused_search_mode': 'deep',
        'focused_search_max_trials': 48,
        'warmstart_search_max_trials': 48,
        'population_size': 3,
        'population_total_frame_budget': 2000000,
        'recheck_topk': 5,
        'allow_final_confirm': True,
        'max_stage_wall_time_hours': 12,
        'max_total_wall_time_hours': 48,
    },
}

BLOCKED_REASON_ENUMS = {
    'search_space_exhausted',
    'resource_limit_reached',
    'repeated_runtime_failure',
    'no_viable_candidate',
    'manual_intervention_required',
    'interface_contract_missing',
    'artifact_inconsistent',
}


def _norm_path(path):
    return str(Path(path)).replace('\\', '/')


def _now_iso():
    return datetime.now().isoformat(timespec='seconds')


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding='utf-8'))


def make_parser():
    parser = argparse.ArgumentParser(description='Flappy Bird DQN auto workflow')
    parser.add_argument('--goal', default='stable_1000')
    parser.add_argument(
        '--profile',
        choices=sorted(PROFILE_PRESETS.keys()),
        default='cpu_normal',
    )
    parser.add_argument('--run-dir', default='runs/auto_run1')
    return parser


def _json_hash(payload):
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def _load_stage_status(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _save_stage_status(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _stage_output_ready(record):
    output_files = record.get('output_files', [])
    if not output_files:
        return False
    return all(Path(path).exists() for path in output_files)


def _json_file_parseable(path: str):
    if not path:
        return False
    target = Path(path)
    if not target.exists():
        return False
    try:
        json.loads(target.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _history_file_readable(path: str):
    if not path:
        return True
    target = Path(path)
    if not target.exists():
        return False
    try:
        HistoryManager(target).load()
    except OSError:
        return False
    return True


def _stage_can_skip(record, input_hash):
    if not record:
        return False
    if record.get('status') != 'completed':
        return False
    if record.get('input_hash') != input_hash:
        return False
    if not _stage_output_ready(record):
        return False
    history_path = record.get('history_path')
    summary_path = record.get('summary_path')
    if not _history_file_readable(history_path):
        return False
    if not _json_file_parseable(summary_path):
        return False
    return True


def _append_once(values, item):
    if item not in values:
        values.append(item)


def _entry_name_from_row(row):
    name = row.get('matrix_entry_name')
    if name:
        return name
    source = row.get('source', '')
    if source.startswith('matrix_'):
        return source[len('matrix_'):]
    return source


def _rank_entry_summaries(entry_summaries):
    return sorted(
        entry_summaries,
        key=lambda item: (
            -item['metrics'].get('stable_success_count', 0),
            -(item['metrics'].get('best_final_success_rate_1000') or 0.0),
            -(item['metrics'].get('best_final_median_score') or 0.0),
            -(item['metrics'].get('best_eval_peak_score') or 0.0),
            item['metrics'].get('best_objective') if item['metrics'].get('best_objective') is not None else float('inf'),
            item['name'],
        ),
    )


def _ordered_unique(values):
    seen = set()
    ordered = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(value)
    return ordered


def _load_search_round_summaries(state: WorkflowState):
    ordered_entries = []
    for key in sorted(state.history_paths):
        if key.startswith('focused_search_round_'):
            ordered_entries.append((key, state.history_paths[key]))
    for key in ('warmstart_round', 'population_round'):
        if state.history_paths.get(key):
            ordered_entries.append((key, state.history_paths[key]))

    summaries = []
    for stage_name, history_path in ordered_entries:
        if not history_path or not Path(history_path).exists():
            continue
        metrics = aggregate_trials_for_workflow(load_raw_trial_rows(history_path))
        summaries.append({
            'stage_name': stage_name,
            'history_path': history_path,
            'metrics': metrics,
        })
    return summaries


def _build_recent_improvement_report(state: WorkflowState):
    summaries = _load_search_round_summaries(state)
    if not summaries:
        return {'rounds': [], 'delta': None}
    recent = summaries[-2:]
    delta = None
    if len(recent) == 2:
        previous_metrics = recent[0]['metrics']
        current_metrics = recent[1]['metrics']
        delta = {
            'from_stage': recent[0]['stage_name'],
            'to_stage': recent[1]['stage_name'],
            'best_final_median_score_delta': (
                (current_metrics.get('best_final_median_score') or 0.0)
                - (previous_metrics.get('best_final_median_score') or 0.0)
            ),
            'best_final_success_rate_1000_delta': (
                (current_metrics.get('best_final_success_rate_1000') or 0.0)
                - (previous_metrics.get('best_final_success_rate_1000') or 0.0)
            ),
            'best_eval_peak_score_delta': (
                (current_metrics.get('best_eval_peak_score') or 0.0)
                - (previous_metrics.get('best_eval_peak_score') or 0.0)
            ),
        }
    return {
        'rounds': recent,
        'delta': delta,
    }


def _update_best_config_meta(state: WorkflowState, **updates):
    meta_path = Path(state.run_dir) / 'best_config_meta.json'
    meta = _read_json(meta_path, default={})
    meta.update({
        'updated_at': _now_iso(),
        **updates,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta


def _best_config_payload(state: WorkflowState):
    best_config_path = Path(state.best_config_path)
    if best_config_path.exists():
        return _read_json(best_config_path, default={})
    return {}


def _success_next_stage(profile_settings):
    if profile_settings['recheck_topk'] > 0:
        return 'recheck_topk'
    if profile_settings['allow_final_confirm']:
        return 'final_confirm'
    return 'report'


def _write_blocked_report(state: WorkflowState, blocked_reason: str, latest_history_path: str):
    if blocked_reason not in BLOCKED_REASON_ENUMS:
        blocked_reason = 'manual_intervention_required'
    latest_rows = load_raw_trial_rows(latest_history_path) if latest_history_path and Path(latest_history_path).exists() else []
    latest_summary = aggregate_trials_for_workflow(latest_rows) if latest_rows else {}
    recent_improvement = _build_recent_improvement_report(state)
    report = {
        'current_stage': state.current_stage,
        'blocked_reason': blocked_reason,
        'current_best_config_path': state.best_config_path,
        'current_best_config': _best_config_payload(state),
        'retained_protocol_entries': state.retained_protocol_entries,
        'retained_structure_entries': state.retained_structure_entries,
        'temporarily_disabled_protocol_entries': state.temporarily_disabled_protocol_entries,
        'temporarily_disabled_structure_entries': state.temporarily_disabled_structure_entries,
        'permanently_eliminated_protocol_entries': state.permanently_eliminated_protocol_entries,
        'permanently_eliminated_structure_entries': state.permanently_eliminated_structure_entries,
        'recent_round_improvements': recent_improvement,
        'failure_reason_counter': latest_summary.get('failure_reason_counter', {}),
        'latest_history_path': latest_history_path,
        'completed_artifacts': sorted(
            str(path).replace('\\', '/')
            for path in Path(state.run_dir).rglob('*')
            if path.is_file()
        ),
        'missing_expected_artifacts': [
            name for name, path in {
                'workflow_state.json': Path(state.run_dir) / 'workflow_state.json',
                'stage_status.json': Path(state.run_dir) / 'stage_status.json',
                'best_config.json': Path(state.run_dir) / 'best_config.json',
                'best_config_meta.json': Path(state.run_dir) / 'best_config_meta.json',
            }.items()
            if not path.exists()
        ],
        'recommended_next_actions': [
            'expand reward parameter ranges',
            're-open temporarily disabled entries',
            'try population_async if not already run',
        ],
    }
    blocked_path = Path(state.run_dir) / 'blocked_report.json'
    blocked_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    state.blocked_reason = blocked_reason
    state.best_trial_summary = _norm_path(blocked_path)
    state.current_stage = 'blocked'
    state.save()
    return state


def _latest_search_history_path(state: WorkflowState):
    for key in (
        'population_round',
        'warmstart_round',
        f'focused_search_round_{state.search_round_index}',
        'focused_search_round_1',
    ):
        path = state.history_paths.get(key)
        if path:
            return path
    return ''


def _mark_stage_running(state: WorkflowState, stage_name: str, input_payload, history_path='', summary_path=''):
    stage_status_path = Path(state.stage_status_path or Path(state.run_dir) / 'stage_status.json')
    stage_status = _load_stage_status(stage_status_path)
    existing = stage_status.get(stage_name, {})
    attempt = int(existing.get('attempt', 0)) + 1
    stage_status[stage_name] = {
        'stage': stage_name,
        'status': 'running',
        'attempt': attempt,
        'started_at': _now_iso(),
        'finished_at': '',
        'input_hash': _json_hash(input_payload),
        'output_files': [],
        'history_path': history_path,
        'summary_path': summary_path,
        'error': '',
    }
    _save_stage_status(stage_status_path, stage_status)
    return stage_status, attempt


def _mark_stage_completed(state: WorkflowState, stage_name: str, input_payload, output_files, history_path='', summary_path=''):
    stage_status_path = Path(state.stage_status_path or Path(state.run_dir) / 'stage_status.json')
    stage_status = _load_stage_status(stage_status_path)
    existing = stage_status.get(stage_name, {})
    stage_status[stage_name] = {
        'stage': stage_name,
        'status': 'completed',
        'attempt': int(existing.get('attempt', 1)),
        'started_at': existing.get('started_at', _now_iso()),
        'finished_at': _now_iso(),
        'input_hash': _json_hash(input_payload),
        'output_files': [_norm_path(path) for path in output_files],
        'history_path': history_path,
        'summary_path': summary_path,
        'error': '',
    }
    _save_stage_status(stage_status_path, stage_status)
    _append_once(state.completed_stages, stage_name)


def _mark_stage_failed(state: WorkflowState, stage_name: str, input_payload, error, history_path='', summary_path=''):
    stage_status_path = Path(state.stage_status_path or Path(state.run_dir) / 'stage_status.json')
    stage_status = _load_stage_status(stage_status_path)
    existing = stage_status.get(stage_name, {})
    stage_status[stage_name] = {
        'stage': stage_name,
        'status': 'failed',
        'attempt': int(existing.get('attempt', 0)) or 1,
        'started_at': existing.get('started_at', _now_iso()),
        'finished_at': _now_iso(),
        'input_hash': existing.get('input_hash', _json_hash(input_payload)),
        'output_files': list(existing.get('output_files', [])),
        'history_path': history_path or existing.get('history_path', ''),
        'summary_path': summary_path or existing.get('summary_path', ''),
        'error': str(error),
    }
    _save_stage_status(stage_status_path, stage_status)
    _append_once(state.failed_stages, stage_name)
    state.last_error = str(error)
    state.save()


def _baseline_stage_input(state: WorkflowState):
    profile_settings = PROFILE_PRESETS[state.profile]
    mode = profile_settings['baseline_mode']
    return {
        'goal': state.goal,
        'profile': state.profile,
        'mode': mode,
        'config': BASELINE_CONFIG,
    }


def run_baseline_stage(state: WorkflowState):
    run_dir = Path(state.run_dir)
    stage_status_path = Path(state.stage_status_path)
    stage_status = _load_stage_status(stage_status_path)
    stage_name = 'baseline_check'
    input_hash = _json_hash(_baseline_stage_input(state))
    existing = stage_status.get(stage_name)

    if _stage_can_skip(existing, input_hash):
        _append_once(state.completed_stages, stage_name)
        state.current_stage = 'protocol_matrix'
        state.history_paths['baseline'] = existing.get('history_path', '')
        state.last_error = ''
        state.save()
        return state

    baseline_dir = run_dir / 'baseline'
    history_path = baseline_dir / 'history.jsonl'
    summary_path = baseline_dir / 'baseline_summary.json'
    checkpoint_dir = baseline_dir / 'checkpoints'
    attempt = int((existing or {}).get('attempt', 0)) + 1

    stage_status[stage_name] = {
        'stage': stage_name,
        'status': 'running',
        'attempt': attempt,
        'started_at': state.updated_at,
        'finished_at': '',
        'input_hash': input_hash,
        'output_files': [],
        'history_path': _norm_path(history_path),
        'summary_path': _norm_path(summary_path),
        'error': '',
    }
    _save_stage_status(stage_status_path, stage_status)

    profile_settings = PROFILE_PRESETS[state.profile]
    presets = get_mode_presets(profile_settings['baseline_mode'])
    result = run_trial(
        config=dict(BASELINE_CONFIG),
        trial_id=-1,
        seed=11,
        source='baseline',
        max_trial_frames=presets['max_trial_frames'],
        eval_interval_frames=presets['eval_interval_frames'],
        eval_episodes=presets['eval_episodes'],
        checkpoint_dir=str(checkpoint_dir),
    )
    result['objective'] = compute_objective(
        success=(result['status'] == 'success'),
        train_raw_env_frames=result['train_raw_env_frames'],
        max_trial_frames=presets['max_trial_frames'],
        best_eval_score=result['best_eval_score'],
    )
    result.setdefault('config', dict(BASELINE_CONFIG))

    baseline_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryManager(history_path)
    history.append(result)

    summary_payload = {
        'stage': stage_name,
        'status': result['status'],
        'failure_reason': result.get('failure_reason'),
        'objective': result['objective'],
        'history_path': _norm_path(history_path),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    stage_status[stage_name] = {
        'stage': stage_name,
        'status': 'completed',
        'attempt': attempt,
        'started_at': stage_status[stage_name]['started_at'],
        'finished_at': state.updated_at,
        'input_hash': input_hash,
        'output_files': [_norm_path(history_path), _norm_path(summary_path)],
        'history_path': _norm_path(history_path),
        'summary_path': _norm_path(summary_path),
        'error': '',
    }
    _save_stage_status(stage_status_path, stage_status)

    _append_once(state.completed_stages, stage_name)
    state.history_paths['baseline'] = _norm_path(history_path)
    state.current_stage = 'protocol_matrix'
    state.last_error = ''
    state.save()
    return state


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def run_matrix_stage(state: WorkflowState, stage_name, matrix_name, matrix, next_stage):
    run_dir = Path(state.run_dir)
    stage_status_path = Path(state.stage_status_path)
    stage_status = _load_stage_status(stage_status_path)
    profile_settings = PROFILE_PRESETS[state.profile]
    input_hash = _json_hash({
        'goal': state.goal,
        'profile': state.profile,
        'matrix_name': matrix_name,
        'mode': profile_settings['matrix_mode'],
        'budget': profile_settings['matrix_budget'],
    })
    existing = stage_status.get(stage_name)
    stage_dir = run_dir / stage_name
    history_path = stage_dir / 'history.jsonl'
    summary_path = stage_dir / 'summary.json'

    if _stage_can_skip(existing, input_hash):
        _append_once(state.completed_stages, stage_name)
        state.current_stage = next_stage
        state.history_paths[stage_name] = existing.get('history_path', '')
        state.last_error = ''
        state.save()
        return state

    attempt = int((existing or {}).get('attempt', 0)) + 1
    stage_status[stage_name] = {
        'stage': stage_name,
        'status': 'running',
        'attempt': attempt,
        'started_at': state.updated_at,
        'finished_at': '',
        'input_hash': input_hash,
        'output_files': [],
        'history_path': _norm_path(history_path),
        'summary_path': _norm_path(summary_path),
        'error': '',
    }
    _save_stage_status(stage_status_path, stage_status)

    results = run_matrix(
        matrix,
        mode=profile_settings['matrix_mode'],
        budget=profile_settings['matrix_budget'],
    )
    summary = summarize_matrix_results(matrix_name, results)

    _write_jsonl(history_path, results)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    stage_status[stage_name] = {
        'stage': stage_name,
        'status': 'completed',
        'attempt': attempt,
        'started_at': stage_status[stage_name]['started_at'],
        'finished_at': state.updated_at,
        'input_hash': input_hash,
        'output_files': [_norm_path(history_path), _norm_path(summary_path)],
        'history_path': _norm_path(history_path),
        'summary_path': _norm_path(summary_path),
        'error': '',
    }
    _save_stage_status(stage_status_path, stage_status)

    _append_once(state.completed_stages, stage_name)
    state.history_paths[stage_name] = _norm_path(history_path)
    state.current_stage = next_stage
    state.last_error = ''
    state.save()
    return state


def derive_candidates_and_search_space(state: WorkflowState):
    stage_name = 'derive_candidates_and_search_space'
    input_payload = {
        'stage': stage_name,
        'protocol_history_path': state.history_paths['protocol_matrix'],
        'structure_history_path': state.history_paths['structure_matrix'],
    }
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        selection = _read_json(Path(state.run_dir) / 'selection_decision.json', default={})
        state.selected_protocol_entry = selection.get('selected_protocol_entry', state.selected_protocol_entry)
        state.selected_structure_entry = selection.get('selected_structure_entry', state.selected_structure_entry)
        state.retained_protocol_entries = list(selection.get('retained_protocol_entries', state.retained_protocol_entries))
        state.retained_structure_entries = list(selection.get('retained_structure_entries', state.retained_structure_entries))
        state.temporarily_disabled_protocol_entries = list(selection.get('temporarily_disabled_protocol_entries', state.temporarily_disabled_protocol_entries))
        state.temporarily_disabled_structure_entries = list(selection.get('temporarily_disabled_structure_entries', state.temporarily_disabled_structure_entries))
        state.permanently_eliminated_protocol_entries = list(selection.get('permanently_eliminated_protocol_entries', state.permanently_eliminated_protocol_entries))
        state.permanently_eliminated_structure_entries = list(selection.get('permanently_eliminated_structure_entries', state.permanently_eliminated_structure_entries))
        state.eliminated_protocol_entries = list(selection.get('eliminated_protocol_entries', state.eliminated_protocol_entries))
        state.eliminated_structure_entries = list(selection.get('eliminated_structure_entries', state.eliminated_structure_entries))
        state.focused_search_space = _norm_path(Path(state.run_dir) / 'focused_search_space.json')
        _append_once(state.completed_stages, stage_name)
        state.current_stage = 'select_base_config'
        state.save()
        return state
    _mark_stage_running(state, stage_name, input_payload)

    protocol_rows = load_raw_trial_rows(state.history_paths['protocol_matrix'])
    structure_rows = load_raw_trial_rows(state.history_paths['structure_matrix'])

    protocol_groups = {}
    for row in protocol_rows:
        protocol_groups.setdefault(_entry_name_from_row(row), []).append(row)
    structure_groups = {}
    for row in structure_rows:
        structure_groups.setdefault(_entry_name_from_row(row), []).append(row)

    protocol_ranked = _rank_entry_summaries([
        {
            'name': name,
            'rows': rows,
            'metrics': aggregate_trials_for_workflow(rows),
            'best_trial': min(
                rows,
                key=lambda row: row.get('objective') if row.get('objective') is not None else float('inf'),
            ),
        }
        for name, rows in protocol_groups.items()
    ])

    def _split_disabled_vs_eliminated(ranked_entries):
        temporarily_disabled = []
        permanently_eliminated = []
        for item in ranked_entries[1:]:
            if item['metrics'].get('trial_count', 0) < 3:
                temporarily_disabled.append(item['name'])
            elif item['metrics'].get('no_learning_count', 0) >= item['metrics'].get('trial_count', 0):
                permanently_eliminated.append(item['name'])
            else:
                temporarily_disabled.append(item['name'])
        return temporarily_disabled, permanently_eliminated
    structure_ranked = _rank_entry_summaries([
        {
            'name': name,
            'rows': rows,
            'metrics': aggregate_trials_for_workflow(rows),
            'best_trial': min(
                rows,
                key=lambda row: row.get('objective') if row.get('objective') is not None else float('inf'),
            ),
        }
        for name, rows in structure_groups.items()
    ])

    retained_protocol = protocol_ranked[:2]
    retained_structure = structure_ranked[:2]
    selected_protocol = retained_protocol[0]
    selected_structure = retained_structure[0]
    temp_disabled_protocol, perm_eliminated_protocol = _split_disabled_vs_eliminated(protocol_ranked)
    temp_disabled_structure, perm_eliminated_structure = _split_disabled_vs_eliminated(structure_ranked)

    selected_protocol_config = dict(selected_protocol['best_trial'].get('config', {}))
    selected_structure_config = dict(selected_structure['best_trial'].get('config', {}))
    selected_config = dict(BASELINE_CONFIG)
    selected_config.update(selected_protocol_config)
    selected_config.update(selected_structure_config)

    focused_search_space = {
        'categorical': {
            'hidden_key': ['medium'],
            'replay_start_size': [selected_config.get('replay_start_size', BASELINE_CONFIG['replay_start_size'])],
            'train_freq': [selected_config.get('train_freq', BASELINE_CONFIG['train_freq'])],
            'reward_scheme_version': _ordered_unique([
                item['best_trial'].get('config', {}).get('reward_scheme_version')
                for item in retained_protocol
            ]),
            'state_representation_version': _ordered_unique([
                item['best_trial'].get('config', {}).get('state_representation_version')
                for item in retained_protocol
            ]),
            'network_backbone': _ordered_unique([
                item['best_trial'].get('config', {}).get('network_backbone')
                for item in retained_structure
            ]),
            'exploration_head': _ordered_unique([
                item['best_trial'].get('config', {}).get('exploration_head')
                for item in retained_structure
            ]),
            'n_step': sorted({
                item['best_trial'].get('config', {}).get('n_step', BASELINE_CONFIG['n_step'])
                for item in retained_protocol + retained_structure
            }),
            'priority': sorted({
                bool(item['best_trial'].get('config', {}).get('priority', BASELINE_CONFIG['priority']))
                for item in retained_protocol + retained_structure
            }),
            'reward_scale': sorted({
                item['best_trial'].get('config', {}).get('reward_scale', BASELINE_CONFIG['reward_scale'])
                for item in retained_protocol
            }),
            'reward_clip': sorted(
                {
                    item['best_trial'].get('config', {}).get('reward_clip', BASELINE_CONFIG['reward_clip'])
                    for item in retained_protocol
                },
                key=lambda value: (value is None, value),
            ),
            'gap_shaping_coef': sorted({
                item['best_trial'].get('config', {}).get('gap_shaping_coef', BASELINE_CONFIG['gap_shaping_coef'])
                for item in retained_protocol
            }),
        },
        'continuous': {
            'lr': {
                'low': selected_config['lr'] * 0.5,
                'high': selected_config['lr'] * 2.0,
                'log': True,
            },
            'gamma': {
                'low': max(0.90, selected_config['gamma'] - 0.02),
                'high': min(0.999, selected_config['gamma'] + 0.005),
            },
        },
        'derived_from': {
            'protocol_history_path': state.history_paths['protocol_matrix'],
            'structure_history_path': state.history_paths['structure_matrix'],
            'top_k': 2,
        },
    }

    selection_reason = {
        **selected_structure['metrics'],
        'selected_protocol_metrics': selected_protocol['metrics'],
        'selected_structure_metrics': selected_structure['metrics'],
    }
    selection_decision = {
        'selected_protocol_entry': selected_protocol['name'],
        'selected_structure_entry': selected_structure['name'],
        'retained_protocol_entries': [item['name'] for item in retained_protocol],
        'retained_structure_entries': [item['name'] for item in retained_structure],
        'temporarily_disabled_protocol_entries': temp_disabled_protocol,
        'temporarily_disabled_structure_entries': temp_disabled_structure,
        'permanently_eliminated_protocol_entries': perm_eliminated_protocol,
        'permanently_eliminated_structure_entries': perm_eliminated_structure,
        'eliminated_protocol_entries': perm_eliminated_protocol,
        'eliminated_structure_entries': perm_eliminated_structure,
        'selected_protocol_trial': selected_protocol['best_trial'],
        'selected_structure_trial': selected_structure['best_trial'],
        'selection_reason': selection_reason,
    }

    focused_path = Path(state.run_dir) / 'focused_search_space.json'
    focused_path.write_text(json.dumps(focused_search_space, ensure_ascii=False, indent=2), encoding='utf-8')
    selection_path = Path(state.run_dir) / 'selection_decision.json'
    selection_path.write_text(json.dumps(selection_decision, ensure_ascii=False, indent=2), encoding='utf-8')
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [focused_path, selection_path],
        summary_path=_norm_path(selection_path),
    )

    state.selected_protocol_entry = selected_protocol['name']
    state.selected_structure_entry = selected_structure['name']
    state.retained_protocol_entries = [item['name'] for item in retained_protocol]
    state.retained_structure_entries = [item['name'] for item in retained_structure]
    state.temporarily_disabled_protocol_entries = temp_disabled_protocol
    state.temporarily_disabled_structure_entries = temp_disabled_structure
    state.permanently_eliminated_protocol_entries = perm_eliminated_protocol
    state.permanently_eliminated_structure_entries = perm_eliminated_structure
    state.eliminated_protocol_entries = perm_eliminated_protocol
    state.eliminated_structure_entries = perm_eliminated_structure
    state.focused_search_space = _norm_path(focused_path)
    state.current_stage = 'select_base_config'
    state.save()
    return state


def select_base_config(state: WorkflowState):
    stage_name = 'select_base_config'
    input_payload = {
        'stage': stage_name,
        'selection_decision_path': _norm_path(Path(state.run_dir) / 'selection_decision.json'),
        'selected_protocol_entry': state.selected_protocol_entry,
        'selected_structure_entry': state.selected_structure_entry,
    }
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        state.best_config_path = _norm_path(Path(state.run_dir) / 'best_config.json')
        state.best_trial_summary = _norm_path(Path(state.run_dir) / 'best_trial_summary.json')
        _append_once(state.completed_stages, stage_name)
        state.current_stage = 'focused_search_round'
        state.save()
        return state
    _mark_stage_running(state, stage_name, input_payload)

    selection_path = Path(state.run_dir) / 'selection_decision.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    protocol_config = dict(selection['selected_protocol_trial'].get('config', {}))
    structure_config = dict(selection['selected_structure_trial'].get('config', {}))
    best_config = dict(BASELINE_CONFIG)
    best_config.update(protocol_config)
    best_config.update(structure_config)

    best_config_path = Path(state.best_config_path or Path(state.run_dir) / 'best_config.json')
    best_config_path.write_text(json.dumps(best_config, ensure_ascii=False, indent=2), encoding='utf-8')

    best_config_meta = {
        'source_stage': 'select_base_config',
        'source_entry': f"{selection['selected_protocol_entry']} + {selection['selected_structure_entry']}",
        'source_trial_id': selection['selected_protocol_trial'].get('trial_id'),
        'source_history_path': state.history_paths.get('protocol_matrix', ''),
        'updated_at': _now_iso(),
        'confidence_level': 'near_miss_high',
        'is_confirmed': False,
        'requires_recheck': True,
        'selection_reason': selection.get('selection_reason', {}),
    }
    meta_path = Path(state.run_dir) / 'best_config_meta.json'
    meta_path.write_text(json.dumps(best_config_meta, ensure_ascii=False, indent=2), encoding='utf-8')
    best_trial_summary = {
        'source_stage': 'select_base_config',
        'selected_protocol_trial': selection['selected_protocol_trial'],
        'selected_structure_trial': selection['selected_structure_trial'],
        'merged_best_config': best_config,
    }
    best_trial_summary_path = Path(state.run_dir) / 'best_trial_summary.json'
    best_trial_summary_path.write_text(
        json.dumps(best_trial_summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [best_config_path, meta_path, best_trial_summary_path],
        summary_path=_norm_path(meta_path),
    )

    state.best_config_path = _norm_path(best_config_path)
    state.best_trial_summary = _norm_path(best_trial_summary_path)
    state.current_stage = 'focused_search_round'
    state.save()
    return state


def run_focused_search_round(state: WorkflowState):
    stage_name = 'focused_search_round'
    profile_settings = PROFILE_PRESETS[state.profile]
    mode = profile_settings['focused_search_mode']
    presets = get_mode_presets(mode)
    round_index = max(1, state.search_round_index or 1)
    round_dir = Path(state.run_dir) / f'focused_search_round_{round_index}'
    round_dir.mkdir(parents=True, exist_ok=True)
    history_path = round_dir / 'history.jsonl'
    study_db_path = round_dir / 'study.db'
    checkpoint_dir = round_dir / 'checkpoints'
    focused_search_space = json.loads(Path(state.focused_search_space).read_text(encoding='utf-8'))
    input_payload = {
        'stage': stage_name,
        'focused_search_space_path': state.focused_search_space,
        'best_config_path': state.best_config_path,
        'round_index': round_index,
        'profile': state.profile,
    }
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        summary = _read_json(Path(state.run_dir) / 'workflow_summary.json', default={})
        state.search_round_index = round_index
        state.history_paths[f'focused_search_round_{round_index}'] = _norm_path(history_path)
        state.study_db_paths[f'focused_search_round_{round_index}'] = _norm_path(study_db_path)
        state.checkpoint_dirs[f'focused_search_round_{round_index}'] = _norm_path(checkpoint_dir)
        _append_once(state.completed_stages, stage_name)
        state.current_stage = summary.get('next_stage') or (
            _success_next_stage(profile_settings)
            if (summary.get('summary', {}).get('stable_success_count') or 0) > 0
            else 'warmstart_search_round'
        )
        state.save()
        return state
    _mark_stage_running(
        state,
        stage_name,
        input_payload,
        history_path=_norm_path(history_path),
        summary_path=_norm_path(Path(state.run_dir) / 'workflow_summary.json'),
    )

    driver = SearchDriver(
        history_path=str(history_path),
        study_db=str(study_db_path),
        max_trials=profile_settings['focused_search_max_trials'],
        max_trial_frames=presets['max_trial_frames'],
        eval_interval_frames=presets['eval_interval_frames'],
        eval_episodes=presets['eval_episodes'],
        checkpoint_dir=str(checkpoint_dir),
        focused_search_space=focused_search_space,
        run_baseline_first=False,
    )
    driver.run()

    export_best_config(driver.history, state.best_config_path or Path(state.run_dir) / 'best_config.json')
    best_trial = driver.history.best_trial()
    if best_trial:
        meta_path = Path(state.run_dir) / 'best_config_meta.json'
        meta_payload = {
            'source_stage': 'focused_search_round',
            'source_entry': f"{state.selected_protocol_entry} + {state.selected_structure_entry}",
            'source_trial_id': best_trial.get('trial_id'),
            'source_history_path': _norm_path(history_path),
            'updated_at': _now_iso(),
            'confidence_level': 'search_success' if best_trial.get('status') == 'success' else 'near_miss_high',
            'is_confirmed': False,
            'requires_recheck': best_trial.get('status') == 'success',
            'selection_reason': aggregate_trials_for_workflow(load_raw_trial_rows(history_path)),
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        best_trial_summary_path = Path(state.run_dir) / 'best_trial_summary.json'
        best_trial_summary_path.write_text(
            json.dumps(best_trial, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        state.best_trial_summary = _norm_path(best_trial_summary_path)

    next_stage = _success_next_stage(profile_settings) if driver.history.success_count() > 0 else 'warmstart_search_round'
    workflow_summary = {
        'current_stage': stage_name,
        'next_stage': next_stage,
        'selected_protocol_entry': state.selected_protocol_entry,
        'selected_structure_entry': state.selected_structure_entry,
        'focused_history_path': _norm_path(history_path),
        'summary': aggregate_trials_for_workflow(load_raw_trial_rows(history_path)),
    }
    summary_path = Path(state.run_dir) / 'workflow_summary.json'
    summary_path.write_text(json.dumps(workflow_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [history_path, summary_path],
        history_path=_norm_path(history_path),
        summary_path=_norm_path(summary_path),
    )

    state.search_round_index = round_index
    state.history_paths[f'focused_search_round_{round_index}'] = _norm_path(history_path)
    state.study_db_paths[f'focused_search_round_{round_index}'] = _norm_path(study_db_path)
    state.checkpoint_dirs[f'focused_search_round_{round_index}'] = _norm_path(checkpoint_dir)
    if driver.history.success_count() > 0:
        state.current_stage = _success_next_stage(profile_settings)
    else:
        widened = widen_search_space_after_stall(focused_search_space)
        Path(state.focused_search_space).write_text(
            json.dumps(widened, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        state.space_widen_count += 1
        state.current_stage = 'warmstart_search_round'
    state.save()
    return state


def run_warmstart_search_round(state: WorkflowState):
    stage_name = 'warmstart_search_round'
    profile_settings = PROFILE_PRESETS[state.profile]
    mode = profile_settings['focused_search_mode']
    presets = get_mode_presets(mode)
    round_dir = Path(state.run_dir) / 'warmstart_round'
    round_dir.mkdir(parents=True, exist_ok=True)
    history_path = round_dir / 'history.jsonl'
    study_db_path = round_dir / 'study.db'
    checkpoint_dir = round_dir / 'checkpoints'
    focused_search_space = json.loads(Path(state.focused_search_space).read_text(encoding='utf-8'))
    input_payload = {
        'stage': stage_name,
        'focused_search_space_path': state.focused_search_space,
        'best_config_path': state.best_config_path,
        'profile': state.profile,
    }
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        summary = _read_json(Path(state.run_dir) / 'workflow_summary.json', default={})
        state.history_paths['warmstart_round'] = _norm_path(history_path)
        state.study_db_paths['warmstart_round'] = _norm_path(study_db_path)
        state.checkpoint_dirs['warmstart_round'] = _norm_path(checkpoint_dir)
        _append_once(state.completed_stages, stage_name)
        state.current_stage = summary.get('next_stage') or (
            _success_next_stage(profile_settings)
            if (summary.get('summary', {}).get('stable_success_count') or 0) > 0
            else 'population_search_round'
        )
        state.save()
        return state
    _mark_stage_running(
        state,
        stage_name,
        input_payload,
        history_path=_norm_path(history_path),
        summary_path=_norm_path(Path(state.run_dir) / 'workflow_summary.json'),
    )

    driver = SearchDriver(
        history_path=str(history_path),
        study_db=str(study_db_path),
        max_trials=profile_settings['warmstart_search_max_trials'],
        max_trial_frames=presets['max_trial_frames'],
        eval_interval_frames=presets['eval_interval_frames'],
        eval_episodes=presets['eval_episodes'],
        checkpoint_dir=str(checkpoint_dir),
        focused_search_space=focused_search_space,
        run_baseline_first=False,
    )
    driver.run_warmstart_tpe()

    summary = aggregate_trials_for_workflow(load_raw_trial_rows(history_path))
    next_stage = _success_next_stage(profile_settings) if driver.history.success_count() > 0 else 'population_search_round'
    summary_path = Path(state.run_dir) / 'workflow_summary.json'
    summary_path.write_text(json.dumps({
        'current_stage': stage_name,
        'next_stage': next_stage,
        'summary': summary,
        'history_path': _norm_path(history_path),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [history_path, summary_path],
        history_path=_norm_path(history_path),
        summary_path=_norm_path(summary_path),
    )

    state.history_paths['warmstart_round'] = _norm_path(history_path)
    state.study_db_paths['warmstart_round'] = _norm_path(study_db_path)
    state.checkpoint_dirs['warmstart_round'] = _norm_path(checkpoint_dir)
    if driver.history.success_count() > 0:
        state.current_stage = _success_next_stage(profile_settings)
    else:
        widened = widen_search_space_after_stall(focused_search_space)
        Path(state.focused_search_space).write_text(
            json.dumps(widened, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        state.space_widen_count += 1
        state.current_stage = 'population_search_round'
    state.save()
    return state


def run_population_search_round(state: WorkflowState):
    stage_name = 'population_search_round'
    profile_settings = PROFILE_PRESETS[state.profile]
    mode = profile_settings['focused_search_mode']
    presets = get_mode_presets(mode)
    round_dir = Path(state.run_dir) / 'population_round'
    round_dir.mkdir(parents=True, exist_ok=True)
    history_path = round_dir / 'history.jsonl'
    study_db_path = round_dir / 'study.db'
    checkpoint_dir = round_dir / 'checkpoints'
    focused_search_space = json.loads(Path(state.focused_search_space).read_text(encoding='utf-8'))
    input_payload = {
        'stage': stage_name,
        'focused_search_space_path': state.focused_search_space,
        'best_config_path': state.best_config_path,
        'profile': state.profile,
    }
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        summary = _read_json(Path(state.run_dir) / 'workflow_summary.json', default={})
        state.history_paths['population_round'] = _norm_path(history_path)
        state.study_db_paths['population_round'] = _norm_path(study_db_path)
        state.checkpoint_dirs['population_round'] = _norm_path(checkpoint_dir)
        _append_once(state.completed_stages, stage_name)
        next_stage = summary.get('next_stage')
        if next_stage:
            state.current_stage = next_stage
            state.save()
            return state
        if (summary.get('summary', {}).get('stable_success_count') or 0) > 0:
            state.current_stage = _success_next_stage(profile_settings)
            state.save()
            return state
        return _write_blocked_report(
            state,
            blocked_reason='search_space_exhausted',
            latest_history_path=_norm_path(history_path),
        )
    _mark_stage_running(
        state,
        stage_name,
        input_payload,
        history_path=_norm_path(history_path),
        summary_path=_norm_path(Path(state.run_dir) / 'workflow_summary.json'),
    )

    driver = SearchDriver(
        history_path=str(history_path),
        study_db=str(study_db_path),
        max_trials=profile_settings['warmstart_search_max_trials'],
        max_trial_frames=presets['max_trial_frames'],
        eval_interval_frames=presets['eval_interval_frames'],
        eval_episodes=presets['eval_episodes'],
        checkpoint_dir=str(checkpoint_dir),
        focused_search_space=focused_search_space,
        run_baseline_first=False,
    )
    driver.run_population_async(
        total_frame_budget=profile_settings['population_total_frame_budget'],
        population_size=profile_settings['population_size'],
    )

    summary = aggregate_trials_for_workflow(load_raw_trial_rows(history_path))
    next_stage = _success_next_stage(profile_settings) if driver.history.success_count() > 0 else 'blocked'
    summary_path = Path(state.run_dir) / 'workflow_summary.json'
    summary_path.write_text(json.dumps({
        'current_stage': stage_name,
        'next_stage': next_stage,
        'summary': summary,
        'history_path': _norm_path(history_path),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [history_path, summary_path],
        history_path=_norm_path(history_path),
        summary_path=_norm_path(summary_path),
    )

    state.history_paths['population_round'] = _norm_path(history_path)
    state.study_db_paths['population_round'] = _norm_path(study_db_path)
    state.checkpoint_dirs['population_round'] = _norm_path(checkpoint_dir)
    if driver.history.success_count() > 0:
        state.current_stage = _success_next_stage(profile_settings)
        state.save()
        return state

    return _write_blocked_report(
        state,
        blocked_reason='search_space_exhausted',
        latest_history_path=_norm_path(history_path),
    )


def run_recheck_topk(state: WorkflowState):
    stage_name = 'recheck_topk'
    profile_settings = PROFILE_PRESETS[state.profile]
    history_path = _latest_search_history_path(state)
    history = HistoryManager(history_path)
    input_payload = {
        'stage': stage_name,
        'history_path': history_path,
        'best_config_path': state.best_config_path,
        'recheck_topk': profile_settings['recheck_topk'],
        'allow_final_confirm': profile_settings['allow_final_confirm'],
    }
    summary_path = Path(state.run_dir) / 'recheck' / 'recheck_summary.json'
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        summary = _read_json(summary_path, default={})
        passed = bool(summary.get('recheck_passed'))
        _append_once(state.completed_stages, stage_name)
        if not passed:
            return _write_blocked_report(state, 'no_viable_candidate', history_path)
        best_trial_id = summary.get('best_candidate_trial_id')
        if best_trial_id is not None:
            _update_best_config_meta(
                state,
                source_stage='recheck_topk',
                source_trial_id=best_trial_id,
                source_history_path=history_path,
                requires_recheck=False,
                confidence_level='recheck_passed',
                is_confirmed=False,
            )
        state.current_stage = 'final_confirm' if profile_settings['allow_final_confirm'] else 'report'
        state.save()
        return state
    _mark_stage_running(
        state,
        stage_name,
        input_payload,
        history_path=history_path,
        summary_path=_norm_path(summary_path),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results = recheck_top_k(
        history,
        k=profile_settings['recheck_topk'],
        max_trial_frames=get_mode_presets(profile_settings['focused_search_mode'])['max_trial_frames'],
    )
    passed = any(item.get('recheck_passed') for item in results)
    best_result = next((item for item in results if item.get('recheck_passed')), results[0] if results else {})
    summary_payload = {
        'stage': stage_name,
        'profile': state.profile,
        'history_path': history_path,
        'k': profile_settings['recheck_topk'],
        'allow_final_confirm': profile_settings['allow_final_confirm'],
        'recheck_passed': passed,
        'best_candidate_trial_id': best_result.get('original_trial_id'),
        'next_stage': (
            'final_confirm' if passed and profile_settings['allow_final_confirm']
            else 'report' if passed
            else 'blocked'
        ),
        'results': results,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [summary_path],
        history_path=history_path,
        summary_path=_norm_path(summary_path),
    )
    if not passed:
        state.current_stage = 'blocked'
        return _write_blocked_report(state, 'no_viable_candidate', history_path)
    _update_best_config_meta(
        state,
        source_stage='recheck_topk',
        source_trial_id=best_result.get('original_trial_id'),
        source_history_path=history_path,
        requires_recheck=False,
        confidence_level='recheck_passed',
        is_confirmed=False,
    )
    state.current_stage = 'final_confirm' if profile_settings['allow_final_confirm'] else 'report'
    state.save()
    return state


def run_final_confirm_stage(state: WorkflowState):
    stage_name = 'final_confirm'
    best_config = json.loads(Path(state.best_config_path).read_text(encoding='utf-8'))
    input_payload = {
        'stage': stage_name,
        'best_config_path': state.best_config_path,
    }
    summary_path = Path(state.run_dir) / 'final_confirm' / 'final_confirm_summary.json'
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        summary = _read_json(summary_path, default={})
        result = summary.get('result', {})
        _update_best_config_meta(
            state,
            source_stage='final_confirm',
            is_confirmed=result.get('status') == 'confirmed',
            requires_recheck=False,
            confidence_level='confirmed' if result.get('status') == 'confirmed' else 'recheck_failed',
        )
        _append_once(state.completed_stages, stage_name)
        state.current_stage = 'report'
        state.save()
        return state
    _mark_stage_running(
        state,
        stage_name,
        input_payload,
        summary_path=_norm_path(summary_path),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result = final_confirm(best_config, checkpoint_dir=str(summary_path.parent / 'checkpoints'))
    summary_payload = {
        'stage': stage_name,
        'profile': state.profile,
        'best_config_path': state.best_config_path,
        'result': result,
        'next_stage': 'report',
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [summary_path],
        summary_path=_norm_path(summary_path),
    )
    _update_best_config_meta(
        state,
        source_stage='final_confirm',
        is_confirmed=result.get('status') == 'confirmed',
        requires_recheck=False,
        confidence_level='confirmed' if result.get('status') == 'confirmed' else 'recheck_failed',
    )
    state.current_stage = 'report'
    state.save()
    return state


def run_report_stage(state: WorkflowState):
    stage_name = 'report'
    input_payload = {
        'stage': stage_name,
        'best_config_path': state.best_config_path,
    }
    report_dir = Path(state.run_dir) / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'final_report.json'
    existing = _load_stage_status(Path(state.stage_status_path)).get(stage_name)
    if _stage_can_skip(existing, _json_hash(input_payload)):
        _append_once(state.completed_stages, stage_name)
        state.current_stage = 'complete'
        state.save()
        return state
    _mark_stage_running(
        state,
        stage_name,
        input_payload,
        summary_path=_norm_path(report_path),
    )
    best_config = _best_config_payload(state)
    best_meta = _read_json(Path(state.run_dir) / 'best_config_meta.json', default={})
    selection_reason = best_meta.get('selection_reason', {})
    if best_meta.get('is_confirmed'):
        unresolved_step = None
    elif (
        best_meta.get('requires_recheck')
        and PROFILE_PRESETS[state.profile]['recheck_topk'] <= 0
    ):
        unresolved_step = 'recheck_skipped_by_profile'
    elif best_meta.get('requires_recheck'):
        unresolved_step = 'recheck_topk'
    elif state.blocked_reason:
        unresolved_step = state.blocked_reason
    elif (
        best_meta.get('confidence_level') == 'recheck_passed'
        and not PROFILE_PRESETS[state.profile]['allow_final_confirm']
    ):
        unresolved_step = 'final_confirm_skipped_by_profile'
    else:
        unresolved_step = 'final_confirm'
    report = {
        'goal': state.goal,
        'profile': state.profile,
        'current_stage': stage_name,
        'best_config_path': state.best_config_path,
        'best_config': best_config,
        'best_config_meta': best_meta,
        'history_paths': state.history_paths,
        'blocked_reason': state.blocked_reason,
        'selected_protocol_entry': state.selected_protocol_entry,
        'selected_structure_entry': state.selected_structure_entry,
        'best_config_explanation': {
            'why_current_config_is_best': {
                'source_stage': best_meta.get('source_stage'),
                'source_entry': best_meta.get('source_entry'),
                'source_trial_id': best_meta.get('source_trial_id'),
                'confidence_level': best_meta.get('confidence_level'),
                'selection_reason': selection_reason,
            },
            'is_confirmed': bool(best_meta.get('is_confirmed')),
            'requires_recheck': bool(best_meta.get('requires_recheck')),
            'unresolved_step': unresolved_step,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    _mark_stage_completed(
        state,
        stage_name,
        input_payload,
        [report_path],
        summary_path=_norm_path(report_path),
    )
    state.current_stage = 'complete'
    state.save()
    return state


def run_workflow(goal='stable_1000', profile='cpu_normal', run_dir='runs/auto_run1'):
    run_dir = Path(run_dir)
    state_path = run_dir / 'workflow_state.json'

    if state_path.exists():
        state = load_workflow_state(state_path)
    else:
        state = WorkflowState.new(goal=goal, profile=profile, run_dir=run_dir)
        state.save(state_path)

    if state.current_stage == 'init':
        state.current_stage = 'baseline_check'
        state.save(state_path)
    handlers = {
        'baseline_check': lambda s: run_baseline_stage(s),
        'protocol_matrix': lambda s: run_matrix_stage(
            state=s,
            stage_name='protocol_matrix',
            matrix_name='protocol',
            matrix=PROTOCOL_ABLATION,
            next_stage='structure_matrix',
        ),
        'structure_matrix': lambda s: run_matrix_stage(
            state=s,
            stage_name='structure_matrix',
            matrix_name='structure',
            matrix=STRUCTURE_ABLATION,
            next_stage='derive_candidates_and_search_space',
        ),
        'derive_candidates_and_search_space': lambda s: derive_candidates_and_search_space(s),
        'select_base_config': lambda s: select_base_config(s),
        'focused_search_round': lambda s: run_focused_search_round(s),
        'warmstart_search_round': lambda s: run_warmstart_search_round(s),
        'population_search_round': lambda s: run_population_search_round(s),
        'recheck_topk': lambda s: run_recheck_topk(s),
        'final_confirm': lambda s: run_final_confirm_stage(s),
        'report': lambda s: run_report_stage(s),
    }

    terminal_stages = {'blocked', 'complete'}
    guard = 0
    while state.current_stage not in terminal_stages:
        guard += 1
        if guard > 20:
            raise RuntimeError(f'Workflow exceeded max stage transitions at {state.current_stage}')
        previous_stage = state.current_stage
        handler = handlers.get(state.current_stage)
        if handler is None:
            return state
        try:
            state = handler(state)
        except Exception as exc:
            _mark_stage_failed(
                state,
                previous_stage,
                {
                    'stage': previous_stage,
                    'goal': state.goal,
                    'profile': state.profile,
                },
                exc,
            )
            stage_record = _load_stage_status(Path(state.stage_status_path)).get(previous_stage, {})
            if int(stage_record.get('attempt', 0)) >= 2:
                return _write_blocked_report(
                    state,
                    blocked_reason='repeated_runtime_failure',
                    latest_history_path=_latest_search_history_path(state),
                )
            raise
        if state.current_stage == previous_stage:
            raise RuntimeError(f'Workflow stage did not advance: {state.current_stage}')
    return state


def main():
    args = make_parser().parse_args()
    state = run_workflow(goal=args.goal, profile=args.profile, run_dir=args.run_dir)
    print(
        json.dumps(
            {
                'goal': state.goal,
                'profile': state.profile,
                'run_dir': state.run_dir,
                'current_stage': state.current_stage,
                'completed_stages': state.completed_stages,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()

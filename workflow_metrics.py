"""Workflow-level metrics and failure-reason normalization helpers."""
from __future__ import annotations

import numpy as np


WORKFLOW_FAILURE_REASON_ENUMS = (
    'success',
    'no_learning_50k',
    'plateau_100k',
    'plateau_300k',
    'high_variance_near_miss',
    'unstable_recheck',
    'oom_or_resource_error',
    'runtime_exception',
    'interrupted',
    'insufficient_evidence',
    'unknown_failure',
)


def normalize_failure_reason(record):
    """Map free-form failure strings to a stable workflow enum."""
    status = str(record.get('status') or '').strip().lower()
    if status == 'success':
        return 'success'

    raw_reason = str(record.get('failure_reason') or '').strip().lower()
    if not raw_reason:
        return 'unknown_failure'
    if raw_reason in WORKFLOW_FAILURE_REASON_ENUMS:
        return raw_reason
    if raw_reason.startswith('no_learning'):
        return 'no_learning_50k'
    if raw_reason.startswith('plateau_'):
        digits = ''.join(ch for ch in raw_reason if ch.isdigit())
        try:
            frames = int(digits) if digits else 0
        except ValueError:
            frames = 0
        return 'plateau_300k' if frames >= 300 else 'plateau_100k'
    if 'variance' in raw_reason and 'near' in raw_reason:
        return 'high_variance_near_miss'
    if 'recheck' in raw_reason and ('unstable' in raw_reason or 'failed' in raw_reason):
        return 'unstable_recheck'
    reason_tokens = raw_reason.replace(':', ' ').replace(',', ' ').split()
    if (
        'oom' in reason_tokens
        or 'out of memory' in raw_reason
        or 'resource' in raw_reason
    ):
        return 'oom_or_resource_error'
    if raw_reason in {'keyboardinterrupt', 'interrupted'} or 'interrupt' in raw_reason:
        return 'interrupted'
    if 'insufficient' in raw_reason and 'evidence' in raw_reason:
        return 'insufficient_evidence'
    if 'exception' in raw_reason or 'traceback' in raw_reason or 'runtime' in raw_reason:
        return 'runtime_exception'
    return 'unknown_failure'


def aggregate_trials_for_workflow(rows):
    """Aggregate raw trial rows into stable workflow comparison metrics."""
    trial_rows = [row for row in rows if row.get('record_type', 'trial') == 'trial']
    median_scores = [
        float(row['median_score'])
        for row in trial_rows
        if row.get('median_score') is not None
    ]
    peak_scores = [
        float(row['best_eval_score'])
        for row in trial_rows
        if row.get('best_eval_score') is not None
    ]
    success_rates = [
        float(row['success_rate_1000'])
        for row in trial_rows
        if row.get('success_rate_1000') is not None
    ]
    objectives = [
        float(row['objective'])
        for row in trial_rows
        if row.get('objective') is not None
    ]

    failure_reason_counter = {}
    plateau_like_count = 0
    no_learning_count = 0
    stable_success_count = 0
    for row in trial_rows:
        normalized_reason = normalize_failure_reason(row)
        if normalized_reason == 'success':
            stable_success_count += 1
            continue
        failure_reason_counter[normalized_reason] = (
            failure_reason_counter.get(normalized_reason, 0) + 1
        )
        if normalized_reason in {'plateau_100k', 'plateau_300k'}:
            plateau_like_count += 1
        if normalized_reason == 'no_learning_50k':
            no_learning_count += 1

    return {
        'trial_count': len(trial_rows),
        'stable_success_count': stable_success_count,
        'best_final_median_score': max(median_scores) if median_scores else None,
        'best_final_success_rate_1000': max(success_rates) if success_rates else None,
        'best_eval_peak_score': max(peak_scores) if peak_scores else None,
        'median_of_final_medians': (
            float(np.median(np.array(median_scores, dtype=np.float64)))
            if median_scores else None
        ),
        'best_objective': min(objectives) if objectives else None,
        'failure_reason_counter': failure_reason_counter,
        'plateau_like_count': plateau_like_count,
        'no_learning_count': no_learning_count,
    }

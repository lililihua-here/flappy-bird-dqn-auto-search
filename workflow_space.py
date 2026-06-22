"""Workflow search-space shaping helpers."""
from __future__ import annotations

import json


def widen_search_space_after_stall(focused_search_space):
    """Widen continuous search ranges conservatively after a stalled round."""
    widened = json.loads(json.dumps(focused_search_space, ensure_ascii=False))
    lr_spec = widened.get('continuous', {}).get('lr')
    if lr_spec:
        lr_spec['low'] = max(1e-5, lr_spec['low'] * 0.5)
        lr_spec['high'] = min(3e-3, lr_spec['high'] * 2.0)
    gamma_spec = widened.get('continuous', {}).get('gamma')
    if gamma_spec:
        gamma_spec['low'] = max(0.90, gamma_spec['low'] - 0.01)
        gamma_spec['high'] = min(0.999, gamma_spec['high'] + 0.002)
    return widened

"""Reward protocol versions for V3."""


def compute_reward_v1_sparse(env_events):
    return -1.0 if env_events.get('hit') else (1.0 if env_events.get('passed_pipe') else 0.0)


def compute_reward_v2_ratio(env_events, death_ratio=1, alive_ratio=0.0, scale=1.0, clip=None):
    r = 0.0
    if env_events.get('hit'):
        r = -float(death_ratio)
    elif env_events.get('passed_pipe'):
        r = 1.0
    else:
        r = float(alive_ratio)
    if clip is not None:
        r = max(-clip, min(clip, r))
    return r * float(scale)


def compute_reward_v3_gap_shaping(env_events, bird_y, gap_center, pipe_gap=400,
                                   death_ratio=1, alive_ratio=0.0, scale=1.0,
                                   clip=None, gap_shaping_coef=0.0):
    r = compute_reward_v2_ratio(env_events, death_ratio, alive_ratio, 1.0, None)
    if gap_shaping_coef > 0 and not env_events.get('hit') and not env_events.get('passed_pipe'):
        gap_dist = abs(bird_y - gap_center) / (pipe_gap / 2)
        r += gap_shaping_coef * (1 - min(1, gap_dist))
    if clip is not None:
        r = max(-clip, min(clip, r))
    return r * float(scale)

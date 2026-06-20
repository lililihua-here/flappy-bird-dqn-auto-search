"""State encoder variants for V3 protocol versioning."""
import numpy as np


class StateEncoderV1:
    STATE_DIM = 7
    VERSION = "low_dim_v1"

    def __init__(self, screen_height=800, screen_width=600, max_fall_speed=10, bird_x=100):
        self.h = float(screen_height)
        self.w = float(screen_width)
        self.ms = float(max_fall_speed)
        self.bx = float(bird_x)

    def encode(self, state):
        return np.array([
            state['bird_y'] / self.h,
            state['bird_velocity'] / self.ms,
            (state['pipe_x'] - self.bx) / self.w,
            state['pipe_gap_top'] / self.h,
            state['pipe_gap_bottom'] / self.h,
            state['pipe_gap_center'] / self.h,
            (state['bird_y'] - state['pipe_gap_center']) / self.h,
        ], dtype=np.float32)

    @property
    def state_dim(self):
        return 7


class StateEncoderV2:
    STATE_DIM = 4
    VERSION = "low_dim_v2"

    def __init__(self, screen_height=800, screen_width=600, max_fall_speed=10, bird_x=100):
        self.h = float(screen_height)
        self.w = float(screen_width)
        self.ms = float(max_fall_speed)
        self.bx = float(bird_x)

    def encode(self, state):
        return np.array([
            state['bird_y'] / self.h,
            state['bird_velocity'] / self.ms,
            (state['pipe_x'] - self.bx) / self.w,
            (state['bird_y'] - state['pipe_gap_center']) / self.h,
        ], dtype=np.float32)

    @property
    def state_dim(self):
        return 4


class StateEncoderV3:
    STATE_DIM = 9
    VERSION = "low_dim_v3"

    def __init__(self, screen_height=800, screen_width=600, max_fall_speed=10, bird_x=100):
        self.h = float(screen_height)
        self.w = float(screen_width)
        self.ms = float(max_fall_speed)
        self.bx = float(bird_x)

    def encode(self, state):
        return np.array([
            state['bird_y'] / self.h,
            state['bird_velocity'] / self.ms,
            (state['pipe_x'] - self.bx) / self.w,
            state['pipe_gap_top'] / self.h,
            state['pipe_gap_bottom'] / self.h,
            state['pipe_gap_center'] / self.h,
            (state['bird_y'] - state['pipe_gap_center']) / self.h,
            (state['bird_y'] - state['pipe_gap_top']) / self.h,
            (state['bird_y'] - state['pipe_gap_bottom']) / self.h,
        ], dtype=np.float32)

    @property
    def state_dim(self):
        return 9


def get_encoder(version, env_config=None):
    env_config = env_config or {}
    kwargs = {
        "screen_height": env_config.get("SCREEN_HEIGHT", 800),
        "screen_width": env_config.get("SCREEN_WIDTH", 600),
        "max_fall_speed": env_config.get("MAX_FALL_SPEED", 10),
        "bird_x": env_config.get("BIRD_X", 100),
    }
    if version == "low_dim_v1":
        return StateEncoderV1(**kwargs)
    if version == "low_dim_v2":
        return StateEncoderV2(**kwargs)
    if version == "low_dim_v3":
        return StateEncoderV3(**kwargs)
    raise ValueError(f"Unknown state version: {version}")

"""Contract tests for state encoder variants (V3.2)."""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


def test_v1_v2_v3_dims():
    from state_encoder_variants import get_encoder
    assert get_encoder("low_dim_v1").state_dim == 7
    assert get_encoder("low_dim_v2").state_dim == 4
    assert get_encoder("low_dim_v3").state_dim == 9
    assert get_encoder("low_dim_v1").STATE_DIM == 7
    assert get_encoder("low_dim_v2").STATE_DIM == 4
    assert get_encoder("low_dim_v3").STATE_DIM == 9


def test_v1_encode_shape():
    from state_encoder_variants import StateEncoderV1
    state = {
        'bird_y': 400, 'bird_velocity': 2, 'pipe_x': 300,
        'pipe_gap_top': 100, 'pipe_gap_bottom': 500, 'pipe_gap_center': 300,
    }
    vec = StateEncoderV1().encode(state)
    assert vec.shape == (7,)
    assert vec.dtype == np.float32


def test_v2_encode_shape():
    from state_encoder_variants import StateEncoderV2
    state = {
        'bird_y': 400, 'bird_velocity': 2, 'pipe_x': 300,
        'pipe_gap_top': 100, 'pipe_gap_bottom': 500, 'pipe_gap_center': 300,
    }
    vec = StateEncoderV2().encode(state)
    assert vec.shape == (4,)
    assert vec.dtype == np.float32


def test_v3_encode_shape():
    from state_encoder_variants import StateEncoderV3
    state = {
        'bird_y': 400, 'bird_velocity': 2, 'pipe_x': 300,
        'pipe_gap_top': 100, 'pipe_gap_bottom': 500, 'pipe_gap_center': 300,
    }
    vec = StateEncoderV3().encode(state)
    assert vec.shape == (9,)
    assert vec.dtype == np.float32


def test_v1_encode_values_in_range():
    from state_encoder_variants import StateEncoderV1
    state = {
        'bird_y': 400, 'bird_velocity': 2, 'pipe_x': 300,
        'pipe_gap_top': 200, 'pipe_gap_bottom': 600, 'pipe_gap_center': 400,
    }
    vec = StateEncoderV1().encode(state)
    # All values should be finite
    assert np.all(np.isfinite(vec))
    # bird_y / 800 should be around 0.5
    assert 0.4 < vec[0] < 0.6


def test_get_encoder_unknown_version_raises():
    from state_encoder_variants import get_encoder
    try:
        get_encoder("nonexistent_version")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_get_encoder_with_env_config():
    from state_encoder_variants import get_encoder
    env_config = {
        "SCREEN_HEIGHT": 720,
        "SCREEN_WIDTH": 480,
        "MAX_FALL_SPEED": 12,
        "BIRD_X": 80,
    }
    encoder = get_encoder("low_dim_v1", env_config=env_config)
    assert encoder.h == 720.0
    assert encoder.w == 480.0
    assert encoder.ms == 12.0
    assert encoder.bx == 80.0


def test_v1_compat_with_original_state_encoder():
    """V1 encoder should produce same output as original StateEncoder."""
    from state_encoder_variants import StateEncoderV1
    from replay_buffer import StateEncoder
    state = {
        'bird_y': 400, 'bird_velocity': 2, 'pipe_x': 300,
        'pipe_gap_top': 100, 'pipe_gap_bottom': 500, 'pipe_gap_center': 300,
    }
    v1_vec = StateEncoderV1().encode(state)
    orig_vec = StateEncoder().encode(state)
    np.testing.assert_array_equal(v1_vec, orig_vec)

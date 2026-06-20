"""Version and hash utilities shared across all modules."""
import subprocess


def get_git_hash():
    """Return short git hash or 'unknown'."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def infer_reward_scheme_version(config):
    """Infer reward scheme version from config, not hardcoded.

    V3.2: If config explicitly contains reward_scheme_version, return it directly.
    Otherwise, infer from reward parameters.
    """
    # V3.2: explicit version from config takes precedence
    if 'reward_scheme_version' in config and config['reward_scheme_version'] is not None:
        return config['reward_scheme_version']

    is_mvp_reward = (
        config.get('death_ratio', 1) == 1
        and config.get('alive_ratio', 0.0) == 0.0
        and config.get('reward_scale', 1.0) == 1.0
        and config.get('reward_clip', None) is None
        and config.get('pipe_reward', 1.0) == 1.0
    )
    if is_mvp_reward:
        return 'mvp_reward_v1'
    return 'v2_reward_search'

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
    """Infer reward scheme version from config, not hardcoded."""
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

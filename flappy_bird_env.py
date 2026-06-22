"""Flappy Bird standard environment — fixed physics, single-pipe recycle."""
import random


class FlappyBirdEnv:
    """Standard fixed-physics Flappy Bird environment.

    P0-1 FIX: total_raw_env_frames survives reset().
    episode_raw_env_frames resets per episode.

    MVP uses single-pipe recycle mode. When a pipe scrolls off-screen, it is
    immediately recycled to the right edge with a new random gap position.
    """

    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 800
    GRAVITY = 0.5
    FLAP_STRENGTH = -3
    PIPE_GAP = 400
    PIPE_VELOCITY = -2.5
    PIPE_WIDTH = 80
    BIRD_X = 100
    BIRD_SIZE = 20
    MAX_FALL_SPEED = 10
    PIPE_SPAWN_X = SCREEN_WIDTH

    def __init__(self, seed=None, reward_config=None):
        self.rng = random.Random(seed)
        self.total_raw_env_frames = 0    # P0-1: never reset
        self.episode_raw_env_frames = 0  # P0-1: reset per episode
        self.reward_config = {
            'pipe_reward': 1.0, 'death_ratio': 1,
            'alive_ratio': 0.0, 'reward_scale': 1.0, 'reward_clip': None,
        }
        if reward_config:
            self.reward_config.update(reward_config)
        self.last_events = {}
        self.reset()

    def reset(self):
        self.bird_y = float(self.SCREEN_HEIGHT // 2)
        self.bird_velocity = 0.0
        self.pipe_x = float(self.PIPE_SPAWN_X)
        self.pipe_gap_center = float(self.rng.randint(200, self.SCREEN_HEIGHT - 200))
        self.score = 0
        self.done = False
        self._scored_current_pipe = False
        self.episode_raw_env_frames = 0  # P0-1: only episode counter resets
        self.last_events = {"hit": False, "passed_pipe": False}
        return self._get_state()

    def step(self, action):
        self.total_raw_env_frames += 1   # P0-1: always increment
        self.episode_raw_env_frames += 1

        if action == 1:
            self.bird_velocity = float(self.FLAP_STRENGTH)
        self.bird_velocity += self.GRAVITY
        self.bird_velocity = max(-self.MAX_FALL_SPEED, min(self.MAX_FALL_SPEED, self.bird_velocity))
        self.bird_y += self.bird_velocity
        self.pipe_x += self.PIPE_VELOCITY

        bird_top = self.bird_y - self.BIRD_SIZE // 2
        bird_bottom = self.bird_y + self.BIRD_SIZE // 2
        pipe_top = self.pipe_gap_center - self.PIPE_GAP // 2
        pipe_bottom = self.pipe_gap_center + self.PIPE_GAP // 2

        hit_pipe = (
            self.BIRD_X + self.BIRD_SIZE // 2 > self.pipe_x
            and self.BIRD_X - self.BIRD_SIZE // 2 < self.pipe_x + self.PIPE_WIDTH
            and (bird_top < pipe_top or bird_bottom > pipe_bottom)
        )
        hit_boundary = bird_top <= 0 or bird_bottom >= self.SCREEN_HEIGHT

        # Reward computation per V2 spec (ratio -> clip -> scale order)
        reward = 0.0
        if hit_pipe or hit_boundary:
            reward = -float(self.reward_config['death_ratio'])
            self.done = True

        if self.pipe_x + self.PIPE_WIDTH < self.BIRD_X and not self._scored_current_pipe:
            self.score += 1
            self._scored_current_pipe = True
            if not self.done:
                reward = float(self.reward_config['pipe_reward'])

        if not self.done and reward == 0.0:
            reward = float(self.reward_config['alive_ratio'])

        clip_val = self.reward_config['reward_clip']
        if clip_val is not None:
            reward = max(-clip_val, min(clip_val, reward))

        reward *= float(self.reward_config['reward_scale'])

        if self.pipe_x < -self.PIPE_WIDTH:
            self.pipe_x = float(self.PIPE_SPAWN_X)
            self.pipe_gap_center = float(self.rng.randint(200, self.SCREEN_HEIGHT - 200))
            self._scored_current_pipe = False

        # V3.2: store events for external reward protocol use
        self.last_events = {
            "hit": hit_pipe or hit_boundary,
            "passed_pipe": self._scored_current_pipe,
        }

        return self._get_state(), reward, self.done

    def capture_runtime_state(self):
        """Return the full mutable runtime state needed for deterministic resume."""
        return {
            'bird_y': self.bird_y,
            'bird_velocity': self.bird_velocity,
            'pipe_x': self.pipe_x,
            'pipe_gap_center': self.pipe_gap_center,
            'score': self.score,
            'done': self.done,
            '_scored_current_pipe': self._scored_current_pipe,
            'total_raw_env_frames': self.total_raw_env_frames,
            'episode_raw_env_frames': self.episode_raw_env_frames,
            'last_events': dict(getattr(self, 'last_events', {})),
            'rng_state': self.rng.getstate(),
        }

    def restore_runtime_state(self, runtime_state):
        """Restore a previously captured runtime state."""
        self.bird_y = float(runtime_state['bird_y'])
        self.bird_velocity = float(runtime_state['bird_velocity'])
        self.pipe_x = float(runtime_state['pipe_x'])
        self.pipe_gap_center = float(runtime_state['pipe_gap_center'])
        self.score = int(runtime_state['score'])
        self.done = bool(runtime_state['done'])
        self._scored_current_pipe = bool(runtime_state['_scored_current_pipe'])
        self.total_raw_env_frames = int(runtime_state['total_raw_env_frames'])
        self.episode_raw_env_frames = int(runtime_state['episode_raw_env_frames'])
        self.last_events = dict(runtime_state.get('last_events', {}))
        self.rng.setstate(runtime_state['rng_state'])

    def _get_state(self):
        pipe_top = self.pipe_gap_center - self.PIPE_GAP // 2
        pipe_bottom = self.pipe_gap_center + self.PIPE_GAP // 2
        return {
            'bird_y': self.bird_y,
            'bird_velocity': self.bird_velocity,
            'pipe_x': self.pipe_x,
            'pipe_gap_top': pipe_top,
            'pipe_gap_bottom': pipe_bottom,
            'pipe_gap_center': self.pipe_gap_center,
        }

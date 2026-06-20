"""DQN network and agent — Double DQN, Adam, Huber loss, soft target update."""
import random
import numpy as np
import torch
import torch.nn as nn
from replay_buffer import ReplayBuffer


# ============================================================================
# DQN network — configurable MLP
# ============================================================================
class DQN(nn.Module):
    """Small MLP Q-network for 7-D low-dimensional state."""

    def __init__(self, state_dim, hidden, n_actions):
        super().__init__()
        dims = [state_dim] + list(hidden) + [n_actions]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


# ============================================================================
# DQN Agent — P0-3/P1-5/P1-8 fixes applied
# ============================================================================
class DQNAgent:
    """DQN agent (1-step Double DQN, Adam, Huber loss, soft target update)."""

    def __init__(self, config, state_dim, n_actions, device):
        # P1-8: Assert MVP fixed parameters
        assert config.get('target_update_mode', 'soft') == 'soft', \
            "MVP fixed: target_update_mode must be 'soft'"
        # P1-3 (v1.3): Assert all MVP-fixed params that env/agent hardcode
        assert config.get('frame_skip', 1) == 1, \
            "MVP fixed: frame_skip must be 1"
        assert config.get('torch_optimizer', 'Adam') == 'Adam', \
            "MVP fixed: torch_optimizer must be 'Adam'"
        assert config.get('loss_type', 'Huber') == 'Huber', \
            "MVP fixed: loss_type must be 'Huber'"
        assert config.get('reward_pipe', 1.0) == 1.0, \
            "MVP fixed: reward_pipe must be 1.0 (env hardcoded)"
        assert config.get('reward_death', -1.0) == -1.0, \
            "MVP fixed: reward_death must be -1.0 (env hardcoded)"
        assert config.get('reward_alive', 0.0) == 0.0, \
            "MVP fixed: reward_alive must be 0.0 (env hardcoded)"

        self.config = config
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        self.q_net = DQN(state_dim, config['hidden'], n_actions).to(device)
        self.target_net = DQN(state_dim, config['hidden'], n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.buffer = ReplayBuffer(config['buffer_sz'])
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config['lr'])
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        self.epsilon = float(config['eps_start'])
        self.decision_steps = 0

    def act(self, state, training=True):
        if training:
            self.decision_steps += 1
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            return int(self.q_net(state_t).argmax(dim=1).item())

    def train(self):
        if not self.buffer.can_sample(self.config['batch_sz']):
            return None

        sample = self.buffer.sample(self.config['batch_sz'])

        # Handle both standard (5-tuple) and n-step (7-tuple) buffers
        gamma_powers_t = None

        if len(sample) == 7:
            states, actions, rewards, next_states, dones, gamma_powers_arr, _actual_ns = sample
            gamma_powers_t = torch.from_numpy(gamma_powers_arr).unsqueeze(1).to(self.device)
        elif len(sample) == 5:
            states, actions, rewards, next_states, dones = sample
        else:
            raise ValueError(f"Unsupported replay sample length: {len(sample)}")

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.config.get('double_q', True):
                next_actions = self.q_net(next_states_t).argmax(1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            # Per-sample discount: gamma^n for n-step, fixed gamma for 1-step
            discount = gamma_powers_t if gamma_powers_t is not None else self.config['gamma']
            target = rewards_t + (1.0 - dones_t) * discount * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config['grad_clip_norm'])
        self.optimizer.step()

        # Soft target update (MVP fixed)
        tau = self.config['tau']
        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        return float(loss.item())

    def decay_epsilon(self):
        # P1-5: use eps_decay_decision_steps, fallback to eps_frames
        decay_steps = self.config.get('eps_decay_decision_steps',
                                      self.config.get('eps_frames', 50000))
        if self.decision_steps >= decay_steps:
            self.epsilon = float(self.config['eps_end'])
        else:
            progress = self.decision_steps / decay_steps
            self.epsilon = float(self.config['eps_start']
                + progress * (self.config['eps_end'] - self.config['eps_start']))

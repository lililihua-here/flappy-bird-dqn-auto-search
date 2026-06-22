"""DQN network and agent — Double DQN, Adam, Huber loss, soft target update."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer


# ============================================================================
# NoisyNet utilities — factorized Gaussian noise (V3.3)
# ============================================================================
def _scale_noise(size, device, dtype):
    """Standard factorized Gaussian noise: f(x) = sign(x) * sqrt(|x|)."""
    x = torch.randn(size, device=device, dtype=dtype)
    return x.sign() * x.abs().sqrt()


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet. Eval mode uses mu (deterministic)."""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))
        self.reset_parameters(sigma_init)

    def reset_parameters(self, sigma_init):
        nn.init.kaiming_uniform_(self.mu_w)
        self.sigma_w.data.fill_(sigma_init / self.in_features ** 0.5)
        nn.init.zeros_(self.mu_b)
        self.sigma_b.data.fill_(sigma_init / self.out_features ** 0.5)

    def forward(self, x):
        if not self.training:
            return x @ self.mu_w.t() + self.mu_b
        eps_in = _scale_noise(self.in_features, x.device, x.dtype)
        eps_out = _scale_noise(self.out_features, x.device, x.dtype)
        weight_noise = eps_out.unsqueeze(1) * eps_in.unsqueeze(0)
        weight = self.mu_w + self.sigma_w * weight_noise
        bias = self.mu_b + self.sigma_b * eps_out
        return x @ weight.t() + bias


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
# NoisyDQN — DQN with NoisyLinear layers (V3.3)
# ============================================================================
class NoisyDQN(nn.Module):
    """DQN with NoisyLinear layers for exploration via parameter noise."""

    def __init__(self, state_dim, hidden, n_actions, sigma_init=0.5):
        super().__init__()
        dims = [state_dim] + list(hidden) + [n_actions]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(NoisyLinear(dims[i], dims[i + 1], sigma_init))
            layers.append(nn.ReLU())
        layers.append(NoisyLinear(dims[-2], dims[-1], sigma_init))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


# ============================================================================
# DuelingMLP — Dueling architecture (V3.3)
# ============================================================================
class DuelingMLP(nn.Module):
    """Dueling architecture with full shared feature stack, then value/advantage streams."""

    def __init__(self, state_dim, hidden, n_actions):
        super().__init__()
        dims = [state_dim] + list(hidden)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.feature = nn.Sequential(*layers)
        last_dim = dims[-1]
        stream_dim = max(16, last_dim // 2)
        self.value = nn.Sequential(nn.Linear(last_dim, stream_dim), nn.ReLU(), nn.Linear(stream_dim, 1))
        self.advantage = nn.Sequential(nn.Linear(last_dim, stream_dim), nn.ReLU(), nn.Linear(stream_dim, n_actions))

    def forward(self, x):
        f = self.feature(x.float())
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)


# ============================================================================
# DQN Agent — P0-3/P1-5/P1-8 fixes applied
# ============================================================================
class DQNAgent:
    """DQN agent (1-step Double DQN, Adam, Huber loss, soft target update)."""

    def __init__(self, config, state_dim, n_actions, device):
        # P1-3 (v1.3): Assert MVP-fixed params that env/agent hardcode
        assert config.get('frame_skip', 1) == 1, \
            "MVP fixed: frame_skip must be 1"
        assert config.get('loss_type', 'Huber') == 'Huber', \
            "MVP fixed: loss_type must be 'Huber'"

        self.config = config
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        # V3.3: network_backbone and exploration_head
        self.network_backbone = config.get("network_backbone", "mlp")
        self.exploration_head = config.get("exploration_head", "epsilon_greedy")

        use_noisy_layers = self.exploration_head == "noisy_net"

        if self.network_backbone == "dueling_mlp":
            self.q_net = DuelingMLP(state_dim, config['hidden'], n_actions).to(device)
            self.target_net = DuelingMLP(state_dim, config['hidden'], n_actions).to(device)
        elif use_noisy_layers:
            self.q_net = NoisyDQN(state_dim, config['hidden'], n_actions).to(device)
            self.target_net = NoisyDQN(state_dim, config['hidden'], n_actions).to(device)
        else:
            self.q_net = DQN(state_dim, config['hidden'], n_actions).to(device)
            self.target_net = DQN(state_dim, config['hidden'], n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.buffer = ReplayBuffer(config['buffer_sz'])

        # V3.3: selectable optimizer
        opt_name = config.get('torch_optimizer', 'Adam')
        if opt_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config['lr'])
        elif opt_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=config['lr'])
        else:
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config['lr'])
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        self.epsilon = float(config['eps_start'])
        self.decision_steps = 0
        self.train_updates = 0

    def act(self, state, training=True):
        # V3.3: NoisyNet exploration — epsilon not used, noise in layers
        if self.exploration_head == "noisy_net":
            if training:
                self.decision_steps += 1
                self.q_net.train()
            else:
                self.q_net.eval()
        elif training:
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

        gamma_powers_t = None
        is_weights = None
        per_indices = None

        if len(sample) == 9:
            # NStepPERBuffer
            states, actions, rewards, next_states, dones, gamma_powers_arr, actual_ns, weights_arr, per_indices = sample
            gamma_powers_t = torch.from_numpy(gamma_powers_arr).unsqueeze(1).to(self.device)
            is_weights = torch.from_numpy(weights_arr).to(self.device)
        elif len(sample) == 7:
            states, actions, rewards, next_states, dones, extra1, extra2 = sample
            if extra1.ndim == 2 and extra1.shape[1] == 1:
                # PERBuffer: weights (B,1), indices
                is_weights = torch.from_numpy(extra1).to(self.device)
                per_indices = extra2
            else:
                # NStepReplayBuffer: gamma_powers (B,), actual_ns
                gamma_powers_t = torch.from_numpy(extra1).unsqueeze(1).to(self.device)
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
            discount = gamma_powers_t if gamma_powers_t is not None else self.config['gamma']
            target = rewards_t + (1.0 - dones_t) * discount * next_q

        td_loss = F.smooth_l1_loss(q_values, target, reduction='none')
        if is_weights is not None:
            loss = (is_weights * td_loss).mean()
        else:
            loss = td_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config['grad_clip_norm'])
        self.optimizer.step()

        self.train_updates += 1

        # V3.3: target update mode (soft or hard)
        if self.config.get('target_update_mode', 'soft') == 'hard':
            hard_freq = self.config.get(
                'hard_update_interval_decision_steps',
                self.config.get('hard_update_freq', 1000),
            )
            if self.train_updates % hard_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            tau = self.config['tau']
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        if per_indices is not None and hasattr(self.buffer, 'update_priorities'):
            with torch.no_grad():
                td_errors = (q_values.detach() - target.detach()).abs().cpu().numpy().reshape(-1)
            self.buffer.update_priorities(per_indices, td_errors)

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

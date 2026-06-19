"""Contract tests for DQN and DQNAgent."""
import numpy as np
import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dqn_agent import DQN, DQNAgent


# ============================================================================
# DQN network tests
# ============================================================================
def test_dqn_forward_shape():
    import torch
    net = DQN(state_dim=7, hidden=[64, 32], n_actions=2)
    x = torch.randn(4, 7)
    y = net(x)
    assert tuple(y.shape) == (4, 2)


def test_dqn_uses_requested_hidden_layers():
    import torch.nn as nn
    net = DQN(state_dim=7, hidden=[128, 64], n_actions=2)
    linears = [m for m in net.net if isinstance(m, nn.Linear)]
    assert [layer.in_features for layer in linears] == [7, 128, 64]
    assert [layer.out_features for layer in linears] == [128, 64, 2]


# ============================================================================
# DQNAgent tests
# ============================================================================
def test_dqn_agent_act_greedy():
    import torch
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.0, 'eps_end': 0.0,
        'eps_decay_decision_steps': 10000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    state = np.array([0.5, 0.0, 0.8, 0.2, 0.6, 0.0, 0.3], dtype=np.float32)
    a1 = agent.act(state, training=False)
    a2 = agent.act(state, training=False)
    assert a1 == a2


def test_dqn_agent_epsilon_decay():
    import torch
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.10, 'eps_end': 0.01,
        'eps_decay_decision_steps': 1000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    assert abs(agent.epsilon - 0.10) < 1e-6
    agent.decision_steps = 500
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.055) < 1e-3
    agent.decision_steps = 1000
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.01) < 1e-6


def test_dqn_agent_eps_frames_backward_compat():
    """P1-5: old 'eps_frames' key still works as fallback."""
    import torch
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.10, 'eps_end': 0.01,
        'eps_frames': 500,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    agent.decision_steps = 500
    agent.decay_epsilon()
    assert abs(agent.epsilon - 0.01) < 1e-6


def test_dqn_agent_config_assertions():
    """P1-8: MVP fixed params are asserted on init."""
    import torch
    import pytest
    config = {
        'hidden': [64, 32], 'lr': 1e-4, 'gamma': 0.99, 'batch_sz': 64,
        'buffer_sz': 1000, 'eps_start': 0.05, 'eps_end': 0.005,
        'eps_decay_decision_steps': 10000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    bad = dict(config, target_update_mode='hard')
    with pytest.raises(AssertionError):
        DQNAgent(bad, state_dim=7, n_actions=2, device='cpu')


def test_dqn_agent_train_returns_loss():
    import torch
    config = {
        'hidden': [64, 32], 'lr': 1e-3, 'gamma': 0.99, 'batch_sz': 16,
        'buffer_sz': 1000, 'eps_start': 0.05, 'eps_end': 0.005,
        'eps_decay_decision_steps': 50000,
        'replay_start_size': 100, 'train_freq': 1,
        'target_update_mode': 'soft', 'tau': 0.005,
        'double_q': True, 'grad_clip_norm': 5,
        'n_step': 1,
    }
    agent = DQNAgent(config, state_dim=7, n_actions=2, device='cpu')
    for _ in range(200):
        s = np.random.randn(7).astype(np.float32)
        ns = np.random.randn(7).astype(np.float32)
        agent.buffer.add(s, random.randint(0, 1), random.random(), ns, random.random() < 0.1)
    loss = agent.train()
    assert isinstance(loss, float) and loss > 0

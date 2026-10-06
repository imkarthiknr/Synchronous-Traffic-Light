import numpy as np
import torch

from junctioniq.rl.dqn import DQNAgent, DQNConfig, ReplayBuffer


def test_epsilon_schedule():
    cfg = DQNConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    assert cfg.epsilon(0) == 1.0
    assert np.isclose(cfg.epsilon(50), 0.55)
    assert np.isclose(cfg.epsilon(1000), 0.1)


def test_replay_buffer_wraps():
    buf = ReplayBuffer(capacity=3, n_obs=2)
    for i in range(5):
        buf.add(np.full(2, i), i, float(i), np.full(2, i + 1), i == 4)
    assert len(buf) == 3
    assert sorted(buf.actions.tolist()) == [2, 3, 4]
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 2) and batch["dones"].dtype == torch.float32


def test_double_dqn_target_uses_online_argmax_and_target_values():
    agent = DQNAgent(1, 2, DQNConfig(gamma=0.5, lr=0.0, target_update_tau=0.0))
    with torch.no_grad():
        for net in (agent.online, agent.target):
            for m in net:
                if isinstance(m, torch.nn.Linear):
                    m.weight.zero_()
                    m.bias.zero_()
        # online prefers action 0 everywhere; target values: action 0 -> 3, action 1 -> 10
        agent.online[-1].bias.copy_(torch.tensor([1.0, 0.0]))
        agent.target[-1].bias.copy_(torch.tensor([3.0, 10.0]))
    batch = {
        "obs": torch.zeros(1, 1),
        "actions": torch.tensor([0]),
        "rewards": torch.tensor([2.0]),
        "next_obs": torch.zeros(1, 1),
        "dones": torch.tensor([0.0]),
    }
    loss = agent.learn(batch)
    # target = 2 + 0.5 * Q_target(s', argmax Q_online) = 2 + 0.5 * 3 = 3.5; Q(s,0) = 1
    # Huber(1 - 3.5) = |2.5| - 0.5 = 2.0  (plain DQN would use max 10 -> target 7)
    assert np.isclose(loss, 2.0)


def test_learning_fits_a_bandit():
    rng = np.random.default_rng(0)
    agent = DQNAgent(2, 3, DQNConfig(hidden=32, lr=1e-2, gamma=0.0), seed=0)
    buf = ReplayBuffer(1000, 2)
    for _ in range(600):
        a = int(rng.integers(3))
        buf.add(np.array([1.0, 0.0]), a, 1.0 if a == 2 else 0.0, np.zeros(2), True)
    for _ in range(300):
        agent.learn(buf.sample(64))
    assert agent.act(np.array([1.0, 0.0]), epsilon=0.0) == 2


def test_save_and_load(tmp_path):
    agent = DQNAgent(4, 3, DQNConfig(hidden=8))
    obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    agent.save(tmp_path / "m.pt", {"env": {"delta": 5}})
    loaded = DQNAgent.load(tmp_path / "m.pt")
    assert np.allclose(agent.q_values(obs), loaded.q_values(obs))
    assert loaded.extra["env"]["delta"] == 5

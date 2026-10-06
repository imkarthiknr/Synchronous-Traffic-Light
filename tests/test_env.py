import numpy as np
import pytest

from junctioniq.control import max_pressure
from junctioniq.sim.env import JunctionEnv


def test_reset_and_observation_bounds(short_env):
    obs, info = short_env.reset(seed=1)
    assert obs.shape == short_env.observation_space.shape
    assert short_env.observation_space.contains(obs)
    assert short_env.action_space.n == 16  # 4 junctions x 2 green phases
    assert info["vehicles_planned"] > 0


def test_same_seed_same_day(short_env):
    results = []
    for _ in range(2):
        short_env.reset(seed=5)
        done = False
        while not done:
            _, _, term, trunc, info = short_env.step(0)
            done = term or trunc
        results.append(info["metrics"])
    assert results[0] == results[1]


def test_switch_shows_yellow_in_sumo(short_env):
    short_env.reset(seed=1)
    jid = short_env.junctions[0].id
    sumo = short_env._sumo
    first = sumo.trafficlight.getRedYellowGreenState(jid)
    short_env.step(1)  # junction 0 asks for its second green
    after = sumo.trafficlight.getRedYellowGreenState(jid)
    # 5 s step with 3 s yellow: the switch has completed
    assert after == short_env.junctions[0].greens[1] != first
    assert short_env.controllers[0].switches == 1


def test_episode_truncates_at_horizon(short_env):
    short_env.reset(seed=2)
    steps, done = 0, False
    while not done:
        _, reward, term, trunc, info = short_env.step(short_env.action_space.sample())
        assert reward <= 0
        done, steps = term or trunc, steps + 1
    assert trunc and not term
    assert steps == 300 // short_env.delta
    m = info["metrics"]
    assert m["simulated_s"] == 300 and m["vehicles"] > 0


@pytest.mark.parametrize("control", ["fixed", "actuated"])
def test_sumo_controlled_days_drain(control):
    env = JunctionEnv(control=control, horizon=300, drain=900, demand_scale=0.5)
    try:
        env.reset(seed=3)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(None)
            done = term or trunc
        m = info["metrics"]
    finally:
        env.close()
    assert m["signal_switches"] == 0
    assert m["arrived"] > 0 and m["mean_travel_time_s"] > 0


def test_agent_control_needs_action(short_env):
    short_env.reset(seed=1)
    with pytest.raises(ValueError):
        short_env.step(None)


def test_max_pressure_and_pressures(short_env):
    obs, _ = short_env.reset(seed=4)
    for _ in range(20):
        obs, *_ = short_env.step(max_pressure(short_env, obs))
    pressures = short_env.pressures()
    assert [len(p) for p in pressures] == short_env.sizes
    assert all(np.isfinite(v) for p in pressures for v in p)


@pytest.mark.parametrize("reward", ["queue", "wait", "speed"])
def test_reward_kinds(reward):
    env = JunctionEnv(control="agent", horizon=60, reward=reward)
    try:
        env.reset(seed=1)
        _, r, *_ = env.step(0)
        assert np.isfinite(r)
    finally:
        env.close()

import pytest

from junctioniq.sim.env import JunctionEnv


@pytest.fixture
def short_env():
    """Agent-controlled environment for 5 simulated minutes."""
    env = JunctionEnv(control="agent", horizon=300)
    yield env
    env.close()

import csv

import pytest

from junctioniq.evaluate import PolicySpec, evaluate, markdown_table
from junctioniq.rl.dqn import DQNAgent, DQNConfig
from junctioniq.rl.train import TrainConfig, train


def test_short_training_run(tmp_path):
    cfg = TrainConfig(
        episodes=2,
        horizon=200,
        validate_every=2,
        validation_seeds=[1],
        dqn=DQNConfig(hidden=16, learning_starts=10, batch_size=8),
    )
    best = train(cfg, tmp_path, log=lambda *_: None)
    assert best.is_file() and (tmp_path / "last.pt").is_file()
    rows = list(csv.DictReader((tmp_path / "training.csv").open()))
    assert [r["episode"] for r in rows] == ["1", "2"]
    assert rows[1]["val_mean_travel_time_s"]
    agent = DQNAgent.load(best)
    assert agent.extra["env"]["min_green"] == cfg.min_green


def test_config_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("episodes: 3\ndemand_scale: [0.5, 1.0]\ndqn:\n  hidden: 32\n")
    cfg = TrainConfig.from_yaml(path)
    assert cfg.episodes == 3 and cfg.demand_scale == (0.5, 1.0) and cfg.dqn.hidden == 32


def test_policy_spec_parsing():
    assert PolicySpec.parse("fixed:20").fixed_green == 20
    assert PolicySpec.parse("fixed").label == "Fixed-time (42 s greens)"
    for bad in ["nope", "actuated:5"]:
        with pytest.raises(ValueError):
            PolicySpec.parse(bad)


@pytest.mark.slow
def test_evaluate_writes_outputs(tmp_path):
    rows = evaluate(
        [PolicySpec.parse("fixed:20"), PolicySpec.parse("max-pressure")],
        seeds=[1],
        scale=0.5,
        out_dir=tmp_path,
        log=lambda *_: None,
    )
    assert len(rows) == 2
    assert "Max-pressure" in markdown_table(rows)
    for name in ["results.csv", "summary.md", "travel_time.png", "queues.png"]:
        assert (tmp_path / name).is_file()

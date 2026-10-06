import json

from junctioniq.cli import _seeds, build_parser, main


def test_seed_ranges():
    assert _seeds("3-5") == [3, 4, 5]
    assert _seeds("1,7") == [1, 7]


def test_parser_defaults():
    args = build_parser().parse_args(["evaluate"])
    assert args.policies == ["fixed", "fixed:20", "actuated", "max-pressure"]


def test_simulate_prints_metrics(capsys):
    main(["simulate", "--policy", "actuated", "--seed", "2", "--scale", "0.4"])
    out = json.loads(capsys.readouterr().out)
    assert out["policy"] == "Actuated (SUMO)" and out["still_in_network"] == 0

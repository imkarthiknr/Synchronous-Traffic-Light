import pytest

from junctioniq.sim.signals import (
    SignalController,
    decode_joint,
    encode_joint,
    green_states,
    joint_action_size,
    yellow_between,
)


def test_green_states_skip_yellows_and_duplicates():
    program = ["GGrr", "yyrr", "rrGG", "rryy", "GGrr"]
    assert green_states(program) == ["GGrr", "rrGG"]


def test_green_states_need_two_greens():
    with pytest.raises(ValueError):
        green_states(["GGGG", "yyyy"])


def test_yellow_only_where_green_is_lost():
    assert yellow_between("GgGr", "rgrG") == "ygyr"
    with pytest.raises(ValueError):
        yellow_between("Gr", "G")


def test_switch_goes_through_yellow_after_min_green():
    c = SignalController("j", ["GGrr", "rrGG"], yellow_time=3, min_green=10)
    assert not c.request(1)  # min green not reached
    for _ in range(10):
        c.tick()
    assert c.request(1)
    assert c.state() == "yyrr" and c.in_yellow
    assert not c.request(0)  # no changes during yellow
    for _ in range(3):
        c.tick()
    assert c.state() == "rrGG"
    assert c.current == 1 and c.time_in_green == 0 and c.switches == 1


def test_requesting_current_green_is_a_no_op():
    c = SignalController("j", ["GGrr", "rrGG"], min_green=0)
    assert not c.request(0)
    with pytest.raises(ValueError):
        c.request(5)


def test_zero_yellow_switches_immediately():
    c = SignalController("j", ["GGrr", "rrGG"], yellow_time=0, min_green=0)
    assert c.request(1)
    assert c.state() == "rrGG"


@pytest.mark.parametrize("sizes", [[2, 2, 2, 2], [2, 3, 4]])
def test_joint_action_round_trip(sizes):
    for action in range(joint_action_size(sizes)):
        assert encode_joint(decode_joint(action, sizes), sizes) == action
    with pytest.raises(ValueError):
        decode_joint(joint_action_size(sizes), sizes)

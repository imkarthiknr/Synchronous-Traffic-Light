import xml.etree.ElementTree as ET

import pytest

from junctioniq import scenario


def test_bundled_scenario_loads():
    sc = scenario.load("kowdenahalli")
    assert scenario.available() == ["kowdenahalli"]
    assert sc.duration == 60 * 60
    assert len(sc.junction_ids) == 4
    assert sc.net.is_file() and sc.osm.is_file()


def test_unknown_scenario():
    with pytest.raises(FileNotFoundError):
        scenario.load("atlantis")


def _trips(path):
    return [t.attrib for t in ET.parse(path).getroot().iter("trip")]


def test_demand_is_reproducible_and_scales(tmp_path):
    sc = scenario.load()
    a, b, c = tmp_path / "a.xml", tmp_path / "b.xml", tmp_path / "c.xml"
    n1 = sc.write_demand(a, seed=7)
    n2 = sc.write_demand(b, seed=7)
    assert n1 == n2 and a.read_text() == b.read_text()
    expected = sum(sum(p.rates.values()) * p.seconds / 3600 for p in sc.periods)
    assert abs(n1 - expected) < 4 * expected**0.5  # Poisson noise
    n_double = sc.write_demand(c, seed=7, scale=2.0)
    assert 1.7 * n1 < n_double < 2.3 * n1


def test_no_u_turns_and_sorted_departures(tmp_path):
    sc = scenario.load()
    sc.write_demand(tmp_path / "t.xml", seed=3)
    trips = _trips(tmp_path / "t.xml")
    entry_of = {v: k for k, v in sc.entries.items()}
    exit_of = {v: k for k, v in sc.exits.items()}
    assert all(entry_of[t["from"]] != exit_of[t["to"]] for t in trips)
    departs = [int(t["depart"]) for t in trips]
    assert departs == sorted(departs)
    assert {t["type"] for t in trips} <= set(sc.vehicle_mix)


def test_network_rebuild_matches_committed_file(tmp_path):
    sc = scenario.load()
    rebuilt = sc.build_network(tmp_path / "net.xml")

    def programs(path):
        root = ET.parse(path).getroot()
        return {
            tl.get("id"): [p.get("state") for p in tl.iter("phase")] for tl in root.iter("tlLogic")
        }

    assert programs(rebuilt) == programs(sc.net)
    assert set(programs(sc.net)) == set(sc.junction_ids)

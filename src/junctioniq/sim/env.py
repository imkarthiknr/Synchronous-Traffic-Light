"""A Gymnasium environment for controlling all signals of a scenario together.

One step is ``delta`` simulated seconds. The action picks a green phase for every junction
at once (a joint action, as in the 2021 project); each junction's ``SignalController``
decides whether the switch can happen yet and inserts the yellow.

Three ways to run the signals:

- ``control="agent"``: the action sets the signals (RL agents, max-pressure, random).
- ``control="fixed"``: a fixed-time plan (the network's own, or ``fixed_green`` seconds per
  green); actions are ignored.
- ``control="actuated"``: SUMO's vehicle-actuated control (gap-based); actions are ignored.
"""

from __future__ import annotations

import contextlib
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np

from junctioniq import scenario as scenarios
from junctioniq.sim.metrics import EpisodeMetrics
from junctioniq.sim.signals import (
    SignalController,
    decode_joint,
    green_states,
    joint_action_size,
)
from junctioniq.sim.sumo import backend, sumo_binary

Control = Literal["agent", "fixed", "actuated"]
Reward = Literal["queue", "wait", "speed"]
VEHICLE_SPACE_M = 7.5  # metres of lane per queued car, used to normalise counts


@dataclass(frozen=True)
class Junction:
    id: str
    name: str
    greens: list[str]
    in_lanes: list[str]
    out_lanes: list[str]
    links: list[tuple[str, str]]  # (incoming lane, outgoing lane) per signal index


def read_junctions(net_file: Path, names: dict[str, str]) -> list[Junction]:
    """Read each junction's static program and lanes straight from the network file."""
    root = ET.parse(net_file).getroot()
    programs = {
        tl.get("id"): [p.get("state") for p in tl.iter("phase")] for tl in root.iter("tlLogic")
    }
    links: dict[str, dict[int, tuple[str, str]]] = {}
    for c in root.iter("connection"):
        tl = c.get("tl")
        if tl:
            in_lane = f"{c.get('from')}_{c.get('fromLane')}"
            out_lane = f"{c.get('to')}_{c.get('toLane')}"
            links.setdefault(tl, {})[int(c.get("linkIndex"))] = (in_lane, out_lane)
    junctions = []
    for jid, name in names.items():
        if jid not in programs:
            raise ValueError(f"Junction {jid} has no traffic light in {net_file.name}")
        ordered = [links[jid][i] for i in sorted(links[jid])]
        junctions.append(
            Junction(
                id=jid,
                name=name,
                greens=green_states(programs[jid]),
                in_lanes=list(dict.fromkeys(i for i, _ in ordered)),
                out_lanes=list(dict.fromkeys(o for _, o in ordered)),
                links=ordered,
            )
        )
    return junctions


def signal_programs(
    net_file: Path, junction_ids: list[str], path: Path, kind: str, green: int = 42
) -> Path:
    """An additional file with a SUMO program per junction, using the network's own phases.

    ``kind="static"`` gives a fixed-time plan with ``green`` seconds per green phase;
    ``kind="actuated"`` gives SUMO's gap-based actuated control (greens of 10-60 s).
    Both are loaded under the program ID ``kind``.
    """
    root = ET.parse(net_file).getroot()
    out = ["<additional>"]
    for tl in root.iter("tlLogic"):
        if tl.get("id") not in junction_ids:
            continue
        out.append(f'  <tlLogic id="{tl.get("id")}" type="{kind}" programID="{kind}" offset="0">')
        if kind == "actuated":
            out.append('    <param key="max-gap" value="3.0"/>')
            out.append('    <param key="detector-gap" value="2.0"/>')
        for p in tl.iter("phase"):
            state = p.get("state")
            if "y" in state.lower():
                out.append(f'    <phase duration="3" state="{state}"/>')
            elif kind == "actuated":
                out.append(f'    <phase duration="30" minDur="10" maxDur="60" state="{state}"/>')
            else:
                out.append(f'    <phase duration="{green}" state="{state}"/>')
        out.append("  </tlLogic>")
    out.append("</additional>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


class JunctionEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        scenario: str | Path = "kowdenahalli",
        control: Control = "agent",
        reward: Reward = "queue",
        delta: int = 5,
        yellow_time: int = 3,
        min_green: int = 10,
        horizon: int | None = None,
        drain: int = 0,
        demand_scale: float | tuple[float, float] = 1.0,
        gui: bool = False,
        teleport_after: int = 300,
        fixed_green: int | None = None,
        work_dir: Path | None = None,
    ) -> None:
        super().__init__()
        if delta < 1 or yellow_time < 0 or min_green < 0:
            raise ValueError("delta must be >= 1 and times must not be negative")
        self.scenario = scenarios.load(scenario)
        self.control = control
        self.reward_kind = reward
        self.delta = delta
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.horizon = horizon or self.scenario.duration
        self.drain = drain
        self.demand_scale = demand_scale
        self.gui = gui
        self.teleport_after = teleport_after
        self.fixed_green = fixed_green  # None keeps the network's own fixed-time plan
        self._tmp = None if work_dir else tempfile.TemporaryDirectory(prefix="junctioniq-")
        self.work_dir = Path(work_dir or self._tmp.name)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.junctions = read_junctions(self.scenario.net, self.scenario.junctions)
        self.sizes = [len(j.greens) for j in self.junctions]
        self._in_lanes = list(dict.fromkeys(lane for j in self.junctions for lane in j.in_lanes))
        self._capacity = {}  # filled from the network on first reset

        n_obs = sum(2 * len(j.in_lanes) + len(j.greens) + 1 for j in self.junctions)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(joint_action_size(self.sizes))

        self._sumo = None
        self.controllers: list[SignalController] = []
        self.metrics = EpisodeMetrics()
        self._prev_wait = 0.0
        self.sim_time = 0

    # ----- episode control -------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.close_sim()
        episode_seed = int(self.np_random.integers(0, 2**31 - 1)) if seed is None else seed
        scale = self.demand_scale
        if isinstance(scale, tuple):
            scale = float(self.np_random.uniform(*scale))
        trips = self.work_dir / "trips.rou.xml"
        n_vehicles = self.scenario.write_demand(trips, seed=episode_seed, scale=scale)

        args = [
            sumo_binary("sumo-gui" if self.gui else "sumo"),
            "-n", str(self.scenario.net),
            "-r", str(trips),
            "--seed", str(episode_seed),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--time-to-teleport", str(self.teleport_after),
            "--waiting-time-memory", "10000",
        ]  # fmt: skip
        program = None
        if self.control == "actuated":
            program = "actuated"
        elif self.control == "fixed" and self.fixed_green is not None:
            program = "static"
        if program:
            add = signal_programs(
                self.scenario.net,
                [j.id for j in self.junctions],
                self.work_dir / f"{program}.add.xml",
                program,
                green=self.fixed_green or 42,
            )
            args += ["-a", str(add)]
        if self.gui:
            args += ["--start", "--quit-on-end"]

        self._sumo = backend(self.gui)
        self._sumo.start(args)
        s = self._sumo
        if not self._capacity:
            self._capacity = {
                lane: max(1.0, s.lane.getLength(lane) / VEHICLE_SPACE_M) for lane in self._in_lanes
            }
        if program:
            for j in self.junctions:
                s.trafficlight.setProgram(j.id, program)

        self.controllers = [
            SignalController(j.id, j.greens, self.yellow_time, self.min_green)
            for j in self.junctions
        ]
        if self.control == "agent":
            for c in self.controllers:
                c.time_in_green = self.min_green  # free to switch from the start
                s.trafficlight.setRedYellowGreenState(c.junction_id, c.state())

        self.metrics = EpisodeMetrics()
        self.sim_time = 0
        self._prev_wait = 0.0
        info = {"vehicles_planned": n_vehicles, "demand_scale": scale, "seed": episode_seed}
        return self._observe(), info

    def step(self, action: int | None) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._sumo is None:
            raise RuntimeError("Call reset() before step()")
        s = self._sumo
        self._steps_this_call = 0
        if self.control == "agent":
            if action is None:
                raise ValueError("control='agent' needs an action")
            for c, green in zip(
                self.controllers, decode_joint(int(action), self.sizes), strict=True
            ):
                c.request(green)

        queue_sum = 0.0
        speed_sum = 0.0
        for _ in range(self.delta):
            self._advance_one_second()
            queue_sum += self._junction_queue
            speed_sum += self._mean_speed()
            if self._finished():
                break

        steps = max(1, self._steps_this_call)
        if self.reward_kind == "queue":
            reward = -queue_sum / steps / 10.0
        elif self.reward_kind == "wait":
            wait = sum(s.lane.getWaitingTime(lane) for lane in self._in_lanes)
            reward = -(wait - self._prev_wait) / 100.0
            self._prev_wait = wait
        else:
            reward = speed_sum / steps
        # An empty network is a real end; stopping at the time limit is a truncation,
        # so a learning agent still bootstraps from the last state.
        terminated = self._empty()
        truncated = not terminated and self._out_of_time()
        info = {"time": self.sim_time}
        if terminated or truncated:
            info["metrics"] = self.metrics.summary()
        return self._observe(), float(reward), terminated, truncated, info

    def close_sim(self) -> None:
        if self._sumo is not None:
            with contextlib.suppress(Exception):  # SUMO may already have exited
                self._sumo.close()
            self._sumo = None

    def close(self) -> None:
        self.close_sim()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    # ----- simulation ------------------------------------------------------

    _steps_this_call = 0
    _junction_queue = 0

    def _advance_one_second(self) -> None:
        s = self._sumo
        if self.control == "agent":
            for c in self.controllers:
                before = c.state()
                c.tick()
                after = c.state()
                if after != before:
                    s.trafficlight.setRedYellowGreenState(c.junction_id, after)
        s.simulationStep()
        self.sim_time += 1
        self._steps_this_call += 1

        now = s.simulation.getTime()
        for vid in s.simulation.getDepartedIDList():
            self.metrics.on_depart(vid, now - s.vehicle.getDepartDelay(vid))
        for vid in s.simulation.getArrivedIDList():
            self.metrics.on_arrive(vid, now)
        self.metrics.teleported += s.simulation.getStartingTeleportNumber()
        stopped = sum(s.edge.getLastStepHaltingNumber(e) for e in s.edge.getIDList())
        self._junction_queue = sum(s.lane.getLastStepHaltingNumber(lane) for lane in self._in_lanes)
        self.metrics.on_second(stopped, self._junction_queue)
        self.metrics.switches = sum(c.switches for c in self.controllers)

    def _mean_speed(self) -> float:
        s = self._sumo
        vehicles = s.vehicle.getIDList()
        if not vehicles:
            return 1.0
        return float(
            np.mean([s.vehicle.getSpeed(v) / s.vehicle.getAllowedSpeed(v) for v in vehicles])
        )

    def _empty(self) -> bool:
        """Demand is over and every vehicle has left."""
        return self.sim_time >= self.horizon and self._sumo.simulation.getMinExpectedNumber() == 0

    def _out_of_time(self) -> bool:
        return self.sim_time >= self.horizon + self.drain

    def _finished(self) -> bool:
        return self._empty() or self._out_of_time()

    def _observe(self) -> np.ndarray:
        s = self._sumo
        parts: list[float] = []
        for j, c in zip(self.junctions, self.controllers, strict=True):
            for lane in j.in_lanes:
                cap = self._capacity[lane]
                parts.append(s.lane.getLastStepHaltingNumber(lane) / cap)
                parts.append(s.lane.getLastStepVehicleNumber(lane) / cap)
            phase = np.zeros(len(j.greens))
            if self.control == "agent":
                phase[c.current] = 1.0
            parts.extend(phase)
            parts.append(1.0 if self.control == "agent" and c.can_switch else 0.0)
        return np.clip(np.asarray(parts, dtype=np.float32), 0.0, 1.0)

    # ----- helpers for rule-based controllers ------------------------------

    def pressures(self) -> list[list[float]]:
        """Per junction and green phase: queued vehicles it would serve minus those downstream."""
        s = self._sumo
        halting = {}

        def q(lane: str) -> int:
            if lane not in halting:
                halting[lane] = s.lane.getLastStepHaltingNumber(lane)
            return halting[lane]

        result = []
        for j in self.junctions:
            per_green = []
            for state in j.greens:
                served = {(i, o) for (i, o), ch in zip(j.links, state, strict=True) if ch in "Gg"}
                per_green.append(float(sum(q(i) - q(o) for i, o in served)))
            result.append(per_green)
        return result

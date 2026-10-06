"""Per-episode traffic measures, collected every simulated second."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EpisodeMetrics:
    departed: int = 0
    arrived: int = 0
    teleported: int = 0
    seconds: int = 0
    stopped_vehicle_seconds: float = 0.0  # all vehicles, whole network
    junction_queue_sum: float = 0.0  # halted vehicles on approach lanes, summed per second
    max_junction_queue: int = 0
    travel_time_sum: float = 0.0  # including time spent waiting to enter the network
    switches: int = 0
    queue_trace: list[int] = field(default_factory=list, repr=False)  # per second
    _planned_depart: dict[str, float] = field(default_factory=dict, repr=False)

    def on_depart(self, vehicle_id: str, planned_depart: float) -> None:
        self.departed += 1
        self._planned_depart[vehicle_id] = planned_depart

    def on_arrive(self, vehicle_id: str, now: float) -> None:
        start = self._planned_depart.pop(vehicle_id, None)
        if start is not None:
            self.arrived += 1
            self.travel_time_sum += now - start

    def on_second(self, stopped: int, junction_queue: int) -> None:
        self.seconds += 1
        self.stopped_vehicle_seconds += stopped
        self.junction_queue_sum += junction_queue
        self.max_junction_queue = max(self.max_junction_queue, junction_queue)
        self.queue_trace.append(junction_queue)

    @property
    def in_network(self) -> int:
        return len(self._planned_depart)

    def summary(self) -> dict[str, float]:
        """Headline numbers. Lower is better except throughput."""
        return {
            "vehicles": self.departed,
            "arrived": self.arrived,
            "still_in_network": self.in_network,
            "teleported": self.teleported,
            "mean_travel_time_s": self.travel_time_sum / self.arrived if self.arrived else 0.0,
            "mean_stopped_time_s": (
                self.stopped_vehicle_seconds / self.departed if self.departed else 0.0
            ),
            "mean_junction_queue": (
                self.junction_queue_sum / self.seconds if self.seconds else 0.0
            ),
            "max_junction_queue": self.max_junction_queue,
            "signal_switches": self.switches,
            "simulated_s": self.seconds,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_planned_depart")
        d.pop("queue_trace")
        return d

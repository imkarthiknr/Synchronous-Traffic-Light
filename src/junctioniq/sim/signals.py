"""Safe signal switching: a requested change goes through yellow, and greens last a minimum time.

The 2021 project set the light's phase index directly every two seconds, so a junction
could flip from green to red with no yellow at all. Here every junction keeps a small
state machine that the agent can only *ask* to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def green_states(program_states: list[str]) -> list[str]:
    """The green phases of a SUMO program: states with no yellow and at least one green."""
    greens = [s for s in program_states if "y" not in s.lower() and any(c in "Gg" for c in s)]
    if len(greens) < 2:
        raise ValueError("A controlled junction needs at least two green phases")
    return list(dict.fromkeys(greens))  # keep order, drop duplicates


def yellow_between(current: str, target: str) -> str:
    """Yellow state for the switch: links that lose their green show yellow, others keep theirs."""
    if len(current) != len(target):
        raise ValueError("Signal states differ in length")
    return "".join(
        "y" if c in "Gg" and t in "rs" else c for c, t in zip(current, target, strict=True)
    )


@dataclass
class SignalController:
    """One junction's signal under agent control."""

    junction_id: str
    greens: list[str]
    yellow_time: int = 3
    min_green: int = 10
    current: int = 0
    time_in_green: int = 0
    yellow_left: int = 0
    pending: int | None = None
    switches: int = field(default=0)

    @property
    def n_greens(self) -> int:
        return len(self.greens)

    @property
    def in_yellow(self) -> bool:
        return self.yellow_left > 0

    @property
    def can_switch(self) -> bool:
        return not self.in_yellow and self.time_in_green >= self.min_green

    def state(self) -> str:
        if self.in_yellow and self.pending is not None:
            return yellow_between(self.greens[self.current], self.greens[self.pending])
        return self.greens[self.current]

    def request(self, green: int) -> bool:
        """Ask for a green phase. Returns True if a switch (through yellow) started."""
        if not 0 <= green < self.n_greens:
            raise ValueError(f"{self.junction_id}: no green phase {green}")
        if green == self.current or not self.can_switch:
            return False
        self.pending = green
        self.yellow_left = self.yellow_time
        if self.yellow_time == 0:
            self._finish_switch()
        return True

    def tick(self) -> None:
        """Advance one simulated second."""
        if self.in_yellow:
            self.yellow_left -= 1
            if self.yellow_left == 0:
                self._finish_switch()
        else:
            self.time_in_green += 1

    def _finish_switch(self) -> None:
        assert self.pending is not None
        self.current = self.pending
        self.pending = None
        self.time_in_green = 0
        self.switches += 1


def joint_action_size(sizes: list[int]) -> int:
    n = 1
    for s in sizes:
        n *= s
    return n


def decode_joint(action: int, sizes: list[int]) -> list[int]:
    """Split a joint action into one green phase per junction (mixed radix, first fastest)."""
    if not 0 <= action < joint_action_size(sizes):
        raise ValueError(f"Action {action} out of range")
    out = []
    for s in sizes:
        out.append(action % s)
        action //= s
    return out


def encode_joint(greens: list[int], sizes: list[int]) -> int:
    action, base = 0, 1
    for g, s in zip(greens, sizes, strict=True):
        if not 0 <= g < s:
            raise ValueError(f"Green {g} out of range for a junction with {s} greens")
        action += g * base
        base *= s
    return action

"""Finding SUMO and starting a simulation through libsumo (fast) or TraCI (for the GUI)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import ModuleType


def sumo_home() -> Path:
    """SUMO's install folder: $SUMO_HOME if set, otherwise the eclipse-sumo package."""
    if env := os.environ.get("SUMO_HOME"):
        return Path(env)
    try:
        import sumo  # the eclipse-sumo wheel

        return Path(sumo.SUMO_HOME)
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "SUMO not found. Install the project dependencies (uv sync) or set SUMO_HOME."
        ) from exc


def sumo_binary(name: str = "sumo") -> str:
    """Path to a SUMO tool such as ``sumo``, ``sumo-gui`` or ``netconvert``."""
    candidate = sumo_home() / "bin" / name
    for path in (candidate, candidate.with_suffix(".exe")):
        if path.is_file():
            return str(path)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"Could not find the SUMO program {name!r} under {sumo_home()}")


def backend(gui: bool = False) -> ModuleType:
    """libsumo runs in-process and is several times faster; the GUI needs TraCI."""
    os.environ.setdefault("SUMO_HOME", str(sumo_home()))
    if gui:
        import traci

        return traci
    import libsumo

    return libsumo

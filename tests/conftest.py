from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from among_us_ai.pathfinding import Pathfinder  # noqa: E402


@pytest.fixture(scope="session")
def pathfinder() -> Pathfinder:
    return Pathfinder()


def make_snapshot(**overrides: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "tick": 1,
        "timestamp": 10.0,
        "phase": "tasks",
        "local_player_id": 0,
        "players": [
            {"id": 0, "name": "Red", "color": "red", "position": [52, 9], "alive": True, "role": "crewmate"},
            {"id": 1, "name": "Blue", "color": "blue", "position": [54, 10], "alive": True},
            {"id": 2, "name": "Green", "color": "green", "position": [34, 38], "alive": True},
            {"id": 3, "name": "Pink", "color": "pink", "position": [92, 28], "alive": True},
            {"id": 4, "name": "Lime", "color": "lime", "position": [66, 30], "alive": True},
        ],
        "tasks": [
            {"id": "t1", "name": "Fix Wiring", "position": [34, 36], "category": "common", "steps_total": 3},
            {"id": "t2", "name": "Swipe Card", "position": [68, 32], "category": "common"},
        ],
        "settings": {"num_impostors": 1},
        "emergency_meetings_left": 1,
    }
    snapshot.update(overrides)
    return snapshot

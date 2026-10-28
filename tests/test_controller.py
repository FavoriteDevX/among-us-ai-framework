from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from among_us_ai.agent import AmongUsAgent
from among_us_ai.controller import (
    Chat,
    GameController,
    Kill,
    MinigameRunner,
    MoveTo,
    PanelGeometry,
    RecordingBackend,
    ScreenCalibration,
    Vote,
)
from among_us_ai.game_state import GameState, GameStateParser, Room, TaskCategory, TaskState, Vec2
from among_us_ai.pathfinding import SKELD, Pathfinder

from .conftest import make_snapshot

PARSER = GameStateParser(SKELD.room_at)


class FakeWorld:
    """Moves the local player according to the keys held on a RecordingBackend."""

    def __init__(self, backend: RecordingBackend, snapshot: dict[str, Any], speed: float = 6.0) -> None:
        self.backend = backend
        self.snapshot = snapshot
        self.speed = speed
        self.last_clock = backend.clock
        self.pos = list(snapshot["players"][0]["position"])
        self.cal = ScreenCalibration()

    def _advance(self) -> None:
        dt = self.backend.clock - self.last_clock
        self.last_clock = self.backend.clock
        if dt <= 0:
            return
        keys = self.cal.keys
        dx = (keys["right"] in self.backend.held) - (keys["left"] in self.backend.held)
        dy = (keys["down"] in self.backend.held) - (keys["up"] in self.backend.held)
        norm = math.hypot(dx, dy)
        if norm:
            nx = self.pos[0] + dx / norm * self.speed * dt
            ny = self.pos[1] + dy / norm * self.speed * dt
            if SKELD.room_at(Vec2(nx, ny)) is not None:
                self.pos = [nx, ny]

    def __call__(self) -> dict[str, Any]:
        self._advance()
        snap = copy.deepcopy(self.snapshot)
        snap["timestamp"] = 10.0 + self.backend.clock
        snap["players"][0]["position"] = list(self.pos)
        return snap

    def state(self) -> GameState:
        return PARSER.parse(self())


def _controller(snapshot: dict[str, Any]) -> tuple[GameController, RecordingBackend, FakeWorld]:
    backend = RecordingBackend()
    world = FakeWorld(backend, snapshot)
    controller = GameController(backend, ScreenCalibration(), Pathfinder(), world.state)
    return controller, backend, world


def test_walk_reaches_target_across_rooms() -> None:
    controller, backend, world = _controller(make_snapshot())
    target = Vec2(66, 30)  # admin
    result = controller.execute(MoveTo(target))
    assert result.success, result.detail
    assert Vec2(*world.pos).distance_to(target) < 1.5
    assert not backend.held
    assert backend.clock < 30


def test_kill_requires_impostor_and_cooldown() -> None:
    snap = make_snapshot()
    controller, backend, _ = _controller(snap)
    assert not controller.execute(Kill(1)).success

    snap = make_snapshot(kill_cooldown=0.0)
    snap["players"][0]["role"] = "impostor"
    controller, backend, _ = _controller(snap)
    result = controller.execute(Kill(1))
    assert result.success, result.detail
    assert [e.args for e in backend.of_kind("press")] == [("q",)]

    snap["kill_cooldown"] = 5.0
    controller, backend, _ = _controller(snap)
    assert "cooldown" in controller.execute(Kill(1)).detail


def test_vote_and_chat_inputs() -> None:
    snap = make_snapshot(phase="voting", meeting={"caller_id": 1, "votes": {}, "chat": []})
    controller, backend, _ = _controller(snap)
    assert controller.execute(Vote(3)).success
    clicks = backend.of_kind("click")
    assert len(clicks) == 2 and clicks[0].args == ScreenCalibration().vote_slot(3)
    assert controller.execute(Vote(None)).success
    assert controller.execute(Chat("  Blue   vented  ")).success
    assert backend.of_kind("type")[-1].args == ("Blue vented",)


def test_wiring_matches_colours() -> None:
    cal = ScreenCalibration(width=1000, height=1000)
    geo = PanelGeometry()
    palette = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    right_order = [2, 0, 3, 1]

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        for i, (px, py) in enumerate(geo.wire_left):
            if abs(x - px * 1000) <= 3 and abs(y - py * 1000) <= 3:
                return palette[i]
        for j, (px, py) in enumerate(geo.wire_right):
            if abs(x - px * 1000) <= 3 and abs(y - py * 1000) <= 3:
                return palette[right_order[j]]
        return (0, 0, 0)

    backend = RecordingBackend(pixel)
    runner = MinigameRunner(backend, cal, geo)
    task = TaskState("w", "Fix Wiring", Room.ELECTRICAL, Vec2(34, 36), TaskCategory.COMMON, 3, 0)
    runner.run(task)
    downs = [i for i, e in enumerate(backend.events) if e.kind == "mouse_down"]
    starts = [backend.events[i - 1].args for i in downs]
    ups = [i for i, e in enumerate(backend.events) if e.kind == "mouse_up"]
    ends = [backend.events[i - 1].args for i in ups]
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        assert start == cal.norm(geo.wire_left[i])
        assert end == cal.norm(geo.wire_right[right_order.index(i)])


def test_prime_shields_clicks_only_red() -> None:
    cal = ScreenCalibration(width=1000, height=1000)
    geo = PanelGeometry()
    red = {0, 3, 6}

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        for i, (px, py) in enumerate(geo.shield_hexes):
            if abs(x - px * 1000) <= 3 and abs(y - py * 1000) <= 3:
                return (202, 40, 40) if i in red else (240, 240, 240)
        return (0, 0, 0)

    backend = RecordingBackend(pixel)
    MinigameRunner(backend, cal, geo).prime_shields(TaskState("s", "Prime Shields", Room.SHIELDS, Vec2(80, 48), TaskCategory.SHORT, 1, 0))
    assert [e.args for e in backend.of_kind("click")] == [cal.norm(geo.shield_hexes[i]) for i in sorted(red)]


def test_calibration_round_trip(tmp_path) -> None:
    cal = ScreenCalibration(width=2560, height=1440, pixels_per_unit=70.0)
    cal.buttons["kill"] = (0.5, 0.5)
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = ScreenCalibration.load(path)
    assert loaded.width == 2560 and loaded.buttons["kill"] == (0.5, 0.5)
    assert loaded.world_to_screen(Vec2(11, 10), Vec2(10, 10)) == (1280 + 70, 720)


def test_agent_reports_nearby_body() -> None:
    snap = make_snapshot()
    snap["players"][2].update(alive=False, position=[53, 12])
    snap["bodies"] = [{"player_id": 2, "position": [53, 12]}]
    backend = RecordingBackend()
    world = FakeWorld(backend, snap)
    agent = AmongUsAgent(world, lambda source: GameController(backend, ScreenCalibration(), Pathfinder(), source))
    result = agent.step()
    assert result is not None and result.success, result
    assert ("r",) in [e.args for e in backend.of_kind("press")]


def test_unknown_task_solver_is_reported() -> None:
    runner = MinigameRunner(RecordingBackend(), ScreenCalibration())
    with pytest.raises(Exception, match="no solver"):
        runner.run(TaskState("x", "Juggle", Room.ADMIN, Vec2(66, 30), TaskCategory.SHORT, 1, 0))

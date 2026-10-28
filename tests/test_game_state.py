from __future__ import annotations

import copy

import pytest

from among_us_ai.game_state import (
    Animation,
    EventType,
    GamePhase,
    GameStateParser,
    GameStateTracker,
    Room,
    StateParseError,
    Vec2,
)
from among_us_ai.pathfinding import SKELD

from .conftest import make_snapshot


@pytest.fixture()
def parser() -> GameStateParser:
    return GameStateParser(SKELD.room_at)


def test_parse_valid_snapshot(parser: GameStateParser) -> None:
    state = parser.parse(make_snapshot())
    assert state.phase is GamePhase.TASKS
    assert state.me.name == "Red"
    assert state.me.room is Room.CAFETERIA
    assert state.players[2].room is Room.ELECTRICAL
    assert state.tasks[0].room is Room.ELECTRICAL
    assert state.settings.num_impostors == 1
    assert len(state.alive_others()) == 4
    assert state.player_by_name("pink").player_id == 3


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda s: s.pop("players"), "snapshot.players"),
        (lambda s: s.update(local_player_id=99), "local_player_id"),
        (lambda s: s["players"][1].update(color="teal"), "players[1].color"),
        (lambda s: s["players"][1].update(position=[1]), "players[1].position"),
        (lambda s: s["players"][1].update(alive="yes"), "players[1].alive"),
        (lambda s: s.update(task_bar=1.5), "task_bar"),
        (lambda s: s["tasks"][0].update(steps_done=5), "tasks[0].steps_done"),
        (lambda s: s["players"].append(dict(s["players"][0])), "players[5].id"),
    ],
)
def test_parse_rejects_invalid(parser: GameStateParser, mutate, path: str) -> None:
    snapshot = make_snapshot()
    mutate(snapshot)
    with pytest.raises(StateParseError) as info:
        parser.parse(snapshot)
    assert info.value.path == path


def test_vec2_math() -> None:
    a, b = Vec2(0, 0), Vec2(3, 4)
    assert a.distance_to(b) == 5
    assert (b - a).normalized() == Vec2(0.6, 0.8)
    assert Vec2(0, 0).normalized() == Vec2(0, 0)


def test_tracker_emits_semantic_events(parser: GameStateParser) -> None:
    tracker = GameStateTracker()
    first = make_snapshot()
    tracker.update(parser.parse(first))

    second = copy.deepcopy(first)
    second.update(tick=2, timestamp=11.0)
    second["players"][1].update(animation="kill")
    second["players"][2].update(alive=False, animation="ghost")
    second["bodies"] = [{"player_id": 2, "position": [54, 11]}]
    second["players"][3].update(animation="visual_task")
    second["players"][4].update(animation="vent_enter")
    second["sabotage"] = {"type": "reactor", "time_remaining": 30, "fix_positions": [[1, 24], [1, 36]]}
    events = tracker.update(parser.parse(second))
    kinds = {e.kind for e in events}
    assert {
        EventType.KILL_WITNESSED,
        EventType.PLAYER_DIED,
        EventType.BODY_DISCOVERED,
        EventType.VISUAL_TASK_WITNESSED,
        EventType.VENT_WITNESSED,
        EventType.SABOTAGE_STARTED,
    } <= kinds
    kill = next(e for e in events if e.kind is EventType.KILL_WITNESSED)
    assert kill.subject_id == 1 and kill.details["victim_id"] == 2
    assert tracker.death_time(2) == 11.0


def test_tracker_sightings_and_queries(parser: GameStateParser) -> None:
    tracker = GameStateTracker(sighting_interval=1.0)
    for i in range(5):
        snap = make_snapshot(tick=i, timestamp=float(i))
        snap["players"][1]["position"] = [54 + i, 10]
        tracker.update(parser.parse(snap))
    assert len(tracker.sightings(1)) == 5
    assert tracker.last_seen(1).position == Vec2(58, 10)
    near = tracker.players_seen_near(Vec2(55, 10), 2.0, 0.0, 4.0)
    assert 1 in near and near[1] == 0.0
    assert tracker.players_seen_in_room(Room.ELECTRICAL, 0.0, 4.0) == {2}
    assert tracker.location_at(1, 2.2).timestamp == 2.0


def test_hidden_players_are_not_sighted(parser: GameStateParser) -> None:
    tracker = GameStateTracker()
    snap = make_snapshot()
    snap["players"][3]["visible"] = False
    tracker.update(parser.parse(snap))
    assert tracker.last_seen(3) is None
    assert tracker.current.me.animation is Animation.IDLE

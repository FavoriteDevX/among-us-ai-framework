from __future__ import annotations

import math

import pytest

from among_us_ai.game_state import Room, Vec2
from among_us_ai.pathfinding import SKELD, NoPathError, Pathfinder, StepKind


def _assert_walkable(pf: Pathfinder, a: Vec2, b: Vec2) -> None:
    steps = max(2, int(a.distance_to(b) / (pf.grid.cell_size / 2)))
    for i in range(steps + 1):
        t = i / steps
        p = Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        assert pf.grid.is_walkable(pf.grid.world_to_cell(p)), f"segment {a}->{b} leaves the map at {p}"


def test_every_room_is_reachable(pathfinder: Pathfinder) -> None:
    matrix = pathfinder.room_distance_matrix()
    assert all(math.isfinite(v) for v in matrix.values())
    assert matrix[(Room.CAFETERIA, Room.CAFETERIA)] == 0


def test_room_lookup() -> None:
    assert SKELD.room_at(Vec2(52, 12)) is Room.CAFETERIA
    assert SKELD.room_at(Vec2(30, 10)) is Room.HALLWAY
    assert SKELD.room_at(Vec2(99, 1)) is None


def test_static_assets_are_walkable(pathfinder: Pathfinder) -> None:
    for station in SKELD.task_stations:
        assert pathfinder.grid.is_walkable(pathfinder.grid.world_to_cell(station.position)), station
    for fixes in SKELD.sabotage_fixes.values():
        for p in fixes:
            assert pathfinder.grid.is_walkable(pathfinder.grid.world_to_cell(p))


def test_path_segments_stay_on_map(pathfinder: Pathfinder) -> None:
    start = pathfinder.room_center(Room.REACTOR)
    goal = pathfinder.room_center(Room.NAVIGATION)
    path = pathfinder.find_path(start, goal)
    points = [start] + [w.position for w in path]
    for a, b in zip(points, points[1:], strict=False):
        _assert_walkable(pathfinder, a, b)
    assert path.waypoints[-1].position == goal
    assert path.distance >= start.distance_to(goal)
    assert path.eta(6.0) == pytest.approx(path.distance / 6.0)


def test_vents_shorten_impostor_routes(pathfinder: Pathfinder) -> None:
    start = pathfinder.room_center(Room.MEDBAY)
    goal = pathfinder.room_center(Room.ELECTRICAL)
    walk = pathfinder.find_path(start, goal)
    vent = pathfinder.find_path(start, goal, allow_vents=True)
    assert vent.vents_used == ("electrical",)
    assert vent.cost < walk.cost
    assert any(w.kind is StepKind.VENT for w in vent)


def test_closed_doors_force_detour_or_block(pathfinder: Pathfinder) -> None:
    admin = pathfinder.room_center(Room.ADMIN)
    upper = pathfinder.room_center(Room.UPPER_ENGINE)
    open_path = pathfinder.find_path(admin, upper)
    detour = pathfinder.find_path(admin, upper, closed_doors=[Room.CAFETERIA])
    assert detour.distance > open_path.distance
    with pytest.raises(NoPathError):
        pathfinder.find_path(pathfinder.room_center(Room.CAFETERIA), upper, closed_doors=[Room.CAFETERIA])


def test_dynamic_obstacles_expire() -> None:
    pf = Pathfinder()
    start = pf.room_center(Room.MEDBAY)
    goal = pf.room_center(Room.UPPER_ENGINE)
    baseline = pf.find_path(start, goal)
    hallway_cells = [(c, r) for c in range(int(22 / 0.5), int(24 / 0.5)) for r in range(int(8 / 0.5), int(12 / 0.5))]
    pf.obstacles.block_cells(hallway_cells, expires_at=100.0)
    detour = pf.find_path(start, goal, now=50.0)
    assert detour.distance > baseline.distance + 20
    restored = pf.find_path(start, goal, now=150.0)
    assert restored.distance == pytest.approx(baseline.distance)


def test_soft_cost_fields_bend_routes() -> None:
    pf = Pathfinder()
    start, goal = Vec2(44, 12), Vec2(60, 12)
    straight = pf.find_path(start, goal)
    pf.obstacles.set_cost_field("crowd", Vec2(52, 12), 4.0, 30.0)
    bent = pf.find_path(start, goal)
    assert bent.distance > straight.distance
    assert min(w.position.distance_to(Vec2(52, 12)) for w in bent) > 1.0


def test_distance_field_matches_astar(pathfinder: Pathfinder) -> None:
    source = pathfinder.room_center(Room.STORAGE)
    target = pathfinder.room_center(Room.SHIELDS)
    field = pathfinder.distance_field(source)
    dijkstra = field[pathfinder.grid.world_to_cell(target)]
    astar = pathfinder.find_path(source, target, smooth=False)
    assert dijkstra <= astar.distance + 1.0


def test_rooms_only_connect_through_hallways() -> None:
    rooms = list(SKELD.rooms.items())
    for i, (a, ra) in enumerate(rooms):
        for b, rb in rooms[i + 1 :]:
            touching = ra.x0 <= rb.x1 and rb.x0 <= ra.x1 and ra.y0 <= rb.y1 and rb.y0 <= ra.y1
            assert not touching, f"{a.value} and {b.value} overlap or share an edge"

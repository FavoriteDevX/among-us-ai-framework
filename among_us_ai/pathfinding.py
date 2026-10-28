"""A* navigation for The Skeld.

The map is modelled as a union of axis-aligned rectangles (rooms and
hallways) in a normalised world coordinate system (x grows right, y grows
down, 100 x 60 units). The layout mirrors The Skeld's topology: room
adjacency, hallways, door-equipped rooms and the four vent networks. The
absolute coordinates are an abstraction; the screen controller converts
them with a calibrated affine transform.

Features:
    * 8-connected grid A* with an octile heuristic and no corner cutting.
    * Wall-clearance costs so routes stay centred in corridors.
    * Vent teleport edges for impostors with an admissible vent-aware heuristic.
    * Door sabotage (closed doors block the room's threshold cells).
    * Dynamic obstacles: timed hard blocks and soft cost fields (crowds,
      danger zones, players to avoid).
    * Line-of-sight path smoothing and ETA estimation.
    * Single-source distance fields for reachability analysis.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from .game_state import Room, Vec2

Cell = tuple[int, int]

SQRT2: Final[float] = math.sqrt(2.0)
NEIGHBOURS: Final[tuple[tuple[int, int, float], ...]] = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
)


class NoPathError(RuntimeError):
    """Raised when no route exists between two points."""


@dataclass(frozen=True, slots=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, p: Vec2) -> bool:
        return self.x0 <= p.x <= self.x1 and self.y0 <= p.y <= self.y1

    @property
    def center(self) -> Vec2:
        return Vec2((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True, slots=True)
class Vent:
    name: str
    room: Room
    position: Vec2
    connections: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskStation:
    name: str
    room: Room
    position: Vec2
    visual: bool = False


@dataclass(frozen=True, slots=True)
class MapDefinition:
    name: str
    width: float
    height: float
    rooms: Mapping[Room, Rect]
    hallways: tuple[Rect, ...]
    door_rooms: frozenset[Room]
    vents: Mapping[str, Vent]
    task_stations: tuple[TaskStation, ...]
    sabotage_fixes: Mapping[str, tuple[Vec2, ...]]
    emergency_button: Vec2
    spawn: Vec2
    admin_table: Vec2
    security_cameras: Vec2

    def room_at(self, position: Vec2) -> Room | None:
        best: tuple[float, Room] | None = None
        for room, rect in self.rooms.items():
            if rect.contains(position) and (best is None or rect.area < best[0]):
                best = (rect.area, room)
        if best is not None:
            return best[1]
        if any(h.contains(position) for h in self.hallways):
            return Room.HALLWAY
        return None


def _skeld() -> MapDefinition:
    rooms = {
        Room.UPPER_ENGINE: Rect(4, 4, 18, 18),
        Room.REACTOR: Rect(0, 22, 12, 38),
        Room.SECURITY: Rect(20, 24, 28, 36),
        Room.LOWER_ENGINE: Rect(4, 42, 18, 56),
        Room.MEDBAY: Rect(30, 14, 40, 26),
        Room.ELECTRICAL: Rect(30, 34, 44, 44),
        Room.CAFETERIA: Rect(42, 2, 62, 22),
        Room.STORAGE: Rect(46, 38, 60, 56),
        Room.ADMIN: Rect(62, 26, 72, 36),
        Room.COMMUNICATIONS: Rect(62, 50, 72, 58),
        Room.WEAPONS: Rect(70, 2, 80, 12),
        Room.O2: Rect(66, 16, 74, 22),
        Room.NAVIGATION: Rect(88, 22, 98, 34),
        Room.SHIELDS: Rect(74, 42, 86, 52),
    }
    hallways = (
        Rect(18, 8, 42, 12),  # upper engine <-> cafeteria
        Rect(32, 12, 36, 14),  # medbay threshold
        Rect(14, 18, 18, 42),  # engine spine
        Rect(12, 28, 14, 32),  # reactor threshold
        Rect(18, 28, 20, 32),  # security threshold
        Rect(18, 46, 46, 50),  # lower engine <-> storage
        Rect(36, 44, 40, 46),  # electrical threshold
        Rect(50, 22, 54, 38),  # cafeteria <-> storage
        Rect(54, 29, 62, 33),  # admin hallway
        Rect(62, 6, 70, 10),  # cafeteria <-> weapons
        Rect(76, 12, 80, 42),  # weapons <-> shields spine
        Rect(74, 18, 76, 21),  # o2 threshold
        Rect(80, 26, 88, 30),  # navigation hallway
        Rect(60, 52, 62, 55),  # storage <-> communications
        Rect(72, 50, 74, 52),  # communications <-> shields
    )
    vent_specs = (
        ("upper_engine", Room.UPPER_ENGINE, Vec2(7, 15), ("reactor_top",)),
        ("reactor_top", Room.REACTOR, Vec2(3, 25), ("upper_engine",)),
        ("reactor_bottom", Room.REACTOR, Vec2(3, 35), ("lower_engine",)),
        ("lower_engine", Room.LOWER_ENGINE, Vec2(7, 45), ("reactor_bottom",)),
        ("medbay", Room.MEDBAY, Vec2(38, 24), ("security", "electrical")),
        ("security", Room.SECURITY, Vec2(26, 26), ("medbay", "electrical")),
        ("electrical", Room.ELECTRICAL, Vec2(32, 42), ("medbay", "security")),
        ("cafeteria", Room.CAFETERIA, Vec2(60, 4), ("admin", "east_hallway")),
        ("admin", Room.ADMIN, Vec2(70, 34), ("cafeteria", "east_hallway")),
        ("east_hallway", Room.HALLWAY, Vec2(78, 34), ("cafeteria", "admin")),
        ("navigation_top", Room.NAVIGATION, Vec2(90, 24), ("weapons",)),
        ("weapons", Room.WEAPONS, Vec2(78, 4), ("navigation_top",)),
        ("navigation_bottom", Room.NAVIGATION, Vec2(90, 32), ("shields",)),
        ("shields", Room.SHIELDS, Vec2(84, 50), ("navigation_bottom",)),
    )
    vents = {name: Vent(name, room, pos, links) for name, room, pos, links in vent_specs}
    stations = (
        TaskStation("Fix Wiring", Room.ELECTRICAL, Vec2(34, 36)),
        TaskStation("Fix Wiring", Room.CAFETERIA, Vec2(44, 4)),
        TaskStation("Fix Wiring", Room.ADMIN, Vec2(64, 28)),
        TaskStation("Fix Wiring", Room.NAVIGATION, Vec2(90, 23)),
        TaskStation("Fix Wiring", Room.STORAGE, Vec2(48, 40)),
        TaskStation("Fix Wiring", Room.SECURITY, Vec2(22, 25)),
        TaskStation("Swipe Card", Room.ADMIN, Vec2(68, 32)),
        TaskStation("Upload Data", Room.ADMIN, Vec2(66, 34)),
        TaskStation("Download Data", Room.CAFETERIA, Vec2(58, 3)),
        TaskStation("Download Data", Room.WEAPONS, Vec2(72, 3)),
        TaskStation("Download Data", Room.NAVIGATION, Vec2(94, 33)),
        TaskStation("Download Data", Room.COMMUNICATIONS, Vec2(66, 56)),
        TaskStation("Download Data", Room.ELECTRICAL, Vec2(40, 36)),
        TaskStation("Fuel Engines", Room.STORAGE, Vec2(50, 52)),
        TaskStation("Fuel Engines", Room.UPPER_ENGINE, Vec2(10, 16)),
        TaskStation("Fuel Engines", Room.LOWER_ENGINE, Vec2(10, 54)),
        TaskStation("Align Engine Output", Room.UPPER_ENGINE, Vec2(12, 6)),
        TaskStation("Align Engine Output", Room.LOWER_ENGINE, Vec2(12, 44)),
        TaskStation("Submit Scan", Room.MEDBAY, Vec2(34, 22), visual=True),
        TaskStation("Inspect Sample", Room.MEDBAY, Vec2(31, 16)),
        TaskStation("Clear Asteroids", Room.WEAPONS, Vec2(75, 7), visual=True),
        TaskStation("Empty Garbage", Room.CAFETERIA, Vec2(60, 20), visual=True),
        TaskStation("Empty Garbage", Room.O2, Vec2(68, 21)),
        TaskStation("Empty Garbage", Room.STORAGE, Vec2(54, 55), visual=True),
        TaskStation("Prime Shields", Room.SHIELDS, Vec2(80, 48), visual=True),
        TaskStation("Calibrate Distributor", Room.ELECTRICAL, Vec2(42, 38)),
        TaskStation("Divert Power", Room.ELECTRICAL, Vec2(38, 35)),
        TaskStation("Accept Diverted Power", Room.SHIELDS, Vec2(76, 44)),
        TaskStation("Accept Diverted Power", Room.WEAPONS, Vec2(78, 10)),
        TaskStation("Accept Diverted Power", Room.NAVIGATION, Vec2(96, 26)),
        TaskStation("Start Reactor", Room.REACTOR, Vec2(2, 30)),
        TaskStation("Unlock Manifolds", Room.REACTOR, Vec2(6, 23)),
        TaskStation("Stabilize Steering", Room.NAVIGATION, Vec2(97, 28)),
        TaskStation("Chart Course", Room.NAVIGATION, Vec2(92, 30)),
        TaskStation("Clean O2 Filter", Room.O2, Vec2(70, 18)),
    )
    return MapDefinition(
        name="The Skeld",
        width=100.0,
        height=60.0,
        rooms=rooms,
        hallways=hallways,
        door_rooms=frozenset(
            {
                Room.CAFETERIA,
                Room.MEDBAY,
                Room.SECURITY,
                Room.ELECTRICAL,
                Room.STORAGE,
                Room.UPPER_ENGINE,
                Room.LOWER_ENGINE,
            }
        ),
        vents=vents,
        task_stations=stations,
        sabotage_fixes={
            "reactor": (Vec2(1, 24), Vec2(1, 36)),
            "oxygen": (Vec2(72, 17), Vec2(64, 35)),
            "lights": (Vec2(43, 37),),
            "communications": (Vec2(70, 52),),
        },
        emergency_button=Vec2(52, 12),
        spawn=Vec2(52, 9),
        admin_table=Vec2(67, 30),
        security_cameras=Vec2(24, 30),
    )


SKELD: Final[MapDefinition] = _skeld()


# --------------------------------------------------------------------------- #
# Dynamic obstacles
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _CostField:
    center: Vec2
    radius: float
    cost: float
    expires_at: float | None


@dataclass(slots=True)
class DynamicObstacles:
    """Time-limited hard blocks and soft cost fields layered over the static grid."""

    _blocked: dict[Cell, float | None] = field(default_factory=dict)
    _fields: dict[str, _CostField] = field(default_factory=dict)

    def block_cells(self, cells: Iterable[Cell], expires_at: float | None = None) -> None:
        for cell in cells:
            self._blocked[cell] = expires_at

    def unblock_cells(self, cells: Iterable[Cell]) -> None:
        for cell in cells:
            self._blocked.pop(cell, None)

    def set_cost_field(self, key: str, center: Vec2, radius: float, cost: float, expires_at: float | None = None) -> None:
        if radius <= 0 or cost < 0:
            raise ValueError("cost field needs positive radius and non-negative cost")
        self._fields[key] = _CostField(center, radius, cost, expires_at)

    def remove_cost_field(self, key: str) -> None:
        self._fields.pop(key, None)

    def clear(self) -> None:
        self._blocked.clear()
        self._fields.clear()

    def prune(self, now: float) -> None:
        for cell in [c for c, t in self._blocked.items() if t is not None and t <= now]:
            del self._blocked[cell]
        for key in [k for k, f in self._fields.items() if f.expires_at is not None and f.expires_at <= now]:
            del self._fields[key]

    def is_blocked(self, cell: Cell) -> bool:
        return cell in self._blocked

    def extra_cost(self, world: Vec2) -> float:
        total = 0.0
        for cost_field in self._fields.values():
            d = world.distance_to(cost_field.center)
            if d < cost_field.radius:
                total += cost_field.cost * (1.0 - d / cost_field.radius)
        return total

    @property
    def has_fields(self) -> bool:
        return bool(self._fields)


# --------------------------------------------------------------------------- #
# Path result
# --------------------------------------------------------------------------- #


class StepKind(str, Enum):
    WALK = "walk"
    VENT = "vent"


@dataclass(frozen=True, slots=True)
class Waypoint:
    position: Vec2
    kind: StepKind = StepKind.WALK
    vent: str | None = None


@dataclass(frozen=True, slots=True)
class Path:
    waypoints: tuple[Waypoint, ...]
    cost: float
    distance: float
    vents_used: tuple[str, ...]

    def eta(self, speed: float, vent_hop_seconds: float = 0.8) -> float:
        if speed <= 0:
            raise ValueError("speed must be positive")
        return self.distance / speed + vent_hop_seconds * len(self.vents_used)

    def __iter__(self) -> Iterator[Waypoint]:
        return iter(self.waypoints)

    def __len__(self) -> int:
        return len(self.waypoints)


# --------------------------------------------------------------------------- #
# Grid & A*
# --------------------------------------------------------------------------- #


class NavGrid:
    """Rasterised walkable grid with door and clearance metadata."""

    def __init__(self, definition: MapDefinition = SKELD, cell_size: float = 0.5, wall_penalty: float = 0.6) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.definition = definition
        self.cell_size = cell_size
        self.cols = int(math.ceil(definition.width / cell_size))
        self.rows = int(math.ceil(definition.height / cell_size))
        shapes = list(definition.rooms.values()) + list(definition.hallways)
        self._walkable: list[bytearray] = []
        for row in range(self.rows):
            line = bytearray(self.cols)
            for col in range(self.cols):
                center = self.cell_center((col, row))
                line[col] = 1 if any(rect.contains(center) for rect in shapes) else 0
            self._walkable.append(line)
        self._clearance = self._compute_clearance()
        self._step_cost = self._compute_step_costs(wall_penalty)
        self.door_cells = self._compute_door_cells()
        self.vent_cells: dict[str, Cell] = {name: self.world_to_cell(v.position) for name, v in definition.vents.items()}
        for name, cell in self.vent_cells.items():
            if not self.is_walkable(cell):
                raise ValueError(f"vent {name} is not on a walkable cell")

    # -- coordinates -------------------------------------------------------- #

    def world_to_cell(self, p: Vec2) -> Cell:
        col = min(self.cols - 1, max(0, int(p.x / self.cell_size)))
        row = min(self.rows - 1, max(0, int(p.y / self.cell_size)))
        return (col, row)

    def cell_center(self, cell: Cell) -> Vec2:
        return Vec2((cell[0] + 0.5) * self.cell_size, (cell[1] + 0.5) * self.cell_size)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows

    def is_walkable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self._walkable[cell[1]][cell[0]] == 1

    def clearance(self, cell: Cell) -> int:
        return self._clearance[cell[1]][cell[0]]

    def step_multiplier(self, cell: Cell) -> float:
        return self._step_cost[cell[1]][cell[0]]

    # -- precomputation ----------------------------------------------------- #

    def _compute_clearance(self) -> list[list[int]]:
        inf = self.cols + self.rows
        dist = [[inf] * self.cols for _ in range(self.rows)]
        queue: deque[Cell] = deque()
        for row in range(self.rows):
            for col in range(self.cols):
                if not self._walkable[row][col]:
                    dist[row][col] = 0
                    queue.append((col, row))
        for col in range(self.cols):
            for row in (-1, self.rows):
                queue.append((col, row))
        for row in range(self.rows):
            for col in (-1, self.cols):
                queue.append((col, row))
        while queue:
            col, row = queue.popleft()
            base = 0 if not (0 <= col < self.cols and 0 <= row < self.rows) else dist[row][col]
            for dx, dy, _ in NEIGHBOURS[:4]:
                nc, nr = col + dx, row + dy
                if 0 <= nc < self.cols and 0 <= nr < self.rows and dist[nr][nc] > base + 1:
                    dist[nr][nc] = base + 1
                    queue.append((nc, nr))
        return dist

    def _compute_step_costs(self, wall_penalty: float) -> list[list[float]]:
        costs: list[list[float]] = []
        for row in range(self.rows):
            line: list[float] = []
            for col in range(self.cols):
                c = self._clearance[row][col]
                line.append(1.0 + (wall_penalty / c if 0 < c <= 3 else 0.0))
            costs.append(line)
        return costs

    def _compute_door_cells(self) -> dict[Room, frozenset[Cell]]:
        result: dict[Room, frozenset[Cell]] = {}
        for room in self.definition.door_rooms:
            rect = self.definition.rooms[room]
            cells: set[Cell] = set()
            for cell in self._cells_in_rect(rect):
                if not self.is_walkable(cell):
                    continue
                for dx, dy, _ in NEIGHBOURS[:4]:
                    other = (cell[0] + dx, cell[1] + dy)
                    if self.is_walkable(other) and not rect.contains(self.cell_center(other)):
                        cells.add(cell)
                        break
            result[room] = frozenset(cells)
        return result

    def _cells_in_rect(self, rect: Rect) -> Iterator[Cell]:
        c0, r0 = self.world_to_cell(Vec2(rect.x0, rect.y0))
        c1, r1 = self.world_to_cell(Vec2(rect.x1, rect.y1))
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                yield (col, row)

    def nearest_walkable(self, p: Vec2, blocked: frozenset[Cell] | set[Cell] = frozenset()) -> Cell:
        start = self.world_to_cell(p)
        if self.is_walkable(start) and start not in blocked:
            return start
        seen = {start}
        queue: deque[Cell] = deque([start])
        while queue:
            cell = queue.popleft()
            for dx, dy, _ in NEIGHBOURS:
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in seen or not self.in_bounds(nxt):
                    continue
                if self.is_walkable(nxt) and nxt not in blocked:
                    return nxt
                seen.add(nxt)
                queue.append(nxt)
        raise NoPathError(f"no walkable cell near {p}")


class Pathfinder:
    """A* planner over a :class:`NavGrid` with vents and dynamic obstacles."""

    def __init__(
        self,
        grid: NavGrid | None = None,
        *,
        vent_cost: float = 3.0,
        max_expansions: int = 200_000,
    ) -> None:
        self.grid = grid or NavGrid()
        self.obstacles = DynamicObstacles()
        self.vent_cost = vent_cost
        self.max_expansions = max_expansions
        self._field_cache: dict[Cell, dict[Cell, float]] = {}
        self._field_cache_size = 256
        self._vent_adjacency: dict[Cell, list[tuple[Cell, str]]] = {}
        for name, vent in self.grid.definition.vents.items():
            here = self.grid.vent_cells[name]
            for target in vent.connections:
                self._vent_adjacency.setdefault(here, []).append((self.grid.vent_cells[target], target))

    @property
    def definition(self) -> MapDefinition:
        return self.grid.definition

    # -- helpers ------------------------------------------------------------ #

    def _closed_door_cells(self, closed_doors: Iterable[Room]) -> set[Cell]:
        cells: set[Cell] = set()
        for room in closed_doors:
            cells |= self.grid.door_cells.get(room, frozenset())
        return cells

    def _passable(self, cell: Cell, door_block: set[Cell]) -> bool:
        return self.grid.is_walkable(cell) and cell not in door_block and not self.obstacles.is_blocked(cell)

    def _octile(self, a: Cell, b: Cell) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    def _vent_cost_cells(self) -> float:
        return self.vent_cost / self.grid.cell_size

    # -- planning ----------------------------------------------------------- #

    def find_path(
        self,
        start: Vec2,
        goal: Vec2,
        *,
        allow_vents: bool = False,
        closed_doors: Iterable[Room] = (),
        now: float | None = None,
        smooth: bool = True,
    ) -> Path:
        if now is not None:
            self.obstacles.prune(now)
        door_block = self._closed_door_cells(closed_doors)
        start_cell = self.grid.nearest_walkable(start)
        goal_cell = self.grid.nearest_walkable(goal, door_block)
        if start_cell in door_block:
            door_block.discard(start_cell)
        if start_cell == goal_cell:
            return Path((Waypoint(goal),), 0.0, start.distance_to(goal), ())

        vent_cells = list(self._vent_adjacency) if allow_vents else []
        vent_hop = self._vent_cost_cells()
        goal_to_nearest_vent = min((self._octile(v, goal_cell) for v in vent_cells), default=math.inf)

        def heuristic(cell: Cell) -> float:
            direct = self._octile(cell, goal_cell)
            if not vent_cells:
                return direct
            via_vent = min(self._octile(cell, v) for v in vent_cells) + vent_hop + goal_to_nearest_vent
            return min(direct, via_vent)

        use_fields = self.obstacles.has_fields
        counter = itertools.count()
        open_heap: list[tuple[float, int, Cell]] = [(heuristic(start_cell), next(counter), start_cell)]
        g_score: dict[Cell, float] = {start_cell: 0.0}
        came_from: dict[Cell, tuple[Cell, str | None]] = {}
        closed: set[Cell] = set()
        expansions = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_cell:
                return self._build_path(came_from, start, goal, start_cell, goal_cell, g_score[current], door_block, smooth)
            closed.add(current)
            expansions += 1
            if expansions > self.max_expansions:
                raise NoPathError("search budget exhausted")
            base = g_score[current]
            cx, cy = current
            for dx, dy, step in NEIGHBOURS:
                nxt = (cx + dx, cy + dy)
                if nxt in closed or not self._passable(nxt, door_block):
                    continue
                if dx != 0 and dy != 0:
                    if not self._passable((cx + dx, cy), door_block) or not self._passable((cx, cy + dy), door_block):
                        continue
                cost = step * self.grid.step_multiplier(nxt)
                if use_fields:
                    cost += self.obstacles.extra_cost(self.grid.cell_center(nxt))
                tentative = base + cost
                if tentative < g_score.get(nxt, math.inf):
                    g_score[nxt] = tentative
                    came_from[nxt] = (current, None)
                    heapq.heappush(open_heap, (tentative + heuristic(nxt), next(counter), nxt))
            if allow_vents:
                for target, vent_name in self._vent_adjacency.get(current, ()):
                    if target in closed or self.obstacles.is_blocked(target):
                        continue
                    tentative = base + vent_hop
                    if tentative < g_score.get(target, math.inf):
                        g_score[target] = tentative
                        came_from[target] = (current, vent_name)
                        heapq.heappush(open_heap, (tentative + heuristic(target), next(counter), target))

        raise NoPathError(f"no path from {start} to {goal}")

    def _build_path(
        self,
        came_from: Mapping[Cell, tuple[Cell, str | None]],
        start: Vec2,
        goal: Vec2,
        start_cell: Cell,
        goal_cell: Cell,
        cost: float,
        door_block: set[Cell],
        smooth: bool,
    ) -> Path:
        # Reconstruct as segments separated by vent hops.
        cells: list[tuple[Cell, str | None]] = [(goal_cell, None)]
        cursor = goal_cell
        while cursor != start_cell:
            parent, vent_name = came_from[cursor]
            if vent_name is not None:
                cells[-1] = (cells[-1][0], vent_name)
            cells.append((parent, None))
            cursor = parent
        cells.reverse()

        segments: list[list[Cell]] = [[]]
        hops: list[str] = []
        for cell, arrived_by_vent in cells:
            if arrived_by_vent is not None:
                hops.append(arrived_by_vent)
                segments.append([])
            segments[-1].append(cell)

        goal_reachable_exactly = self.grid.nearest_walkable(goal, door_block) == goal_cell
        waypoints: list[Waypoint] = []
        distance = 0.0
        last_index = len(segments) - 1
        for index, segment in enumerate(segments):
            pts = self._smooth(segment, door_block) if smooth else list(segment)
            world = [self.grid.cell_center(c) for c in pts]
            if index == 0:
                world[0] = start
            if index == last_index and goal_reachable_exactly:
                world[-1] = goal
            for a, b in zip(world, world[1:]):
                distance += a.distance_to(b)
            if index == 0:
                waypoints.extend(Waypoint(p) for p in world[1:])
            else:
                waypoints.append(Waypoint(world[0], StepKind.VENT, hops[index - 1]))
                waypoints.extend(Waypoint(p) for p in world[1:])
        return Path(tuple(waypoints), cost * self.grid.cell_size, distance, tuple(hops))

    def _line_of_sight(self, a: Cell, b: Cell, door_block: set[Cell]) -> bool:
        """Supercover traversal: every cell the segment touches must be passable."""
        x, y = a
        x1, y1 = b
        dx, dy = abs(x1 - x), abs(y1 - y)
        sx = 1 if x1 > x else -1
        sy = 1 if y1 > y else -1
        err = dx - dy
        while True:
            if not self._passable((x, y), door_block):
                return False
            if (x, y) == (x1, y1):
                return True
            e2 = 2 * err
            step_x = e2 > -dy
            step_y = e2 < dx
            if step_x and step_y:
                if not self._passable((x + sx, y), door_block) or not self._passable((x, y + sy), door_block):
                    return False
            if step_x:
                err -= dy
                x += sx
            if step_y:
                err += dx
                y += sy

    def _smooth(self, cells: Sequence[Cell], door_block: set[Cell]) -> list[Cell]:
        if len(cells) <= 2:
            return list(cells)
        if self.obstacles.has_fields:
            # Soft costs shape the route; keep its corners rather than cutting through fields.
            return self._collapse_collinear(cells)
        result = [cells[0]]
        anchor = 0
        probe = 2
        while probe < len(cells):
            if not self._line_of_sight(cells[anchor], cells[probe], door_block):
                result.append(cells[probe - 1])
                anchor = probe - 1
            probe += 1
        result.append(cells[-1])
        return result

    @staticmethod
    def _collapse_collinear(cells: Sequence[Cell]) -> list[Cell]:
        result = [cells[0]]
        for prev, cur, nxt in zip(cells, cells[1:], cells[2:]):
            if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
                result.append(cur)
        result.append(cells[-1])
        return result

    # -- reachability ------------------------------------------------------- #

    def distance_field(
        self,
        source: Vec2,
        *,
        allow_vents: bool = False,
        closed_doors: Iterable[Room] = (),
        max_distance: float = math.inf,
    ) -> dict[Cell, float]:
        """Dijkstra from ``source``; returns world-unit path length to every reachable cell."""
        door_block = self._closed_door_cells(closed_doors)
        start = self.grid.nearest_walkable(source)
        limit = max_distance / self.grid.cell_size
        dist: dict[Cell, float] = {start: 0.0}
        heap: list[tuple[float, Cell]] = [(0.0, start)]
        vent_hop = self._vent_cost_cells()
        while heap:
            d, cell = heapq.heappop(heap)
            if d > dist.get(cell, math.inf) or d > limit:
                continue
            cx, cy = cell
            for dx, dy, step in NEIGHBOURS:
                nxt = (cx + dx, cy + dy)
                if not self._passable(nxt, door_block):
                    continue
                if dx and dy and (not self._passable((cx + dx, cy), door_block) or not self._passable((cx, cy + dy), door_block)):
                    continue
                nd = d + step
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    heapq.heappush(heap, (nd, nxt))
            if allow_vents:
                for target, _ in self._vent_adjacency.get(cell, ()):
                    nd = d + vent_hop
                    if nd < dist.get(target, math.inf):
                        dist[target] = nd
                        heapq.heappush(heap, (nd, target))
        return {cell: value * self.grid.cell_size for cell, value in dist.items() if value <= limit}

    def static_distance(self, start: Vec2, goal: Vec2) -> float:
        """Walking distance on the static map (no doors, obstacles or vents), cached per goal cell.

        Intended for frequent ranking queries such as "which task is closest";
        use :meth:`find_path` when the exact current route matters.
        """
        goal_cell = self.grid.nearest_walkable(goal)
        field_ = self._field_cache.get(goal_cell)
        if field_ is None:
            saved_blocked, saved_fields = self.obstacles._blocked, self.obstacles._fields
            self.obstacles._blocked, self.obstacles._fields = {}, {}
            try:
                field_ = self.distance_field(self.grid.cell_center(goal_cell))
            finally:
                self.obstacles._blocked, self.obstacles._fields = saved_blocked, saved_fields
            if len(self._field_cache) >= self._field_cache_size:
                self._field_cache.pop(next(iter(self._field_cache)))
            self._field_cache[goal_cell] = field_
        return field_.get(self.grid.nearest_walkable(start), math.inf)

    def has_line_of_sight(self, a: Vec2, b: Vec2, closed_doors: Iterable[Room] = ()) -> bool:
        """True when the straight segment between two points crosses only walkable, open cells."""
        door_block = self._closed_door_cells(closed_doors)
        start, end = self.grid.world_to_cell(a), self.grid.world_to_cell(b)
        if not self.grid.is_walkable(start) or not self.grid.is_walkable(end):
            return False
        door_block.discard(start)
        door_block.discard(end)
        saved = self.obstacles._blocked
        self.obstacles._blocked = {}
        try:
            return self._line_of_sight(start, end, door_block)
        finally:
            self.obstacles._blocked = saved

    def travel_distance(self, start: Vec2, goal: Vec2, **kwargs: object) -> float:
        return self.find_path(start, goal, **kwargs).distance  # type: ignore[arg-type]

    def nearest_vent(self, position: Vec2) -> Vent:
        return min(self.definition.vents.values(), key=lambda v: v.position.distance_to(position))

    def room_center(self, room: Room) -> Vec2:
        rect = self.definition.rooms.get(room)
        if rect is None:
            raise KeyError(f"{room.value} has no rectangle in {self.definition.name}")
        return self.grid.cell_center(self.grid.nearest_walkable(rect.center))

    def room_distance_matrix(self) -> dict[tuple[Room, Room], float]:
        matrix: dict[tuple[Room, Room], float] = {}
        rooms = list(self.definition.rooms)
        for origin in rooms:
            field_ = self.distance_field(self.room_center(origin))
            for target in rooms:
                matrix[(origin, target)] = field_.get(self.grid.world_to_cell(self.room_center(target)), math.inf)
        return matrix

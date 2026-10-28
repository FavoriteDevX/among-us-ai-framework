"""Strictly typed game state model, snapshot parser and temporal tracker.

The framework does not read game memory. A snapshot is a JSON-compatible
mapping produced by a perception source: the bundled headless simulator,
a client-side telemetry mod, or a screen-reading pipeline. This module
validates that mapping into immutable dataclasses and turns consecutive
snapshots into a stream of semantic events (deaths, sightings, vents,
sabotages) that the discussion engine uses as evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Final


class StateParseError(ValueError):
    """Raised when a snapshot does not satisfy the expected schema."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Role(str, Enum):
    CREWMATE = "crewmate"
    IMPOSTOR = "impostor"
    UNKNOWN = "unknown"


class GamePhase(str, Enum):
    LOBBY = "lobby"
    TASKS = "tasks"
    MEETING = "meeting"
    VOTING = "voting"
    EJECTION = "ejection"
    GAME_OVER = "game_over"


class PlayerColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    PINK = "pink"
    ORANGE = "orange"
    YELLOW = "yellow"
    BLACK = "black"
    WHITE = "white"
    PURPLE = "purple"
    BROWN = "brown"
    CYAN = "cyan"
    LIME = "lime"
    MAROON = "maroon"
    ROSE = "rose"
    BANANA = "banana"
    GRAY = "gray"
    TAN = "tan"
    CORAL = "coral"


class Room(str, Enum):
    CAFETERIA = "cafeteria"
    WEAPONS = "weapons"
    O2 = "o2"
    NAVIGATION = "navigation"
    SHIELDS = "shields"
    COMMUNICATIONS = "communications"
    STORAGE = "storage"
    ADMIN = "admin"
    ELECTRICAL = "electrical"
    LOWER_ENGINE = "lower_engine"
    SECURITY = "security"
    REACTOR = "reactor"
    UPPER_ENGINE = "upper_engine"
    MEDBAY = "medbay"
    HALLWAY = "hallway"


class TaskCategory(str, Enum):
    COMMON = "common"
    SHORT = "short"
    LONG = "long"


class SabotageType(str, Enum):
    REACTOR = "reactor"
    OXYGEN = "oxygen"
    LIGHTS = "lights"
    COMMUNICATIONS = "communications"
    DOORS = "doors"

    @property
    def is_critical(self) -> bool:
        """Critical sabotages end the game for the crew when the timer expires."""
        return self in (SabotageType.REACTOR, SabotageType.OXYGEN)


class Animation(str, Enum):
    IDLE = "idle"
    WALKING = "walking"
    TASK = "task"
    VISUAL_TASK = "visual_task"
    VENT_ENTER = "vent_enter"
    VENT_EXIT = "vent_exit"
    KILL = "kill"
    GHOST = "ghost"


class EventType(str, Enum):
    PHASE_CHANGED = "phase_changed"
    PLAYER_SIGHTED = "player_sighted"
    PLAYER_DIED = "player_died"
    BODY_DISCOVERED = "body_discovered"
    KILL_WITNESSED = "kill_witnessed"
    VENT_WITNESSED = "vent_witnessed"
    VISUAL_TASK_WITNESSED = "visual_task_witnessed"
    TASK_COMPLETED = "task_completed"
    TASK_BAR_CHANGED = "task_bar_changed"
    SABOTAGE_STARTED = "sabotage_started"
    SABOTAGE_RESOLVED = "sabotage_resolved"
    DOORS_CLOSED = "doors_closed"
    DOORS_OPENED = "doors_opened"
    MEETING_CALLED = "meeting_called"
    PLAYER_EJECTED = "player_ejected"


# Visual tasks, when enabled, prove the performer is a crewmate.
VISUAL_TASK_NAMES: Final[frozenset[str]] = frozenset(
    {"submit scan", "clear asteroids", "empty garbage", "prime shields"}
)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def distance_to(self, other: Vec2) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def normalized(self) -> Vec2:
        n = self.length()
        if n == 0.0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / n, self.y / n)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class GameSettings:
    num_impostors: int = 2
    kill_distance: float = 2.5
    kill_cooldown: float = 25.0
    player_speed: float = 6.0
    crewmate_vision: float = 9.0
    impostor_vision: float = 13.0
    discussion_time: float = 30.0
    voting_time: float = 60.0
    emergency_meetings: int = 1
    emergency_cooldown: float = 15.0
    confirm_ejects: bool = True
    visual_tasks: bool = True
    anonymous_votes: bool = False


@dataclass(frozen=True, slots=True)
class PlayerState:
    player_id: int
    name: str
    color: PlayerColor
    position: Vec2
    alive: bool
    role: Role
    visible: bool
    in_vent: bool
    animation: Animation
    room: Room | None
    tasks_completed: int | None = None


@dataclass(frozen=True, slots=True)
class TaskState:
    task_id: str
    name: str
    room: Room
    position: Vec2
    category: TaskCategory
    steps_total: int
    steps_done: int
    is_fake: bool = False

    @property
    def completed(self) -> bool:
        return self.steps_done >= self.steps_total

    @property
    def is_visual(self) -> bool:
        return self.name.lower() in VISUAL_TASK_NAMES


@dataclass(frozen=True, slots=True)
class DeadBody:
    player_id: int
    position: Vec2
    room: Room | None


@dataclass(frozen=True, slots=True)
class ActiveSabotage:
    kind: SabotageType
    time_remaining: float | None
    fix_positions: tuple[Vec2, ...]
    fixes_required: int


@dataclass(frozen=True, slots=True)
class ChatMessage:
    sender_id: int
    text: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class MeetingState:
    caller_id: int
    reported_body_id: int | None
    time_remaining: float
    votes: Mapping[int, int | None]
    chat: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class GameState:
    tick: int
    timestamp: float
    phase: GamePhase
    local_player_id: int
    players: Mapping[int, PlayerState]
    tasks: tuple[TaskState, ...]
    bodies: tuple[DeadBody, ...]
    sabotage: ActiveSabotage | None
    closed_doors: frozenset[Room]
    task_bar: float
    kill_cooldown: float
    emergency_meetings_left: int
    settings: GameSettings
    meeting: MeetingState | None
    impostor_ids: frozenset[int] = frozenset()
    ejected_player_id: int | None = None

    # -- convenience accessors -------------------------------------------- #

    @property
    def me(self) -> PlayerState:
        return self.players[self.local_player_id]

    @property
    def is_impostor(self) -> bool:
        return self.me.role is Role.IMPOSTOR

    def alive_players(self) -> list[PlayerState]:
        return [p for p in self.players.values() if p.alive]

    def alive_others(self) -> list[PlayerState]:
        return [p for p in self.players.values() if p.alive and p.player_id != self.local_player_id]

    def visible_others(self) -> list[PlayerState]:
        return [p for p in self.alive_others() if p.visible and not p.in_vent]

    def pending_tasks(self) -> list[TaskState]:
        return [t for t in self.tasks if not t.completed]

    def player_by_name(self, name: str) -> PlayerState | None:
        lowered = name.strip().lower()
        for player in self.players.values():
            if player.name.lower() == lowered or player.color.value == lowered:
                return player
        return None

    def known_teammates(self) -> frozenset[int]:
        if not self.is_impostor:
            return frozenset()
        return frozenset(pid for pid in self.impostor_ids if pid != self.local_player_id)

    def players_within(self, center: Vec2, radius: float, *, include_self: bool = False) -> list[PlayerState]:
        result: list[PlayerState] = []
        for player in self.alive_players():
            if not include_self and player.player_id == self.local_player_id:
                continue
            if player.visible and not player.in_vent and player.position.distance_to(center) <= radius:
                result.append(player)
        return result

    def alive_impostor_estimate(self) -> int:
        """Number of impostors still alive, using ejection confirmations when known."""
        if self.impostor_ids:
            return sum(1 for pid in self.impostor_ids if self.players[pid].alive)
        return self.settings.num_impostors


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

RoomResolver = Callable[[Vec2], Room | None]


class GameStateParser:
    """Validates raw snapshot mappings into :class:`GameState` objects.

    ``room_resolver`` maps a world position to a room. When omitted the parser
    relies on an explicit ``room`` field in each record.
    """

    def __init__(self, room_resolver: RoomResolver | None = None) -> None:
        self._room_resolver = room_resolver

    # -- primitive readers -------------------------------------------------- #

    @staticmethod
    def _require(data: Mapping[str, Any], key: str, path: str) -> Any:
        if key not in data:
            raise StateParseError(f"{path}.{key}", "missing required field")
        return data[key]

    @staticmethod
    def _as_int(value: Any, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise StateParseError(path, f"expected int, got {type(value).__name__}")
        return value

    @staticmethod
    def _as_float(value: Any, path: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StateParseError(path, f"expected number, got {type(value).__name__}")
        result = float(value)
        if not math.isfinite(result):
            raise StateParseError(path, "number must be finite")
        return result

    @staticmethod
    def _as_bool(value: Any, path: str) -> bool:
        if not isinstance(value, bool):
            raise StateParseError(path, f"expected bool, got {type(value).__name__}")
        return value

    @staticmethod
    def _as_str(value: Any, path: str) -> str:
        if not isinstance(value, str):
            raise StateParseError(path, f"expected string, got {type(value).__name__}")
        return value

    @staticmethod
    def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise StateParseError(path, f"expected object, got {type(value).__name__}")
        return value

    @staticmethod
    def _as_list(value: Any, path: str) -> Sequence[Any]:
        if not isinstance(value, (list, tuple)):
            raise StateParseError(path, f"expected array, got {type(value).__name__}")
        return value

    @staticmethod
    def _as_enum(enum_cls: type[Enum], value: Any, path: str) -> Any:
        if not isinstance(value, str):
            raise StateParseError(path, f"expected string for {enum_cls.__name__}")
        try:
            return enum_cls(value.lower())
        except ValueError as exc:
            allowed = ", ".join(member.value for member in enum_cls)  # type: ignore[attr-defined]
            raise StateParseError(path, f"invalid {enum_cls.__name__} {value!r}; allowed: {allowed}") from exc

    def _as_vec2(self, value: Any, path: str) -> Vec2:
        if isinstance(value, Mapping):
            return Vec2(
                self._as_float(self._require(value, "x", path), f"{path}.x"),
                self._as_float(self._require(value, "y", path), f"{path}.y"),
            )
        items = self._as_list(value, path)
        if len(items) != 2:
            raise StateParseError(path, "position must have exactly two components")
        return Vec2(self._as_float(items[0], f"{path}[0]"), self._as_float(items[1], f"{path}[1]"))

    def _resolve_room(self, record: Mapping[str, Any], position: Vec2, path: str) -> Room | None:
        if record.get("room") is not None:
            return self._as_enum(Room, record["room"], f"{path}.room")
        if self._room_resolver is not None:
            return self._room_resolver(position)
        return None

    # -- section parsers ---------------------------------------------------- #

    def _parse_settings(self, raw: Any) -> GameSettings:
        if raw is None:
            return GameSettings()
        data = self._as_mapping(raw, "settings")
        defaults = GameSettings()
        values: dict[str, Any] = {}
        for name in GameSettings.__dataclass_fields__:
            if name not in data:
                continue
            current = getattr(defaults, name)
            path = f"settings.{name}"
            if isinstance(current, bool):
                values[name] = self._as_bool(data[name], path)
            elif isinstance(current, int):
                values[name] = self._as_int(data[name], path)
            else:
                values[name] = self._as_float(data[name], path)
        settings = replace(defaults, **values)
        if settings.num_impostors < 1:
            raise StateParseError("settings.num_impostors", "must be at least 1")
        return settings

    def _parse_player(self, raw: Any, index: int) -> PlayerState:
        path = f"players[{index}]"
        data = self._as_mapping(raw, path)
        player_id = self._as_int(self._require(data, "id", path), f"{path}.id")
        color = self._as_enum(PlayerColor, self._require(data, "color", path), f"{path}.color")
        name = self._as_str(data.get("name", color.value.capitalize()), f"{path}.name")
        position = self._as_vec2(self._require(data, "position", path), f"{path}.position")
        alive = self._as_bool(self._require(data, "alive", path), f"{path}.alive")
        role = self._as_enum(Role, data.get("role", "unknown"), f"{path}.role")
        visible = self._as_bool(data.get("visible", True), f"{path}.visible")
        in_vent = self._as_bool(data.get("in_vent", False), f"{path}.in_vent")
        default_anim = Animation.IDLE.value if alive else Animation.GHOST.value
        animation = self._as_enum(Animation, data.get("animation", default_anim), f"{path}.animation")
        tasks_completed = data.get("tasks_completed")
        if tasks_completed is not None:
            tasks_completed = self._as_int(tasks_completed, f"{path}.tasks_completed")
        return PlayerState(
            player_id=player_id,
            name=name,
            color=color,
            position=position,
            alive=alive,
            role=role,
            visible=visible,
            in_vent=in_vent,
            animation=animation,
            room=self._resolve_room(data, position, path),
            tasks_completed=tasks_completed,
        )

    def _parse_task(self, raw: Any, index: int) -> TaskState:
        path = f"tasks[{index}]"
        data = self._as_mapping(raw, path)
        position = self._as_vec2(self._require(data, "position", path), f"{path}.position")
        room = self._resolve_room(data, position, path)
        if room is None:
            raise StateParseError(f"{path}.room", "task room could not be resolved")
        steps_total = self._as_int(data.get("steps_total", 1), f"{path}.steps_total")
        steps_done = self._as_int(data.get("steps_done", 0), f"{path}.steps_done")
        if steps_total < 1:
            raise StateParseError(f"{path}.steps_total", "must be >= 1")
        if not 0 <= steps_done <= steps_total:
            raise StateParseError(f"{path}.steps_done", "must be between 0 and steps_total")
        return TaskState(
            task_id=self._as_str(self._require(data, "id", path), f"{path}.id"),
            name=self._as_str(self._require(data, "name", path), f"{path}.name"),
            room=room,
            position=position,
            category=self._as_enum(TaskCategory, data.get("category", "short"), f"{path}.category"),
            steps_total=steps_total,
            steps_done=steps_done,
            is_fake=self._as_bool(data.get("is_fake", False), f"{path}.is_fake"),
        )

    def _parse_body(self, raw: Any, index: int) -> DeadBody:
        path = f"bodies[{index}]"
        data = self._as_mapping(raw, path)
        position = self._as_vec2(self._require(data, "position", path), f"{path}.position")
        return DeadBody(
            player_id=self._as_int(self._require(data, "player_id", path), f"{path}.player_id"),
            position=position,
            room=self._resolve_room(data, position, path),
        )

    def _parse_sabotage(self, raw: Any) -> ActiveSabotage | None:
        if raw is None:
            return None
        data = self._as_mapping(raw, "sabotage")
        kind = self._as_enum(SabotageType, self._require(data, "type", "sabotage"), "sabotage.type")
        remaining = data.get("time_remaining")
        time_remaining = None if remaining is None else self._as_float(remaining, "sabotage.time_remaining")
        fixes = tuple(
            self._as_vec2(item, f"sabotage.fix_positions[{i}]")
            for i, item in enumerate(self._as_list(data.get("fix_positions", []), "sabotage.fix_positions"))
        )
        default_required = 2 if kind.is_critical else 1
        return ActiveSabotage(
            kind=kind,
            time_remaining=time_remaining,
            fix_positions=fixes,
            fixes_required=self._as_int(data.get("fixes_required", default_required), "sabotage.fixes_required"),
        )

    def _parse_meeting(self, raw: Any, player_ids: set[int]) -> MeetingState | None:
        if raw is None:
            return None
        data = self._as_mapping(raw, "meeting")
        caller_id = self._as_int(self._require(data, "caller_id", "meeting"), "meeting.caller_id")
        body = data.get("reported_body_id")
        reported = None if body is None else self._as_int(body, "meeting.reported_body_id")
        votes_raw = self._as_mapping(data.get("votes", {}), "meeting.votes")
        votes: dict[int, int | None] = {}
        for key, target in votes_raw.items():
            try:
                voter = int(key)
            except (TypeError, ValueError) as exc:
                raise StateParseError(f"meeting.votes[{key!r}]", "voter key must be an integer") from exc
            if voter not in player_ids:
                raise StateParseError(f"meeting.votes[{key!r}]", "unknown voter")
            if target is None:
                votes[voter] = None
            else:
                target_id = self._as_int(target, f"meeting.votes[{key!r}]")
                if target_id not in player_ids:
                    raise StateParseError(f"meeting.votes[{key!r}]", "unknown vote target")
                votes[voter] = target_id
        chat: list[ChatMessage] = []
        for i, item in enumerate(self._as_list(data.get("chat", []), "meeting.chat")):
            msg = self._as_mapping(item, f"meeting.chat[{i}]")
            chat.append(
                ChatMessage(
                    sender_id=self._as_int(self._require(msg, "sender_id", f"meeting.chat[{i}]"), f"meeting.chat[{i}].sender_id"),
                    text=self._as_str(self._require(msg, "text", f"meeting.chat[{i}]"), f"meeting.chat[{i}].text"),
                    timestamp=self._as_float(msg.get("timestamp", 0.0), f"meeting.chat[{i}].timestamp"),
                )
            )
        return MeetingState(
            caller_id=caller_id,
            reported_body_id=reported,
            time_remaining=self._as_float(data.get("time_remaining", 0.0), "meeting.time_remaining"),
            votes=votes,
            chat=tuple(chat),
        )

    # -- entry point -------------------------------------------------------- #

    def parse(self, raw: Mapping[str, Any]) -> GameState:
        data = self._as_mapping(raw, "snapshot")
        players_list = self._as_list(self._require(data, "players", "snapshot"), "players")
        players: dict[int, PlayerState] = {}
        for index, item in enumerate(players_list):
            player = self._parse_player(item, index)
            if player.player_id in players:
                raise StateParseError(f"players[{index}].id", f"duplicate player id {player.player_id}")
            players[player.player_id] = player
        local_id = self._as_int(self._require(data, "local_player_id", "snapshot"), "local_player_id")
        if local_id not in players:
            raise StateParseError("local_player_id", "does not reference a known player")

        tasks = tuple(
            self._parse_task(item, i) for i, item in enumerate(self._as_list(data.get("tasks", []), "tasks"))
        )
        bodies = tuple(
            self._parse_body(item, i) for i, item in enumerate(self._as_list(data.get("bodies", []), "bodies"))
        )
        for body in bodies:
            if body.player_id not in players:
                raise StateParseError("bodies", f"body references unknown player {body.player_id}")
        closed_doors = frozenset(
            self._as_enum(Room, item, f"closed_doors[{i}]")
            for i, item in enumerate(self._as_list(data.get("closed_doors", []), "closed_doors"))
        )
        impostor_ids = frozenset(
            self._as_int(item, f"impostor_ids[{i}]")
            for i, item in enumerate(self._as_list(data.get("impostor_ids", []), "impostor_ids"))
        )
        if not impostor_ids <= players.keys():
            raise StateParseError("impostor_ids", "references unknown players")
        task_bar = self._as_float(data.get("task_bar", 0.0), "task_bar")
        if not 0.0 <= task_bar <= 1.0:
            raise StateParseError("task_bar", "must be within [0, 1]")
        ejected = data.get("ejected_player_id")

        return GameState(
            tick=self._as_int(self._require(data, "tick", "snapshot"), "tick"),
            timestamp=self._as_float(self._require(data, "timestamp", "snapshot"), "timestamp"),
            phase=self._as_enum(GamePhase, self._require(data, "phase", "snapshot"), "phase"),
            local_player_id=local_id,
            players=players,
            tasks=tasks,
            bodies=bodies,
            sabotage=self._parse_sabotage(data.get("sabotage")),
            closed_doors=closed_doors,
            task_bar=task_bar,
            kill_cooldown=max(0.0, self._as_float(data.get("kill_cooldown", 0.0), "kill_cooldown")),
            emergency_meetings_left=self._as_int(data.get("emergency_meetings_left", 0), "emergency_meetings_left"),
            settings=self._parse_settings(data.get("settings")),
            meeting=self._parse_meeting(data.get("meeting"), set(players)),
            impostor_ids=impostor_ids,
            ejected_player_id=None if ejected is None else self._as_int(ejected, "ejected_player_id"),
        )


# --------------------------------------------------------------------------- #
# Temporal tracking
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Sighting:
    player_id: int
    timestamp: float
    position: Vec2
    room: Room | None
    animation: Animation


@dataclass(frozen=True, slots=True)
class GameEvent:
    kind: EventType
    timestamp: float
    subject_id: int | None = None
    room: Room | None = None
    position: Vec2 | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def describe(self, names: Mapping[int, str]) -> str:
        who = names.get(self.subject_id, "unknown") if self.subject_id is not None else ""
        where = f" in {self.room.value}" if self.room else ""
        stamp = f"[t={self.timestamp:.0f}s]"
        match self.kind:
            case EventType.PLAYER_SIGHTED:
                return f"{stamp} saw {who}{where}"
            case EventType.PLAYER_DIED:
                return f"{stamp} {who} died{where}"
            case EventType.BODY_DISCOVERED:
                return f"{stamp} found {who}'s body{where}"
            case EventType.KILL_WITNESSED:
                victim = names.get(int(self.details.get("victim_id", -1)), "someone")
                return f"{stamp} WITNESSED {who} kill {victim}{where}"
            case EventType.VENT_WITNESSED:
                return f"{stamp} WITNESSED {who} use a vent{where}"
            case EventType.VISUAL_TASK_WITNESSED:
                return f"{stamp} watched {who} do a visual task{where}"
            case EventType.TASK_COMPLETED:
                return f"{stamp} completed {self.details.get('task', 'a task')}{where}"
            case EventType.SABOTAGE_STARTED:
                return f"{stamp} sabotage started: {self.details.get('type')}"
            case EventType.SABOTAGE_RESOLVED:
                return f"{stamp} sabotage resolved: {self.details.get('type')}"
            case EventType.MEETING_CALLED:
                return f"{stamp} meeting called by {who}"
            case EventType.PLAYER_EJECTED:
                return f"{stamp} {who} was ejected"
            case _:
                return f"{stamp} {self.kind.value} {who}{where}".rstrip()


class GameStateTracker:
    """Accumulates snapshots and derives events and per-player sighting history."""

    def __init__(self, *, history_size: int = 600, sighting_interval: float = 1.0) -> None:
        self._states: deque[GameState] = deque(maxlen=history_size)
        self._sightings: dict[int, list[Sighting]] = defaultdict(list)
        self._events: list[GameEvent] = []
        self._sighting_interval = sighting_interval
        self._known_bodies: set[int] = set()
        self._death_times: dict[int, float] = {}

    # -- read accessors ----------------------------------------------------- #

    @property
    def current(self) -> GameState | None:
        return self._states[-1] if self._states else None

    @property
    def events(self) -> Sequence[GameEvent]:
        return tuple(self._events)

    def events_since(self, timestamp: float, kinds: Iterable[EventType] | None = None) -> list[GameEvent]:
        wanted = set(kinds) if kinds is not None else None
        return [e for e in self._events if e.timestamp >= timestamp and (wanted is None or e.kind in wanted)]

    def sightings(self, player_id: int) -> Sequence[Sighting]:
        return tuple(self._sightings.get(player_id, ()))

    def last_seen(self, player_id: int) -> Sighting | None:
        history = self._sightings.get(player_id)
        return history[-1] if history else None

    def death_time(self, player_id: int) -> float | None:
        return self._death_times.get(player_id)

    def sightings_between(self, start: float, end: float) -> Iterator[Sighting]:
        for history in self._sightings.values():
            for sighting in history:
                if start <= sighting.timestamp <= end:
                    yield sighting

    def players_seen_near(self, position: Vec2, radius: float, start: float, end: float) -> dict[int, float]:
        """Return ``player_id -> closest observed distance`` within a time window."""
        closest: dict[int, float] = {}
        for sighting in self.sightings_between(start, end):
            distance = sighting.position.distance_to(position)
            if distance <= radius and distance < closest.get(sighting.player_id, math.inf):
                closest[sighting.player_id] = distance
        return closest

    def players_seen_in_room(self, room: Room, start: float, end: float) -> set[int]:
        return {s.player_id for s in self.sightings_between(start, end) if s.room is room}

    def location_at(self, player_id: int, timestamp: float, tolerance: float = 6.0) -> Sighting | None:
        """Best sighting of a player near a point in time (used for alibi checks)."""
        best: Sighting | None = None
        best_gap = tolerance
        for sighting in self._sightings.get(player_id, ()):
            gap = abs(sighting.timestamp - timestamp)
            if gap <= best_gap:
                best, best_gap = sighting, gap
        return best

    # -- update ------------------------------------------------------------- #

    def reset_round_memory(self) -> None:
        """Forget bodies after a meeting; the game removes them from the map."""
        self._known_bodies.clear()

    def update(self, state: GameState) -> list[GameEvent]:
        previous = self.current
        new_events: list[GameEvent] = []
        now = state.timestamp

        if previous is None or previous.phase is not state.phase:
            new_events.append(
                GameEvent(
                    EventType.PHASE_CHANGED,
                    now,
                    details={"from": previous.phase.value if previous else None, "to": state.phase.value},
                )
            )
            if previous is not None and previous.phase in (GamePhase.MEETING, GamePhase.VOTING, GamePhase.EJECTION):
                if state.phase is GamePhase.TASKS:
                    self.reset_round_memory()

        if state.meeting is not None and (previous is None or previous.meeting is None):
            new_events.append(
                GameEvent(
                    EventType.MEETING_CALLED,
                    now,
                    subject_id=state.meeting.caller_id,
                    details={"reported_body_id": state.meeting.reported_body_id},
                )
            )

        if state.ejected_player_id is not None and (previous is None or previous.ejected_player_id != state.ejected_player_id):
            new_events.append(GameEvent(EventType.PLAYER_EJECTED, now, subject_id=state.ejected_player_id))

        new_events.extend(self._diff_players(previous, state))
        new_events.extend(self._diff_bodies(state))
        new_events.extend(self._diff_tasks(previous, state))
        new_events.extend(self._diff_sabotage(previous, state))

        self._record_sightings(state)
        self._states.append(state)
        self._events.extend(new_events)
        return new_events

    def _diff_players(self, previous: GameState | None, state: GameState) -> list[GameEvent]:
        events: list[GameEvent] = []
        now = state.timestamp
        me = state.me
        for player in state.players.values():
            if player.player_id == state.local_player_id:
                continue
            before = previous.players.get(player.player_id) if previous else None
            if before is not None and before.alive and not player.alive:
                self._death_times[player.player_id] = now
                events.append(GameEvent(EventType.PLAYER_DIED, now, player.player_id, player.room, player.position))
            if not player.alive or not me.alive:
                continue
            if player.animation is Animation.KILL and player.visible:
                victim = self._nearest_fresh_victim(previous, state, player)
                if before is None or before.animation is not Animation.KILL:
                    events.append(
                        GameEvent(
                            EventType.KILL_WITNESSED,
                            now,
                            player.player_id,
                            player.room,
                            player.position,
                            {"victim_id": victim if victim is not None else -1},
                        )
                    )
            vent_now = player.animation in (Animation.VENT_ENTER, Animation.VENT_EXIT)
            vent_before = before is not None and before.animation in (Animation.VENT_ENTER, Animation.VENT_EXIT)
            entered_vent = before is not None and before.visible and not before.in_vent and player.in_vent
            if (vent_now and player.visible and not vent_before) or entered_vent:
                events.append(GameEvent(EventType.VENT_WITNESSED, now, player.player_id, player.room, player.position))
            if (
                state.settings.visual_tasks
                and player.animation is Animation.VISUAL_TASK
                and player.visible
                and (before is None or before.animation is not Animation.VISUAL_TASK)
            ):
                events.append(
                    GameEvent(EventType.VISUAL_TASK_WITNESSED, now, player.player_id, player.room, player.position)
                )
        return events

    @staticmethod
    def _nearest_fresh_victim(previous: GameState | None, state: GameState, killer: PlayerState) -> int | None:
        best: tuple[float, int] | None = None
        for body in state.bodies:
            if previous is not None:
                before = previous.players.get(body.player_id)
                if before is None or not before.alive:
                    continue
            d = body.position.distance_to(killer.position)
            if best is None or d < best[0]:
                best = (d, body.player_id)
        return best[1] if best else None

    def _diff_bodies(self, state: GameState) -> list[GameEvent]:
        events: list[GameEvent] = []
        for body in state.bodies:
            if body.player_id in self._known_bodies:
                continue
            self._known_bodies.add(body.player_id)
            self._death_times.setdefault(body.player_id, state.timestamp)
            events.append(
                GameEvent(EventType.BODY_DISCOVERED, state.timestamp, body.player_id, body.room, body.position)
            )
        return events

    @staticmethod
    def _diff_tasks(previous: GameState | None, state: GameState) -> list[GameEvent]:
        if previous is None:
            return []
        events: list[GameEvent] = []
        before = {t.task_id: t for t in previous.tasks}
        for task in state.tasks:
            old = before.get(task.task_id)
            if old is not None and not old.completed and task.completed:
                events.append(
                    GameEvent(
                        EventType.TASK_COMPLETED,
                        state.timestamp,
                        state.local_player_id,
                        task.room,
                        task.position,
                        {"task": task.name},
                    )
                )
        if not math.isclose(previous.task_bar, state.task_bar, abs_tol=1e-6):
            events.append(
                GameEvent(
                    EventType.TASK_BAR_CHANGED,
                    state.timestamp,
                    details={"from": previous.task_bar, "to": state.task_bar},
                )
            )
        return events

    @staticmethod
    def _diff_sabotage(previous: GameState | None, state: GameState) -> list[GameEvent]:
        events: list[GameEvent] = []
        old = previous.sabotage if previous else None
        new = state.sabotage
        if new is not None and (old is None or old.kind is not new.kind):
            events.append(GameEvent(EventType.SABOTAGE_STARTED, state.timestamp, details={"type": new.kind.value}))
        if old is not None and (new is None or new.kind is not old.kind):
            events.append(GameEvent(EventType.SABOTAGE_RESOLVED, state.timestamp, details={"type": old.kind.value}))
        old_doors = previous.closed_doors if previous else frozenset()
        for room in state.closed_doors - old_doors:
            events.append(GameEvent(EventType.DOORS_CLOSED, state.timestamp, room=room))
        for room in old_doors - state.closed_doors:
            events.append(GameEvent(EventType.DOORS_OPENED, state.timestamp, room=room))
        return events

    def _record_sightings(self, state: GameState) -> None:
        if state.phase is not GamePhase.TASKS:
            return
        if not state.me.alive:
            observed = [p for p in state.players.values() if p.alive and p.visible]
        else:
            observed = [p for p in state.players.values() if p.alive and p.visible and not p.in_vent]
        for player in observed:
            history = self._sightings[player.player_id]
            last = history[-1] if history else None
            moved_room = last is None or last.room is not player.room
            if last is None or moved_room or state.timestamp - last.timestamp >= self._sighting_interval:
                history.append(Sighting(player.player_id, state.timestamp, player.position, player.room, player.animation))
                if moved_room and player.player_id != state.local_player_id:
                    self._events.append(
                        GameEvent(EventType.PLAYER_SIGHTED, state.timestamp, player.player_id, player.room, player.position)
                    )

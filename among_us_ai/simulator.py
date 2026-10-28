"""Headless simulation of The Skeld for self-play, benchmarking and tests.

The simulator owns the authoritative game state and renders a separate
snapshot for every player, in the same schema the real perception layer
produces. Each snapshot respects that player's information: vision radius
and walls, lights sabotage, who they know is dead, which roles they know,
and the impostor team's shared knowledge. Agents therefore run unchanged
against the simulator and against the real game.

Actions are executed through :class:`SimController`, which implements the
same ``execute(action) -> ActionResult`` contract as the screen
controller. Long actions (walking, tasks, fixing) become *intents* that
advance with simulated time and can be pre-empted by a newer decision.

Run a batch from the command line::

    python -m among_us_ai.simulator --games 20 --players 8 --impostors 2 --seed 7
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .agent import AgentConfig, AmongUsAgent
from .controller import (
    Action,
    ActionResult,
    CallEmergency,
    Chat,
    DoTask,
    FakeTask,
    FixSabotage,
    Kill,
    MoveTo,
    Report,
    Sabotage,
    Vote,
    Wait,
)
from .discussion_engine import tally_votes
from .game_state import Animation, GamePhase, GameSettings, PlayerColor, Role, Room, SabotageType, TaskCategory, Vec2
from .llm_bridge import LLMBridge
from .pathfinding import NoPathError, Pathfinder, StepKind, TaskStation, Waypoint

logger = logging.getLogger(__name__)

# Seconds of work per task step; anything unlisted takes DEFAULT_TASK_SECONDS.
TASK_SECONDS: Mapping[str, float] = {
    "Upload Data": 9.0,
    "Submit Scan": 10.0,
    "Clear Asteroids": 12.0,
    "Start Reactor": 12.0,
    "Inspect Sample": 8.0,
    "Fuel Engines": 4.0,
    "Swipe Card": 3.0,
    "Prime Shields": 2.0,
    "Unlock Manifolds": 4.0,
    "Chart Course": 3.0,
    "Clean O2 Filter": 5.0,
}
DEFAULT_TASK_SECONDS = 3.5

FIX_SECONDS: Mapping[SabotageType, float] = {
    SabotageType.LIGHTS: 2.5,
    SabotageType.COMMUNICATIONS: 3.0,
    SabotageType.OXYGEN: 3.0,
}

VENT_ENTER_SECONDS = 0.5
VENT_TRAVEL_SECONDS = 0.4
VENT_EXIT_SECONDS = 0.5
KILL_ANIMATION_SECONDS = 0.6


@dataclass(frozen=True, slots=True)
class SimConfig:
    num_players: int = 8
    settings: GameSettings = field(default_factory=lambda: GameSettings(num_impostors=2))
    common_tasks: int = 2
    short_tasks: int = 3
    long_tasks: int = 1
    dt: float = 0.25
    decision_interval: float = 1.0
    report_distance: float = 3.5
    use_distance: float = 1.5
    sabotage_cooldown: float = 30.0
    door_close_seconds: float = 10.0
    door_cooldown: float = 20.0
    reactor_timer: float = 45.0
    oxygen_timer: float = 40.0
    reactor_hold_limit: float = 20.0
    lights_vision_factor: float = 0.35
    ejection_seconds: float = 4.0
    max_game_seconds: float = 900.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 4 <= self.num_players <= len(PlayerColor):
            raise ValueError(f"num_players must be between 4 and {len(PlayerColor)}")
        if not 1 <= self.settings.num_impostors < self.num_players / 2:
            raise ValueError("impostors must be at least 1 and fewer than half the players")
        if self.dt <= 0:
            raise ValueError("dt must be positive")


# --------------------------------------------------------------------------- #
# World entities
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SimTask:
    task_id: str
    name: str
    category: TaskCategory
    stations: tuple[TaskStation, ...]
    is_fake: bool = False
    steps_done: int = 0

    @property
    def steps_total(self) -> int:
        return len(self.stations)

    @property
    def completed(self) -> bool:
        return self.steps_done >= self.steps_total

    @property
    def station(self) -> TaskStation:
        return self.stations[min(self.steps_done, self.steps_total - 1)]

    def to_snapshot(self) -> dict[str, Any]:
        station = self.station
        return {
            "id": self.task_id,
            "name": self.name,
            "room": station.room.value,
            "position": [station.position.x, station.position.y],
            "category": self.category.value,
            "steps_total": self.steps_total,
            "steps_done": self.steps_done,
            "is_fake": self.is_fake,
        }


@dataclass(slots=True)
class Intent:
    action: Action
    path: deque[Waypoint] = field(default_factory=deque)
    stage: str = "walk"
    timer: float = 0.0
    elapsed: float = 0.0
    repath_at: float = 0.0
    vent_stage: str | None = None
    vent_timer: float = 0.0
    vent_target: Vec2 | None = None
    fix_index: int | None = None


@dataclass(slots=True)
class SimPlayer:
    player_id: int
    name: str
    color: PlayerColor
    role: Role
    position: Vec2
    tasks: list[SimTask]
    alive: bool = True
    in_vent: bool = False
    animation: Animation = Animation.IDLE
    animation_until: float = 0.0
    kill_cooldown: float = 0.0
    emergency_left: int = 1
    intent: Intent | None = None

    @property
    def is_impostor(self) -> bool:
        return self.role is Role.IMPOSTOR


@dataclass(slots=True)
class SimSabotage:
    kind: SabotageType
    time_remaining: float | None
    fix_positions: tuple[Vec2, ...]
    fixed: set[int] = field(default_factory=set)
    holders: dict[int, int] = field(default_factory=dict)
    frozen_task_bar: float = 0.0

    @property
    def fixes_required(self) -> int:
        return len(self.fix_positions)


@dataclass(slots=True)
class SimMeeting:
    caller_id: int
    reported_body_id: int | None
    started_at: float
    phase_until: float
    votes: dict[int, int | None] = field(default_factory=dict)
    chat: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KillRecord:
    timestamp: float
    killer_id: int
    victim_id: int
    room: Room | None


@dataclass(frozen=True, slots=True)
class EjectionRecord:
    timestamp: float
    player_id: int | None
    role: Role | None
    votes: Mapping[int, int | None]


@dataclass(frozen=True, slots=True)
class GameResult:
    winner: Role | None
    reason: str
    duration: float
    kills: tuple[KillRecord, ...]
    ejections: tuple[EjectionRecord, ...]
    meetings: int
    task_bar: float
    impostor_ids: frozenset[int]

    @property
    def correct_ejections(self) -> int:
        return sum(1 for e in self.ejections if e.role is Role.IMPOSTOR)

    @property
    def wrong_ejections(self) -> int:
        return sum(1 for e in self.ejections if e.role is Role.CREWMATE)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #


class SkeldSimulator:
    """Authoritative game engine for The Skeld."""

    def __init__(self, config: SimConfig | None = None, pathfinder: Pathfinder | None = None) -> None:
        self.config = config or SimConfig()
        self.settings = self.config.settings
        self.pathfinder = pathfinder or Pathfinder()
        self.map = self.pathfinder.definition
        self.rng = random.Random(self.config.seed)
        self.time = 0.0
        self.tick_count = 0
        self.phase = GamePhase.TASKS
        self.players: dict[int, SimPlayer] = {}
        self.bodies: dict[int, Vec2] = {}
        self.sabotage: SimSabotage | None = None
        self.sabotage_ready_at = 10.0
        self.closed_doors: dict[Room, float] = {}
        self.doors_ready_at = 10.0
        self.meeting: SimMeeting | None = None
        self.last_meeting_end = 0.0
        self.last_ejected: int | None = None
        self.revealed_roles: dict[int, Role] = {}
        self.known_dead: dict[int, set[int]] = {}
        self.last_seen: dict[int, dict[int, Vec2]] = {}
        self.kills: list[KillRecord] = []
        self.ejections: list[EjectionRecord] = []
        self.meetings_held = 0
        self.winner: Role | None = None
        self.end_reason = ""
        self.log: list[str] = []
        self._visibility: dict[tuple[int, int], bool] = {}
        self._setup_players()
        self._refresh_visibility()

    # -- setup -------------------------------------------------------------- #

    def _spawn_points(self) -> list[Vec2]:
        center = self.map.spawn
        n = len(self.players) or self.config.num_players
        points = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            points.append(Vec2(center.x + 3.5 * math.cos(angle), center.y + 3.0 * math.sin(angle)))
        return points

    def _stations(self, name: str, room: Room | None = None) -> list[TaskStation]:
        return [s for s in self.map.task_stations if s.name == name and (room is None or s.room is room)]

    def _task_pool(self) -> dict[TaskCategory, list[Callable[[], tuple[str, tuple[TaskStation, ...]]]]]:
        rng = self.rng

        def single(name: str) -> Callable[[], tuple[str, tuple[TaskStation, ...]]]:
            return lambda: (name, (rng.choice(self._stations(name)),))

        def wiring() -> tuple[str, tuple[TaskStation, ...]]:
            stations = self._stations("Fix Wiring")
            return "Fix Wiring", tuple(sorted(rng.sample(stations, 3), key=lambda s: s.position.x))

        def upload() -> tuple[str, tuple[TaskStation, ...]]:
            return "Upload Data", (rng.choice(self._stations("Download Data")), self._stations("Upload Data")[0])

        def divert() -> tuple[str, tuple[TaskStation, ...]]:
            return "Divert Power", (self._stations("Divert Power")[0], rng.choice(self._stations("Accept Diverted Power")))

        def fuel() -> tuple[str, tuple[TaskStation, ...]]:
            storage = self._stations("Fuel Engines", Room.STORAGE)[0]
            engine = rng.choice([s for s in self._stations("Fuel Engines") if s.room is not Room.STORAGE])
            return "Fuel Engines", (storage, engine)

        def garbage() -> tuple[str, tuple[TaskStation, ...]]:
            first = rng.choice([s for s in self._stations("Empty Garbage") if s.room is not Room.STORAGE])
            return "Empty Garbage", (first, self._stations("Empty Garbage", Room.STORAGE)[0])

        def engines() -> tuple[str, tuple[TaskStation, ...]]:
            return "Align Engine Output", tuple(self._stations("Align Engine Output"))

        return {
            TaskCategory.COMMON: [wiring, single("Swipe Card")],
            TaskCategory.SHORT: [
                single("Prime Shields"), single("Calibrate Distributor"), single("Chart Course"),
                single("Stabilize Steering"), single("Clean O2 Filter"), single("Unlock Manifolds"),
                divert, garbage,
            ],
            TaskCategory.LONG: [
                single("Submit Scan"), single("Start Reactor"), single("Inspect Sample"),
                single("Clear Asteroids"), upload, fuel, engines,
            ],
        }

    def _deal_tasks(self, owner: int, *, fake: bool, common: Sequence[tuple[str, tuple[TaskStation, ...]]]) -> list[SimTask]:
        pool = self._task_pool()
        chosen: list[tuple[TaskCategory, tuple[str, tuple[TaskStation, ...]]]] = [(TaskCategory.COMMON, item) for item in common]
        for category, count in ((TaskCategory.SHORT, self.config.short_tasks), (TaskCategory.LONG, self.config.long_tasks)):
            makers = self.rng.sample(pool[category], min(count, len(pool[category])))
            chosen.extend((category, make()) for make in makers)
        return [
            SimTask(f"{owner}-{i}", name, category, stations, is_fake=fake)
            for i, (category, (name, stations)) in enumerate(chosen)
        ]

    def _setup_players(self) -> None:
        cfg = self.config
        colors = list(PlayerColor)[: cfg.num_players]
        impostors = set(self.rng.sample(range(cfg.num_players), cfg.settings.num_impostors))
        common_makers = self.rng.sample(self._task_pool()[TaskCategory.COMMON], min(cfg.common_tasks, 2))
        # Common tasks share the same stations for every crewmate.
        fixed_common = [maker() for maker in common_makers]
        for pid, color in enumerate(colors):
            role = Role.IMPOSTOR if pid in impostors else Role.CREWMATE
            self.players[pid] = SimPlayer(
                player_id=pid,
                name=color.value.capitalize(),
                color=color,
                role=role,
                position=self.map.spawn,
                tasks=self._deal_tasks(pid, fake=role is Role.IMPOSTOR, common=fixed_common),
                kill_cooldown=min(10.0, self.settings.kill_cooldown) if role is Role.IMPOSTOR else 0.0,
                emergency_left=self.settings.emergency_meetings,
            )
            self.known_dead[pid] = set()
            self.last_seen[pid] = {}
        for player, point in zip(self.players.values(), self._spawn_points(), strict=True):
            player.position = point
        self._note(f"impostors: {', '.join(self.players[i].name for i in sorted(impostors))}")

    def _note(self, message: str) -> None:
        line = f"[{self.time:6.1f}s] {message}"
        self.log.append(line)
        logger.debug(line)

    # -- derived state ------------------------------------------------------ #

    @property
    def impostor_ids(self) -> frozenset[int]:
        return frozenset(pid for pid, p in self.players.items() if p.is_impostor)

    @property
    def finished(self) -> bool:
        return self.winner is not None or self.end_reason == "timeout"

    def task_bar(self) -> float:
        total = done = 0
        for player in self.players.values():
            if player.is_impostor:
                continue
            for task in player.tasks:
                total += task.steps_total
                done += min(task.steps_done, task.steps_total)
        return done / total if total else 1.0

    def vision(self, player: SimPlayer) -> float:
        if player.is_impostor:
            return self.settings.impostor_vision
        if self.sabotage is not None and self.sabotage.kind is SabotageType.LIGHTS:
            return self.settings.crewmate_vision * self.config.lights_vision_factor
        return self.settings.crewmate_vision

    def _sees_point(self, viewer: SimPlayer, point: Vec2) -> bool:
        if not viewer.alive:
            return True
        if viewer.position.distance_to(point) > self.vision(viewer):
            return False
        return self.pathfinder.has_line_of_sight(viewer.position, point, self.closed_doors)

    def _refresh_visibility(self) -> None:
        self._visibility.clear()
        for viewer in self.players.values():
            for target in self.players.values():
                if viewer is target:
                    continue
                if viewer.alive:
                    seen = target.alive and not target.in_vent and self._sees_point(viewer, target.position)
                else:
                    seen = not target.in_vent
                self._visibility[(viewer.player_id, target.player_id)] = seen
                if seen:
                    self.last_seen[viewer.player_id][target.player_id] = target.position
            for body_id, body_pos in self.bodies.items():
                if self._sees_point(viewer, body_pos):
                    self.known_dead[viewer.player_id].add(body_id)

    def can_see(self, viewer_id: int, target_id: int) -> bool:
        return viewer_id == target_id or self._visibility.get((viewer_id, target_id), False)

    # -- snapshot ----------------------------------------------------------- #

    def snapshot(self, viewer_id: int) -> dict[str, Any]:
        viewer = self.players[viewer_id]
        knows_team = viewer.is_impostor
        players: list[dict[str, Any]] = []
        for target in self.players.values():
            pid = target.player_id
            visible = self.can_see(viewer_id, pid)
            if pid == viewer_id or not viewer.alive:
                alive = target.alive
            else:
                alive = target.alive or pid not in self.known_dead[viewer_id]
            if pid == viewer_id or (knows_team and target.is_impostor):
                role = target.role
            else:
                role = self.revealed_roles.get(pid, Role.UNKNOWN)
            position = target.position if visible else self.last_seen[viewer_id].get(pid, self.map.spawn)
            animation = target.animation if visible else (Animation.IDLE if alive else Animation.GHOST)
            if not target.alive and animation is not Animation.GHOST and (visible or pid == viewer_id):
                animation = Animation.GHOST
            players.append(
                {
                    "id": pid,
                    "name": target.name,
                    "color": target.color.value,
                    "position": [position.x, position.y],
                    "alive": alive,
                    "role": role.value,
                    "visible": visible,
                    "in_vent": target.in_vent if pid == viewer_id else False,
                    "animation": animation.value,
                }
            )
        bodies = []
        if self.phase is GamePhase.TASKS:
            bodies = [
                {"player_id": pid, "position": [pos.x, pos.y]}
                for pid, pos in self.bodies.items()
                if self._sees_point(viewer, pos)
            ]
        sabotage = None
        if self.sabotage is not None:
            sabotage = {
                "type": self.sabotage.kind.value,
                "time_remaining": self.sabotage.time_remaining,
                "fix_positions": [[p.x, p.y] for i, p in enumerate(self.sabotage.fix_positions) if i not in self.sabotage.fixed],
                "fixes_required": self.sabotage.fixes_required,
            }
        comms_down = self.sabotage is not None and self.sabotage.kind is SabotageType.COMMUNICATIONS
        meeting = None
        if self.meeting is not None:
            meeting = {
                "caller_id": self.meeting.caller_id,
                "reported_body_id": self.meeting.reported_body_id,
                "time_remaining": max(0.0, self.meeting.phase_until - self.time),
                "votes": {str(k): v for k, v in self.meeting.votes.items()},
                "chat": list(self.meeting.chat),
            }
        impostor_ids = sorted(self.impostor_ids) if knows_team else sorted(
            pid for pid, role in self.revealed_roles.items() if role is Role.IMPOSTOR
        )
        return {
            "tick": self.tick_count,
            "timestamp": self.time,
            "phase": self.phase.value,
            "local_player_id": viewer_id,
            "players": players,
            "tasks": [t.to_snapshot() for t in viewer.tasks],
            "bodies": bodies,
            "sabotage": sabotage,
            "closed_doors": [room.value for room in self.closed_doors],
            "task_bar": self.sabotage.frozen_task_bar if comms_down and self.sabotage else self.task_bar(),
            "kill_cooldown": viewer.kill_cooldown if viewer.is_impostor else 0.0,
            "emergency_meetings_left": viewer.emergency_left,
            "settings": asdict(self.settings),
            "meeting": meeting,
            "impostor_ids": impostor_ids,
            "ejected_player_id": self.last_ejected,
        }

    # -- instant actions ---------------------------------------------------- #

    def try_kill(self, killer_id: int, victim_id: int) -> tuple[bool, str]:
        killer = self.players[killer_id]
        victim = self.players.get(victim_id)
        if self.phase is not GamePhase.TASKS:
            return False, "kills only happen during free roam"
        if not killer.is_impostor or not killer.alive:
            return False, "only living impostors can kill"
        if killer.in_vent:
            return False, "cannot kill from a vent"
        if killer.kill_cooldown > 0:
            return False, f"kill on cooldown ({killer.kill_cooldown:.1f}s)"
        if victim is None or not victim.alive or victim.is_impostor or victim.in_vent:
            return False, "invalid kill target"
        if killer.position.distance_to(victim.position) > self.settings.kill_distance:
            return False, "target out of range"
        victim.alive = False
        victim.intent = None
        victim.animation = Animation.GHOST
        self.bodies[victim_id] = victim.position
        killer.position = victim.position
        killer.intent = None
        killer.animation = Animation.KILL
        killer.animation_until = self.time + KILL_ANIMATION_SECONDS
        killer.kill_cooldown = self.settings.kill_cooldown
        self.known_dead[killer_id].add(victim_id)
        room = self.map.room_at(victim.position)
        self.kills.append(KillRecord(self.time, killer_id, victim_id, room))
        self._note(f"{killer.name} killed {victim.name} in {room.value if room else '?'}")
        self._refresh_visibility()
        self._check_win()
        return True, "ok"

    def try_report(self, reporter_id: int, body_id: int) -> tuple[bool, str]:
        reporter = self.players[reporter_id]
        if self.phase is not GamePhase.TASKS or not reporter.alive:
            return False, "cannot report now"
        body = self.bodies.get(body_id)
        if body is None:
            return False, "no such body"
        if reporter.position.distance_to(body) > self.config.report_distance:
            return False, "body out of reach"
        self._start_meeting(reporter_id, body_id)
        return True, "ok"

    def try_emergency(self, caller_id: int) -> tuple[bool, str]:
        caller = self.players[caller_id]
        if self.phase is not GamePhase.TASKS or not caller.alive:
            return False, "cannot call a meeting now"
        if caller.emergency_left <= 0:
            return False, "no emergency meetings left"
        if self.sabotage is not None and self.sabotage.kind.is_critical:
            return False, "emergency button disabled during a critical sabotage"
        if self.time - self.last_meeting_end < self.settings.emergency_cooldown:
            return False, "emergency button cooling down"
        if caller.position.distance_to(self.map.emergency_button) > self.config.use_distance * 2:
            return False, "too far from the emergency button"
        caller.emergency_left -= 1
        self._start_meeting(caller_id, None)
        return True, "ok"

    def try_sabotage(self, player_id: int, kind: SabotageType, door_room: str | None) -> tuple[bool, str]:
        player = self.players[player_id]
        if self.phase is not GamePhase.TASKS or not player.is_impostor:
            return False, "only impostors can sabotage during free roam"
        if kind is SabotageType.DOORS:
            try:
                room = Room(door_room or "")
            except ValueError:
                return False, f"unknown door room {door_room!r}"
            if room not in self.map.door_rooms:
                return False, f"{room.value} has no doors"
            if self.time < self.doors_ready_at:
                return False, "doors on cooldown"
            self.closed_doors[room] = self.time + self.config.door_close_seconds
            self.doors_ready_at = self.time + self.config.door_cooldown
            self._note(f"{player.name} closed the {room.value} doors")
            return True, "ok"
        if self.sabotage is not None:
            return False, "another sabotage is active"
        if self.time < self.sabotage_ready_at:
            return False, "sabotage on cooldown"
        timer = {SabotageType.REACTOR: self.config.reactor_timer, SabotageType.OXYGEN: self.config.oxygen_timer}.get(kind)
        positions = self.map.sabotage_fixes[kind.value]
        self.sabotage = SimSabotage(kind, timer, tuple(positions), frozen_task_bar=self.task_bar())
        self._note(f"{player.name} sabotaged {kind.value}")
        return True, "ok"

    def try_chat(self, player_id: int, text: str) -> tuple[bool, str]:
        player = self.players[player_id]
        if self.meeting is None or self.phase not in (GamePhase.MEETING, GamePhase.VOTING):
            return False, "chat is only available in meetings"
        if not player.alive:
            return False, "ghosts cannot talk to the living"
        clean = " ".join(text.split())[:100]
        if not clean:
            return False, "empty message"
        self.meeting.chat.append({"sender_id": player_id, "text": clean, "timestamp": self.time})
        self._note(f"{player.name}: {clean}")
        return True, "ok"

    def try_vote(self, voter_id: int, target_id: int | None) -> tuple[bool, str]:
        voter = self.players[voter_id]
        if self.meeting is None or self.phase is not GamePhase.VOTING:
            return False, "voting has not started"
        if not voter.alive:
            return False, "dead players cannot vote"
        if voter_id in self.meeting.votes:
            return False, "already voted"
        if target_id is not None and (target_id not in self.players or not self.players[target_id].alive):
            return False, "invalid vote target"
        self.meeting.votes[voter_id] = target_id
        target = self.players[target_id].name if target_id is not None else "skip"
        self._note(f"{voter.name} voted {target}")
        if all(pid in self.meeting.votes for pid, p in self.players.items() if p.alive):
            self._end_voting()
        return True, "ok"

    # -- meetings ----------------------------------------------------------- #

    def _start_meeting(self, caller_id: int, body_id: int | None) -> None:
        caller = self.players[caller_id]
        reason = f"reported {self.players[body_id].name}'s body" if body_id is not None else "called an emergency meeting"
        self._note(f"{caller.name} {reason}")
        self.meetings_held += 1
        self.phase = GamePhase.MEETING
        self.meeting = SimMeeting(caller_id, body_id, self.time, self.time + self.settings.discussion_time)
        dead = {pid for pid, p in self.players.items() if not p.alive}
        for known in self.known_dead.values():
            known |= dead
        self.bodies.clear()
        self.sabotage = None
        self.sabotage_ready_at = self.time + self.config.sabotage_cooldown
        self.closed_doors.clear()
        for player in self.players.values():
            player.intent = None
            player.in_vent = False
            player.animation = Animation.IDLE if player.alive else Animation.GHOST

    def _end_voting(self) -> None:
        assert self.meeting is not None
        alive_votes = {voter: target for voter, target in self.meeting.votes.items() if self.players[voter].alive}
        ejected, tie = tally_votes(alive_votes)
        role: Role | None = None
        if ejected is not None:
            victim = self.players[ejected]
            victim.alive = False
            victim.intent = None
            victim.animation = Animation.GHOST
            role = victim.role
            if self.settings.confirm_ejects:
                self.revealed_roles[ejected] = role
            for known in self.known_dead.values():
                known.add(ejected)
            self._note(f"{victim.name} was ejected ({role.value if self.settings.confirm_ejects else 'role hidden'})")
        else:
            self._note("no one was ejected" + (" (tie)" if tie else ""))
        self.last_ejected = ejected
        self.ejections.append(EjectionRecord(self.time, ejected, role, dict(alive_votes)))
        self.phase = GamePhase.EJECTION
        self.meeting.phase_until = self.time + self.config.ejection_seconds
        self._check_win()

    def _resume_tasks(self) -> None:
        self.phase = GamePhase.TASKS
        self.meeting = None
        self.last_meeting_end = self.time
        for player, point in zip(self.players.values(), self._spawn_points(), strict=True):
            if player.alive:
                player.position = point
            if player.is_impostor:
                player.kill_cooldown = self.settings.kill_cooldown
        for viewer in self.last_seen.values():
            viewer.clear()
        self._refresh_visibility()

    # -- intents ------------------------------------------------------------ #

    def set_intent(self, player_id: int, action: Action, target: Vec2 | None, *, allow_vents: bool = False) -> tuple[bool, str]:
        player = self.players[player_id]
        intent = Intent(action)
        if target is not None:
            try:
                path = self.pathfinder.find_path(
                    player.position, target,
                    allow_vents=allow_vents and player.is_impostor and player.alive,
                    closed_doors=self.closed_doors if player.alive else (),
                )
            except NoPathError as exc:
                return False, str(exc)
            intent.path = deque(path.waypoints)
        else:
            intent.stage = "work"
        player.intent = intent
        return True, "scheduled"

    def _step_path(self, player: SimPlayer, intent: Intent, dt: float) -> bool:
        """Advance along the path; returns True once the path is exhausted."""
        if intent.vent_stage is not None:
            intent.vent_timer -= dt
            if intent.vent_timer > 0:
                return False
            if intent.vent_stage == "entering":
                player.in_vent = True
                intent.vent_stage = "travelling"
                intent.vent_timer = VENT_TRAVEL_SECONDS
                return False
            if intent.vent_stage == "travelling":
                assert intent.vent_target is not None
                player.position = intent.vent_target
                player.in_vent = False
                player.animation = Animation.VENT_EXIT
                player.animation_until = self.time + VENT_EXIT_SECONDS
                intent.vent_stage = "exiting"
                intent.vent_timer = VENT_EXIT_SECONDS
                return False
            intent.vent_stage = None
        budget = self.settings.player_speed * dt
        moved = False
        while budget > 0 and intent.path:
            waypoint = intent.path[0]
            if waypoint.kind is StepKind.VENT:
                intent.path.popleft()
                intent.vent_stage = "entering"
                intent.vent_timer = VENT_ENTER_SECONDS
                intent.vent_target = waypoint.position
                player.animation = Animation.VENT_ENTER
                player.animation_until = self.time + VENT_ENTER_SECONDS
                return False
            offset = waypoint.position - player.position
            distance = offset.length()
            if distance <= budget:
                player.position = waypoint.position
                budget -= distance
                intent.path.popleft()
            else:
                player.position = player.position + offset.normalized() * budget
                budget = 0.0
            moved = True
        if moved and player.animation_until <= self.time:
            player.animation = Animation.WALKING if player.alive else Animation.GHOST
        return not intent.path

    def _finish_intent(self, player: SimPlayer) -> None:
        player.intent = None
        if player.animation_until <= self.time:
            player.animation = Animation.IDLE if player.alive else Animation.GHOST

    def _advance_intent(self, player: SimPlayer, dt: float) -> None:
        intent = player.intent
        if intent is None:
            return
        intent.elapsed += dt
        action = intent.action
        if isinstance(action, Kill):
            self._advance_chase(player, intent, action, dt)
            return
        if intent.stage == "walk":
            if not self._step_path(player, intent, dt):
                return
            intent.stage = "work"
            self._on_arrival(player, intent)
            return
        self._work(player, intent, dt)

    def _advance_chase(self, player: SimPlayer, intent: Intent, action: Kill, dt: float) -> None:
        victim = self.players.get(action.target_id)
        if victim is None or not victim.alive or intent.elapsed > 6.0 or not self.can_see(player.player_id, victim.player_id):
            self._finish_intent(player)
            return
        if player.position.distance_to(victim.position) <= self.settings.kill_distance:
            ok, detail = self.try_kill(player.player_id, victim.player_id)
            if not ok:
                logger.debug("chase kill failed: %s", detail)
                self._finish_intent(player)
            return
        if self.time >= intent.repath_at or not intent.path:
            try:
                path = self.pathfinder.find_path(player.position, victim.position, closed_doors=self.closed_doors)
            except NoPathError:
                self._finish_intent(player)
                return
            intent.path = deque(path.waypoints)
            intent.repath_at = self.time + 0.5
        self._step_path(player, intent, dt)

    def _on_arrival(self, player: SimPlayer, intent: Intent) -> None:
        action = intent.action
        match action:
            case MoveTo():
                self._finish_intent(player)
            case DoTask(task_id=task_id):
                task = self._own_task(player, task_id)
                if task is None or task.completed:
                    self._finish_intent(player)
                    return
                intent.timer = TASK_SECONDS.get(task.name, DEFAULT_TASK_SECONDS)
                visual = task.station.visual and self.settings.visual_tasks and player.alive
                player.animation = Animation.VISUAL_TASK if visual else Animation.TASK
                player.animation_until = self.time + intent.timer
            case FakeTask(duration=duration):
                intent.timer = duration
                player.animation = Animation.TASK
                player.animation_until = self.time + duration
            case FixSabotage(position=position):
                sabotage = self.sabotage
                if sabotage is None:
                    self._finish_intent(player)
                    return
                index = min(range(len(sabotage.fix_positions)), key=lambda i: sabotage.fix_positions[i].distance_to(position))
                intent.fix_index = index
                intent.timer = self.config.reactor_hold_limit if sabotage.kind is SabotageType.REACTOR else FIX_SECONDS.get(sabotage.kind, 3.0)
                player.animation = Animation.TASK
            case CallEmergency():
                ok, detail = self.try_emergency(player.player_id)
                if not ok:
                    logger.debug("%s emergency failed: %s", player.name, detail)
                    self._finish_intent(player)
            case Report(body_id=body_id):
                ok, detail = self.try_report(player.player_id, body_id)
                if not ok:
                    logger.debug("%s report failed: %s", player.name, detail)
                    self._finish_intent(player)
            case Wait(seconds=seconds):
                intent.timer = seconds
            case _:
                self._finish_intent(player)

    def _work(self, player: SimPlayer, intent: Intent, dt: float) -> None:
        action = intent.action
        intent.timer -= dt
        if isinstance(action, FixSabotage):
            sabotage = self.sabotage
            if sabotage is None or intent.fix_index is None:
                self._finish_intent(player)
                return
            if sabotage.kind is SabotageType.REACTOR:
                sabotage.holders[intent.fix_index] = player.player_id
                if intent.timer <= 0:
                    sabotage.holders.pop(intent.fix_index, None)
                    self._finish_intent(player)
                return
            if intent.timer <= 0:
                sabotage.fixed.add(intent.fix_index)
                self._note(f"{player.name} fixed a {sabotage.kind.value} panel")
                if len(sabotage.fixed) >= sabotage.fixes_required:
                    self._resolve_sabotage()
                self._finish_intent(player)
            return
        if intent.timer > 0:
            return
        if isinstance(action, DoTask):
            task = self._own_task(player, action.task_id)
            if task is not None and not task.completed:
                task.steps_done += 1
                if task.completed:
                    self._note(f"{player.name} finished {task.name}")
            self._finish_intent(player)
            self._check_win()
            return
        self._finish_intent(player)

    def _own_task(self, player: SimPlayer, task_id: str) -> SimTask | None:
        return next((t for t in player.tasks if t.task_id == task_id), None)

    def _resolve_sabotage(self) -> None:
        if self.sabotage is None:
            return
        self._note(f"{self.sabotage.kind.value} sabotage resolved")
        self.sabotage = None
        self.sabotage_ready_at = self.time + self.config.sabotage_cooldown

    # -- clock -------------------------------------------------------------- #

    def tick(self) -> None:
        if self.finished:
            return
        dt = self.config.dt
        self.time += dt
        self.tick_count += 1

        if self.phase is GamePhase.TASKS:
            for player in self.players.values():
                if player.is_impostor and player.alive:
                    player.kill_cooldown = max(0.0, player.kill_cooldown - dt)
                if player.animation_until and player.animation_until <= self.time and player.intent is None:
                    player.animation = Animation.IDLE if player.alive else Animation.GHOST
                    player.animation_until = 0.0
            if self.sabotage is not None:
                self.sabotage.holders.clear()
            for player in list(self.players.values()):
                if self.phase is not GamePhase.TASKS:
                    break
                self._advance_intent(player, dt)
            if self.phase is GamePhase.TASKS:
                self._advance_sabotage(dt)
                for room in [r for r, until in self.closed_doors.items() if until <= self.time]:
                    del self.closed_doors[room]
        elif self.phase is GamePhase.MEETING:
            assert self.meeting is not None
            if self.time >= self.meeting.phase_until:
                self.phase = GamePhase.VOTING
                self.meeting.phase_until = self.time + self.settings.voting_time
        elif self.phase is GamePhase.VOTING:
            assert self.meeting is not None
            if self.time >= self.meeting.phase_until:
                self._end_voting()
        elif self.phase is GamePhase.EJECTION:
            assert self.meeting is not None
            if self.time >= self.meeting.phase_until and not self.finished:
                self._resume_tasks()

        if self.phase is GamePhase.TASKS:
            self._refresh_visibility()
        if self.time >= self.config.max_game_seconds and self.winner is None:
            self.end_reason = "timeout"
            self.phase = GamePhase.GAME_OVER
            self._note("game timed out")

    def _advance_sabotage(self, dt: float) -> None:
        sabotage = self.sabotage
        if sabotage is None:
            return
        if sabotage.kind is SabotageType.REACTOR and len(sabotage.holders) >= sabotage.fixes_required:
            self._resolve_sabotage()
            return
        if sabotage.time_remaining is not None:
            sabotage.time_remaining -= dt
            if sabotage.time_remaining <= 0:
                self._declare(Role.IMPOSTOR, f"{sabotage.kind.value} meltdown")

    def _check_win(self) -> None:
        if self.winner is not None:
            return
        impostors = sum(1 for p in self.players.values() if p.alive and p.is_impostor)
        crew = sum(1 for p in self.players.values() if p.alive and not p.is_impostor)
        if impostors == 0:
            self._declare(Role.CREWMATE, "all impostors ejected")
        elif impostors >= crew:
            self._declare(Role.IMPOSTOR, "impostors reached parity")
        elif self.task_bar() >= 1.0:
            self._declare(Role.CREWMATE, "all tasks completed")

    def _declare(self, winner: Role, reason: str) -> None:
        self.winner = winner
        self.end_reason = reason
        self.phase = GamePhase.GAME_OVER
        self._note(f"{winner.value}s win: {reason}")

    def result(self) -> GameResult:
        return GameResult(
            winner=self.winner,
            reason=self.end_reason or "in progress",
            duration=self.time,
            kills=tuple(self.kills),
            ejections=tuple(self.ejections),
            meetings=self.meetings_held,
            task_bar=self.task_bar(),
            impostor_ids=self.impostor_ids,
        )


# --------------------------------------------------------------------------- #
# Controller adapter
# --------------------------------------------------------------------------- #


class SimController:
    """``ActionExecutor`` that turns agent actions into simulator intents."""

    def __init__(self, sim: SkeldSimulator, player_id: int, *, kill_chase_radius: float = 6.0) -> None:
        self.sim = sim
        self.player_id = player_id
        self.kill_chase_radius = kill_chase_radius

    @property
    def player(self) -> SimPlayer:
        return self.sim.players[self.player_id]

    @property
    def idle(self) -> bool:
        return self.player.intent is None

    def _result(self, action: Action, ok: bool, detail: str) -> ActionResult:
        return ActionResult(action, ok, detail, 0.0)

    def execute(self, action: Action) -> ActionResult:
        sim = self.sim
        player = self.player
        current = player.intent
        if current is not None and current.action == action:
            return self._result(action, True, "in progress")
        match action:
            case Chat(text=text):
                return self._result(action, *sim.try_chat(self.player_id, text))
            case Vote(target_id=target):
                return self._result(action, *sim.try_vote(self.player_id, target))
            case Sabotage(kind=kind, door_room=room):
                return self._result(action, *sim.try_sabotage(self.player_id, kind, room))
            case Kill(target_id=target):
                victim = sim.players.get(target)
                if victim is None:
                    return self._result(action, False, "unknown target")
                if player.position.distance_to(victim.position) <= sim.settings.kill_distance:
                    return self._result(action, *sim.try_kill(self.player_id, target))
                if player.kill_cooldown > 0 or not player.is_impostor:
                    return self._result(action, False, "cannot kill now")
                if player.position.distance_to(victim.position) > self.kill_chase_radius:
                    return self._result(action, False, "target too far to chase")
                player.intent = Intent(action)
                return self._result(action, True, "chasing")
            case Report(body_id=body_id):
                body = sim.bodies.get(body_id)
                if body is None:
                    return self._result(action, False, "no such body")
                if player.position.distance_to(body) <= sim.config.report_distance:
                    return self._result(action, *sim.try_report(self.player_id, body_id))
                return self._result(action, *sim.set_intent(self.player_id, action, body))
            case MoveTo(target=target, allow_vents=vents):
                return self._result(action, *sim.set_intent(self.player_id, action, target, allow_vents=vents))
            case DoTask(task_id=task_id):
                task = sim._own_task(player, task_id)
                if task is None or task.completed or task.is_fake:
                    return self._result(action, False, "task unavailable")
                return self._result(action, *sim.set_intent(self.player_id, action, task.station.position))
            case FakeTask(task_id=task_id):
                task = sim._own_task(player, task_id)
                if task is None:
                    return self._result(action, False, "unknown task")
                return self._result(action, *sim.set_intent(self.player_id, action, task.station.position))
            case FixSabotage(position=position):
                if sim.sabotage is None:
                    return self._result(action, False, "nothing to fix")
                return self._result(action, *sim.set_intent(self.player_id, action, position))
            case CallEmergency():
                return self._result(action, *sim.set_intent(self.player_id, action, sim.map.emergency_button))
            case Wait():
                return self._result(action, *sim.set_intent(self.player_id, action, None))
            case _:
                return self._result(action, False, f"unsupported action {action!r}")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _snapshot_source(sim: SkeldSimulator, player_id: int) -> Callable[[], dict[str, Any]]:
    return lambda: sim.snapshot(player_id)


def _fixed_executor(controller: SimController) -> Callable[[Callable[[], Any]], SimController]:
    return lambda _source: controller


class SimulationRunner:
    """Steps the simulator and every agent until the game ends."""

    def __init__(
        self,
        sim: SkeldSimulator,
        *,
        bridge: LLMBridge | None = None,
        agent_config: AgentConfig | None = None,
        bridges: Mapping[int, LLMBridge | None] | None = None,
    ) -> None:
        self.sim = sim
        self.controllers = {pid: SimController(sim, pid) for pid in sim.players}
        self.agents: dict[int, AmongUsAgent] = {}
        for pid in sim.players:
            player_bridge = bridges.get(pid, bridge) if bridges is not None else bridge
            controller = self.controllers[pid]
            self.agents[pid] = AmongUsAgent(
                _snapshot_source(sim, pid),
                _fixed_executor(controller),
                bridge=player_bridge,
                pathfinder=sim.pathfinder,
                config=agent_config,
            )
        self._next_decision = dict.fromkeys(sim.players, 0.0)
        self._meeting_queue: deque[int] = deque()
        self._meeting_turn_at = 0.0
        self._meeting_interval = 1.0
        self._meeting_seen: float | None = None

    def _plan_meeting_turns(self) -> None:
        sim = self.sim
        assert sim.meeting is not None
        statements = max(1, next(iter(self.agents.values())).config.statements_per_meeting)
        alive = [pid for pid, p in sim.players.items() if p.alive]
        caller = sim.meeting.caller_id
        order = ([caller] if caller in alive else []) + [pid for pid in alive if pid != caller]
        self._meeting_queue = deque(order * statements)
        self._meeting_interval = max(0.25, sim.settings.discussion_time / max(1, len(self._meeting_queue) + 1))
        self._meeting_turn_at = sim.time
        self._meeting_seen = sim.meeting.started_at

    def step(self) -> None:
        sim = self.sim
        sim.tick()
        if sim.finished:
            return
        states = {pid: agent.refresh() for pid, agent in self.agents.items()}
        phase = sim.phase
        if phase is GamePhase.TASKS:
            for pid, agent in self.agents.items():
                if sim.phase is not GamePhase.TASKS:
                    break
                if self.controllers[pid].idle or sim.time >= self._next_decision[pid]:
                    agent.act(states[pid])
                    self._next_decision[pid] = sim.time + sim.config.decision_interval
        elif phase is GamePhase.MEETING:
            if sim.meeting is not None and self._meeting_seen != sim.meeting.started_at:
                self._plan_meeting_turns()
            if self._meeting_queue and sim.time >= self._meeting_turn_at:
                speaker = self._meeting_queue.popleft()
                self.agents[speaker].act(self.agents[speaker].refresh())
                self._meeting_turn_at = sim.time + self._meeting_interval
        elif phase in (GamePhase.VOTING, GamePhase.EJECTION):
            for pid, agent in self.agents.items():
                if sim.phase is not phase:
                    break
                agent.act(states[pid])

    def run(self) -> GameResult:
        while not self.sim.finished:
            self.step()
        return self.sim.result()


def play_game(config: SimConfig | None = None, *, bridge: LLMBridge | None = None, pathfinder: Pathfinder | None = None,
              agent_config: AgentConfig | None = None) -> tuple[GameResult, list[str]]:
    sim = SkeldSimulator(config, pathfinder)
    result = SimulationRunner(sim, bridge=bridge, agent_config=agent_config).run()
    return result, sim.log


@dataclass(frozen=True, slots=True)
class SeriesSummary:
    games: int
    wins: Mapping[str, int]
    reasons: Mapping[str, int]
    mean_duration: float
    mean_kills: float
    correct_ejections: int
    wrong_ejections: int

    def render(self) -> str:
        lines = [f"games: {self.games}"]
        for side, count in sorted(self.wins.items()):
            lines.append(f"  {side:<10} {count:>4}  ({count / self.games:.0%})")
        lines.append("end reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items())))
        lines.append(f"mean duration: {self.mean_duration:.0f}s   mean kills: {self.mean_kills:.1f}")
        lines.append(f"ejections: {self.correct_ejections} impostors, {self.wrong_ejections} crewmates")
        return "\n".join(lines)


def run_series(games: int, config: SimConfig | None = None, *, bridge: LLMBridge | None = None) -> SeriesSummary:
    base = config or SimConfig()
    pathfinder = Pathfinder()
    results: list[GameResult] = []
    for index in range(games):
        seed = None if base.seed is None else base.seed + index
        result, _ = play_game(replace(base, seed=seed), bridge=bridge, pathfinder=pathfinder)
        results.append(result)
    wins = Counter(r.winner.value if r.winner else "none" for r in results)
    reasons = Counter(r.reason for r in results)
    return SeriesSummary(
        games=games,
        wins=dict(wins),
        reasons=dict(reasons),
        mean_duration=sum(r.duration for r in results) / games,
        mean_kills=sum(len(r.kills) for r in results) / games,
        correct_ejections=sum(r.correct_ejections for r in results),
        wrong_ejections=sum(r.wrong_ejections for r in results),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run headless Among Us games on The Skeld.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--impostors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--llm", choices=["claude", "gpt", "gemini"], default=None,
                        help="use an LLM for meeting statements and votes (needs the provider's API key)")
    parser.add_argument("--log", action="store_true", help="print the event log of each game")
    args = parser.parse_args(argv)

    bridge = LLMBridge.from_environment(preferred=args.llm) if args.llm else None
    config = SimConfig(
        num_players=args.players,
        settings=GameSettings(num_impostors=args.impostors),
        max_game_seconds=args.max_seconds,
        seed=args.seed,
    )
    if args.log or args.games == 1:
        pathfinder = Pathfinder()
        for index in range(args.games):
            seed = None if config.seed is None else config.seed + index
            result, log = play_game(replace(config, seed=seed), bridge=bridge, pathfinder=pathfinder)
            print("\n".join(log))
            winner = result.winner.value if result.winner else "nobody"
            print(f"=> {winner} win ({result.reason}) after {result.duration:.0f}s, "
                  f"{len(result.kills)} kills, {result.meetings} meetings\n")
        return 0
    print(run_series(args.games, config, bridge=bridge).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

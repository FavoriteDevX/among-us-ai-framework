"""Agent runtime: perception -> decision -> action loop for one player.

:class:`AmongUsAgent` glues the modules together. Every state refresh goes
through the tracker, so evidence keeps accumulating while long actions
(walking, tasks) are running. Free-roam decisions come from
:class:`TaskPhasePlanner`, a rule-based policy that can optionally defer to
an LLM which chooses among the actions the planner considers legal.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from .controller import (
    Action,
    ActionResult,
    CallEmergency,
    Chat,
    DoTask,
    FakeTask,
    FixSabotage,
    GameController,
    Kill,
    MoveTo,
    Report,
    Sabotage,
    Vote,
    Wait,
)
from .discussion_engine import DiscussionAgent, EvidenceKind, MeetingContext, Statement
from .game_state import (
    GamePhase,
    GameState,
    GameStateParser,
    GameStateTracker,
    PlayerState,
    Role,
    SabotageType,
)
from .llm_bridge import TASK_PHASE_TEMPLATE, LLMBridge, LLMError
from .pathfinding import NoPathError, Pathfinder

logger = logging.getLogger(__name__)

SnapshotSource = Callable[[], Mapping[str, Any]]


class HttpSnapshotSource:
    """Polls a telemetry endpoint (for example a client mod) that serves snapshot JSON."""

    def __init__(self, url: str, *, timeout: float = 2.0, client: httpx.Client | None = None) -> None:
        self.url = url
        self._client = client or httpx.Client(timeout=timeout)

    def __call__(self) -> Mapping[str, Any]:
        response = self._client.get(self.url)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, Mapping):
            raise ValueError("snapshot endpoint did not return an object")
        return data

    def close(self) -> None:
        self._client.close()


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    report_radius: float = 9.0
    kill_scan_radius: float = 14.0
    witness_radius: float = 9.0
    sabotage_min_cooldown: float = 12.0
    fix_minor_sabotage_radius: float = 25.0
    llm_decisions: bool = False


@dataclass(slots=True)
class Option:
    label: str
    action: Action
    score: float
    target: str | None = None


class TaskPhasePlanner:
    """Free-roam policy. Produces scored legal options and picks the best."""

    def __init__(self, pathfinder: Pathfinder, bridge: LLMBridge | None = None, config: PlannerConfig | None = None) -> None:
        self.pathfinder = pathfinder
        self.bridge = bridge
        self.config = config or PlannerConfig()
        self._failed_tasks: dict[str, float] = {}
        self._last_kill_at: float | None = None

    def mark_task_failed(self, task_id: str, now: float) -> None:
        self._failed_tasks[task_id] = now

    def note_kill(self, now: float) -> None:
        self._last_kill_at = now

    def _distance(self, state: GameState, target: Any) -> float:
        try:
            return self.pathfinder.find_path(state.me.position, target, closed_doors=state.closed_doors).distance
        except NoPathError:
            return math.inf

    # -- option generation -------------------------------------------------- #

    def crewmate_options(self, state: GameState, discussion: DiscussionAgent) -> list[Option]:
        cfg = self.config
        options: list[Option] = []
        me = state.me
        for body in state.bodies:
            if me.position.distance_to(body.position) <= cfg.report_radius:
                options.append(Option("report_body", Report(body.player_id), 100.0, state.players[body.player_id].name))
        decisive = [
            e for e in discussion.model.evidence
            if e.kind in (EvidenceKind.KILL_WITNESSED, EvidenceKind.VENT_WITNESSED) and state.players[e.subject_id].alive
        ]
        if decisive and state.emergency_meetings_left > 0:
            options.append(Option("call_emergency", CallEmergency(), 90.0))
        if state.sabotage is not None and state.sabotage.fix_positions:
            nearest = min(state.sabotage.fix_positions, key=lambda p: me.position.distance_to(p))
            distance = self._distance(state, nearest)
            if state.sabotage.kind.is_critical:
                options.append(Option("fix_sabotage", FixSabotage(nearest), 80.0, state.sabotage.kind.value))
            elif state.sabotage.kind is not SabotageType.DOORS and distance <= cfg.fix_minor_sabotage_radius:
                options.append(Option("fix_sabotage", FixSabotage(nearest), 45.0, state.sabotage.kind.value))
        threats = [e.subject_id for e in decisive]
        for pid in threats:
            threat = state.players[pid]
            if threat.visible and me.position.distance_to(threat.position) < 6.0:
                flee = self.pathfinder.room_center(self._safest_room(state, threat))
                options.append(Option("flee", MoveTo(flee), 95.0, threat.name))
        for task in state.pending_tasks():
            if task.task_id in self._failed_tasks and state.timestamp - self._failed_tasks[task.task_id] < 60.0:
                continue
            distance = self._distance(state, task.position)
            if math.isfinite(distance):
                options.append(Option("do_task", DoTask(task.task_id), 40.0 - distance * 0.25, task.task_id))
        if not any(o.label == "do_task" for o in options):
            crowd = self._crowd_center(state)
            options.append(Option("group_up", MoveTo(crowd), 10.0))
        return options

    def impostor_options(self, state: GameState, discussion: DiscussionAgent) -> list[Option]:
        cfg = self.config
        options: list[Option] = []
        me = state.me
        teammates = state.known_teammates()
        crew_visible = [p for p in state.visible_others() if p.player_id not in teammates]

        if state.kill_cooldown <= 0.0:
            for victim in crew_visible:
                d = me.position.distance_to(victim.position)
                if d > cfg.kill_scan_radius:
                    continue
                witnesses = [
                    p for p in crew_visible
                    if p.player_id != victim.player_id and p.position.distance_to(victim.position) <= cfg.witness_radius
                ]
                if witnesses:
                    continue
                suspicion_of_victim = discussion.model.probabilities(state, exclude_ids=teammates).get(victim.player_id, 0.0)
                score = 85.0 - d - 10.0 * suspicion_of_victim
                options.append(Option("kill", Kill(victim.player_id), score, victim.name))

        if self._last_kill_at is not None and state.timestamp - self._last_kill_at < 4.0:
            far_room = max(
                self.pathfinder.definition.rooms,
                key=lambda r: self.pathfinder.room_center(r).distance_to(me.position),
            )
            options.append(Option("escape", MoveTo(self.pathfinder.room_center(far_room), allow_vents=True), 92.0, far_room.value))

        for body in state.bodies:
            close = me.position.distance_to(body.position) <= cfg.report_radius
            onlookers = [p for p in crew_visible if p.position.distance_to(body.position) <= cfg.witness_radius * 1.5]
            if close and onlookers:
                options.append(Option("self_report", Report(body.player_id), 88.0, state.players[body.player_id].name))

        if state.sabotage is None and state.kill_cooldown >= cfg.sabotage_min_cooldown:
            crowd = len(crew_visible)
            kind = SabotageType.LIGHTS if crowd >= 2 else SabotageType.REACTOR
            options.append(Option("sabotage", Sabotage(kind), 30.0 + 5.0 * crowd, kind.value))
        if state.sabotage is not None and state.sabotage.kind.is_critical and crew_visible:
            # Fixing occasionally while watched builds credibility.
            nearest = min(state.sabotage.fix_positions, key=lambda p: me.position.distance_to(p))
            options.append(Option("fake_fix", MoveTo(nearest), 25.0, state.sabotage.kind.value))

        isolated = self._most_isolated(state, teammates)
        if isolated is not None and state.kill_cooldown < 8.0:
            options.append(Option("hunt", MoveTo(isolated.position), 50.0 - state.kill_cooldown, isolated.name))
        for task in state.tasks:
            distance = self._distance(state, task.position)
            if math.isfinite(distance):
                options.append(Option("fake_task", FakeTask(task.task_id, duration=5.0), 20.0 - distance * 0.2, task.task_id))
        if not options:
            options.append(Option("wait", Wait(1.0), 0.0))
        return options

    def _safest_room(self, state: GameState, threat: PlayerState) -> Any:
        def score(room: Any) -> float:
            center = self.pathfinder.room_center(room)
            people = sum(1 for p in state.visible_others() if p.room is room)
            return center.distance_to(threat.position) * 0.1 + people * 3.0

        return max(self.pathfinder.definition.rooms, key=score)

    def _crowd_center(self, state: GameState) -> Any:
        others = state.visible_others()
        if not others:
            return self.pathfinder.definition.emergency_button
        target = max(others, key=lambda p: sum(1 for o in others if o.position.distance_to(p.position) < 8.0))
        return target.position

    @staticmethod
    def _most_isolated(state: GameState, teammates: frozenset[int]) -> PlayerState | None:
        crew = [p for p in state.visible_others() if p.player_id not in teammates]
        if not crew:
            return None
        return min(crew, key=lambda p: sum(1 for o in crew if o is not p and o.position.distance_to(p.position) < 10.0))

    # -- choice ------------------------------------------------------------- #

    def decide(self, state: GameState, tracker: GameStateTracker, discussion: DiscussionAgent) -> Option:
        options = self.impostor_options(state, discussion) if state.is_impostor else self.crewmate_options(state, discussion)
        options.sort(key=lambda o: o.score, reverse=True)
        best = options[0]
        if not self.config.llm_decisions or self.bridge is None or best.score >= 80.0:
            return best
        shortlist = options[:6]
        names = {pid: p.name for pid, p in state.players.items()}
        events = "\n".join(e.describe(names) for e in tracker.events_since(state.timestamp - 45.0)[-12:]) or "none"
        try:
            result = self.bridge.run_template(
                TASK_PHASE_TEMPLATE,
                {
                    "name": state.me.name,
                    "color": state.me.color.value,
                    "role": state.me.role.value,
                    "room": state.me.room.value if state.me.room else "unknown",
                    "timestamp": f"{state.timestamp:.0f}",
                    "task_bar": f"{state.task_bar:.0%}",
                    "kill_cooldown": f"{state.kill_cooldown:.0f}s" if state.is_impostor else "n/a",
                    "sabotage": state.sabotage.kind.value if state.sabotage else "none",
                    "visible": ", ".join(f"{p.name} in {p.room.value if p.room else '?'}" for p in state.visible_others()) or "nobody",
                    "tasks": ", ".join(f"{t.name} ({t.room.value})" for t in state.pending_tasks()) or "none",
                    "events": events,
                    "actions": "; ".join(f"{i}: {o.label}{' ' + o.target if o.target else ''}" for i, o in enumerate(shortlist)),
                },
            )
        except LLMError as exc:
            logger.debug("LLM decision unavailable: %s", exc)
            return best
        choice = str(result.data.get("action", "")).strip().lower()
        target = str(result.data.get("target") or "").strip().lower()
        for index, option in enumerate(shortlist):
            if choice in (str(index), f"{index}: {option.label}") or (
                choice == option.label and (not target or not option.target or option.target.lower() == target)
            ):
                return option
        return best


@dataclass(slots=True)
class AgentConfig:
    statements_per_meeting: int = 2
    reasoning_budget: int | None = None
    planner: PlannerConfig = field(default_factory=PlannerConfig)


class AmongUsAgent:
    """A complete autonomous player."""

    def __init__(
        self,
        source: SnapshotSource,
        controller_factory: Callable[[Callable[[], GameState]], GameController],
        *,
        bridge: LLMBridge | None = None,
        pathfinder: Pathfinder | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.source = source
        self.bridge = bridge
        self.pathfinder = pathfinder or Pathfinder()
        self.config = config or AgentConfig()
        self.parser = GameStateParser(self.pathfinder.definition.room_at)
        self.tracker = GameStateTracker()
        self.planner = TaskPhasePlanner(self.pathfinder, bridge, self.config.planner)
        self.controller = controller_factory(self.refresh)
        self.discussion: DiscussionAgent | None = None
        self._round_start = 0.0
        self._meeting_key: tuple[int, float] | None = None
        self._statements_made = 0
        self._heard = 0
        self._voted = False
        self.history: list[ActionResult] = []

    # -- perception --------------------------------------------------------- #

    def refresh(self) -> GameState:
        state = self.parser.parse(self.source())
        previous = self.tracker.current
        self.tracker.update(state)
        if self.discussion is None or self.discussion.player_id != state.local_player_id:
            self.discussion = DiscussionAgent(
                state.local_player_id, self.pathfinder, self.bridge, reasoning_budget=self.config.reasoning_budget
            )
        self.discussion.observe_tick(state, self.tracker)
        if previous is not None and previous.phase is not GamePhase.TASKS and state.phase is GamePhase.TASKS:
            self._round_start = state.timestamp
        return state

    # -- loop --------------------------------------------------------------- #

    def step(self) -> ActionResult | None:
        state = self.refresh()
        assert self.discussion is not None
        if state.phase is GamePhase.TASKS:
            self._meeting_key = None
            if not state.me.alive:
                return self._ghost_step(state)
            option = self.planner.decide(state, self.tracker, self.discussion)
            result = self.controller.execute(option.action)
            self.history.append(result)
            if result.success and isinstance(option.action, Kill):
                self.planner.note_kill(self.tracker.current.timestamp if self.tracker.current else state.timestamp)
            if not result.success and isinstance(option.action, DoTask):
                self.planner.mark_task_failed(option.action.task_id, state.timestamp)
            return result
        if state.phase in (GamePhase.MEETING, GamePhase.VOTING):
            return self._meeting_step(state)
        if state.phase is GamePhase.EJECTION:
            self._finish_meeting(state)
        return None

    def _ghost_step(self, state: GameState) -> ActionResult | None:
        if state.is_impostor:
            if state.sabotage is None:
                result = self.controller.execute(Sabotage(SabotageType.REACTOR))
                self.history.append(result)
                return result
            return None
        pending = [t for t in state.pending_tasks()]
        if not pending:
            return None
        task = min(pending, key=lambda t: t.position.distance_to(state.me.position))
        result = self.controller.execute(DoTask(task.task_id))
        self.history.append(result)
        return result

    def _meeting_step(self, state: GameState) -> ActionResult | None:
        assert self.discussion is not None and state.meeting is not None
        key = (state.meeting.caller_id, round(self._round_start, 1))
        if self._meeting_key != key:
            self._meeting_key = key
            self._statements_made = 0
            self._heard = 0
            self._voted = False
            body = next((b for b in self.tracker.events if b.kind.value == "body_discovered" and b.subject_id == state.meeting.reported_body_id), None)
            context = MeetingContext(
                caller_id=state.meeting.caller_id,
                reported_body_id=state.meeting.reported_body_id,
                body_position=body.position if body else None,
                body_room=body.room if body else None,
                round_start=self._round_start,
                started_at=state.timestamp,
            )
            self.discussion.begin_meeting(state, self.tracker, context)

        for message in state.meeting.chat[self._heard :]:
            self.discussion.hear(Statement(message.sender_id, message.text, message.timestamp, claimed_room=None))
        self._heard = len(state.meeting.chat)

        if not state.me.alive:
            return None
        if state.phase is GamePhase.MEETING and self._statements_made < self.config.statements_per_meeting:
            statement = self.discussion.compose_statement()
            self._statements_made += 1
            result = self.controller.execute(Chat(statement.text))
            self.history.append(result)
            return result
        if not self._voted and (state.phase is GamePhase.VOTING or self._statements_made >= self.config.statements_per_meeting):
            decision = self.discussion.decide_vote(state.meeting.votes)
            result = self.controller.execute(Vote(decision.target_id))
            self._voted = result.success
            self.history.append(result)
            return result
        return None

    def _finish_meeting(self, state: GameState) -> None:
        if self.discussion is None or self.discussion.context is None:
            return
        votes = dict(state.meeting.votes) if state.meeting else {}
        ejected = state.ejected_player_id
        role = None
        if ejected is not None and state.settings.confirm_ejects:
            role = Role.IMPOSTOR if ejected in state.impostor_ids else state.players[ejected].role
            if role is Role.UNKNOWN:
                role = None
        self.discussion.end_meeting(votes, ejected, role)

    def run(self, max_steps: int | None = None) -> None:
        steps = 0
        while max_steps is None or steps < max_steps:
            state = self.tracker.current
            if state is not None and state.phase is GamePhase.GAME_OVER:
                return
            self.step()
            steps += 1


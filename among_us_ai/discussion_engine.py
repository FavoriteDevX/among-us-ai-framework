"""Meeting debate engine: suspicion modelling, claim auditing, deception and voting.

Components
----------
* :class:`SuspicionModel` keeps a log-odds score per player and calibrates
  the scores into probabilities whose sum equals the number of impostors
  still alive.
* :class:`EvidenceCollector` turns tracker history into weighted evidence:
  witnessed kills and vents, proximity to bodies, and travel-time alibis
  computed with the pathfinder's distance fields.
* :class:`ClaimLedger` extracts claims from chat (LLM with a deterministic
  parser fallback) and audits them against first-hand observations.
* :class:`DeceptionPlanner` is the impostor's strategic lying module: it
  fabricates low-risk cover stories, chooses scapegoats and selects a
  meeting strategy from the current threat level.
* :class:`VotingPolicy` produces a safe baseline vote that also guards LLM
  output against self-destructive choices.
* :class:`DiscussionAgent` and :class:`MeetingSession` orchestrate a
  multi-agent debate round by round.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from .game_state import (
    EventType,
    GameEvent,
    GameState,
    GameStateTracker,
    Role,
    Room,
    Sighting,
    Vec2,
)
from .llm_bridge import (
    CLAIM_EXTRACTION_TEMPLATE,
    MEETING_STATEMENT_TEMPLATE,
    VOTE_TEMPLATE,
    LLMBridge,
    LLMError,
)
from .pathfinding import Pathfinder

logger = logging.getLogger(__name__)

LOGIT_CAP: Final[float] = 15.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
# Evidence & suspicion
# --------------------------------------------------------------------------- #


class EvidenceKind(str, Enum):
    KILL_WITNESSED = "kill_witnessed"
    VENT_WITNESSED = "vent_witnessed"
    VISUAL_TASK = "visual_task"
    NEAR_BODY = "near_body"
    LAST_WITH_VICTIM = "last_with_victim"
    TRAVEL_ALIBI = "travel_alibi"
    CLAIM_CONTRADICTED = "claim_contradicted"
    CLAIM_CORROBORATED = "claim_corroborated"
    ACCUSED = "accused"
    VOUCHED = "vouched"
    DEFENDED_SUSPECT = "defended_suspect"
    REPORTED_BODY = "reported_body"
    VOTED_CONFIRMED_CREW = "voted_confirmed_crew"
    FOLLOWING = "following"
    CONFIRMED_ROLE = "confirmed_role"


DEFAULT_WEIGHTS: Final[Mapping[EvidenceKind, float]] = {
    EvidenceKind.KILL_WITNESSED: 14.0,
    EvidenceKind.VENT_WITNESSED: 11.0,
    EvidenceKind.VISUAL_TASK: -11.0,
    EvidenceKind.NEAR_BODY: 1.4,
    EvidenceKind.LAST_WITH_VICTIM: 1.6,
    EvidenceKind.TRAVEL_ALIBI: -2.5,
    EvidenceKind.CLAIM_CONTRADICTED: 2.6,
    EvidenceKind.CLAIM_CORROBORATED: -0.7,
    EvidenceKind.ACCUSED: 0.6,
    EvidenceKind.VOUCHED: -0.4,
    EvidenceKind.DEFENDED_SUSPECT: 0.7,
    EvidenceKind.REPORTED_BODY: -0.3,
    EvidenceKind.VOTED_CONFIRMED_CREW: 0.9,
    EvidenceKind.FOLLOWING: 0.5,
    EvidenceKind.CONFIRMED_ROLE: 0.0,
}


@dataclass(frozen=True, slots=True)
class Evidence:
    subject_id: int
    kind: EvidenceKind
    weight: float
    timestamp: float
    note: str
    key: str

    def describe(self, names: Mapping[int, str]) -> str:
        sign = "+" if self.weight >= 0 else ""
        return f"{names.get(self.subject_id, self.subject_id)}: {self.kind.value} ({sign}{self.weight:.1f}) - {self.note}"


class SuspicionModel:
    """Log-odds suspicion with calibration to the number of living impostors.

    Each piece of evidence carries a unique ``key`` so re-deriving the same
    fact (for example on every tick) does not double-count it.
    """

    def __init__(self, owner_id: int, weights: Mapping[EvidenceKind, float] | None = None) -> None:
        self.owner_id = owner_id
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self._evidence: dict[str, Evidence] = {}
        self._known_roles: dict[int, Role] = {}

    def weight_for(self, kind: EvidenceKind, scale: float = 1.0) -> float:
        return self.weights[kind] * scale

    def add(self, evidence: Evidence) -> bool:
        if evidence.subject_id == self.owner_id or evidence.key in self._evidence:
            return False
        self._evidence[evidence.key] = evidence
        return True

    def add_many(self, items: Iterable[Evidence]) -> int:
        return sum(1 for item in items if self.add(item))

    def set_known_role(self, player_id: int, role: Role) -> None:
        self._known_roles[player_id] = role

    def known_role(self, player_id: int) -> Role | None:
        return self._known_roles.get(player_id)

    def evidence_for(self, player_id: int) -> list[Evidence]:
        return sorted((e for e in self._evidence.values() if e.subject_id == player_id), key=lambda e: e.timestamp)

    @property
    def evidence(self) -> list[Evidence]:
        return sorted(self._evidence.values(), key=lambda e: e.timestamp)

    def raw_scores(self, candidates: Iterable[int]) -> dict[int, float]:
        totals: dict[int, float] = {pid: 0.0 for pid in candidates}
        for item in self._evidence.values():
            if item.subject_id in totals:
                totals[item.subject_id] += item.weight
        return {pid: max(-LOGIT_CAP, min(LOGIT_CAP, score)) for pid, score in totals.items()}

    def probabilities(self, state: GameState, *, exclude_ids: Iterable[int] = ()) -> dict[int, float]:
        """Impostor probability for every living candidate, summing to the impostors alive."""
        excluded = set(exclude_ids) | {self.owner_id}
        candidates = [p.player_id for p in state.alive_players() if p.player_id not in excluded]
        if not candidates:
            return {}
        result: dict[int, float] = {}
        open_candidates: list[int] = []
        impostors_left = float(state.alive_impostor_estimate())
        if state.me.role is Role.IMPOSTOR:
            impostors_left -= 1.0
        for pid in candidates:
            role = self._known_roles.get(pid)
            if role is Role.IMPOSTOR:
                result[pid] = 1.0
                impostors_left -= 1.0
            elif role is Role.CREWMATE:
                result[pid] = 0.0
            else:
                open_candidates.append(pid)
        impostors_left = max(0.0, min(impostors_left, float(len(open_candidates))))
        if not open_candidates:
            return result
        if impostors_left == 0.0:
            result.update({pid: 0.0 for pid in open_candidates})
            return result
        if impostors_left == float(len(open_candidates)):
            result.update({pid: 1.0 for pid in open_candidates})
            return result
        scores = self.raw_scores(open_candidates)
        prior = _logit(impostors_left / len(open_candidates))
        lo, hi = -60.0, 60.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            total = sum(_sigmoid(prior + scores[pid] + mid) for pid in open_candidates)
            if total > impostors_left:
                hi = mid
            else:
                lo = mid
        bias = (lo + hi) / 2.0
        result.update({pid: _sigmoid(prior + scores[pid] + bias) for pid in open_candidates})
        return result

    def ranked(self, state: GameState, *, exclude_ids: Iterable[int] = ()) -> list[tuple[int, float]]:
        probs = self.probabilities(state, exclude_ids=exclude_ids)
        return sorted(probs.items(), key=lambda item: item[1], reverse=True)

    def render_table(self, state: GameState, *, exclude_ids: Iterable[int] = ()) -> str:
        rows = []
        for pid, p in self.ranked(state, exclude_ids=exclude_ids):
            top = sorted(self.evidence_for(pid), key=lambda e: abs(e.weight), reverse=True)[:2]
            why = "; ".join(e.note for e in top) or "no evidence"
            rows.append(f"- {state.players[pid].name}: {p:.0%} ({why})")
        return "\n".join(rows) or "- (no candidates)"


# --------------------------------------------------------------------------- #
# Evidence collection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class KillWindow:
    victim_id: int
    body_position: Vec2
    body_room: Room | None
    start: float
    end: float


class EvidenceCollector:
    """Derives evidence from the tracker's events and sighting history."""

    def __init__(
        self,
        pathfinder: Pathfinder,
        model: SuspicionModel,
        *,
        near_body_radius: float = 10.0,
        follow_window: float = 20.0,
    ) -> None:
        self.pathfinder = pathfinder
        self.model = model
        self.near_body_radius = near_body_radius
        self.follow_window = follow_window
        self._processed_events = 0

    def ingest_events(self, tracker: GameStateTracker, state: GameState) -> list[Evidence]:
        events = tracker.events
        fresh = events[self._processed_events :]
        self._processed_events = len(events)
        found: list[Evidence] = []
        for event in fresh:
            found.extend(self._from_event(event, state))
        self.model.add_many(found)
        return found

    def _from_event(self, event: GameEvent, state: GameState) -> list[Evidence]:
        if event.subject_id is None:
            return []
        name = state.players[event.subject_id].name if event.subject_id in state.players else str(event.subject_id)
        room = event.room.value if event.room else "unknown room"
        w = self.model.weight_for
        match event.kind:
            case EventType.KILL_WITNESSED:
                return [
                    Evidence(event.subject_id, EvidenceKind.KILL_WITNESSED, w(EvidenceKind.KILL_WITNESSED), event.timestamp,
                             f"I saw {name} kill in {room}", f"kill:{event.subject_id}:{event.timestamp:.1f}")
                ]
            case EventType.VENT_WITNESSED:
                return [
                    Evidence(event.subject_id, EvidenceKind.VENT_WITNESSED, w(EvidenceKind.VENT_WITNESSED), event.timestamp,
                             f"I saw {name} vent in {room}", f"vent:{event.subject_id}:{event.timestamp:.1f}")
                ]
            case EventType.VISUAL_TASK_WITNESSED:
                self.model.set_known_role(event.subject_id, Role.CREWMATE)
                return [
                    Evidence(event.subject_id, EvidenceKind.VISUAL_TASK, w(EvidenceKind.VISUAL_TASK), event.timestamp,
                             f"I watched {name} do a visual task in {room}", f"visual:{event.subject_id}")
                ]
            case _:
                return []

    def kill_window(self, tracker: GameStateTracker, victim_id: int, body_position: Vec2, body_room: Room | None, found_at: float) -> KillWindow:
        last_alive = tracker.last_seen(victim_id)
        start = last_alive.timestamp if last_alive is not None else max(0.0, found_at - 60.0)
        died = tracker.death_time(victim_id)
        end = min(found_at, died) if died is not None else found_at
        return KillWindow(victim_id, body_position, body_room, min(start, end), end)

    def analyse_body(self, tracker: GameStateTracker, state: GameState, window: KillWindow, reporter_id: int | None) -> list[Evidence]:
        found: list[Evidence] = []
        names = {pid: p.name for pid, p in state.players.items()}
        w = self.model.weight_for
        victim = names.get(window.victim_id, "the victim")

        near = tracker.players_seen_near(window.body_position, self.near_body_radius, window.start - 5.0, window.end)
        for pid, distance in near.items():
            if pid in (window.victim_id, self.model.owner_id):
                continue
            scale = 1.0 - 0.5 * (distance / self.near_body_radius)
            found.append(
                Evidence(pid, EvidenceKind.NEAR_BODY, w(EvidenceKind.NEAR_BODY, scale), window.end,
                         f"seen {distance:.0f}m from {victim}'s body", f"near:{window.victim_id}:{pid}")
            )

        victim_seen = tracker.last_seen(window.victim_id)
        if victim_seen is not None:
            with_victim = tracker.players_seen_near(victim_seen.position, 6.0, victim_seen.timestamp - 4.0, victim_seen.timestamp + 4.0)
            for pid in with_victim:
                if pid in (window.victim_id, self.model.owner_id):
                    continue
                found.append(
                    Evidence(pid, EvidenceKind.LAST_WITH_VICTIM, w(EvidenceKind.LAST_WITH_VICTIM), window.end,
                             f"last seen together with {victim}", f"lastwith:{window.victim_id}:{pid}")
                )

        speed = state.settings.player_speed
        field_ = self.pathfinder.distance_field(window.body_position, allow_vents=True)
        for player in state.alive_players():
            pid = player.player_id
            if pid == self.model.owner_id:
                continue
            if self._has_travel_alibi(tracker.sightings(pid), field_, speed, window):
                found.append(
                    Evidence(pid, EvidenceKind.TRAVEL_ALIBI, w(EvidenceKind.TRAVEL_ALIBI), window.end,
                             f"too far from {victim}'s body during the kill window", f"alibi:{window.victim_id}:{pid}")
                )

        if reporter_id is not None and reporter_id != self.model.owner_id:
            found.append(
                Evidence(reporter_id, EvidenceKind.REPORTED_BODY, w(EvidenceKind.REPORTED_BODY), window.end,
                         f"reported {victim}'s body", f"report:{window.victim_id}:{reporter_id}")
            )
        self.model.add_many(found)
        return found

    def _has_travel_alibi(
        self,
        sightings: Sequence[Sighting],
        field_: Mapping[tuple[int, int], float],
        speed: float,
        window: KillWindow,
    ) -> bool:
        """True when no moment in the kill window is reachable given every sighting.

        Each sighting at time ``t`` and travel distance ``d`` forbids being at
        the body during ``(t - d/speed, t + d/speed)``. If the union of these
        intervals covers the whole kill window, the player has an alibi.
        """
        if speed <= 0 or window.end <= window.start:
            return False
        forbidden: list[tuple[float, float]] = []
        for sighting in sightings:
            if sighting.timestamp < window.start - 60.0 or sighting.timestamp > window.end + 60.0:
                continue
            cell = self.pathfinder.grid.world_to_cell(sighting.position)
            distance = field_.get(cell)
            if distance is None:
                continue
            slack = distance / speed
            if slack <= 0.5:
                continue
            forbidden.append((sighting.timestamp - slack, sighting.timestamp + slack))
        if not forbidden:
            return False
        forbidden.sort()
        cursor = window.start
        for lo, hi in forbidden:
            if lo > cursor:
                return False
            cursor = max(cursor, hi)
            if cursor >= window.end:
                return True
        return cursor >= window.end

    def detect_following(self, tracker: GameStateTracker, state: GameState) -> list[Evidence]:
        """Flag players who stayed close to us across several rooms."""
        me = state.local_player_id
        mine = [s for s in tracker.sightings(me) if s.timestamp >= state.timestamp - self.follow_window]
        if len(mine) < 3:
            return []
        found: list[Evidence] = []
        for player in state.alive_others():
            theirs = tracker.sightings(player.player_id)
            shared_rooms = {
                s.room
                for s in mine
                for o in theirs
                if s.room is not None and s.room is o.room and s.room is not Room.HALLWAY and abs(s.timestamp - o.timestamp) < 3.0
            }
            if len(shared_rooms) >= 3:
                found.append(
                    Evidence(player.player_id, EvidenceKind.FOLLOWING, self.model.weight_for(EvidenceKind.FOLLOWING),
                             state.timestamp, f"{player.name} followed me through {len(shared_rooms)} rooms",
                             f"follow:{player.player_id}:{int(state.timestamp // self.follow_window)}")
                )
        self.model.add_many(found)
        return found


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #


class ClaimKind(str, Enum):
    LOCATION = "location"
    ACCUSE = "accuse"
    VOUCH = "vouch"
    SIGHTING = "sighting"


@dataclass(frozen=True, slots=True)
class Claim:
    speaker_id: int
    kind: ClaimKind
    timestamp: float
    meeting_index: int
    target_id: int | None = None
    room: Room | None = None


@dataclass(frozen=True, slots=True)
class Statement:
    speaker_id: int
    text: str
    timestamp: float
    accuse_id: int | None = None
    vouch_id: int | None = None
    claimed_room: Room | None = None


_ROOM_ALIASES: Final[Mapping[str, Room]] = {
    "cafeteria": Room.CAFETERIA, "cafe": Room.CAFETERIA, "caf": Room.CAFETERIA,
    "weapons": Room.WEAPONS, "weps": Room.WEAPONS,
    "o2": Room.O2, "oxygen": Room.O2,
    "navigation": Room.NAVIGATION, "nav": Room.NAVIGATION,
    "shields": Room.SHIELDS,
    "communications": Room.COMMUNICATIONS, "comms": Room.COMMUNICATIONS,
    "storage": Room.STORAGE,
    "admin": Room.ADMIN,
    "electrical": Room.ELECTRICAL, "elec": Room.ELECTRICAL, "electric": Room.ELECTRICAL,
    "lower engine": Room.LOWER_ENGINE, "lower": Room.LOWER_ENGINE,
    "upper engine": Room.UPPER_ENGINE, "upper": Room.UPPER_ENGINE,
    "security": Room.SECURITY, "secu": Room.SECURITY, "cams": Room.SECURITY,
    "reactor": Room.REACTOR,
    "medbay": Room.MEDBAY, "med": Room.MEDBAY, "medical": Room.MEDBAY,
    "hallway": Room.HALLWAY, "hall": Room.HALLWAY,
}

_ACCUSE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:vote|it'?s|its|sus(?:sy)?|kill(?:ed|er)?|vented|impostor|imposter)\b", re.IGNORECASE
)
_VOUCH_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:safe|clear(?:ed)?|innocent|trust|with me|vouch|not)\b", re.IGNORECASE)
_SELF_LOC_RE: Final[re.Pattern[str]] = re.compile(r"\bi\s*(?:was|am|'m|went)\b[^.!?]*", re.IGNORECASE)


def _find_room(text: str) -> Room | None:
    lowered = text.lower()
    best: tuple[int, Room] | None = None
    for alias, room in _ROOM_ALIASES.items():
        match = re.search(rf"\b{re.escape(alias)}\b", lowered)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), room)
    return best[1] if best else None


def _mentioned_players(text: str, state: GameState, exclude: int) -> list[int]:
    lowered = text.lower()
    found: list[tuple[int, int]] = []
    for player in state.players.values():
        if player.player_id == exclude:
            continue
        for token in {player.name.lower(), player.color.value}:
            match = re.search(rf"\b{re.escape(token)}\b", lowered)
            if match:
                found.append((match.start(), player.player_id))
                break
    return [pid for _, pid in sorted(found)]


def parse_claims_heuristically(statement: Statement, state: GameState, meeting_index: int) -> list[Claim]:
    """Deterministic claim extraction used when the LLM is unavailable."""
    claims: list[Claim] = []
    text = statement.text
    if statement.claimed_room is not None:
        claims.append(Claim(statement.speaker_id, ClaimKind.LOCATION, statement.timestamp, meeting_index, room=statement.claimed_room))
    else:
        self_loc = _SELF_LOC_RE.search(text)
        if self_loc:
            room = _find_room(self_loc.group(0))
            if room is not None:
                claims.append(Claim(statement.speaker_id, ClaimKind.LOCATION, statement.timestamp, meeting_index, room=room))
    mentioned = _mentioned_players(text, state, statement.speaker_id)
    for pid in mentioned:
        pattern = rf"{re.escape(state.players[pid].name)}|{state.players[pid].color.value}"
        window = re.search(rf"(?:.{{0,30}})(?:{pattern})(?:.{{0,30}})", text, re.IGNORECASE)
        context = window.group(0) if window else text
        if statement.accuse_id == pid or (statement.accuse_id is None and _ACCUSE_RE.search(context) and not _VOUCH_RE.search(context)):
            claims.append(Claim(statement.speaker_id, ClaimKind.ACCUSE, statement.timestamp, meeting_index, target_id=pid))
        elif statement.vouch_id == pid or _VOUCH_RE.search(context):
            claims.append(Claim(statement.speaker_id, ClaimKind.VOUCH, statement.timestamp, meeting_index, target_id=pid))
        saw = re.search(rf"saw\s+(?:{pattern})\s+(?:in|at|near|by)\s+([a-z0-9 ]+)", text, re.IGNORECASE)
        if saw:
            room = _find_room(saw.group(1))
            if room is not None:
                claims.append(Claim(statement.speaker_id, ClaimKind.SIGHTING, statement.timestamp, meeting_index, target_id=pid, room=room))
    return claims


class ClaimLedger:
    """Stores all claims and audits them against first-hand observations."""

    def __init__(self, model: SuspicionModel, *, bridge: LLMBridge | None = None, adjacency_tolerance: float = 4.0) -> None:
        self.model = model
        self.bridge = bridge
        self.adjacency_tolerance = adjacency_tolerance
        self.claims: list[Claim] = []
        self.meeting_index = 0

    def new_meeting(self) -> None:
        self.meeting_index += 1

    def claims_by(self, speaker_id: int, kind: ClaimKind | None = None) -> list[Claim]:
        return [c for c in self.claims if c.speaker_id == speaker_id and (kind is None or c.kind is kind)]

    def extract(self, statement: Statement, state: GameState) -> list[Claim]:
        if self.bridge is None:
            return parse_claims_heuristically(statement, state, self.meeting_index)
        players = ", ".join(p.name for p in state.players.values())
        try:
            result = self.bridge.run_template(
                CLAIM_EXTRACTION_TEMPLATE,
                {
                    "players": players,
                    "rooms": ", ".join(r.value for r in Room),
                    "speaker": state.players[statement.speaker_id].name,
                    "message": statement.text.replace('"', "'"),
                },
            )
        except LLMError as exc:
            logger.debug("claim extraction fell back to heuristics: %s", exc)
            return parse_claims_heuristically(statement, state, self.meeting_index)
        data = result.data
        claims: list[Claim] = []
        room = _find_room(str(data.get("claimed_room") or "")) if data.get("claimed_room") else statement.claimed_room
        if room is not None:
            claims.append(Claim(statement.speaker_id, ClaimKind.LOCATION, statement.timestamp, self.meeting_index, room=room))
        for kind, key in ((ClaimKind.ACCUSE, "accused"), (ClaimKind.VOUCH, "vouched")):
            for name in data.get(key) or []:
                target = state.player_by_name(str(name))
                if target is not None and target.player_id != statement.speaker_id:
                    claims.append(Claim(statement.speaker_id, kind, statement.timestamp, self.meeting_index, target_id=target.player_id))
        for item in data.get("saw") or []:
            if not isinstance(item, Mapping):
                continue
            target = state.player_by_name(str(item.get("player", "")))
            seen_room = _find_room(str(item.get("room", "")))
            if target is not None and seen_room is not None:
                claims.append(Claim(statement.speaker_id, ClaimKind.SIGHTING, statement.timestamp, self.meeting_index, target_id=target.player_id, room=seen_room))
        return claims

    def record(self, statement: Statement, state: GameState, tracker: GameStateTracker, round_start: float) -> list[Evidence]:
        claims = self.extract(statement, state)
        self.claims.extend(claims)
        return self.audit(claims, state, tracker, round_start)

    def _rooms_seen(self, tracker: GameStateTracker, player_id: int, start: float, end: float) -> set[Room]:
        return {
            s.room for s in tracker.sightings(player_id)
            if start <= s.timestamp <= end and s.room is not None and s.room is not Room.HALLWAY
        }

    def audit(self, claims: Sequence[Claim], state: GameState, tracker: GameStateTracker, round_start: float) -> list[Evidence]:
        found: list[Evidence] = []
        names = {pid: p.name for pid, p in state.players.items()}
        w = self.model.weight_for
        end = state.timestamp
        probs = self.model.probabilities(state)
        for claim in claims:
            speaker = names.get(claim.speaker_id, "?")
            if claim.kind is ClaimKind.LOCATION and claim.room is not None:
                seen = self._rooms_seen(tracker, claim.speaker_id, round_start, end)
                earlier = [c for c in self.claims_by(claim.speaker_id, ClaimKind.LOCATION) if c.meeting_index == claim.meeting_index and c.room is not claim.room and c is not claim]
                if seen and claim.room not in seen:
                    found.append(Evidence(claim.speaker_id, EvidenceKind.CLAIM_CONTRADICTED, w(EvidenceKind.CLAIM_CONTRADICTED), claim.timestamp,
                                          f"{speaker} claimed {claim.room.value} but I saw them in {', '.join(sorted(r.value for r in seen))}",
                                          f"locx:{claim.meeting_index}:{claim.speaker_id}:{claim.room.value}"))
                elif claim.room in seen:
                    found.append(Evidence(claim.speaker_id, EvidenceKind.CLAIM_CORROBORATED, w(EvidenceKind.CLAIM_CORROBORATED), claim.timestamp,
                                          f"I can confirm {speaker} was in {claim.room.value}",
                                          f"locok:{claim.meeting_index}:{claim.speaker_id}"))
                if earlier:
                    found.append(Evidence(claim.speaker_id, EvidenceKind.CLAIM_CONTRADICTED, w(EvidenceKind.CLAIM_CONTRADICTED, 0.8), claim.timestamp,
                                          f"{speaker} changed their story from {earlier[-1].room.value if earlier[-1].room else '?'} to {claim.room.value}",
                                          f"locchg:{claim.meeting_index}:{claim.speaker_id}"))
            elif claim.kind is ClaimKind.SIGHTING and claim.target_id is not None and claim.room is not None:
                seen = self._rooms_seen(tracker, claim.target_id, round_start, end)
                if seen and claim.room not in seen and claim.target_id != self.model.owner_id:
                    found.append(Evidence(claim.speaker_id, EvidenceKind.CLAIM_CONTRADICTED, w(EvidenceKind.CLAIM_CONTRADICTED, 0.6), claim.timestamp,
                                          f"{speaker} placed {names.get(claim.target_id)} in {claim.room.value}, which contradicts what I saw",
                                          f"sightx:{claim.meeting_index}:{claim.speaker_id}:{claim.target_id}"))
                elif claim.target_id == self.model.owner_id:
                    my_rooms = self._rooms_seen(tracker, self.model.owner_id, round_start, end)
                    if my_rooms and claim.room not in my_rooms:
                        found.append(Evidence(claim.speaker_id, EvidenceKind.CLAIM_CONTRADICTED, w(EvidenceKind.CLAIM_CONTRADICTED), claim.timestamp,
                                              f"{speaker} lied about seeing me in {claim.room.value}",
                                              f"sightme:{claim.meeting_index}:{claim.speaker_id}"))
            elif claim.kind is ClaimKind.ACCUSE and claim.target_id is not None:
                if claim.target_id == self.model.owner_id and state.me.role is Role.CREWMATE:
                    found.append(Evidence(claim.speaker_id, EvidenceKind.CLAIM_CONTRADICTED, w(EvidenceKind.CLAIM_CONTRADICTED, 0.7), claim.timestamp,
                                          f"{speaker} accused me and I know I'm innocent",
                                          f"accme:{claim.meeting_index}:{claim.speaker_id}"))
                elif self.model.known_role(claim.target_id) is Role.CREWMATE:
                    found.append(Evidence(claim.speaker_id, EvidenceKind.DEFENDED_SUSPECT, w(EvidenceKind.DEFENDED_SUSPECT), claim.timestamp,
                                          f"{speaker} accused {names.get(claim.target_id)} who is confirmed crew",
                                          f"acccrew:{claim.meeting_index}:{claim.speaker_id}:{claim.target_id}"))
                else:
                    found.append(Evidence(claim.target_id, EvidenceKind.ACCUSED, w(EvidenceKind.ACCUSED, 1.0 - probs.get(claim.speaker_id, 0.5)), claim.timestamp,
                                          f"accused by {speaker}",
                                          f"acc:{claim.meeting_index}:{claim.speaker_id}:{claim.target_id}"))
            elif claim.kind is ClaimKind.VOUCH and claim.target_id is not None:
                target_p = probs.get(claim.target_id, 0.0)
                if target_p >= 0.75 or self.model.known_role(claim.target_id) is Role.IMPOSTOR:
                    found.append(Evidence(claim.speaker_id, EvidenceKind.DEFENDED_SUSPECT, w(EvidenceKind.DEFENDED_SUSPECT), claim.timestamp,
                                          f"{speaker} defended {names.get(claim.target_id)} despite strong evidence",
                                          f"defend:{claim.meeting_index}:{claim.speaker_id}:{claim.target_id}"))
                else:
                    found.append(Evidence(claim.target_id, EvidenceKind.VOUCHED, w(EvidenceKind.VOUCHED, 1.0 - probs.get(claim.speaker_id, 0.5)), claim.timestamp,
                                          f"vouched for by {speaker}",
                                          f"vouch:{claim.meeting_index}:{claim.speaker_id}:{claim.target_id}"))
        self.model.add_many(found)
        return found


# --------------------------------------------------------------------------- #
# Strategic deception (impostor)
# --------------------------------------------------------------------------- #


class MeetingStrategy(str, Enum):
    TRUTHFUL = "truthful"
    LAY_LOW = "lay_low"
    DEFLECT = "deflect"
    ACCUSE = "accuse"
    BANDWAGON = "bandwagon"
    SOW_DOUBT = "sow_doubt"
    BUS_TEAMMATE = "bus_teammate"


STRATEGY_GUIDANCE: Final[Mapping[MeetingStrategy, str]] = {
    MeetingStrategy.TRUTHFUL: "Share your genuine observations clearly and push the strongest suspect.",
    MeetingStrategy.LAY_LOW: "Say little. Give your cover story once, agree with reasonable points, avoid strong accusations.",
    MeetingStrategy.DEFLECT: "You are under suspicion. Calmly restate your cover story and redirect attention to the scapegoat with one concrete detail.",
    MeetingStrategy.ACCUSE: "Confidently accuse the scapegoat with a specific, plausible observation consistent with your cover story.",
    MeetingStrategy.BANDWAGON: "Support the accusation that is already gaining momentum against the scapegoat; add a small corroborating detail.",
    MeetingStrategy.SOW_DOUBT: "Argue that the evidence is too weak, point out holes in the accusation, and push for a skip.",
    MeetingStrategy.BUS_TEAMMATE: "Your teammate is doomed. Join the vote against them early to build trust for yourself.",
}


@dataclass(frozen=True, slots=True)
class CoverStory:
    room: Room
    task: str | None
    companions: tuple[int, ...]
    risk: float
    truthful: bool

    def describe(self, names: Mapping[int, str]) -> str:
        task = f" doing {self.task}" if self.task else ""
        with_ = f" with {', '.join(names[c] for c in self.companions)}" if self.companions else ""
        return f"I was in {self.room.value}{task}{with_}"


@dataclass(frozen=True, slots=True)
class DeceptionPlan:
    strategy: MeetingStrategy
    cover: CoverStory
    scapegoat_id: int | None
    threat: float
    rationale: str


class PublicPerception:
    """Estimates how suspicious each player looks to the *other* players.

    Impostors use it to judge their own exposure and to pick scapegoats that
    the table will believe.
    """

    def __init__(self) -> None:
        self._scores: dict[int, float] = defaultdict(float)

    def observe_claim(self, claim: Claim) -> None:
        if claim.target_id is None:
            return
        if claim.kind is ClaimKind.ACCUSE:
            self._scores[claim.target_id] += 1.0
        elif claim.kind is ClaimKind.VOUCH:
            self._scores[claim.target_id] -= 0.6

    def observe_votes(self, votes: Mapping[int, int | None]) -> None:
        for target in votes.values():
            if target is not None:
                self._scores[target] += 0.5

    def bump(self, player_id: int, amount: float) -> None:
        self._scores[player_id] += amount

    def heat(self, player_id: int) -> float:
        return self._scores.get(player_id, 0.0)

    def normalized(self, player_ids: Iterable[int]) -> dict[int, float]:
        ids = list(player_ids)
        if not ids:
            return {}
        values = {pid: math.exp(self._scores.get(pid, 0.0) * 0.8) for pid in ids}
        total = sum(values.values())
        return {pid: v / total for pid, v in values.items()}

    def decay(self, factor: float = 0.6) -> None:
        for pid in list(self._scores):
            self._scores[pid] *= factor


class DeceptionPlanner:
    """Plans impostor lies that minimise the chance of contradiction."""

    def __init__(self, pathfinder: Pathfinder, perception: PublicPerception) -> None:
        self.pathfinder = pathfinder
        self.perception = perception
        self.committed: dict[int, CoverStory] = {}

    def _witnesses_of_me(self, tracker: GameStateTracker, state: GameState, start: float) -> dict[Room, set[int]]:
        """Rooms where living players were near me (they could vouch or contradict)."""
        rooms: dict[Room, set[int]] = defaultdict(set)
        my_sightings = [s for s in tracker.sightings(state.local_player_id) if s.timestamp >= start]
        for mine in my_sightings:
            if mine.room is None or mine.room is Room.HALLWAY:
                continue
            others = tracker.players_seen_near(mine.position, state.settings.crewmate_vision * 0.6, mine.timestamp - 1.5, mine.timestamp + 1.5)
            for pid in others:
                if pid != state.local_player_id and state.players[pid].alive:
                    rooms[mine.room].add(pid)
        return rooms

    def fabricate_cover(
        self,
        state: GameState,
        tracker: GameStateTracker,
        *,
        round_start: float,
        kill_room: Room | None,
        meeting_index: int,
    ) -> CoverStory:
        if meeting_index in self.committed:
            return self.committed[meeting_index]
        teammates = state.known_teammates()
        witnessed = self._witnesses_of_me(tracker, state, round_start)
        crew_witness_rooms = {room: {p for p in pids if p not in teammates} for room, pids in witnessed.items()}

        # Partial truth: if innocent crewmates saw us somewhere other than the kill room, use it.
        honest = [(room, pids) for room, pids in crew_witness_rooms.items() if pids and room is not kill_room]
        if honest:
            room, pids = max(honest, key=lambda item: len(item[1]))
            cover = CoverStory(room, self._task_in(room), tuple(sorted(pids)), risk=0.05, truthful=True)
            self.committed[meeting_index] = cover
            return cover

        occupied: dict[Room, set[int]] = defaultdict(set)
        for sighting in tracker.sightings_between(round_start, state.timestamp):
            if sighting.room is not None and sighting.player_id not in teammates and sighting.player_id != state.local_player_id:
                occupied[sighting.room].add(sighting.player_id)

        my_last = tracker.last_seen(state.local_player_id)
        origin = my_last.position if my_last else state.me.position
        speed = max(0.1, state.settings.player_speed)
        span = max(1.0, state.timestamp - round_start)
        field_ = self.pathfinder.distance_field(origin)
        best: tuple[float, Room] | None = None
        for room in self.pathfinder.definition.rooms:
            if room is kill_room:
                continue
            if kill_room is not None and self._adjacent(room, kill_room):
                adjacency_penalty = 0.35
            else:
                adjacency_penalty = 0.0
            alive_occupants = {p for p in occupied.get(room, set()) if state.players[p].alive}
            dead_occupants = {p for p in occupied.get(room, set()) if not state.players[p].alive}
            contradiction_risk = 0.25 * len(alive_occupants)
            travel = field_.get(self.pathfinder.grid.world_to_cell(self.pathfinder.room_center(room)), math.inf) / speed
            travel_penalty = 0.0 if travel <= span * 0.5 else 0.3
            task_bonus = -0.1 if self._task_in(room) else 0.15
            dead_bonus = -0.05 * len(dead_occupants)
            heat_bonus = -0.05 * sum(self.perception.heat(p) for p in alive_occupants)
            risk = max(0.0, contradiction_risk + adjacency_penalty + travel_penalty + task_bonus + dead_bonus + heat_bonus)
            if best is None or risk < best[0]:
                best = (risk, room)
        room = best[1] if best else Room.CAFETERIA
        risk = best[0] if best else 1.0
        companions = tuple(sorted(p for p in teammates if state.players[p].alive)) if risk < 0.2 else ()
        cover = CoverStory(room, self._task_in(room), companions, risk=min(1.0, risk), truthful=False)
        self.committed[meeting_index] = cover
        return cover

    def _task_in(self, room: Room) -> str | None:
        for station in self.pathfinder.definition.task_stations:
            if station.room is room and not station.visual:
                return station.name
        return None

    def _adjacent(self, a: Room, b: Room) -> bool:
        rooms = self.pathfinder.definition.rooms
        ra, rb = rooms[a], rooms[b]
        gap_x = max(0.0, max(ra.x0, rb.x0) - min(ra.x1, rb.x1))
        gap_y = max(0.0, max(ra.y0, rb.y0) - min(ra.y1, rb.y1))
        return math.hypot(gap_x, gap_y) <= 10.0

    def choose_scapegoat(self, state: GameState, model: SuspicionModel, accusers_of_me: Iterable[int] = ()) -> int | None:
        teammates = state.known_teammates()
        accusers = set(accusers_of_me)
        best: tuple[float, int] | None = None
        for player in state.alive_others():
            pid = player.player_id
            if pid in teammates or model.known_role(pid) is Role.CREWMATE:
                continue
            score = self.perception.heat(pid) * 1.0
            score += sum(e.weight for e in model.evidence_for(pid) if e.kind in (EvidenceKind.NEAR_BODY, EvidenceKind.LAST_WITH_VICTIM))
            if pid in accusers:
                score += 0.8
            if any(e.kind is EvidenceKind.TRAVEL_ALIBI for e in model.evidence_for(pid)):
                score -= 1.5
            if best is None or score > best[0]:
                best = (score, pid)
        return best[1] if best else None

    def plan(
        self,
        state: GameState,
        tracker: GameStateTracker,
        model: SuspicionModel,
        *,
        round_start: float,
        kill_room: Room | None,
        meeting_index: int,
        accusers_of_me: Iterable[int] = (),
    ) -> DeceptionPlan:
        me = state.local_player_id
        alive_ids = [p.player_id for p in state.alive_players()]
        public = self.perception.normalized(alive_ids)
        my_heat = public.get(me, 0.0)
        fair_share = 1.0 / max(1, len(alive_ids))
        threat = min(1.0, my_heat / (fair_share * 3.0))
        cover = self.fabricate_cover(state, tracker, round_start=round_start, kill_room=kill_room, meeting_index=meeting_index)
        scapegoat = self.choose_scapegoat(state, model, accusers_of_me)

        teammates = [t for t in state.known_teammates() if state.players[t].alive]
        doomed_teammate = next((t for t in teammates if public.get(t, 0.0) >= 0.5), None)
        crew_alive = len(alive_ids) - len(teammates) - 1
        impostors_alive = len(teammates) + 1

        if doomed_teammate is not None and impostors_alive > 1 and crew_alive > impostors_alive + 1:
            strategy = MeetingStrategy.BUS_TEAMMATE
            scapegoat = doomed_teammate
            rationale = "teammate is about to be ejected; voting with the crowd buys credibility"
        elif threat >= 0.6:
            strategy = MeetingStrategy.DEFLECT if scapegoat is not None else MeetingStrategy.SOW_DOUBT
            rationale = "I am the main suspect; redirect attention"
        elif scapegoat is not None and public.get(scapegoat, 0.0) >= 2.0 * fair_share:
            strategy = MeetingStrategy.BANDWAGON
            rationale = "momentum already exists against a crewmate"
        elif scapegoat is not None and cover.risk <= 0.3 and threat < 0.3:
            strategy = MeetingStrategy.ACCUSE
            rationale = "low exposure and a credible scapegoat; take initiative"
        elif any(public.get(t, 0.0) >= 2.0 * fair_share for t in teammates):
            strategy = MeetingStrategy.SOW_DOUBT
            rationale = "protect a teammate under pressure by pushing a skip"
        else:
            strategy = MeetingStrategy.LAY_LOW
            rationale = "no pressure; stay unremarkable"
        return DeceptionPlan(strategy, cover, scapegoat, threat, rationale)


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VoteDecision:
    target_id: int | None
    confidence: float
    reasoning: str
    source: str


class VotingPolicy:
    def __init__(self, *, threshold: float = 0.55, margin: float = 0.12, desperate_threshold: float = 0.35) -> None:
        self.threshold = threshold
        self.margin = margin
        self.desperate_threshold = desperate_threshold

    def crewmate_vote(self, state: GameState, model: SuspicionModel) -> VoteDecision:
        ranked = model.ranked(state)
        if not ranked:
            return VoteDecision(None, 1.0, "no candidates", "policy")
        top_id, top_p = ranked[0]
        second_p = ranked[1][1] if len(ranked) > 1 else 0.0
        alive = len(state.alive_players())
        impostors = state.alive_impostor_estimate()
        # If skipping could let the impostors reach parity after the next kill, act on weaker evidence.
        desperate = alive - impostors <= impostors + 2
        threshold = self.desperate_threshold if desperate else self.threshold
        if top_p >= threshold and top_p - second_p >= self.margin:
            return VoteDecision(top_id, top_p, f"{state.players[top_id].name} is most likely impostor ({top_p:.0%})", "policy")
        return VoteDecision(None, 1.0 - top_p, f"evidence too weak (top {top_p:.0%})", "policy")

    def impostor_vote(self, state: GameState, plan: DeceptionPlan | None, perception: PublicPerception, current_votes: Mapping[int, int | None]) -> VoteDecision:
        teammates = state.known_teammates()
        tally = Counter(t for t in current_votes.values() if t is not None)
        alive = len(state.alive_players())
        if tally:
            leader, count = tally.most_common(1)[0]
            if leader not in teammates and leader != state.local_player_id and count >= max(1, (alive - 1) // 3):
                return VoteDecision(leader, 0.8, "bandwagon on a crewmate who is already losing the vote", "policy")
        if plan is not None and plan.strategy is MeetingStrategy.BUS_TEAMMATE and plan.scapegoat_id is not None:
            return VoteDecision(plan.scapegoat_id, 0.7, "bus a doomed teammate to build trust", "policy")
        if plan is not None and plan.strategy in (MeetingStrategy.ACCUSE, MeetingStrategy.DEFLECT, MeetingStrategy.BANDWAGON) and plan.scapegoat_id is not None:
            return VoteDecision(plan.scapegoat_id, 0.6, "follow through on the accusation", "policy")
        return VoteDecision(None, 0.5, "skip to avoid drawing attention", "policy")


# --------------------------------------------------------------------------- #
# Agents & meeting orchestration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MeetingContext:
    caller_id: int
    reported_body_id: int | None
    body_position: Vec2 | None
    body_room: Room | None
    round_start: float
    started_at: float

    def reason(self, names: Mapping[int, str]) -> str:
        caller = names.get(self.caller_id, "someone")
        if self.reported_body_id is None:
            return f"emergency meeting called by {caller}"
        room = self.body_room.value if self.body_room else "unknown room"
        return f"{caller} reported {names.get(self.reported_body_id, 'a')}'s body in {room}"


class DiscussionAgent:
    """One player's meeting brain: evidence, claims, deception and voting."""

    def __init__(
        self,
        player_id: int,
        pathfinder: Pathfinder,
        bridge: LLMBridge | None,
        *,
        voting_policy: VotingPolicy | None = None,
        reasoning_budget: int | None = None,
    ) -> None:
        self.player_id = player_id
        self.pathfinder = pathfinder
        self.bridge = bridge
        self.model = SuspicionModel(player_id)
        self.collector = EvidenceCollector(pathfinder, self.model)
        self.ledger = ClaimLedger(self.model, bridge=bridge)
        self.perception = PublicPerception()
        self.deception = DeceptionPlanner(pathfinder, self.perception)
        self.voting = voting_policy or VotingPolicy()
        self.reasoning_budget = reasoning_budget
        self.own_statements: list[Statement] = []
        self.transcript: list[Statement] = []
        self.context: MeetingContext | None = None
        self.plan: DeceptionPlan | None = None
        self.state: GameState | None = None
        self.tracker: GameStateTracker | None = None
        self._accusers_of_me: set[int] = set()

    # -- lifecycle ---------------------------------------------------------- #

    def observe_tick(self, state: GameState, tracker: GameStateTracker) -> None:
        self.state = state
        self.tracker = tracker
        self.collector.ingest_events(tracker, state)
        if state.is_impostor:
            for pid in state.impostor_ids:
                self.model.set_known_role(pid, Role.IMPOSTOR)

    def begin_meeting(self, state: GameState, tracker: GameStateTracker, context: MeetingContext) -> None:
        self.observe_tick(state, tracker)
        self.context = context
        self.transcript = []
        self._accusers_of_me = set()
        self.ledger.new_meeting()
        self.perception.decay()
        if context.reported_body_id is not None and context.body_position is not None:
            window = self.collector.kill_window(tracker, context.reported_body_id, context.body_position, context.body_room, context.started_at)
            evidence = self.collector.analyse_body(tracker, state, window, context.caller_id)
            for item in evidence:
                if item.kind in (EvidenceKind.NEAR_BODY, EvidenceKind.LAST_WITH_VICTIM):
                    self.perception.bump(item.subject_id, 0.4)
        self.collector.detect_following(tracker, state)
        if state.is_impostor:
            self.plan = self.deception.plan(
                state, tracker, self.model,
                round_start=context.round_start,
                kill_room=context.body_room,
                meeting_index=self.ledger.meeting_index,
            )
        else:
            self.plan = None

    def end_meeting(self, votes: Mapping[int, int | None], ejected_id: int | None, ejected_role: Role | None) -> None:
        self.perception.observe_votes(votes)
        if ejected_id is not None and ejected_role is not None:
            self.model.set_known_role(ejected_id, ejected_role)
            if ejected_role is Role.CREWMATE and self.state is not None:
                for voter, target in votes.items():
                    if target == ejected_id and voter != self.player_id:
                        self.model.add(Evidence(voter, EvidenceKind.VOTED_CONFIRMED_CREW, self.model.weight_for(EvidenceKind.VOTED_CONFIRMED_CREW),
                                                self.state.timestamp, "voted out a confirmed crewmate",
                                                f"badvote:{self.ledger.meeting_index}:{voter}"))
        self.context = None

    # -- discussion --------------------------------------------------------- #

    def _names(self) -> dict[int, str]:
        assert self.state is not None
        return {pid: p.name for pid, p in self.state.players.items()}

    def _evidence_summary(self, limit: int = 8) -> str:
        names = self._names()
        items = sorted(self.model.evidence, key=lambda e: abs(e.weight), reverse=True)[:limit]
        return "\n".join(f"- {e.describe(names)}" for e in items) or "- nothing notable"

    def _transcript_text(self, limit: int = 20) -> str:
        names = self._names()
        return "\n".join(f"{names.get(s.speaker_id, '?')}: {s.text}" for s in self.transcript[-limit:]) or "(no messages yet)"

    def _truthful_cover(self) -> str:
        assert self.state is not None and self.tracker is not None and self.context is not None
        rooms = [s.room for s in self.tracker.sightings(self.player_id) if s.timestamp >= self.context.round_start and s.room and s.room is not Room.HALLWAY]
        if not rooms:
            return "no specific location"
        common = Counter(rooms).most_common(1)[0][0]
        return f"I was mostly in {common.value}"

    def compose_statement(self) -> Statement:
        if self.state is None or self.context is None:
            raise RuntimeError("compose_statement called outside a meeting")
        state = self.state
        names = self._names()
        is_impostor = state.is_impostor
        teammates = state.known_teammates()
        exclude = teammates if is_impostor else frozenset()
        strategy = self.plan.strategy if self.plan else MeetingStrategy.TRUTHFUL
        cover = self.plan.cover.describe(names) if self.plan else self._truthful_cover()
        variables = {
            "name": state.me.name,
            "color": state.me.color.value,
            "role": state.me.role.value,
            "reason": self.context.reason(names),
            "evidence": self._evidence_summary(),
            "suspicion": self.model.render_table(state, exclude_ids=exclude),
            "strategy": STRATEGY_GUIDANCE[strategy]
            + (f" Scapegoat: {names[self.plan.scapegoat_id]}." if self.plan and self.plan.scapegoat_id is not None else ""),
            "cover_story": cover,
            "own_history": " | ".join(s.text for s in self.own_statements[-4:]) or "(none)",
            "transcript": self._transcript_text(),
        }
        statement: Statement | None = None
        if self.bridge is not None:
            try:
                result = self.bridge.run_template(MEETING_STATEMENT_TEMPLATE, variables, reasoning_budget=self.reasoning_budget)
                data = result.data
                accuse = state.player_by_name(str(data["accuse"])) if data.get("accuse") else None
                vouch = state.player_by_name(str(data["vouch"])) if data.get("vouch") else None
                if is_impostor and accuse is not None and accuse.player_id in teammates and strategy is not MeetingStrategy.BUS_TEAMMATE:
                    accuse = None
                claimed = _find_room(str(data.get("claimed_room") or "")) if data.get("claimed_room") else None
                if is_impostor and self.plan is not None and claimed is not None and claimed is not self.plan.cover.room:
                    claimed = self.plan.cover.room
                statement = Statement(
                    self.player_id,
                    str(data["message"]).strip()[:200],
                    state.timestamp,
                    accuse.player_id if accuse else None,
                    vouch.player_id if vouch else None,
                    claimed,
                )
            except LLMError as exc:
                logger.warning("statement generation failed for %s: %s", state.me.name, exc)
        if statement is None:
            statement = self._fallback_statement()
        self.own_statements.append(statement)
        return statement

    def _fallback_statement(self) -> Statement:
        assert self.state is not None and self.context is not None
        state = self.state
        names = self._names()
        if self.plan is not None:
            cover = self.plan.cover
            target = self.plan.scapegoat_id
            if self.plan.strategy in (MeetingStrategy.ACCUSE, MeetingStrategy.DEFLECT, MeetingStrategy.BANDWAGON, MeetingStrategy.BUS_TEAMMATE) and target is not None:
                text = f"{cover.describe(names)}. {names[target]} was acting weird near there, I'm voting {names[target]}."
                return Statement(self.player_id, text, state.timestamp, accuse_id=target, claimed_room=cover.room)
            if self.plan.strategy is MeetingStrategy.SOW_DOUBT:
                text = f"{cover.describe(names)}. Nobody has real proof, we should skip this one."
                return Statement(self.player_id, text, state.timestamp, claimed_room=cover.room)
            return Statement(self.player_id, f"{cover.describe(names)}. Didn't see anything.", state.timestamp, claimed_room=cover.room)
        ranked = self.model.ranked(state)
        decisive = [e for e in self.model.evidence if e.kind in (EvidenceKind.KILL_WITNESSED, EvidenceKind.VENT_WITNESSED)]
        where = self._truthful_cover()
        if decisive:
            e = decisive[-1]
            return Statement(self.player_id, f"{e.note}. Vote {names[e.subject_id]}!", state.timestamp, accuse_id=e.subject_id)
        if ranked and ranked[0][1] >= self.voting.threshold:
            pid, p = ranked[0]
            top_note = sorted(self.model.evidence_for(pid), key=lambda x: abs(x.weight), reverse=True)
            reason = top_note[0].note if top_note else "their story doesn't add up"
            return Statement(self.player_id, f"{where}. {names[pid]} is sus: {reason}.", state.timestamp, accuse_id=pid)
        cleared = [e.subject_id for e in self.model.evidence if e.kind is EvidenceKind.VISUAL_TASK]
        if cleared:
            pid = cleared[-1]
            return Statement(self.player_id, f"{where}. {names[pid]} is clear, I saw their visual task.", state.timestamp, vouch_id=pid)
        return Statement(self.player_id, f"{where}. I don't have enough info, skip unless someone has proof.", state.timestamp)

    def hear(self, statement: Statement) -> None:
        if self.state is None or self.tracker is None or self.context is None:
            raise RuntimeError("hear called outside a meeting")
        self.transcript.append(statement)
        if statement.speaker_id == self.player_id:
            return
        claims = self.ledger.extract(statement, self.state)
        self.ledger.claims.extend(claims)
        for claim in claims:
            self.perception.observe_claim(claim)
            if claim.kind is ClaimKind.ACCUSE and claim.target_id == self.player_id:
                self._accusers_of_me.add(statement.speaker_id)
        self.ledger.audit(claims, self.state, self.tracker, self.context.round_start)
        if self.state.is_impostor and self.plan is not None and self._accusers_of_me:
            self.plan = self.deception.plan(
                self.state, self.tracker, self.model,
                round_start=self.context.round_start,
                kill_room=self.context.body_room,
                meeting_index=self.ledger.meeting_index,
                accusers_of_me=self._accusers_of_me,
            )

    # -- voting ------------------------------------------------------------- #

    def decide_vote(self, current_votes: Mapping[int, int | None] | None = None) -> VoteDecision:
        if self.state is None:
            raise RuntimeError("decide_vote called outside a meeting")
        state = self.state
        votes = dict(current_votes or {})
        if state.is_impostor:
            baseline = self.voting.impostor_vote(state, self.plan, self.perception, votes)
        else:
            baseline = self.voting.crewmate_vote(state, self.model)
        if self.bridge is None:
            return baseline
        names = self._names()
        teammates = state.known_teammates()
        targets = [p for p in state.alive_others()]
        try:
            result = self.bridge.run_template(
                VOTE_TEMPLATE,
                {
                    "name": state.me.name,
                    "color": state.me.color.value,
                    "role": state.me.role.value,
                    "alive": ", ".join(p.name for p in state.alive_players()),
                    "impostors_alive": state.alive_impostor_estimate(),
                    "suspicion": self.model.render_table(state, exclude_ids=teammates),
                    "recommendation": f"{names[baseline.target_id] if baseline.target_id is not None else 'skip'} ({baseline.reasoning})",
                    "transcript": self._transcript_text(30),
                    "targets": ", ".join(p.name for p in targets),
                },
                reasoning_budget=self.reasoning_budget,
            )
        except LLMError as exc:
            logger.warning("vote generation failed for %s: %s", state.me.name, exc)
            return baseline
        raw_vote = str(result.data["vote"]).strip()
        confidence = max(0.0, min(1.0, float(result.data["confidence"])))
        reasoning = str(result.data.get("reasoning", ""))
        if raw_vote.lower() in ("skip", "none", ""):
            return VoteDecision(None, confidence, reasoning, "llm")
        chosen = state.player_by_name(raw_vote)
        if chosen is None or not chosen.alive or chosen.player_id == self.player_id:
            return baseline
        return self._guard_vote(VoteDecision(chosen.player_id, confidence, reasoning, "llm"), baseline)

    def _guard_vote(self, proposed: VoteDecision, baseline: VoteDecision) -> VoteDecision:
        """Reject LLM votes that contradict hard evidence."""
        assert self.state is not None and proposed.target_id is not None
        state = self.state
        target = proposed.target_id
        if state.is_impostor:
            if target in state.known_teammates() and not (self.plan and self.plan.strategy is MeetingStrategy.BUS_TEAMMATE):
                return baseline
            return proposed
        if self.model.known_role(target) is Role.CREWMATE:
            return baseline
        decisive = {e.subject_id for e in self.model.evidence if e.kind in (EvidenceKind.KILL_WITNESSED, EvidenceKind.VENT_WITNESSED)}
        if decisive and target not in decisive:
            return baseline
        probs = self.model.probabilities(state)
        if probs.get(target, 0.0) < 0.15:
            return baseline
        return proposed


@dataclass(frozen=True, slots=True)
class MeetingOutcome:
    statements: tuple[Statement, ...]
    votes: Mapping[int, int | None]
    ejected_id: int | None
    tie: bool


class MeetingSession:
    """Runs a round-robin debate between several :class:`DiscussionAgent` instances.

    Used for self-play in the simulator, where every player is an agent that
    sees its own perspective of the game state.
    """

    def __init__(self, agents: Mapping[int, DiscussionAgent], *, rounds: int = 2, speak_order: Sequence[int] | None = None) -> None:
        if not agents:
            raise ValueError("a meeting needs at least one agent")
        self.agents = dict(agents)
        self.rounds = rounds
        self.speak_order = list(speak_order) if speak_order else sorted(self.agents)

    def run(
        self,
        states: Mapping[int, GameState],
        trackers: Mapping[int, GameStateTracker],
        context: MeetingContext,
        human_statements: Iterable[Statement] = (),
    ) -> MeetingOutcome:
        alive = [pid for pid in self.speak_order if states[pid].me.alive]
        for pid in alive:
            self.agents[pid].begin_meeting(states[pid], trackers[pid], context)
        statements: list[Statement] = list(human_statements)
        for statement in statements:
            for pid in alive:
                self.agents[pid].hear(statement)
        order = [context.caller_id] + [pid for pid in alive if pid != context.caller_id] if context.caller_id in alive else alive
        for _ in range(self.rounds):
            for speaker in order:
                statement = self.agents[speaker].compose_statement()
                statements.append(statement)
                for listener in alive:
                    self.agents[listener].hear(statement)
        votes: dict[int, int | None] = {}
        for voter in order:
            decision = self.agents[voter].decide_vote(votes)
            votes[voter] = decision.target_id
        ejected, tie = tally_votes(votes)
        return MeetingOutcome(tuple(statements), votes, ejected, tie)


def tally_votes(votes: Mapping[int, int | None]) -> tuple[int | None, bool]:
    """Return ``(ejected_id, tie)``. Skips count as a candidate; ties eject nobody."""
    counts: Counter[int | None] = Counter(votes.values())
    if not counts:
        return None, False
    ranked = counts.most_common()
    top_target, top_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_count:
        return None, True
    return top_target, False

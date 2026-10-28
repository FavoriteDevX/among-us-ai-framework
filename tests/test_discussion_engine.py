from __future__ import annotations

import copy

import pytest

from among_us_ai.discussion_engine import (
    ClaimKind,
    DiscussionAgent,
    Evidence,
    EvidenceKind,
    MeetingContext,
    MeetingSession,
    MeetingStrategy,
    Statement,
    SuspicionModel,
    parse_claims_heuristically,
    tally_votes,
)
from among_us_ai.game_state import EventType, GameEvent, GameState, GameStateParser, GameStateTracker, Role, Room, Vec2
from among_us_ai.llm_bridge import LLMBridge, ScriptedProvider
from among_us_ai.pathfinding import SKELD, Pathfinder

from .conftest import make_snapshot

PARSER = GameStateParser(SKELD.room_at)


def _timeline(local: int = 0, roles: dict[int, str] | None = None) -> tuple[list[GameState], GameStateTracker]:
    """Green dies in electrical at t=20 next to Blue; Pink stays in navigation the whole round."""
    tracker = GameStateTracker()
    states: list[GameState] = []
    for t in range(0, 31):
        snap = make_snapshot(tick=t, timestamp=float(t), local_player_id=local)
        players = snap["players"]
        if roles:
            for p in players:
                p["role"] = roles.get(p["id"], "unknown")
        players[0]["position"] = [36, 40]  # Red watches electrical
        players[1]["position"] = [38 if t < 20 else 40, 39]
        players[2]["position"] = [37, 39]
        players[3]["position"] = [93, 28]
        players[4]["position"] = [60, 12] if t < 15 else [52, 30]
        if t >= 20:
            players[2]["alive"] = False
            snap["bodies"] = [{"player_id": 2, "position": [37, 39]}]
        state = PARSER.parse(snap)
        tracker.update(state)
        states.append(state)
    return states, tracker


def test_suspicion_calibration_sums_to_impostors() -> None:
    state = PARSER.parse(make_snapshot(settings={"num_impostors": 2}))
    model = SuspicionModel(0)
    model.add(Evidence(1, EvidenceKind.NEAR_BODY, 1.4, 1.0, "near", "k1"))
    model.add(Evidence(3, EvidenceKind.TRAVEL_ALIBI, -2.5, 1.0, "far", "k2"))
    assert not model.add(Evidence(1, EvidenceKind.NEAR_BODY, 1.4, 1.0, "near", "k1"))
    probs = model.probabilities(state)
    assert sum(probs.values()) == pytest.approx(2.0, abs=1e-6)
    assert probs[1] > probs[2] > probs[3]
    model.set_known_role(4, Role.CREWMATE)
    probs = model.probabilities(state)
    assert probs[4] == 0.0 and sum(probs.values()) == pytest.approx(2.0, abs=1e-6)


def test_witnessed_vent_dominates() -> None:
    state = PARSER.parse(make_snapshot())
    model = SuspicionModel(0)
    model.add(Evidence(4, EvidenceKind.VENT_WITNESSED, 11.0, 1.0, "vent", "v"))
    ranked = model.ranked(state)
    assert ranked[0][0] == 4 and ranked[0][1] > 0.99


def test_body_analysis_produces_alibis_and_proximity(pathfinder: Pathfinder) -> None:
    states, tracker = _timeline()
    agent = DiscussionAgent(0, pathfinder, None)
    context = MeetingContext(0, 2, Vec2(37, 39), Room.ELECTRICAL, round_start=0.0, started_at=30.0)
    agent.begin_meeting(states[-1], tracker, context)
    kinds = {(e.subject_id, e.kind) for e in agent.model.evidence}
    assert (1, EvidenceKind.NEAR_BODY) in kinds
    assert (1, EvidenceKind.LAST_WITH_VICTIM) in kinds
    assert (3, EvidenceKind.TRAVEL_ALIBI) in kinds
    assert (1, EvidenceKind.TRAVEL_ALIBI) not in kinds
    ranked = agent.model.ranked(states[-1])
    assert ranked[0][0] == 1


def test_claim_contradiction_is_detected(pathfinder: Pathfinder) -> None:
    states, tracker = _timeline()
    agent = DiscussionAgent(0, pathfinder, None)
    agent.begin_meeting(states[-1], tracker, MeetingContext(0, 2, Vec2(37, 39), Room.ELECTRICAL, 0.0, 30.0))
    agent.hear(Statement(1, "I was in medbay doing scan, Pink is sus", 31.0))
    notes = [e for e in agent.model.evidence_for(1) if e.kind is EvidenceKind.CLAIM_CONTRADICTED]
    assert notes and "medbay" in notes[0].note


def test_heuristic_claim_parser() -> None:
    state = PARSER.parse(make_snapshot())
    claims = parse_claims_heuristically(Statement(1, "I was in nav. vote pink, I saw lime in admin. green is safe", 1.0), state, 1)
    kinds = {(c.kind, c.target_id, c.room) for c in claims}
    assert (ClaimKind.LOCATION, None, Room.NAVIGATION) in kinds
    assert (ClaimKind.ACCUSE, 3, None) in kinds
    assert (ClaimKind.SIGHTING, 4, Room.ADMIN) in kinds
    assert (ClaimKind.VOUCH, 2, None) in kinds


def test_impostor_cover_story_avoids_kill_room(pathfinder: Pathfinder) -> None:
    roles = {0: "crewmate", 1: "impostor"}
    states, tracker = _timeline(local=1, roles=roles)
    final = copy.copy(states[-1])
    agent = DiscussionAgent(1, pathfinder, None)
    agent.begin_meeting(final, tracker, MeetingContext(0, 2, Vec2(37, 39), Room.ELECTRICAL, 0.0, 30.0))
    assert agent.plan is not None
    assert agent.plan.cover.room is not Room.ELECTRICAL
    assert agent.plan.scapegoat_id not in (1, None)
    statement = agent.compose_statement()
    assert statement.claimed_room is agent.plan.cover.room
    assert agent.plan.strategy in set(MeetingStrategy)


def test_tally_votes() -> None:
    assert tally_votes({0: 1, 2: 1, 3: None}) == (1, False)
    assert tally_votes({0: 1, 2: 3}) == (None, True)
    assert tally_votes({0: None, 2: None, 3: 1}) == (None, False)
    assert tally_votes({}) == (None, False)


def test_meeting_session_ejects_the_killer(pathfinder: Pathfinder) -> None:
    perspectives: dict[int, GameState] = {}
    trackers: dict[int, GameStateTracker] = {}
    for pid in (0, 1, 3, 4):
        r = {p: ("crewmate" if p == pid else "unknown") for p in range(5)}
        if pid == 1:
            r = {0: "crewmate", 1: "impostor", 2: "crewmate", 3: "crewmate", 4: "crewmate"}
        states, tracker = _timeline(local=pid, roles=r)
        perspectives[pid], trackers[pid] = states[-1], tracker
    # Red (0) witnessed the kill directly.
    trackers[0]._events.append(GameEvent(EventType.KILL_WITNESSED, 20.0, 1, Room.ELECTRICAL, Vec2(38, 39), {"victim_id": 2}))
    agents = {pid: DiscussionAgent(pid, pathfinder, None) for pid in perspectives}
    for pid, agent in agents.items():
        agent.observe_tick(perspectives[pid], trackers[pid])
    session = MeetingSession(agents, rounds=1)
    outcome = session.run(perspectives, trackers, MeetingContext(0, 2, Vec2(37, 39), Room.ELECTRICAL, 0.0, 30.0))
    assert outcome.votes[0] == 1
    assert outcome.ejected_id == 1
    assert any("Vote Blue" in s.text for s in outcome.statements)


def test_llm_vote_is_guarded_against_cleared_players(pathfinder: Pathfinder) -> None:
    states, tracker = _timeline()
    reply = '{"vote": "Pink", "confidence": 0.9, "reasoning": "gut feeling"}'
    statement = '{"message": "Blue was next to the body", "accuse": "Blue", "reasoning": "proximity"}'
    claim = '{"reasoning": "none"}'

    def responder(request) -> str:
        template = request.metadata.get("template")
        return {"vote_decision": reply, "meeting_statement": statement}.get(template, claim)

    bridge = LLMBridge({"s": ScriptedProvider(responder)})
    agent = DiscussionAgent(0, pathfinder, bridge)
    agent.begin_meeting(states[-1], tracker, MeetingContext(0, 2, Vec2(37, 39), Room.ELECTRICAL, 0.0, 30.0))
    agent.model.set_known_role(3, Role.CREWMATE)
    said = agent.compose_statement()
    assert said.accuse_id == 1
    decision = agent.decide_vote()
    assert decision.target_id != 3

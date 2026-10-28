from __future__ import annotations

import pytest

from among_us_ai.controller import DoTask, FixSabotage, Kill, MoveTo, Sabotage, Vote
from among_us_ai.game_state import GamePhase, GameSettings, GameStateParser, Role, Room, SabotageType, Vec2
from among_us_ai.pathfinding import SKELD, Pathfinder
from among_us_ai.simulator import SimConfig, SimController, SimulationRunner, SkeldSimulator, play_game

PARSER = GameStateParser(SKELD.room_at)


@pytest.fixture(scope="module")
def shared_pathfinder() -> Pathfinder:
    return Pathfinder()


def _sim(pf: Pathfinder, **kwargs) -> SkeldSimulator:
    settings = kwargs.pop("settings", GameSettings(num_impostors=1, kill_cooldown=20.0))
    return SkeldSimulator(SimConfig(num_players=6, settings=settings, seed=kwargs.pop("seed", 3), **kwargs), pf)


def _roles(sim: SkeldSimulator) -> tuple[int, list[int]]:
    impostor = next(pid for pid, p in sim.players.items() if p.is_impostor)
    crew = [pid for pid, p in sim.players.items() if not p.is_impostor]
    return impostor, crew


def _run(sim: SkeldSimulator, seconds: float) -> None:
    end = sim.time + seconds
    while sim.time < end and not sim.finished:
        sim.tick()


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SimConfig(num_players=3)
    with pytest.raises(ValueError):
        SimConfig(num_players=6, settings=GameSettings(num_impostors=3))


def test_snapshots_parse_and_hide_information(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder)
    impostor, crew = _roles(sim)
    for pid in sim.players:
        state = PARSER.parse(sim.snapshot(pid))
        assert state.local_player_id == pid
        assert state.me.role is sim.players[pid].role
    crew_view = PARSER.parse(sim.snapshot(crew[0]))
    assert crew_view.players[impostor].role is Role.UNKNOWN
    assert crew_view.impostor_ids == frozenset()
    assert crew_view.kill_cooldown == 0.0
    imp_view = PARSER.parse(sim.snapshot(impostor))
    assert imp_view.impostor_ids == frozenset({impostor})
    assert all(t.is_fake for t in imp_view.tasks)
    assert not any(t.is_fake for t in crew_view.tasks)


def test_walls_block_vision(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder)
    a, b = 0, 1
    sim.players[a].position = Vec2(52, 12)
    sim.players[b].position = Vec2(52, 18)
    sim._refresh_visibility()
    assert sim.can_see(a, b)
    sim.players[b].position = Vec2(34, 20)  # medbay, behind walls
    sim._refresh_visibility()
    assert not sim.can_see(a, b)
    hidden = next(p for p in sim.snapshot(a)["players"] if p["id"] == b)
    assert hidden["visible"] is False and hidden["position"] == [52, 18]


def test_kill_rules_and_death_knowledge(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder)
    impostor, crew = _roles(sim)
    victim, bystander = crew[0], crew[1]
    sim.players[impostor].position = Vec2(34, 38)
    sim.players[victim].position = Vec2(35, 38)
    sim.players[bystander].position = Vec2(92, 28)
    ok, detail = sim.try_kill(impostor, victim)
    assert not ok and "cooldown" in detail
    sim.players[impostor].kill_cooldown = 0.0
    assert sim.try_kill(impostor, victim) == (True, "ok")
    assert victim in sim.bodies and sim.players[impostor].kill_cooldown == 20.0
    assert not sim.try_kill(impostor, crew[2])[0]

    killer_view = PARSER.parse(sim.snapshot(impostor))
    assert not killer_view.players[victim].alive
    far_view = PARSER.parse(sim.snapshot(bystander))
    assert far_view.players[victim].alive, "a distant crewmate cannot know about the kill yet"
    assert far_view.bodies == ()
    ghost_view = PARSER.parse(sim.snapshot(victim))
    assert not ghost_view.me.alive


def test_report_meeting_and_ejection(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder)
    impostor, crew = _roles(sim)
    victim, reporter = crew[0], crew[1]
    sim.players[impostor].kill_cooldown = 0.0
    sim.players[victim].position = sim.players[impostor].position
    assert sim.try_kill(impostor, victim)[0]
    sim.players[reporter].position = sim.bodies[victim]
    assert sim.try_report(reporter, victim) == (True, "ok")
    assert sim.phase is GamePhase.MEETING and not sim.bodies
    view = PARSER.parse(sim.snapshot(crew[2]))
    assert not view.players[victim].alive
    assert sim.try_vote(reporter, impostor)[0] is False  # discussion first
    _run(sim, sim.settings.discussion_time + 0.5)
    assert sim.phase is GamePhase.VOTING
    for pid, player in sim.players.items():
        if player.alive:
            assert sim.try_vote(pid, impostor)[0]
    assert sim.winner is Role.CREWMATE
    assert sim.result().correct_ejections == 1


def test_reactor_needs_two_simultaneous_holders(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder, reactor_timer=60.0)
    impostor, crew = _roles(sim)
    assert not sim.try_sabotage(crew[0], SabotageType.REACTOR, None)[0]
    sim.time = 20.0
    assert sim.try_sabotage(impostor, SabotageType.REACTOR, None)[0]
    top, bottom = SKELD.sabotage_fixes["reactor"]
    a, b = SimController(sim, crew[0]), SimController(sim, crew[1])
    assert a.execute(FixSabotage(top)).success
    _run(sim, 25.0)
    assert sim.sabotage is not None, "one holder must not be enough"
    end = sim.time + 30.0
    while sim.sabotage is not None and sim.time < end:
        a.execute(FixSabotage(top))
        b.execute(FixSabotage(bottom))
        sim.tick()
    assert sim.sabotage is None
    assert sim.winner is None


def test_reactor_meltdown_wins_for_impostors(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder, reactor_timer=5.0)
    impostor, _ = _roles(sim)
    sim.time = 20.0
    assert sim.try_sabotage(impostor, SabotageType.REACTOR, None)[0]
    _run(sim, 6.0)
    assert sim.winner is Role.IMPOSTOR and "reactor" in sim.end_reason


def test_vent_travel_hides_the_impostor(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder)
    impostor, crew = _roles(sim)
    sim.players[impostor].position = Vec2(38, 24)  # medbay vent
    controller = SimController(sim, impostor)
    assert controller.execute(MoveTo(Vec2(32, 41), allow_vents=True)).success
    saw_in_vent = False
    for _ in range(20):
        sim.tick()
        saw_in_vent = saw_in_vent or sim.players[impostor].in_vent
        if controller.idle:
            break
    assert saw_in_vent
    assert sim.players[impostor].position.distance_to(Vec2(32, 41)) < 1.0
    assert sim.time < 5.0, "venting should be much faster than walking"


def test_multi_step_tasks_advance_the_task_bar(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder)
    _, crew = _roles(sim)
    pid = crew[0]
    controller = SimController(sim, pid)
    task = next(t for t in sim.players[pid].tasks if t.steps_total > 1)
    before = sim.task_bar()
    first_station = task.station.position
    assert controller.execute(DoTask(task.task_id)).success
    assert controller.execute(DoTask(task.task_id)).detail == "in progress"
    _run(sim, 40.0)
    assert task.steps_done == 1
    assert task.station.position != first_station
    assert sim.task_bar() > before
    impostor, _ = _roles(sim)
    fake = sim.players[impostor].tasks[0]
    assert not SimController(sim, impostor).execute(DoTask(fake.task_id)).success


def test_controller_rejects_invalid_actions(shared_pathfinder: Pathfinder) -> None:
    sim = _sim(shared_pathfinder)
    impostor, crew = _roles(sim)
    assert not SimController(sim, crew[0]).execute(Vote(None)).success
    assert not SimController(sim, crew[0]).execute(Kill(crew[1])).success
    assert not SimController(sim, crew[0]).execute(Sabotage(SabotageType.LIGHTS)).success
    sim.time = 20.0
    assert SimController(sim, impostor).execute(Sabotage(SabotageType.DOORS, Room.ELECTRICAL.value)).success
    assert Room.ELECTRICAL in sim.closed_doors
    assert not SimController(sim, impostor).execute(Sabotage(SabotageType.DOORS, Room.ADMIN.value)).success


def test_full_games_are_deterministic_and_finish(shared_pathfinder: Pathfinder) -> None:
    config = SimConfig(num_players=6, settings=GameSettings(num_impostors=1), seed=11, max_game_seconds=600.0)
    first, log_a = play_game(config, pathfinder=shared_pathfinder)
    second, log_b = play_game(config, pathfinder=shared_pathfinder)
    assert log_a == log_b
    assert first.winner == second.winner
    assert first.winner is not None or first.reason == "timeout"
    assert first.duration <= 600.0 + config.dt


def test_runner_agents_play_both_roles(shared_pathfinder: Pathfinder) -> None:
    sim = SkeldSimulator(SimConfig(num_players=7, settings=GameSettings(num_impostors=2), seed=5), shared_pathfinder)
    runner = SimulationRunner(sim)
    result = runner.run()
    assert result.winner in (Role.CREWMATE, Role.IMPOSTOR)
    completed = sum(t.steps_done for p in sim.players.values() if not p.is_impostor for t in p.tasks)
    assert completed > 0
    assert any(isinstance(h.action, (MoveTo, DoTask)) for a in runner.agents.values() for h in a.history)

# among-us-ai-framework

A multi-agent framework in which LLM-driven agents play Among Us on The Skeld: perception, navigation, task play, meeting debate, deception and voting.

## Architecture

| Module | Responsibility |
| --- | --- |
| `among_us_ai/game_state.py` | Strictly typed, immutable state model; a validating snapshot parser; a tracker that turns consecutive snapshots into events (deaths, bodies, witnessed kills/vents, visual tasks, sabotages) and a per-player sighting history. |
| `among_us_ai/pathfinding.py` | The Skeld as rooms and hallways; 8-connected A* with an octile heuristic, wall-clearance costs, vent teleport edges with an admissible heuristic, door sabotage, timed obstacles, soft cost fields, line-of-sight smoothing and Dijkstra distance fields. |
| `among_us_ai/llm_bridge.py` | Claude (Messages API), GPT (Chat Completions) and Gemini (`generateContent`) providers; a thread-safe bridge with hot swapping, fallback chain, retry/backoff and JSON repair; reasoning templates for free-roam decisions, meeting statements, claim extraction and votes. |
| `among_us_ai/discussion_engine.py` | Log-odds suspicion calibrated to the number of living impostors; kill-window travel-time alibis; claim extraction and auditing; the impostor deception planner (cover stories, scapegoats, strategy selection); a guarded voting policy; round-robin multi-agent meetings. |
| `among_us_ai/controller.py` | Input backends (pyautogui/mss, or a recording backend for dry runs), screen calibration, closed-loop movement with stuck detection and replanning, vent hops, kill/report/sabotage, pixel-based solvers for the task minigames, chat and voting. |
| `among_us_ai/agent.py` | The perceive → decide → act loop for one player, with a rule-based planner that can defer to an LLM. |

## Perception input

The framework does not read game memory. It consumes JSON snapshots (see `tests/conftest.py::make_snapshot` for the schema) produced by a client-side telemetry mod, a simulator, or a vision pipeline. `HttpSnapshotSource` polls an HTTP endpoint that serves them.

## Calibration

`ScreenCalibration` and `PanelGeometry` ship with default normalized coordinates. You must calibrate them against screenshots of your own resolution and game version before using the desktop backend: button positions, `pixels_per_unit`, and the minigame panel points. Save the result with `ScreenCalibration.save()`.

Coordinates in `pathfinding.SKELD` follow The Skeld's topology (room adjacency, door rooms, the four vent networks) in a normalized 100×60 space, not the game's internal units. The telemetry source must map positions into this space.

## Usage

```bash
pip install -e ".[dev]"
```

```bash
export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY, GEMINI_API_KEY
```

```python
from among_us_ai import AmongUsAgent, GameController, HttpSnapshotSource, LLMBridge, Pathfinder, PyAutoGUIBackend, ScreenCalibration

bridge = LLMBridge.from_environment(preferred="claude")
pathfinder = Pathfinder()
calibration = ScreenCalibration.load("calibration.local.json")
backend = PyAutoGUIBackend()

agent = AmongUsAgent(
    HttpSnapshotSource("http://127.0.0.1:8765/snapshot"),
    lambda source: GameController(backend, calibration, pathfinder, source),
    bridge=bridge,
    pathfinder=pathfinder,
)
agent.run()
```

`bridge.swap("gemini")` switches providers at runtime without restarting the agent.

## Limitations

- The oxygen keypad needs a code reader (OCR or a vision model) passed as `o2_code_reader`.
- The minigame solvers are pixel heuristics. Their colours and positions depend on calibration.
- Use bots only in private lobbies or test environments with people who have agreed to play with them. Running them in public matchmaking breaks the game's terms and ruins matches for other players.

## Tests

```bash
pytest
```

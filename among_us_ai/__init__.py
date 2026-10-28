"""Multi-agent framework for LLM-driven Among Us players."""

from .agent import AgentConfig, AmongUsAgent, HttpSnapshotSource, PlannerConfig, TaskPhasePlanner
from .controller import GameController, MinigameRunner, PyAutoGUIBackend, RecordingBackend, ScreenCalibration
from .discussion_engine import DiscussionAgent, MeetingSession, SuspicionModel
from .game_state import GameState, GameStateParser, GameStateTracker, Role, Room, Vec2
from .llm_bridge import AnthropicProvider, GeminiProvider, LLMBridge, OpenAIProvider, ScriptedProvider
from .pathfinding import SKELD, NavGrid, Pathfinder
from .simulator import SimConfig, SimController, SimulationRunner, SkeldSimulator, play_game, run_series

__all__ = [
    "SKELD",
    "SimConfig",
    "SimController",
    "SimulationRunner",
    "SkeldSimulator",
    "AgentConfig",
    "AmongUsAgent",
    "AnthropicProvider",
    "DiscussionAgent",
    "GameController",
    "GameState",
    "GameStateParser",
    "GameStateTracker",
    "GeminiProvider",
    "HttpSnapshotSource",
    "LLMBridge",
    "MeetingSession",
    "MinigameRunner",
    "NavGrid",
    "OpenAIProvider",
    "Pathfinder",
    "PlannerConfig",
    "PyAutoGUIBackend",
    "RecordingBackend",
    "Role",
    "Room",
    "ScreenCalibration",
    "ScriptedProvider",
    "SuspicionModel",
    "TaskPhasePlanner",
    "Vec2",
    "play_game",
    "run_series",
]

__version__ = "0.1.0"

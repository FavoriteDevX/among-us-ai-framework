"""Screen and input automation: movement, kills, vents, tasks, meetings.

The controller is split into three layers:

* :class:`InputBackend` - raw keyboard, mouse and screen capture. The
  :class:`PyAutoGUIBackend` drives a real desktop; :class:`RecordingBackend`
  records every input for dry runs and tests.
* :class:`ScreenCalibration` - the mapping between world coordinates,
  normalised UI coordinates and pixels. It is loaded from JSON so the same
  code works at any resolution once calibrated against real screenshots.
* :class:`GameController` - executes high-level :data:`Action` objects with
  closed-loop feedback from a state source (follow a path, vent, kill,
  report, run task minigames, chat and vote).

Intended use is private lobbies and local testing environments. Do not run
automated players in public matchmaking against people who have not agreed
to play with bots.
"""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path as FilePath
from typing import Any, Protocol

from .game_state import GamePhase, GameState, SabotageType, TaskState, Vec2
from .pathfinding import NoPathError, Path, Pathfinder, StepKind

logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]
NormPoint = tuple[float, float]


class ControllerError(RuntimeError):
    """Base error for action execution failures."""


class StuckError(ControllerError):
    """Movement made no progress; the caller should replan."""


class ActionInterrupted(ControllerError):
    """The game phase changed (meeting, death) while an action was running."""


class TaskAutomationError(ControllerError):
    """A minigame could not be completed from the available screen information."""


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class ScreenImage(Protocol):
    @property
    def size(self) -> tuple[int, int]: ...

    def getpixel(self, xy: tuple[int, int]) -> Any: ...


class InputBackend(ABC):
    @abstractmethod
    def key_down(self, key: str) -> None: ...

    @abstractmethod
    def key_up(self, key: str) -> None: ...

    @abstractmethod
    def press(self, key: str) -> None: ...

    @abstractmethod
    def move_mouse(self, x: int, y: int) -> None: ...

    @abstractmethod
    def mouse_down(self) -> None: ...

    @abstractmethod
    def mouse_up(self) -> None: ...

    @abstractmethod
    def click(self, x: int, y: int) -> None: ...

    @abstractmethod
    def type_text(self, text: str) -> None: ...

    @abstractmethod
    def screenshot(self, region: tuple[int, int, int, int]) -> ScreenImage: ...

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def now(self) -> float:
        return time.monotonic()

    def drag(self, start: tuple[int, int], end: tuple[int, int], duration: float, steps: int = 20) -> None:
        self.move_mouse(*start)
        self.mouse_down()
        for i in range(1, steps + 1):
            t = i / steps
            self.move_mouse(round(start[0] + (end[0] - start[0]) * t), round(start[1] + (end[1] - start[1]) * t))
            self.sleep(duration / steps)
        self.mouse_up()

    def drag_through(self, points: Sequence[tuple[int, int]], duration: float) -> None:
        if len(points) < 2:
            raise ValueError("drag_through needs at least two points")
        self.move_mouse(*points[0])
        self.mouse_down()
        per_segment = duration / (len(points) - 1)
        for a, b in zip(points, points[1:]):
            for i in range(1, 9):
                t = i / 8
                self.move_mouse(round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t))
                self.sleep(per_segment / 8)
        self.mouse_up()

    def release_all(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.key_up(key)


class PyAutoGUIBackend(InputBackend):
    """Real desktop input through ``pyautogui`` and capture through ``mss``."""

    def __init__(self, *, pause: float = 0.0) -> None:
        try:
            import pyautogui  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on host
            raise ControllerError("pyautogui is required: pip install 'among-us-ai[desktop]'") from exc
        try:
            import mss  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on host
            raise ControllerError("mss and Pillow are required: pip install 'among-us-ai[desktop]'") from exc
        pyautogui.PAUSE = pause
        pyautogui.FAILSAFE = True
        self._gui = pyautogui
        self._mss = mss.mss()
        self._image = Image

    def key_down(self, key: str) -> None:
        self._gui.keyDown(key)

    def key_up(self, key: str) -> None:
        self._gui.keyUp(key)

    def press(self, key: str) -> None:
        self._gui.press(key)

    def move_mouse(self, x: int, y: int) -> None:
        self._gui.moveTo(x, y)

    def mouse_down(self) -> None:
        self._gui.mouseDown()

    def mouse_up(self) -> None:
        self._gui.mouseUp()

    def click(self, x: int, y: int) -> None:
        self._gui.click(x, y)

    def type_text(self, text: str) -> None:
        self._gui.write(text, interval=0.01)

    def screenshot(self, region: tuple[int, int, int, int]) -> ScreenImage:
        left, top, width, height = region
        shot = self._mss.grab({"left": left, "top": top, "width": width, "height": height})
        return self._image.frombytes("RGB", shot.size, shot.rgb)


@dataclass(frozen=True, slots=True)
class InputEvent:
    kind: str
    args: tuple[Any, ...]
    at: float


class SolidImage:
    """Screen image that returns a pixel function's colour; used by the recording backend."""

    def __init__(self, size: tuple[int, int], pixel: Callable[[int, int], RGB]) -> None:
        self._size = size
        self._pixel = pixel

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def getpixel(self, xy: tuple[int, int]) -> RGB:
        return self._pixel(*xy)


class RecordingBackend(InputBackend):
    """Records inputs with a virtual clock; screenshots come from ``pixel_source``."""

    def __init__(self, pixel_source: Callable[[int, int], RGB] | None = None) -> None:
        self.events: list[InputEvent] = []
        self.held: set[str] = set()
        self.clock = 0.0
        self._pixel_source = pixel_source or (lambda x, y: (0, 0, 0))

    def _log(self, kind: str, *args: Any) -> None:
        self.events.append(InputEvent(kind, args, self.clock))

    def key_down(self, key: str) -> None:
        self.held.add(key)
        self._log("key_down", key)

    def key_up(self, key: str) -> None:
        if key in self.held:
            self.held.discard(key)
            self._log("key_up", key)

    def press(self, key: str) -> None:
        self._log("press", key)

    def move_mouse(self, x: int, y: int) -> None:
        self._log("move", x, y)

    def mouse_down(self) -> None:
        self._log("mouse_down")

    def mouse_up(self) -> None:
        self._log("mouse_up")

    def click(self, x: int, y: int) -> None:
        self._log("click", x, y)

    def type_text(self, text: str) -> None:
        self._log("type", text)

    def screenshot(self, region: tuple[int, int, int, int]) -> ScreenImage:
        left, top, width, height = region
        return SolidImage((width, height), lambda x, y: self._pixel_source(left + x, top + y))

    def sleep(self, seconds: float) -> None:
        self.clock += max(0.0, seconds)

    def now(self) -> float:
        return self.clock

    def of_kind(self, kind: str) -> list[InputEvent]:
        return [e for e in self.events if e.kind == kind]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def _default_buttons() -> dict[str, NormPoint]:
    return {
        "use": (0.925, 0.86),
        "kill": (0.795, 0.86),
        "report": (0.925, 0.70),
        "vent": (0.795, 0.70),
        "sabotage": (0.925, 0.86),
        "chat_open": (0.905, 0.07),
        "chat_input": (0.45, 0.88),
        "chat_send": (0.78, 0.88),
        "chat_close": (0.82, 0.07),
        "emergency_button": (0.50, 0.50),
        "vote_skip": (0.13, 0.88),
        "map_close": (0.12, 0.10),
        "task_close": (0.13, 0.08),
    }


def _default_sabotage_map() -> dict[str, NormPoint]:
    return {
        "reactor": (0.17, 0.53),
        "oxygen": (0.64, 0.33),
        "lights": (0.38, 0.62),
        "communications": (0.62, 0.80),
        "doors_cafeteria": (0.53, 0.20),
        "doors_medbay": (0.37, 0.38),
        "doors_security": (0.27, 0.53),
        "doors_electrical": (0.38, 0.67),
        "doors_storage": (0.53, 0.73),
        "doors_upper_engine": (0.15, 0.25),
        "doors_lower_engine": (0.15, 0.80),
    }


@dataclass(slots=True)
class ScreenCalibration:
    """Pixel geometry of the game window.

    ``pixels_per_unit`` converts framework world units into screen pixels at
    the camera zoom used in matches. The local player is always drawn at the
    window centre, so world offsets from the player map linearly to pixels.
    """

    left: int = 0
    top: int = 0
    width: int = 1920
    height: int = 1080
    pixels_per_unit: float = 58.0
    vent_arrow_radius: float = 0.16
    meeting_vote_grid: tuple[NormPoint, NormPoint] = ((0.27, 0.27), (0.52, 0.11))
    meeting_columns: int = 2
    buttons: dict[str, NormPoint] = field(default_factory=_default_buttons)
    sabotage_map: dict[str, NormPoint] = field(default_factory=_default_sabotage_map)
    keys: dict[str, str] = field(
        default_factory=lambda: {
            "up": "w", "down": "s", "left": "a", "right": "d",
            "use": "e", "kill": "q", "report": "r", "vent": "v", "map": "tab",
        }
    )

    @classmethod
    def load(cls, path: str | FilePath) -> ScreenCalibration:
        data = json.loads(FilePath(path).read_text(encoding="utf-8"))
        calibration = cls()
        for key, value in data.items():
            if not hasattr(calibration, key):
                raise ValueError(f"unknown calibration key {key!r}")
            if key in ("buttons", "sabotage_map"):
                merged = getattr(calibration, key)
                merged.update({k: (float(v[0]), float(v[1])) for k, v in value.items()})
            elif key == "keys":
                calibration.keys.update({str(k): str(v) for k, v in value.items()})
            elif key == "meeting_vote_grid":
                calibration.meeting_vote_grid = (tuple(value[0]), tuple(value[1]))  # type: ignore[assignment]
            else:
                setattr(calibration, key, type(getattr(calibration, key))(value))
        return calibration

    def save(self, path: str | FilePath) -> None:
        FilePath(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @property
    def region(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    def norm(self, point: NormPoint) -> tuple[int, int]:
        return (round(self.left + point[0] * self.width), round(self.top + point[1] * self.height))

    def button(self, name: str) -> tuple[int, int]:
        try:
            return self.norm(self.buttons[name])
        except KeyError as exc:
            raise ControllerError(f"button {name!r} is not calibrated") from exc

    def world_to_screen(self, world: Vec2, player: Vec2) -> tuple[int, int]:
        cx, cy = self.center
        return (round(cx + (world.x - player.x) * self.pixels_per_unit), round(cy + (world.y - player.y) * self.pixels_per_unit))

    def vote_slot(self, index: int) -> tuple[int, int]:
        (x0, y0), (dx, dy) = self.meeting_vote_grid
        col = index % self.meeting_columns
        row = index // self.meeting_columns
        return self.norm((x0 + col * dx, y0 + row * dy))


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MoveTo:
    target: Vec2
    allow_vents: bool = False
    tolerance: float = 0.6


@dataclass(frozen=True, slots=True)
class Kill:
    target_id: int


@dataclass(frozen=True, slots=True)
class Report:
    body_id: int


@dataclass(frozen=True, slots=True)
class DoTask:
    task_id: str


@dataclass(frozen=True, slots=True)
class FakeTask:
    task_id: str
    duration: float = 6.0


@dataclass(frozen=True, slots=True)
class Sabotage:
    kind: SabotageType
    door_room: str | None = None


@dataclass(frozen=True, slots=True)
class FixSabotage:
    position: Vec2


@dataclass(frozen=True, slots=True)
class CallEmergency:
    """Walk to the cafeteria button and call an emergency meeting."""


@dataclass(frozen=True, slots=True)
class Chat:
    text: str


@dataclass(frozen=True, slots=True)
class Vote:
    target_id: int | None


@dataclass(frozen=True, slots=True)
class Wait:
    seconds: float


Action = MoveTo | Kill | Report | DoTask | FakeTask | Sabotage | FixSabotage | CallEmergency | Chat | Vote | Wait

StateSource = Callable[[], GameState]


@dataclass(frozen=True, slots=True)
class ActionResult:
    action: Action
    success: bool
    detail: str
    duration: float


class ActionExecutor(Protocol):
    """Anything that can carry out an :data:`Action` (screen controller or simulator)."""

    def execute(self, action: Action) -> ActionResult: ...


# --------------------------------------------------------------------------- #
# Minigame automation
# --------------------------------------------------------------------------- #


def colour_distance(a: RGB, b: RGB) -> float:
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)))


def _rgb(value: Any) -> RGB:
    if isinstance(value, int):
        return (value, value, value)
    return (int(value[0]), int(value[1]), int(value[2]))


@dataclass(frozen=True, slots=True)
class PanelGeometry:
    """Normalised panel coordinates for each minigame (relative to the whole window)."""

    wire_left: tuple[NormPoint, ...] = ((0.30, 0.27), (0.30, 0.43), (0.30, 0.59), (0.30, 0.75))
    wire_right: tuple[NormPoint, ...] = ((0.70, 0.27), (0.70, 0.43), (0.70, 0.59), (0.70, 0.75))
    card_wallet: NormPoint = (0.38, 0.80)
    card_start: NormPoint = (0.27, 0.40)
    card_end: NormPoint = (0.78, 0.40)
    data_button: NormPoint = (0.50, 0.63)
    fuel_button: NormPoint = (0.80, 0.80)
    garbage_lever_top: NormPoint = (0.66, 0.40)
    garbage_lever_bottom: NormPoint = (0.66, 0.72)
    shield_hexes: tuple[NormPoint, ...] = (
        (0.50, 0.30), (0.42, 0.40), (0.58, 0.40), (0.50, 0.50), (0.42, 0.60), (0.58, 0.60), (0.50, 0.70),
    )
    divert_sliders: tuple[NormPoint, ...] = tuple((0.29 + 0.0614 * i, 0.72) for i in range(8))
    divert_slider_travel: float = 0.22
    accept_power_switch: NormPoint = (0.50, 0.50)
    distributor_dials: tuple[NormPoint, ...] = ((0.44, 0.27), (0.44, 0.51), (0.44, 0.75))
    distributor_targets: tuple[NormPoint, ...] = ((0.44, 0.19), (0.44, 0.43), (0.44, 0.67))
    distributor_buttons: tuple[NormPoint, ...] = ((0.66, 0.27), (0.66, 0.51), (0.66, 0.75))
    reactor_display: tuple[NormPoint, ...] = tuple((0.28 + 0.06 * (i % 3), 0.42 + 0.10 * (i // 3)) for i in range(9))
    reactor_keypad: tuple[NormPoint, ...] = tuple((0.60 + 0.06 * (i % 3), 0.42 + 0.10 * (i // 3)) for i in range(9))
    manifold_buttons: tuple[NormPoint, ...] = tuple((0.29 + 0.085 * (i % 5), 0.45 + 0.13 * (i // 5)) for i in range(10))
    engine_slider: NormPoint = (0.62, 0.70)
    engine_target: NormPoint = (0.62, 0.50)
    steering_center: NormPoint = (0.50, 0.50)
    course_row_y: float = 0.50
    course_x_range: tuple[float, float] = (0.22, 0.80)
    asteroid_box: tuple[NormPoint, NormPoint] = ((0.25, 0.15), (0.75, 0.85))
    o2_leaf_box: tuple[NormPoint, NormPoint] = ((0.40, 0.20), (0.75, 0.80))
    o2_exit: NormPoint = (0.26, 0.50)
    sample_start: NormPoint = (0.69, 0.82)
    sample_tubes: tuple[NormPoint, ...] = tuple((0.36 + 0.07 * i, 0.55) for i in range(5))
    sample_buttons: tuple[NormPoint, ...] = tuple((0.36 + 0.07 * i, 0.78) for i in range(5))


@dataclass(frozen=True, slots=True)
class PanelColours:
    shield_red: RGB = (202, 40, 40)
    lit_green: RGB = (60, 210, 60)
    lit_blue: RGB = (60, 150, 245)
    lit_yellow: RGB = (240, 220, 50)
    asteroid: RGB = (120, 115, 105)
    leaf: RGB = (90, 160, 60)
    anomaly_red: RGB = (220, 50, 50)
    course_node: RGB = (240, 240, 240)
    tolerance: float = 60.0


class MinigameRunner:
    """Pixel-driven solvers for The Skeld's task minigames."""

    def __init__(
        self,
        backend: InputBackend,
        calibration: ScreenCalibration,
        geometry: PanelGeometry | None = None,
        colours: PanelColours | None = None,
    ) -> None:
        self.backend = backend
        self.cal = calibration
        self.geo = geometry or PanelGeometry()
        self.colours = colours or PanelColours()
        self._solvers: dict[str, Callable[[TaskState], None]] = {
            "fix wiring": self.fix_wiring,
            "swipe card": self.swipe_card,
            "upload data": self.transfer_data,
            "download data": self.transfer_data,
            "fuel engines": self.fuel_engines,
            "empty garbage": self.empty_garbage,
            "prime shields": self.prime_shields,
            "divert power": self.divert_power,
            "accept diverted power": self.accept_power,
            "calibrate distributor": self.calibrate_distributor,
            "start reactor": self.start_reactor,
            "unlock manifolds": self.unlock_manifolds,
            "align engine output": self.align_engine,
            "stabilize steering": self.stabilize_steering,
            "chart course": self.chart_course,
            "clear asteroids": self.clear_asteroids,
            "clean o2 filter": self.clean_o2_filter,
            "submit scan": self.submit_scan,
            "inspect sample": self.inspect_sample,
        }

    def supports(self, task_name: str) -> bool:
        return task_name.lower() in self._solvers

    def run(self, task: TaskState) -> None:
        solver = self._solvers.get(task.name.lower())
        if solver is None:
            raise TaskAutomationError(f"no solver registered for {task.name!r}")
        solver(task)

    def register(self, task_name: str, solver: Callable[[TaskState], None]) -> None:
        self._solvers[task_name.lower()] = solver

    # -- pixel helpers ------------------------------------------------------ #

    def _px(self, point: NormPoint) -> tuple[int, int]:
        return self.cal.norm(point)

    def _sample(self, image: ScreenImage, point: NormPoint, radius: int = 2) -> RGB:
        x, y = round(point[0] * self.cal.width), round(point[1] * self.cal.height)
        w, h = image.size
        acc = [0, 0, 0]
        n = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px, py = min(w - 1, max(0, x + dx)), min(h - 1, max(0, y + dy))
                r, g, b = _rgb(image.getpixel((px, py)))
                acc[0] += r
                acc[1] += g
                acc[2] += b
                n += 1
        return (acc[0] // n, acc[1] // n, acc[2] // n)

    def _grab(self) -> ScreenImage:
        return self.backend.screenshot(self.cal.region)

    def _matches(self, colour: RGB, reference: RGB) -> bool:
        return colour_distance(colour, reference) <= self.colours.tolerance

    @staticmethod
    def _brightness(colour: RGB) -> float:
        return 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]

    def _scan_box(self, image: ScreenImage, box: tuple[NormPoint, NormPoint], reference: RGB, step: float = 0.025) -> list[NormPoint]:
        (x0, y0), (x1, y1) = box
        hits: list[NormPoint] = []
        y = y0
        while y <= y1:
            x = x0
            while x <= x1:
                if self._matches(self._sample(image, (x, y), radius=1), reference):
                    if all(math.hypot(x - hx, y - hy) > step * 1.5 for hx, hy in hits):
                        hits.append((x, y))
                x += step
            y += step
        return hits

    # -- solvers ------------------------------------------------------------ #

    def fix_wiring(self, _: TaskState) -> None:
        image = self._grab()
        left = [self._sample(image, p) for p in self.geo.wire_left]
        right = [self._sample(image, p) for p in self.geo.wire_right]
        used: set[int] = set()
        for i, colour in enumerate(left):
            choices = [(colour_distance(colour, r), j) for j, r in enumerate(right) if j not in used]
            if not choices:
                raise TaskAutomationError("wiring: ran out of right-hand endpoints")
            j = min(choices)[1]
            used.add(j)
            self.backend.drag(self._px(self.geo.wire_left[i]), self._px(self.geo.wire_right[j]), 0.35)
            self.backend.sleep(0.1)

    def swipe_card(self, _: TaskState) -> None:
        self.backend.click(*self._px(self.geo.card_wallet))
        self.backend.sleep(0.8)
        # The game accepts swipes that take roughly 0.8 to 1.2 seconds.
        self.backend.drag(self._px(self.geo.card_start), self._px(self.geo.card_end), 1.0, steps=40)

    def transfer_data(self, _: TaskState) -> None:
        self.backend.click(*self._px(self.geo.data_button))
        self.backend.sleep(9.5)

    def fuel_engines(self, _: TaskState) -> None:
        self.backend.move_mouse(*self._px(self.geo.fuel_button))
        self.backend.mouse_down()
        self.backend.sleep(3.8)
        self.backend.mouse_up()

    def empty_garbage(self, _: TaskState) -> None:
        self.backend.move_mouse(*self._px(self.geo.garbage_lever_top))
        self.backend.mouse_down()
        self.backend.move_mouse(*self._px(self.geo.garbage_lever_bottom))
        self.backend.sleep(3.2)
        self.backend.mouse_up()

    def prime_shields(self, _: TaskState) -> None:
        image = self._grab()
        for hex_point in self.geo.shield_hexes:
            if self._matches(self._sample(image, hex_point), self.colours.shield_red):
                self.backend.click(*self._px(hex_point))
                self.backend.sleep(0.12)

    def divert_power(self, _: TaskState) -> None:
        image = self._grab()
        brightness = [self._brightness(self._sample(image, p)) for p in self.geo.divert_sliders]
        target = max(range(len(brightness)), key=brightness.__getitem__)
        x, y = self.geo.divert_sliders[target]
        self.backend.drag(self._px((x, y)), self._px((x, y - self.geo.divert_slider_travel)), 0.4)

    def accept_power(self, _: TaskState) -> None:
        self.backend.click(*self._px(self.geo.accept_power_switch))
        self.backend.sleep(0.3)

    def calibrate_distributor(self, _: TaskState, timeout: float = 12.0) -> None:
        refs = (self.colours.lit_yellow, self.colours.lit_blue, self.colours.lit_blue)
        for dial, target, button, ref in zip(self.geo.distributor_dials, self.geo.distributor_targets, self.geo.distributor_buttons, refs):
            deadline = self.backend.now() + timeout
            while self.backend.now() < deadline:
                image = self._grab()
                if self._brightness(self._sample(image, target)) > self._brightness(self._sample(image, dial)) + 40 or self._matches(self._sample(image, target), ref):
                    self.backend.click(*self._px(button))
                    break
                self.backend.sleep(0.02)
            else:
                raise TaskAutomationError("distributor: dial never aligned")
            self.backend.sleep(0.3)

    def start_reactor(self, _: TaskState, rounds: int = 5) -> None:
        """Simon-says: watch the display light up, then replay the sequence."""
        for round_index in range(1, rounds + 1):
            sequence: list[int] = []
            last_lit: int | None = None
            deadline = self.backend.now() + 2.0 + 1.0 * round_index
            while len(sequence) < round_index and self.backend.now() < deadline:
                image = self._grab()
                levels = [self._brightness(self._sample(image, p)) for p in self.geo.reactor_display]
                brightest = max(range(9), key=levels.__getitem__)
                lit = brightest if levels[brightest] > 150 else None
                if lit is not None and lit != last_lit:
                    sequence.append(lit)
                last_lit = lit
                self.backend.sleep(0.05)
            if len(sequence) < round_index:
                raise TaskAutomationError(f"reactor: observed {len(sequence)} of {round_index} flashes")
            self.backend.sleep(0.4)
            for index in sequence:
                self.backend.click(*self._px(self.geo.reactor_keypad[index]))
                self.backend.sleep(0.25)
            self.backend.sleep(0.8)

    def unlock_manifolds(self, _: TaskState) -> None:
        """Find the 1..10 order by probing: a correct press stays lit, a wrong one resets."""
        order: list[int] = []
        remaining = list(range(len(self.geo.manifold_buttons)))
        while remaining:
            for candidate in list(remaining):
                for known in order:
                    self.backend.click(*self._px(self.geo.manifold_buttons[known]))
                    self.backend.sleep(0.05)
                self.backend.click(*self._px(self.geo.manifold_buttons[candidate]))
                self.backend.sleep(0.12)
                if self._matches(self._sample(self._grab(), self.geo.manifold_buttons[candidate]), self.colours.lit_blue):
                    order.append(candidate)
                    remaining.remove(candidate)
                    break
            else:
                raise TaskAutomationError("manifolds: no candidate stayed lit")

    def align_engine(self, _: TaskState) -> None:
        self.backend.drag(self._px(self.geo.engine_slider), self._px(self.geo.engine_target), 0.5)

    def stabilize_steering(self, _: TaskState) -> None:
        self.backend.click(*self._px(self.geo.steering_center))
        self.backend.sleep(0.3)

    def chart_course(self, _: TaskState) -> None:
        image = self._grab()
        x0, x1 = self.geo.course_x_range
        nodes: list[NormPoint] = []
        x = x0
        while x <= x1:
            best: tuple[float, float] | None = None
            for dy in range(-30, 31, 2):
                y = self.geo.course_row_y + dy / 100.0
                colour = self._sample(image, (x, y), radius=1)
                d = colour_distance(colour, self.colours.course_node)
                if d <= self.colours.tolerance and (best is None or d < best[0]):
                    best = (d, y)
            if best is not None and (not nodes or x - nodes[-1][0] > 0.08):
                nodes.append((x, best[1]))
            x += 0.01
        if len(nodes) < 2:
            raise TaskAutomationError("chart course: could not detect checkpoints")
        self.backend.drag_through([self._px(n) for n in nodes], duration=1.5)

    def clear_asteroids(self, _: TaskState, duration: float = 25.0) -> None:
        deadline = self.backend.now() + duration
        while self.backend.now() < deadline:
            hits = self._scan_box(self._grab(), self.geo.asteroid_box, self.colours.asteroid, step=0.04)
            for point in hits[:4]:
                self.backend.click(*self._px(point))
            self.backend.sleep(0.15)

    def clean_o2_filter(self, _: TaskState, max_passes: int = 8) -> None:
        for _pass in range(max_passes):
            leaves = self._scan_box(self._grab(), self.geo.o2_leaf_box, self.colours.leaf, step=0.04)
            if not leaves:
                return
            for leaf in leaves:
                self.backend.drag(self._px(leaf), self._px(self.geo.o2_exit), 0.25)
        raise TaskAutomationError("o2 filter: leaves remain after maximum passes")

    def submit_scan(self, _: TaskState) -> None:
        self.backend.sleep(10.5)

    def inspect_sample(self, task: TaskState) -> None:
        if task.steps_done == 0:
            self.backend.click(*self._px(self.geo.sample_start))
            self.backend.sleep(1.0)
            return
        image = self._grab()
        reds = [colour_distance(self._sample(image, tube), self.colours.anomaly_red) for tube in self.geo.sample_tubes]
        anomaly = min(range(len(reds)), key=reds.__getitem__)
        self.backend.click(*self._px(self.geo.sample_buttons[anomaly]))


# --------------------------------------------------------------------------- #
# Game controller
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MovementTuning:
    poll_interval: float = 0.05
    stuck_timeout: float = 1.2
    stuck_epsilon: float = 0.15
    waypoint_tolerance: float = 0.45
    segment_timeout_factor: float = 3.0
    diagonal_deadzone: float = 0.38


class GameController:
    """Executes actions against the game with closed-loop state feedback."""

    def __init__(
        self,
        backend: InputBackend,
        calibration: ScreenCalibration,
        pathfinder: Pathfinder,
        state_source: StateSource,
        *,
        minigames: MinigameRunner | None = None,
        tuning: MovementTuning | None = None,
        o2_code_reader: Callable[[ScreenImage], str] | None = None,
    ) -> None:
        self.o2_code_reader = o2_code_reader
        self.backend = backend
        self.cal = calibration
        self.pathfinder = pathfinder
        self.state_source = state_source
        self.minigames = minigames or MinigameRunner(backend, calibration)
        self.tuning = tuning or MovementTuning()
        self._held: set[str] = set()

    # -- low level ---------------------------------------------------------- #

    def _keys_for(self, direction: Vec2) -> set[str]:
        d = direction.normalized()
        dz = self.tuning.diagonal_deadzone
        keys: set[str] = set()
        if d.x > dz:
            keys.add(self.cal.keys["right"])
        elif d.x < -dz:
            keys.add(self.cal.keys["left"])
        if d.y > dz:
            keys.add(self.cal.keys["down"])
        elif d.y < -dz:
            keys.add(self.cal.keys["up"])
        return keys

    def _set_keys(self, keys: set[str]) -> None:
        for key in self._held - keys:
            self.backend.key_up(key)
        for key in keys - self._held:
            self.backend.key_down(key)
        self._held = set(keys)

    def stop(self) -> None:
        self._set_keys(set())

    def _state(self, expect_phase: GamePhase = GamePhase.TASKS) -> GameState:
        state = self.state_source()
        if state.phase is not expect_phase:
            self.stop()
            raise ActionInterrupted(f"phase changed to {state.phase.value}")
        return state

    # -- movement ----------------------------------------------------------- #

    def plan(self, target: Vec2, *, allow_vents: bool = False) -> Path:
        state = self.state_source()
        return self.pathfinder.find_path(
            state.me.position,
            target,
            allow_vents=allow_vents and state.is_impostor,
            closed_doors=state.closed_doors,
            now=state.timestamp,
        )

    def walk_to(self, target: Vec2, *, allow_vents: bool = False, tolerance: float = 0.6, max_replans: int = 3) -> None:
        for attempt in range(max_replans + 1):
            path = self.plan(target, allow_vents=allow_vents)
            try:
                self.follow(path)
            except StuckError:
                if attempt == max_replans:
                    raise
                state = self.state_source()
                cell = self.pathfinder.grid.world_to_cell(state.me.position)
                self.pathfinder.obstacles.set_cost_field(
                    f"stuck:{cell}", state.me.position, 1.5, 4.0, expires_at=state.timestamp + 10.0
                )
                logger.info("movement stuck at %s; replanning (%d)", state.me.position, attempt + 1)
                continue
            if self.state_source().me.position.distance_to(target) <= tolerance + self.tuning.waypoint_tolerance:
                return
        raise StuckError(f"could not reach {target}")

    def follow(self, path: Path) -> None:
        try:
            for waypoint in path:
                if waypoint.kind is StepKind.VENT:
                    self._vent_hop(waypoint.position, waypoint.vent)
                else:
                    self._walk_segment(waypoint.position)
        finally:
            self.stop()

    def _walk_segment(self, target: Vec2) -> None:
        tuning = self.tuning
        state = self._state()
        speed = max(0.1, state.settings.player_speed)
        budget = state.me.position.distance_to(target) / speed * tuning.segment_timeout_factor + 1.0
        started = self.backend.now()
        last_progress_at = started
        best_distance = math.inf
        while True:
            state = self._state()
            if not state.me.alive:
                raise ActionInterrupted("player died")
            offset = target - state.me.position
            distance = offset.length()
            if distance <= tuning.waypoint_tolerance:
                return
            now = self.backend.now()
            if distance < best_distance - tuning.stuck_epsilon:
                best_distance = distance
                last_progress_at = now
            if now - last_progress_at > tuning.stuck_timeout or now - started > budget:
                self.stop()
                raise StuckError(f"no progress toward {target}")
            self._set_keys(self._keys_for(offset))
            self.backend.sleep(tuning.poll_interval)

    def _vent_hop(self, exit_position: Vec2, vent_name: str | None) -> None:
        """Enter the vent under the player, click the arrow toward the next vent, exit."""
        self.stop()
        state = self._state()
        if not state.me.in_vent:
            self.backend.press(self.cal.keys["vent"])
            self.backend.sleep(0.45)
        state = self._state()
        direction = (exit_position - state.me.position).normalized()
        radius = self.cal.vent_arrow_radius * self.cal.height
        cx, cy = self.cal.center
        self.backend.click(round(cx + direction.x * radius), round(cy + direction.y * radius))
        self.backend.sleep(0.35)
        state = self._state()
        if state.me.position.distance_to(exit_position) > 2.0:
            raise StuckError(f"vent hop to {vent_name} did not arrive")
        self.backend.press(self.cal.keys["vent"])
        self.backend.sleep(0.45)

    # -- interactions ------------------------------------------------------- #

    def kill(self, target_id: int) -> None:
        state = self._state()
        if not state.is_impostor:
            raise ControllerError("only impostors can kill")
        if state.kill_cooldown > 0:
            raise ControllerError(f"kill on cooldown ({state.kill_cooldown:.1f}s)")
        target = state.players.get(target_id)
        if target is None or not target.alive:
            raise ControllerError("kill target is not alive")
        reach = state.settings.kill_distance
        if state.me.position.distance_to(target.position) > reach:
            self.walk_to(target.position, tolerance=reach * 0.6)
            state = self._state()
            target = state.players[target_id]
            if state.me.position.distance_to(target.position) > reach:
                raise ControllerError("target moved out of kill range")
        self.backend.press(self.cal.keys["kill"])
        self.backend.sleep(0.3)

    def report(self, body_id: int) -> None:
        state = self._state()
        body = next((b for b in state.bodies if b.player_id == body_id), None)
        if body is None:
            raise ControllerError(f"no visible body for player {body_id}")
        if state.me.position.distance_to(body.position) > 2.0:
            self.walk_to(body.position, tolerance=1.2)
        self.backend.press(self.cal.keys["report"])
        self.backend.sleep(0.3)

    def do_task(self, task_id: str) -> None:
        state = self._state()
        task = next((t for t in state.tasks if t.task_id == task_id), None)
        if task is None:
            raise ControllerError(f"unknown task {task_id}")
        if task.completed:
            return
        if task.is_fake:
            raise ControllerError("impostor tasks are fake; use FakeTask")
        self.walk_to(task.position, tolerance=0.8)
        self.backend.press(self.cal.keys["use"])
        self.backend.sleep(0.6)
        try:
            self.minigames.run(task)
        except TaskAutomationError:
            self.backend.click(*self.cal.button("task_close"))
            raise
        self.backend.sleep(0.5)

    def fake_task(self, task_id: str, duration: float) -> None:
        state = self._state()
        task = next((t for t in state.tasks if t.task_id == task_id), None)
        if task is None:
            raise ControllerError(f"unknown task {task_id}")
        self.walk_to(task.position, tolerance=0.8)
        waited = 0.0
        while waited < duration:
            self._state()
            self.backend.sleep(0.5)
            waited += 0.5

    def sabotage(self, kind: SabotageType, door_room: str | None = None) -> None:
        state = self._state()
        if not state.is_impostor:
            raise ControllerError("only impostors can sabotage")
        self.stop()
        self.backend.press(self.cal.keys["map"])
        self.backend.sleep(0.4)
        key = f"doors_{door_room}" if kind is SabotageType.DOORS else kind.value
        if key not in self.cal.sabotage_map:
            raise ControllerError(f"sabotage target {key!r} is not calibrated")
        self.backend.click(*self.cal.norm(self.cal.sabotage_map[key]))
        self.backend.sleep(0.2)
        self.backend.press(self.cal.keys["map"])

    def fix_sabotage(self, position: Vec2) -> None:
        self.walk_to(position, tolerance=0.8)
        state = self._state()
        self.backend.press(self.cal.keys["use"])
        self.backend.sleep(0.5)
        kind = state.sabotage.kind if state.sabotage else None
        if kind in (SabotageType.REACTOR,):
            cx, cy = self.cal.center
            self.backend.move_mouse(cx, cy)
            self.backend.mouse_down()
            deadline = self.backend.now() + 15.0
            while self.backend.now() < deadline:
                current = self.state_source()
                if current.sabotage is None or current.phase is not GamePhase.TASKS:
                    break
                self.backend.sleep(0.25)
            self.backend.mouse_up()
        elif kind is SabotageType.LIGHTS:
            self._fix_lights()
        elif kind is SabotageType.COMMUNICATIONS:
            self._fix_comms()
        elif kind is SabotageType.OXYGEN:
            self._fix_oxygen()

    def _fix_oxygen(self) -> None:
        """Type the five-digit code shown on the sticky note into the keypad.

        Reading the note needs OCR or a vision model, supplied as ``o2_code_reader``.
        """
        if self.o2_code_reader is None:
            raise TaskAutomationError("oxygen keypad needs an o2_code_reader (OCR or vision model)")
        code = "".join(ch for ch in self.o2_code_reader(self.backend.screenshot(self.cal.region)) if ch.isdigit())
        if len(code) != 5:
            raise TaskAutomationError(f"oxygen code reader returned {code!r}")
        keypad = {str(d): (0.43 + 0.07 * ((d - 1) % 3), 0.38 + 0.12 * ((d - 1) // 3)) for d in range(1, 10)}
        keypad["0"] = (0.50, 0.74)
        for digit in code:
            self.backend.click(*self.cal.norm(keypad[digit]))
            self.backend.sleep(0.1)
        self.backend.click(*self.cal.norm((0.57, 0.74)))

    def _fix_lights(self) -> None:
        switches = [(0.25 + 0.125 * i, 0.68) for i in range(5)]
        image = self.backend.screenshot(self.cal.region)
        runner = self.minigames
        for point in switches:
            colour = runner._sample(image, point)
            if not runner._matches(colour, runner.colours.lit_green):
                self.backend.click(*self.cal.norm(point))
                self.backend.sleep(0.1)

    def _fix_comms(self, timeout: float = 10.0) -> None:
        cx, cy = self.cal.norm((0.50, 0.55))
        radius = 0.12 * self.cal.height
        runner = self.minigames
        deadline = self.backend.now() + timeout
        angle = 0.0
        while self.backend.now() < deadline:
            image = self.backend.screenshot(self.cal.region)
            if runner._matches(runner._sample(image, (0.82, 0.30)), runner.colours.lit_green):
                return
            self.backend.drag((round(cx + radius), cy), (round(cx + radius * math.cos(angle)), round(cy + radius * math.sin(angle))), 0.1)
            angle += math.radians(12)
        raise TaskAutomationError("comms dial never locked")

    def call_emergency(self) -> None:
        state = self._state()
        if state.emergency_meetings_left <= 0:
            raise ControllerError("no emergency meetings left")
        self.walk_to(self.pathfinder.definition.emergency_button, tolerance=1.0)
        self.backend.press(self.cal.keys["use"])
        self.backend.sleep(0.4)
        self.backend.click(*self.cal.button("emergency_button"))

    def chat(self, text: str) -> None:
        clean = " ".join(text.split())[:100]
        self.backend.click(*self.cal.button("chat_open"))
        self.backend.sleep(0.3)
        self.backend.click(*self.cal.button("chat_input"))
        self.backend.type_text(clean)
        self.backend.press("enter")
        self.backend.sleep(0.2)
        self.backend.click(*self.cal.button("chat_close"))

    def vote(self, target_id: int | None) -> None:
        state = self.state_source()
        if state.phase not in (GamePhase.MEETING, GamePhase.VOTING):
            raise ActionInterrupted("not in a meeting")
        if target_id is None:
            point = self.cal.button("vote_skip")
        else:
            ordered = sorted(state.players)
            if target_id not in ordered:
                raise ControllerError(f"unknown vote target {target_id}")
            point = self.cal.vote_slot(ordered.index(target_id))
        self.backend.click(*point)
        self.backend.sleep(0.3)
        confirm = (point[0] + round(0.09 * self.cal.width), point[1])
        self.backend.click(*confirm)

    # -- dispatch ----------------------------------------------------------- #

    def execute(self, action: Action) -> ActionResult:
        started = self.backend.now()
        try:
            match action:
                case MoveTo(target=target, allow_vents=vents, tolerance=tol):
                    self.walk_to(target, allow_vents=vents, tolerance=tol)
                case Kill(target_id=target):
                    self.kill(target)
                case Report(body_id=body):
                    self.report(body)
                case DoTask(task_id=task_id):
                    self.do_task(task_id)
                case FakeTask(task_id=task_id, duration=duration):
                    self.fake_task(task_id, duration)
                case Sabotage(kind=kind, door_room=room):
                    self.sabotage(kind, room)
                case FixSabotage(position=position):
                    self.fix_sabotage(position)
                case CallEmergency():
                    self.call_emergency()
                case Chat(text=text):
                    self.chat(text)
                case Vote(target_id=target):
                    self.vote(target)
                case Wait(seconds=seconds):
                    self.backend.sleep(seconds)
                case _:
                    raise ControllerError(f"unsupported action {action!r}")
        except (ControllerError, NoPathError) as exc:
            self.stop()
            return ActionResult(action, False, str(exc), self.backend.now() - started)
        return ActionResult(action, True, "ok", self.backend.now() - started)

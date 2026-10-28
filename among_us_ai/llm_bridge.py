"""Hot-swappable LLM connectors (Claude, GPT, Gemini) with reasoning templates.

Every provider implements :class:`LLMProvider`. The :class:`LLMBridge` owns a
registry of providers, an active provider that can be swapped at runtime
from any thread, an ordered fallback chain, retries with exponential
backoff, and structured JSON decoding of model output.

Reasoning templates (:class:`ReasoningTemplate`) encode a deliberate
"observe -> hypothesise -> evaluate -> decide" procedure. Models are asked to
return a single JSON object whose ``decision`` field is machine-readable and
whose ``reasoning`` field holds a short audit trail. The bridge never relies
on hidden chain-of-thought being returned verbatim; it requests concise
justifications that can be logged and evaluated.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any, Final

import httpx

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class LLMError(RuntimeError):
    """Base error for provider failures."""


class LLMConfigurationError(LLMError):
    """Provider is missing credentials or has invalid configuration."""


class LLMTransientError(LLMError):
    """Retryable failure: rate limit, timeout, overloaded or 5xx."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMResponseFormatError(LLMError):
    """The model returned output that could not be decoded into the schema."""


# --------------------------------------------------------------------------- #
# Request / response types
# --------------------------------------------------------------------------- #


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system: str
    messages: tuple[ChatTurn, ...]
    max_tokens: int = 1024
    temperature: float = 0.7
    json_mode: bool = True
    reasoning_budget: int | None = None
    stop: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def single(cls, system: str, prompt: str, **kwargs: Any) -> LLMRequest:
        return cls(system=system, messages=(ChatTurn(MessageRole.USER, prompt),), **kwargs)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: TokenUsage
    latency_s: float
    finish_reason: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- #
# Provider base
# --------------------------------------------------------------------------- #


def _retry_after_seconds(response: httpx.Response) -> float | None:
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


class LLMProvider(ABC):
    """Abstract provider. Subclasses translate :class:`LLMRequest` into a vendor API call."""

    name: str = "provider"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._lock = threading.Lock()

    @property
    def client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self._timeout)
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None and self._owns_client:
                self._client.close()
            self._client = None

    def _require_key(self) -> str:
        if not self._api_key:
            raise LLMConfigurationError(f"{self.name}: API key is not configured")
        return self._api_key

    def _post(self, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            response = self.client.post(url, headers=dict(headers), json=payload, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"{self.name}: request timed out") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(f"{self.name}: transport error: {exc}") from exc
        if response.status_code in (408, 409, 425, 429) or response.status_code >= 500:
            raise LLMTransientError(
                f"{self.name}: HTTP {response.status_code}: {response.text[:300]}",
                retry_after=_retry_after_seconds(response),
            )
        if response.status_code in (401, 403):
            raise LLMConfigurationError(f"{self.name}: authentication failed (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise LLMError(f"{self.name}: HTTP {response.status_code}: {response.text[:500]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.name}: response was not JSON") from exc
        if not isinstance(body, Mapping):
            raise LLMError(f"{self.name}: unexpected response shape")
        return body

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a single completion."""


# --------------------------------------------------------------------------- #
# Concrete providers
# --------------------------------------------------------------------------- #


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic Messages API."""

    name = "claude"
    DEFAULT_MODEL: Final[str] = "claude-opus-5-5"
    API_VERSION: Final[str] = "2023-06-01"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model or os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL),
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        }
        if request.reasoning_budget:
            # Extended thinking requires temperature 1 and a budget below max_tokens.
            budget = max(1024, request.reasoning_budget)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["max_tokens"] = max(request.max_tokens, budget + 1024)
        else:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = list(request.stop)
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = self._require_key()
        started = time.monotonic()
        body = self._post(
            f"{self.base_url}/v1/messages",
            headers={"x-api-key": key, "anthropic-version": self.API_VERSION, "content-type": "application/json"},
            payload=self.build_payload(request),
        )
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, Mapping) and b.get("type") == "text")
        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=str(body.get("model", self.model)),
            usage=TokenUsage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))),
            latency_s=time.monotonic() - started,
            finish_reason=body.get("stop_reason"),
            raw=body,
        )


class OpenAIProvider(LLMProvider):
    """GPT via the OpenAI Chat Completions API (also works with compatible gateways)."""

    name = "gpt"
    DEFAULT_MODEL: Final[str] = "gpt-5"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model or os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.organization = organization or os.environ.get("OPENAI_ORG_ID")

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        messages = [{"role": "system", "content": request.system}]
        messages.extend({"role": m.role.value, "content": m.content} for m in request.messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens + (request.reasoning_budget or 0),
        }
        if request.reasoning_budget:
            effort = "high" if request.reasoning_budget >= 8000 else "medium" if request.reasoning_budget >= 2000 else "low"
            payload["reasoning_effort"] = effort
        else:
            payload["temperature"] = request.temperature
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.stop:
            payload["stop"] = list(request.stop)
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = self._require_key()
        headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        if self.organization:
            headers["openai-organization"] = self.organization
        started = time.monotonic()
        body = self._post(f"{self.base_url}/chat/completions", headers=headers, payload=self.build_payload(request))
        choices = body.get("choices") or []
        if not choices:
            raise LLMError("gpt: response contained no choices")
        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}
        return LLMResponse(
            text=str(message.get("content") or ""),
            provider=self.name,
            model=str(body.get("model", self.model)),
            usage=TokenUsage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
            latency_s=time.monotonic() - started,
            finish_reason=choices[0].get("finish_reason"),
            raw=body,
        )


class GeminiProvider(LLMProvider):
    """Gemini via the Google Generative Language API."""

    name = "gemini"
    DEFAULT_MODEL: Final[str] = "gemini-2.5-pro"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model or os.environ.get("GEMINI_MODEL", self.DEFAULT_MODEL),
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        contents = [
            {"role": "user" if m.role is MessageRole.USER else "model", "parts": [{"text": m.content}]}
            for m in request.messages
        ]
        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens + (request.reasoning_budget or 0),
        }
        if request.json_mode:
            generation["responseMimeType"] = "application/json"
        if request.stop:
            generation["stopSequences"] = list(request.stop)
        if request.reasoning_budget:
            generation["thinkingConfig"] = {"thinkingBudget": request.reasoning_budget}
        return {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": contents,
            "generationConfig": generation,
        }

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = self._require_key()
        started = time.monotonic()
        body = self._post(
            f"{self.base_url}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            payload=self.build_payload(request),
        )
        candidates = body.get("candidates") or []
        if not candidates:
            feedback = body.get("promptFeedback") or {}
            raise LLMError(f"gemini: no candidates returned (blockReason={feedback.get('blockReason')})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, Mapping) and not p.get("thought"))
        usage = body.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=TokenUsage(int(usage.get("promptTokenCount", 0)), int(usage.get("candidatesTokenCount", 0))),
            latency_s=time.monotonic() - started,
            finish_reason=candidates[0].get("finishReason"),
            raw=body,
        )


class ScriptedProvider(LLMProvider):
    """Deterministic provider for offline runs, replays and tests.

    ``responder`` receives the request and returns the raw text the model
    would have produced. A sequence of strings is replayed in order and
    cycled when exhausted.
    """

    name = "scripted"

    def __init__(self, responder: Callable[[LLMRequest], str] | Sequence[str], model: str = "scripted-v1") -> None:
        super().__init__(model, api_key="offline")
        if callable(responder):
            self._responder: Callable[[LLMRequest], str] = responder
        else:
            replies = list(responder)
            if not replies:
                raise LLMConfigurationError("scripted provider needs at least one reply")
            index = {"i": 0}

            def replay(_: LLMRequest) -> str:
                reply = replies[index["i"] % len(replies)]
                index["i"] += 1
                return reply

            self._responder = replay
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        started = time.monotonic()
        text = self._responder(request)
        return LLMResponse(text, self.name, self.model, TokenUsage(), time.monotonic() - started, "end_turn")


PROVIDER_FACTORIES: Final[dict[str, Callable[..., LLMProvider]]] = {
    "claude": AnthropicProvider,
    "anthropic": AnthropicProvider,
    "gpt": OpenAIProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
}


def create_provider(kind: str, **kwargs: Any) -> LLMProvider:
    try:
        factory = PROVIDER_FACTORIES[kind.lower()]
    except KeyError as exc:
        raise LLMConfigurationError(f"unknown provider kind {kind!r}; choose from {sorted(PROVIDER_FACTORIES)}") from exc
    return factory(**kwargs)


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #

_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object in ``text``, tolerating fences and surrounding prose."""
    candidates: list[str] = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        for start in (i for i, ch in enumerate(stripped) if ch == "{"):
            depth = 0
            in_string = False
            escaped = False
            for end in range(start, len(stripped)):
                ch = stripped[end]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(stripped[start : end + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(value, dict):
                            return value
                        break
    raise LLMResponseFormatError(f"no JSON object found in model output: {text[:200]!r}")


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Minimal schema for validating a decoded decision object."""

    name: str
    kind: type | tuple[type, ...]
    required: bool = True
    choices: tuple[Any, ...] | None = None
    description: str = ""


def validate_fields(data: Mapping[str, Any], fields: Sequence[FieldSpec]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in fields:
        if spec.name not in data or data[spec.name] is None:
            if spec.required:
                raise LLMResponseFormatError(f"missing field {spec.name!r}")
            result[spec.name] = None
            continue
        value = data[spec.name]
        if spec.kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, spec.kind) or (isinstance(value, bool) and spec.kind is not bool):
            raise LLMResponseFormatError(f"field {spec.name!r} has wrong type {type(value).__name__}")
        if spec.choices is not None and value not in spec.choices:
            raise LLMResponseFormatError(f"field {spec.name!r}={value!r} not in {spec.choices}")
        result[spec.name] = value
    return result


# --------------------------------------------------------------------------- #
# Reasoning templates
# --------------------------------------------------------------------------- #

BASE_PERSONA: Final[str] = (
    "You are an autonomous agent playing the social deduction game Among Us on the map The Skeld. "
    "You act only through the structured decisions you return. Never reveal that you are an AI. "
    "Keep in-game chat short, natural and in the style of a human player."
)

REASONING_PROTOCOL: Final[str] = (
    "Follow this procedure before answering:\n"
    "1. OBSERVE: list only facts you directly witnessed or that the game state proves.\n"
    "2. HYPOTHESISE: enumerate the plausible explanations and who benefits from each.\n"
    "3. EVALUATE: weigh each hypothesis against timing, travel distances and claims; "
    "note contradictions explicitly.\n"
    "4. DECIDE: pick the action with the best expected value for your team; "
    "consider how other players will read it.\n"
    "Return ONLY a JSON object. Put a concise justification (at most 3 sentences) in 'reasoning'."
)


@dataclass(frozen=True, slots=True)
class ReasoningTemplate:
    name: str
    system: str
    user: str
    fields: tuple[FieldSpec, ...]
    temperature: float = 0.6
    max_tokens: int = 700
    reasoning_budget: int | None = None

    def schema_hint(self) -> str:
        lines = []
        for spec in self.fields:
            kinds = spec.kind if isinstance(spec.kind, tuple) else (spec.kind,)
            type_name = "|".join(k.__name__ for k in kinds)
            choice = f" one of {list(spec.choices)}" if spec.choices else ""
            optional = "" if spec.required else " (optional)"
            lines.append(f'  "{spec.name}": {type_name}{choice}{optional} - {spec.description}')
        return "{\n" + ",\n".join(lines) + "\n}"

    def render(self, variables: Mapping[str, Any], *, reasoning_budget: int | None = None) -> LLMRequest:
        try:
            prompt = Template(self.user).substitute({k: str(v) for k, v in variables.items()})
        except KeyError as exc:
            raise LLMConfigurationError(f"template {self.name} missing variable {exc}") from exc
        prompt = f"{prompt}\n\nRespond with JSON matching:\n{self.schema_hint()}"
        return LLMRequest.single(
            f"{BASE_PERSONA}\n\n{self.system}\n\n{REASONING_PROTOCOL}",
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            json_mode=True,
            reasoning_budget=reasoning_budget if reasoning_budget is not None else self.reasoning_budget,
            metadata={"template": self.name},
        )


TASK_PHASE_TEMPLATE: Final[ReasoningTemplate] = ReasoningTemplate(
    name="task_phase_decision",
    system=(
        "Decide your next high-level action during the free-roam phase. As a crewmate prioritise "
        "survival, task completion, fixing critical sabotages and reporting bodies. As an impostor "
        "prioritise isolated kills, alibi building, sabotage timing and never being seen near bodies or vents."
    ),
    user=(
        "You are $name ($color), role: $role.\n"
        "Current room: $room. Time: ${timestamp}s. Task bar: $task_bar.\n"
        "Kill cooldown: $kill_cooldown. Active sabotage: $sabotage.\n"
        "Visible players: $visible.\n"
        "Pending tasks: $tasks.\n"
        "Recent events:\n$events\n"
        "Allowed actions: $actions."
    ),
    fields=(
        FieldSpec("action", str, description="one of the allowed actions"),
        FieldSpec("target", (str, int), required=False, description="player name, task id, room or sabotage"),
        FieldSpec("reasoning", str, description="short justification"),
    ),
    temperature=0.4,
    max_tokens=400,
)

MEETING_STATEMENT_TEMPLATE: Final[ReasoningTemplate] = ReasoningTemplate(
    name="meeting_statement",
    system=(
        "You are speaking in an emergency meeting. Be persuasive and specific: cite rooms, times and "
        "who you were with. Crewmates must share genuine information. Impostors must stay consistent "
        "with the cover story provided and with everything they said earlier."
    ),
    user=(
        "You are $name ($color), role: $role. Meeting reason: $reason.\n"
        "Your private evidence:\n$evidence\n"
        "Suspicion table (probability each player is an impostor):\n$suspicion\n"
        "Your strategy for this meeting: $strategy\n"
        "Cover story you must stay consistent with: $cover_story\n"
        "Your earlier statements: $own_history\n"
        "Discussion so far:\n$transcript\n"
        "Write your next chat message (max 25 words)."
    ),
    fields=(
        FieldSpec("message", str, description="the chat message to send"),
        FieldSpec("accuse", str, required=False, description="name of player you accuse, if any"),
        FieldSpec("vouch", str, required=False, description="name of player you vouch for, if any"),
        FieldSpec("claimed_room", str, required=False, description="room you claim you were in"),
        FieldSpec("reasoning", str, description="short justification"),
    ),
    temperature=0.8,
    max_tokens=350,
)

CLAIM_EXTRACTION_TEMPLATE: Final[ReasoningTemplate] = ReasoningTemplate(
    name="claim_extraction",
    system="Extract structured claims from Among Us meeting chat. Do not invent claims that are not stated.",
    user=(
        "Players: $players. Rooms: $rooms.\n"
        "Message from $speaker: \"$message\"\n"
        "Return the claims made in this message."
    ),
    fields=(
        FieldSpec("claimed_room", str, required=False, description="room the speaker says they were in"),
        FieldSpec("accused", list, required=False, description="names the speaker accuses"),
        FieldSpec("vouched", list, required=False, description="names the speaker vouches for"),
        FieldSpec("saw", list, required=False, description='list of {"player": name, "room": room} sightings claimed'),
        FieldSpec("reasoning", str, description="short justification"),
    ),
    temperature=0.0,
    max_tokens=300,
)

VOTE_TEMPLATE: Final[ReasoningTemplate] = ReasoningTemplate(
    name="vote_decision",
    system=(
        "Cast your vote. Skipping is correct when evidence is weak and a wrong ejection would cost your team. "
        "Impostors should avoid voting for teammates unless that teammate is certainly going to be ejected."
    ),
    user=(
        "You are $name ($color), role: $role. Alive players: $alive. Impostors alive (estimate): $impostors_alive.\n"
        "Suspicion table:\n$suspicion\n"
        "Engine recommendation: $recommendation\n"
        "Discussion transcript:\n$transcript\n"
        "Valid vote targets: $targets (or 'skip')."
    ),
    fields=(
        FieldSpec("vote", str, description="player name or 'skip'"),
        FieldSpec("confidence", float, description="0..1 confidence in the vote"),
        FieldSpec("reasoning", str, description="short justification"),
    ),
    temperature=0.2,
    max_tokens=300,
    reasoning_budget=None,
)


# --------------------------------------------------------------------------- #
# Bridge
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 20.0
    jitter: float = 0.25

    def delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(self.max_delay, retry_after)
        raw = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))


@dataclass(slots=True)
class BridgeStats:
    calls: int = 0
    failures: int = 0
    fallbacks: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    by_provider: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StructuredResult:
    data: dict[str, Any]
    response: LLMResponse


class LLMBridge:
    """Thread-safe registry with a hot-swappable active provider and fallback chain."""

    def __init__(
        self,
        providers: Mapping[str, LLMProvider] | None = None,
        *,
        active: str | None = None,
        fallback_order: Iterable[str] = (),
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._providers: dict[str, LLMProvider] = dict(providers or {})
        self._lock = threading.RLock()
        self._active = active or next(iter(self._providers), None)
        self._fallback = [name for name in fallback_order]
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self.stats = BridgeStats()
        if self._active is not None and self._active not in self._providers:
            raise LLMConfigurationError(f"active provider {self._active!r} is not registered")

    @classmethod
    def from_environment(cls, preferred: str | None = None) -> LLMBridge:
        """Register every provider whose API key is present in the environment."""
        candidates = {
            "claude": ("ANTHROPIC_API_KEY", AnthropicProvider),
            "gpt": ("OPENAI_API_KEY", OpenAIProvider),
            "gemini": ("GEMINI_API_KEY", GeminiProvider),
        }
        providers: dict[str, LLMProvider] = {}
        for name, (env_key, factory) in candidates.items():
            if os.environ.get(env_key) or (name == "gemini" and os.environ.get("GOOGLE_API_KEY")):
                providers[name] = factory()
        if not providers:
            raise LLMConfigurationError("no LLM API keys found (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY)")
        preferred = preferred or os.environ.get("AMONG_US_LLM")
        active = preferred if preferred in providers else next(iter(providers))
        fallback = [n for n in providers if n != active]
        return cls(providers, active=active, fallback_order=fallback)

    # -- registry ----------------------------------------------------------- #

    def register(self, name: str, provider: LLMProvider, *, activate: bool = False) -> None:
        with self._lock:
            old = self._providers.get(name)
            self._providers[name] = provider
            if activate or self._active is None:
                self._active = name
        if old is not None and old is not provider:
            old.close()

    def unregister(self, name: str) -> None:
        with self._lock:
            provider = self._providers.pop(name, None)
            self._fallback = [n for n in self._fallback if n != name]
            if self._active == name:
                self._active = next(iter(self._providers), None)
        if provider is not None:
            provider.close()

    def swap(self, name: str) -> None:
        """Atomically change the active provider; in-flight calls finish on the old one."""
        with self._lock:
            if name not in self._providers:
                raise LLMConfigurationError(f"provider {name!r} is not registered")
            self._active = name
        logger.info("LLM provider swapped to %s", name)

    def set_fallback_order(self, order: Iterable[str]) -> None:
        with self._lock:
            self._fallback = [n for n in order]

    @property
    def active(self) -> str | None:
        with self._lock:
            return self._active

    @property
    def providers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._providers)

    def _chain(self) -> list[tuple[str, LLMProvider]]:
        with self._lock:
            if self._active is None:
                raise LLMConfigurationError("no provider registered")
            order = [self._active] + [n for n in self._fallback if n != self._active]
            return [(n, self._providers[n]) for n in order if n in self._providers]

    # -- calls -------------------------------------------------------------- #

    def _call_with_retry(self, name: str, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                return provider.complete(request)
            except LLMTransientError as exc:
                if attempt >= self._retry.max_attempts:
                    raise
                wait = self._retry.delay(attempt, exc.retry_after)
                logger.warning("%s transient failure (attempt %d): %s; retrying in %.1fs", name, attempt, exc, wait)
                self._sleep(wait)

    def complete(self, request: LLMRequest) -> LLMResponse:
        errors: list[str] = []
        for index, (name, provider) in enumerate(self._chain()):
            try:
                response = self._call_with_retry(name, provider, request)
            except LLMConfigurationError as exc:
                errors.append(f"{name}: {exc}")
                continue
            except LLMError as exc:
                errors.append(f"{name}: {exc}")
                with self._lock:
                    self.stats.failures += 1
                continue
            with self._lock:
                self.stats.calls += 1
                self.stats.usage = self.stats.usage + response.usage
                self.stats.by_provider[name] = self.stats.by_provider.get(name, 0) + 1
                if index > 0:
                    self.stats.fallbacks += 1
            return response
        raise LLMError("all providers failed: " + " | ".join(errors))

    def complete_json(
        self,
        request: LLMRequest,
        fields: Sequence[FieldSpec] = (),
        *,
        repair_attempts: int = 1,
    ) -> StructuredResult:
        """Complete and decode a JSON object, asking the model to repair malformed output."""
        current = request
        last_error: Exception | None = None
        for _ in range(repair_attempts + 1):
            response = self.complete(current)
            try:
                data = extract_json_object(response.text)
                if fields:
                    data = {**data, **validate_fields(data, fields)}
                return StructuredResult(data, response)
            except LLMResponseFormatError as exc:
                last_error = exc
                current = LLMRequest(
                    system=request.system,
                    messages=request.messages
                    + (
                        ChatTurn(MessageRole.ASSISTANT, response.text or "(empty)"),
                        ChatTurn(
                            MessageRole.USER,
                            f"Your reply was invalid ({exc}). Reply again with ONLY the JSON object.",
                        ),
                    ),
                    max_tokens=request.max_tokens,
                    temperature=0.0 if request.reasoning_budget is None else request.temperature,
                    json_mode=True,
                    reasoning_budget=request.reasoning_budget,
                    metadata=request.metadata,
                )
        raise LLMResponseFormatError(str(last_error))

    def run_template(
        self,
        template: ReasoningTemplate,
        variables: Mapping[str, Any],
        *,
        reasoning_budget: int | None = None,
    ) -> StructuredResult:
        return self.complete_json(template.render(variables, reasoning_budget=reasoning_budget), template.fields)

    def close(self) -> None:
        with self._lock:
            providers = list(self._providers.values())
        for provider in providers:
            provider.close()

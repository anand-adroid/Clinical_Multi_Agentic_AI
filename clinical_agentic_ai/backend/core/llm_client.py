"""
LLM client — a thin facade over Anthropic and OpenAI SDKs with a stub for
offline operation. Every prompt-using agent imports the ``llm`` singleton from
this module so a single class controls the call contract, retry policy, and
telemetry.

Design choices that matter in a regulated setting:

1. Strict JSON contracts. Every call returns a parsed dict, never raw text.
   If the model deviates from the schema the response is repaired (three
   extraction strategies) or raises ``LLMError``. This is the first line
   of defence against hallucinated structure leaking into the pipeline.

2. Multi-provider failover. Anthropic is the primary provider; OpenAI is the
   automatic fallback. If Anthropic is down (5xx, overload, network) the
   client retries the same prompt against OpenAI before escalating to the
   deterministic / HITL path. Provider selection is invisible to agents.

3. Offline / air-gapped stub. When neither API key is configured the client
   routes to a deterministic stub. Every agent has its own fallback or HITL
   escalation path so the workflow completes end-to-end without network egress
   — important for CI, demos, and pharma environments with private gateways.

4. Idempotency + bounded retries. Tenacity wraps each SDK call with three
   attempts and exponential back-off. Temperature is fixed at zero; no
   randomness creeps in across retries.

5. Telemetry. Every call emits a structured log line with backend, model,
   token counts, and latency. The companion ``console`` narrator prints a
   human-readable line so the uvicorn terminal stays scannable.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.config import settings
from backend.utils import console
from backend.utils.logging_setup import get_logger


# Exception classes that should trigger a tenacity retry per provider.
# Wrapped in try/except so the offline path works without either SDK installed.

try:
    from anthropic import (
        APIConnectionError as _APIConnectionError,
        APITimeoutError as _APITimeoutError,
        InternalServerError as _InternalServerError,
        RateLimitError as _RateLimitError,
    )
    _ANTHROPIC_TRANSIENT: tuple[type[BaseException], ...] = (
        _APITimeoutError, _APIConnectionError,
        _RateLimitError, _InternalServerError,
    )
except ImportError:  # pragma: no cover — stub-only environments
    _ANTHROPIC_TRANSIENT = ()

try:
    from openai import (
        APIConnectionError as _OAIConnectionError,
        APITimeoutError as _OAITimeoutError,
        InternalServerError as _OAIInternalServerError,
        RateLimitError as _OAIRateLimitError,
    )
    _OPENAI_TRANSIENT: tuple[type[BaseException], ...] = (
        _OAITimeoutError, _OAIConnectionError,
        _OAIRateLimitError, _OAIInternalServerError,
    )
except ImportError:  # pragma: no cover — OpenAI not installed
    _OPENAI_TRANSIENT = ()

_TRANSIENT_RETRY: tuple[type[BaseException], ...] = (
    TimeoutError, ConnectionError, OSError,
) + _ANTHROPIC_TRANSIENT + _OPENAI_TRANSIENT

log = get_logger("llm")


class LLMError(RuntimeError):
    """Raised when the LLM (or its stub) cannot produce a valid response."""


@dataclass
class LLMResponse:
    text: str
    parsed: dict[str, Any] | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str
    backend: str  # "anthropic" | "stub"


# --------------------------------------------------------------------------- #
#  JSON extraction & repair                                                   #
# --------------------------------------------------------------------------- #

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _repair_common_claude_issues(s: str) -> str:
    """Apply repairs for the common malformations Claude emits when its
    response contains code with newlines, quotes, or backslashes.

    Order matters: each repair targets a specific failure mode I have seen
    in the wild on this codebase's prompts.
    """
    # 1. Trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # 2. Smart quotes -> ASCII quotes
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    # 3. Python None/True/False that occasionally bleed through
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction. Model output is parsed, never ``exec``ed.

    Four strategies, tried in order. If all four fail this raises LLMError
    so the caller can take its fallback path (heuristic review, HITL
    escalation, etc.). I have deliberately kept this synchronous and quick;
    retrying the LLM on a parse failure rarely helps because temperature is
    zero, so the second response is usually identical to the first.
    """
    text = (text or "").strip()
    # 1. Plain JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Wrapped in ```json ... ``` fences (Claude does this often)
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_common_claude_issues(candidate))
            except json.JSONDecodeError:
                pass
    # 3. Greedy outermost braces
    m = _JSON_BLOCK.search(text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # 4. Apply repairs to the brace-bounded substring and try again.
        repaired = _repair_common_claude_issues(candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Model output is not parseable as JSON after repair: {exc}"
            ) from exc
    raise LLMError("Model output did not contain valid JSON")


# --------------------------------------------------------------------------- #
#  Anthropic backend                                                          #
# --------------------------------------------------------------------------- #


class _AnthropicBackend:
    def __init__(self) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    @retry(
        # Retry on transient transport AND anthropic-side overload / rate
        # limit / 5xx errors. Parse failures (LLMError) are NOT retried —
        # with temperature=0 the second response is identical to the
        # first, so retrying just burns wall-clock and tokens.
        #
        # Fail-fast budget: 3 attempts, exponential backoff capped at 8s.
        # Total worst-case wait is ~2+4+8 = 14s per call. Long enough to
        # ride out a brief Anthropic blip; short enough that a sustained
        # outage cascades into the deterministic fallback (planner) or
        # HITL escalation (code generator) within a couple of minutes
        # rather than 20+. The circuit breaker below short-circuits
        # subsequent calls once a few consecutive failures are seen.
        retry=retry_if_exception_type(_TRANSIENT_RETRY),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _raw_call(self, system: str, user: str) -> tuple[str, int, int, int]:
        """Network-side only. Returns ``(text, tokens_in, tokens_out, latency_ms)``.
        Kept separate from ``call`` so ``_extract_json`` failures do not
        trigger the retry decorator."""
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = int((time.perf_counter() - t0) * 1000)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        tokens_in = getattr(msg.usage, "input_tokens", 0)
        tokens_out = getattr(msg.usage, "output_tokens", 0)
        return text, tokens_in, tokens_out, latency

    def call(self, system: str, user: str, expect_json: bool) -> LLMResponse:
        text, tokens_in, tokens_out, latency = self._raw_call(system, user)
        parsed = _extract_json(text) if expect_json else None
        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            model=self.model,
            backend="anthropic",
        )


# --------------------------------------------------------------------------- #
#  OpenAI backend (automatic fallback)                                        #
# --------------------------------------------------------------------------- #


class _OpenAIBackend:
    """Fallback LLM backend. Activated automatically when the Anthropic call
    fails (5xx, overload, circuit breaker open). Satisfies the same interface
    as ``_AnthropicBackend`` so ``LLMClient`` can swap providers transparently
    without any agent knowing a failover occurred.

    The same JSON-extraction and repair pipeline is used for both providers
    because GPT-4o also occasionally wraps output in ```json fences.
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    @retry(
        retry=retry_if_exception_type(_OPENAI_TRANSIENT or _TRANSIENT_RETRY),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _raw_call(self, system: str, user: str) -> tuple[str, int, int, int]:
        """Network-side only. Same contract as ``_AnthropicBackend._raw_call``."""
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency = int((time.perf_counter() - t0) * 1000)
        text = resp.choices[0].message.content or ""
        tokens_in = resp.usage.prompt_tokens if resp.usage else 0
        tokens_out = resp.usage.completion_tokens if resp.usage else 0
        return text, tokens_in, tokens_out, latency

    def call(self, system: str, user: str, expect_json: bool) -> LLMResponse:
        text, tokens_in, tokens_out, latency = self._raw_call(system, user)
        parsed = _extract_json(text) if expect_json else None
        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            model=self.model,
            backend="openai",
        )


# --------------------------------------------------------------------------- #
#  Deterministic stub backend                                                 #
# --------------------------------------------------------------------------- #


class _StubBackend:
    """A rule-based generator used when no LLM is configured.

    It implements just enough behaviour to make the *whole* workflow runnable
    end-to-end with deterministic, auditable outputs. Each agent that uses the
    LLM is responsible for providing a stub-friendly fallback (see the
    individual agent modules)."""

    model = "rule-based-stub-v1"

    def call(self, system: str, user: str, expect_json: bool) -> LLMResponse:  # noqa: ARG002
        # This backend never actually invokes a model — the agents short-circuit
        # to their built-in deterministic generators when `llm.enabled` is
        # False. The call path here exists only for graceful messaging.
        return LLMResponse(
            text="{}",
            parsed={} if expect_json else None,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            model=self.model,
            backend="stub",
        )


# --------------------------------------------------------------------------- #
#  Public facade                                                              #
# --------------------------------------------------------------------------- #


class LLMClient:
    # Circuit-breaker state (process-global). Tracks failures across ALL
    # providers — the breaker only opens when both Anthropic AND OpenAI
    # have failed ``_FAIL_LIMIT`` times consecutively. A single success
    # from either provider resets the counter and closes the breaker.
    _FAIL_LIMIT = 2
    _OPEN_FOR_SECONDS = 60.0
    _consecutive_failures = 0
    _open_until = 0.0

    def __init__(self) -> None:
        self._primary: Any = None    # Anthropic — tried first
        self._secondary: Any = None  # OpenAI   — automatic fallback

        if settings.anthropic_api_key:
            try:
                self._primary = _AnthropicBackend()
                log.info("llm.initialised", backend="anthropic", model=settings.anthropic_model)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("llm.anthropic_init_failed", reason=str(exc))

        _openai_key = settings.openai_api_key
        if _openai_key and _openai_key != "your-openai-api-key-here":
            try:
                self._secondary = _OpenAIBackend()
                log.info("llm.initialised", backend="openai", model=settings.openai_model)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("llm.openai_init_failed", reason=str(exc))

        self.enabled = self._primary is not None or self._secondary is not None
        if not self.enabled:
            log.info("llm.initialised", backend="stub")

    # -------------------- circuit breaker helpers ----------------------

    @classmethod
    def _breaker_open(cls) -> bool:
        """True when the breaker is currently open (calls should fail
        fast). Auto-closes when ``_open_until`` is in the past."""
        if cls._open_until and time.monotonic() < cls._open_until:
            return True
        if cls._open_until and time.monotonic() >= cls._open_until:
            # Time elapsed — half-open: allow one probing call. Reset
            # the open timer; if the probing call fails it'll be opened
            # again with a fresh window by ``_record_failure``.
            cls._open_until = 0.0
            cls._consecutive_failures = 0
        return False

    @classmethod
    def _record_failure(cls) -> None:
        cls._consecutive_failures += 1
        if cls._consecutive_failures >= cls._FAIL_LIMIT:
            cls._open_until = time.monotonic() + cls._OPEN_FOR_SECONDS
            log.warning(
                "llm.circuit_breaker_open",
                failures=cls._consecutive_failures,
                cooldown_s=cls._OPEN_FOR_SECONDS,
            )

    @classmethod
    def _record_success(cls) -> None:
        cls._consecutive_failures = 0
        cls._open_until = 0.0

    def complete(
        self,
        *,
        system: str,
        user: str,
        expect_json: bool = True,
        purpose: str = "completion",
    ) -> LLMResponse:
        # Stub path — no LLM configured; agents use their own deterministic
        # fallbacks or escalate to HITL when they receive an empty response.
        if not self.enabled:
            return LLMResponse(
                text="{}",
                parsed={} if expect_json else None,
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                model="rule-based-stub-v1",
                backend="stub",
            )

        # Circuit-breaker short-circuit. Only fires after BOTH providers have
        # failed _FAIL_LIMIT times in a row. Lets the pipeline cascade through
        # deterministic / HITL fallbacks in seconds rather than retrying 14s
        # per call across 7+ derivations.
        if LLMClient._breaker_open():
            raise LLMError(
                "LLM circuit breaker is open after consecutive failures across "
                "all providers. Falling back to deterministic / HITL path."
            )

        primary_model = getattr(self._primary, "model", None) or getattr(self._secondary, "model", "?")
        console.llm_call_start(primary_model, purpose)

        last_exc: Exception | None = None

        # ── Try primary (Anthropic) ──────────────────────────────────────────
        if self._primary is not None:
            try:
                resp = self._primary.call(system, user, expect_json)
                LLMClient._record_success()
                console.llm_call_done(resp.latency_ms, resp.tokens_in, resp.tokens_out)
                self._log_call(resp, purpose)
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc if isinstance(exc, LLMError) else LLMError(f"{type(exc).__name__}: {exc}")
                log.warning(
                    "llm.primary_failed_trying_fallback",
                    purpose=purpose,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        # ── Try secondary (OpenAI) ───────────────────────────────────────────
        if self._secondary is not None:
            try:
                log.info("llm.using_fallback", backend="openai", purpose=purpose)
                resp = self._secondary.call(system, user, expect_json)
                LLMClient._record_success()
                console.llm_call_done(resp.latency_ms, resp.tokens_in, resp.tokens_out)
                self._log_call(resp, purpose)
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc if isinstance(exc, LLMError) else LLMError(f"{type(exc).__name__}: {exc}")
                log.warning(
                    "llm.secondary_failed",
                    purpose=purpose,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        # ── Both providers failed ────────────────────────────────────────────
        LLMClient._record_failure()
        raise last_exc or LLMError("All LLM providers failed with no recorded error.")

    def _log_call(self, resp: LLMResponse, purpose: str) -> None:  # noqa: ARG002
        log.info(
            "llm.call",
            backend=resp.backend,
            model=resp.model,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
        )


# Singleton — agents import this directly.
llm = LLMClient()

"""Test doubles and tiny assertion helpers (no third-party imports)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def raises(exc_type: type[BaseException], match: str | None = None):
    """Minimal `pytest.raises` replacement so tests run under any runner."""
    try:
        yield
    except exc_type as exc:  # noqa: PERF203
        if match is not None and match.lower() not in str(exc).lower():
            raise AssertionError(f"{exc!r} does not contain {match!r}") from None
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


class ScriptedTestUser:
    """Deterministic stand-in for `DynamicTestUser` (no Ollama required)."""

    def __init__(self, replies: list[str] | None = None, default: str = "D'accord, merci.") -> None:
        self.replies = list(replies or [])
        self.default = default
        self.calls: list[dict[str, Any]] = []

    def generate_autonomous_response(
        self,
        va_prompt: str,
        customer_goal: str,
        customer_profile: str,
        language: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        self.calls.append({
            "va_prompt": va_prompt,
            "goal": customer_goal,
            "profile": customer_profile,
            "language": language,
            "history_len": len(conversation_history),
        })
        if self.replies:
            return self.replies.pop(0)
        return self.default


class RecordingAdapter:
    """DFCX adapter double returning a preset list of turns."""

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = list(turns)
        self.sent: list[dict[str, Any]] = []

    def _next(self) -> dict[str, Any]:
        return self.turns.pop(0) if self.turns else {"response_text": "", "end_session": True}

    def continue_handoff(self, target_environment, parameters, utterance, session_id,
                         language_code="fr-CA"):
        self.sent.append({"kind": "handoff", "text": utterance, "language": language_code})
        return self._next()

    def send_text(self, target_environment, session_id, text, language_code="fr-CA",
                  parameters=None):
        self.sent.append({"kind": "text", "text": text, "language": language_code})
        return self._next()


class RecordingNgaAdapter:
    """NGA adapter double returning a preset list of post-handback turns."""

    def __init__(self, turns: list[dict[str, Any]] | None = None) -> None:
        self.turns = list(turns or [])
        self.sent: list[dict[str, Any]] = []
        self.resumed: list[dict[str, Any]] = []

    def _next(self) -> dict[str, Any]:
        turn = self.turns.pop(0) if self.turns else {
            "response_text": "", "end_session": True, "end_reason": "end_interaction"
        }
        return {"channel": "nga", **turn}

    def resume_from_handback(self, session_id, handback, *, language_code="fr-CA"):
        self.resumed.append({"session_id": session_id, "handback": handback,
                             "language": language_code})
        return self._next()

    def send_text(self, session_id, text, *, language_code="fr-CA"):
        self.sent.append({"kind": "text", "text": text, "language": language_code})
        return self._next()


def silent(_message: str) -> None:
    """Printer that swallows runner output during tests."""


STAGING_ENV = {
    "app_name": "projects/prj-test/locations/us/apps/app-id",
    "flow_environment": "projects/prj-test/locations/us-central1/agents/agent-id/environments/env-id",
    "language_code": "fr-CA",
}


def make_case(**overrides: Any) -> dict[str, Any]:
    """A minimal but valid offline-runnable case."""
    case: dict[str, Any] = {
        "name": "unit_case",
        "tags": ["unit", "billing"],
        "steering_utterance": "J'ai une offre promotionnelle expirée.",
        "expected_transfer": {
            "environment": STAGING_ENV["flow_environment"],
            "route": "bill_promo_expired",
            "language": "fr-ca",
            "handoff_to": "post_steering",
        },
        "autonomous_flow": {
            "enabled": True,
            "max_turns": 5,
            "customer_goal": "Understand why the promotion expired.",
            "customer_profile": "Brief, polite, Canadian French.",
            "stop_phrases": ["bonne journée"],
        },
    }
    case.update(overrides)
    return case

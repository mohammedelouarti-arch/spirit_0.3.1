"""Deterministic stand-in for the Ollama customer.

Used by `--fake-llm` so CI can exercise the full autonomous loop (and the
`expected_conversation` assertions) on a machine with no Ollama daemon.
"""

from __future__ import annotations

from itertools import cycle
from typing import Any

DEFAULT_REPLIES = (
    "Oui, s'il vous pla\u00eet, pouvez-vous v\u00e9rifier?",
    "D'accord, allez-y.",
    "Merci, c'est bien ce que je voulais savoir.",
    "Non merci, c'est tout pour moi.",
)


class CannedTestUser:
    """Implements the `TestUser` protocol without any network calls."""

    def __init__(self, replies: tuple[str, ...] | list[str] = DEFAULT_REPLIES) -> None:
        self.replies = list(replies) or list(DEFAULT_REPLIES)
        self._cycle = cycle(self.replies)
        self.calls: list[dict[str, Any]] = []

    def health_check(self) -> None:
        """No-op: nothing to preflight."""

    def generate_autonomous_response(
        self,
        va_prompt: str,
        customer_goal: str,
        customer_profile: str,
        language: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        self.calls.append({"va_prompt": va_prompt, "language": language})
        return next(self._cycle)

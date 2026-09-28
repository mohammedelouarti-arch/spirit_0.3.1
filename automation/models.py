from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransferPayload:
    target: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class JourneyResult:
    session_id: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    transfer: TransferPayload | None = None


@dataclass
class ConversationTurn:
    """One VA prompt plus the Ollama-generated customer reply.

    `channel` records which system produced `va_response` -- "dfcx" or "nga" --
    because a single call can bounce between them several times.
    `leg` is the 1-based index of the contiguous stretch on that channel.
    """

    turn: int
    va_response: str
    user_response: str
    matched_intent: str | None = None
    current_page: str | None = None
    channel: str = "dfcx"
    leg: int = 1


@dataclass
class LegTransition:
    """A control transfer between NGA and DFCX during the conversation.

    `kind` is "handback" (DFCX -> NGA, a `transferToNga` payload) or
    "retransfer" (NGA -> DFCX, a `transferToDialogflow` payload).
    `text` is whatever the outgoing system said on the transition turn; it is
    spoken to the caller but never answered, so it has no ConversationTurn.
    """

    kind: str
    leg: int
    after_turn: int
    payload: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    # `$session.params.x` references DFCX passed through unsubstituted.
    unresolved: list[str] = field(default_factory=list)
    # The target actually used, after any configured fallback substitution.
    resolved_target: str | None = None

    @property
    def target(self) -> str | None:
        return self.payload.get("transferToNga") or self.payload.get("transferToDialogflow")

    @property
    def effective_target(self) -> str | None:
        return self.resolved_target or self.target


@dataclass
class ConversationResult:
    """Outcome of the autonomous, Ollama-driven post-steering conversation.

    The conversation is multi-leg: it starts on DFCX after the steering
    utterance and may bounce back to NGA (handback) and forward again
    (retransfer) any number of times until a real terminal condition.
    """

    turns: list[ConversationTurn] = field(default_factory=list)
    stop_reason: str = "not_started"
    final_dfcx: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, str]] = field(default_factory=list)
    transitions: list[LegTransition] = field(default_factory=list)
    final_turn: dict[str, Any] = field(default_factory=dict)
    final_channel: str = "dfcx"
    legs: int = 1
    # Page the session was stuck on when a NO_MATCH run ended the call.
    stranded_page: str | None = None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def handbacks(self) -> list[dict[str, Any]]:
        """Every DFCX -> NGA transfer payload, in order."""
        return [item.payload for item in self.transitions if item.kind == "handback"]

    @property
    def defects(self) -> list[str]:
        """Unresolved `$session.params.*` references across every transition.

        A transfer carrying one of these still *fires* -- which is why the
        symptom is "the transfer works but the experience is wrong".
        """
        found: list[str] = []
        for item in self.transitions:
            found += [f"{item.kind} leg {item.leg}: {ref}" for ref in item.unresolved]
        return found

    @property
    def retransfers(self) -> list[dict[str, Any]]:
        """Every NGA -> DFCX transfer payload, in order."""
        return [item.payload for item in self.transitions if item.kind == "retransfer"]

    @property
    def handback(self) -> dict[str, Any] | None:
        """The most recent handback payload (back-compat with 0.3.x)."""
        found = self.handbacks
        return found[-1] if found else None

    @property
    def va_text(self) -> str:
        """All VA text concatenated - handy for phrase assertions.

        Includes transition turns, so a farewell said on the way out of DFCX
        still counts for `must_mention` / `must_not_mention`.
        """
        spoken = [turn.va_response for turn in self.turns]
        spoken += [item.text for item in self.transitions if item.text]
        return " ".join(spoken)

    def turns_on(self, channel: str) -> list[ConversationTurn]:
        return [turn for turn in self.turns if turn.channel == channel]

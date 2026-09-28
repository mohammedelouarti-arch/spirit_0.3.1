"""Offline stand-in for the NGA side of a resumed call.

`fake_dfcx.FakeDfcxAdapter` covers the DFCX legs; this covers what happens
after DFCX hands the call back with a `transferToNga` payload. Scripts are
keyed by the handback's `variables.handoff_to`, so a case controls which NGA
menu it lands in. A script step carrying `transfer` re-enters DFCX, which is
what makes NGA -> DFCX -> NGA -> DFCX ping-pong testable with no tenant.
"""

from __future__ import annotations

from typing import Any

from automation.language import DEFAULT_LANGUAGE

# Ordered NGA turns played after a handback, keyed by `variables.handoff_to`.
# A step with `transfer` sends the call back to DFCX on the named route.
SCRIPTS: dict[str, list[dict[str, Any]]] = {
    "billing_menu": [
        {"text": "Bon retour. Je suis de nouveau avec vous. "
                 "Voulez-vous continuer avec votre facture ou parler d'un probl\u00e8me technique?"},
        {"text": "Tr\u00e8s bien, je vous transf\u00e8re au service technique.",
         "transfer": {"route": "tech_internet_outage", "handoff_to": "post_handback"}},
    ],
    "main_menu": [
        {"text": "Je vous ram\u00e8ne au menu principal. Que puis-je faire pour vous?"},
        {"text": "Compris, je vous transf\u00e8re.",
         "transfer": {"route": "bill_explain_charges", "handoff_to": "post_handback"}},
    ],
    "close_call": [
        {"text": "Merci d'avoir appel\u00e9. Bonne journ\u00e9e!",
         "end_session": True, "end_reason": "end_interaction"},
    ],
}

FALLBACK_SCRIPT: list[dict[str, Any]] = [
    {"text": "Je reprends la conversation. Comment puis-je vous aider?"},
    {"text": "Merci d'avoir appel\u00e9. Bonne journ\u00e9e!",
     "end_session": True, "end_reason": "end_interaction"},
]


class FakeNgaAdapter:
    """Deterministic, stateful fake implementing the `NgaAdapter` protocol."""

    def __init__(
        self,
        scripts: dict[str, list[dict[str, Any]]] | None = None,
        *,
        default_environment: str | None = None,
    ) -> None:
        self.scripts = scripts if scripts is not None else SCRIPTS
        # The DFCX environment a `transfer` step re-enters when the script does
        # not name one explicitly -- normally the same environment the call was
        # steered into originally.
        self.default_environment = default_environment
        # session_id -> {"handoff_to": str, "turn": int, "variables": dict}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def _script_for(self, handoff_to: str | None) -> list[dict[str, Any]]:
        return self.scripts.get(str(handoff_to or ""), FALLBACK_SCRIPT)

    def _turn(self, session_id: str, text: str, language_code: str) -> dict[str, Any]:
        state = self.sessions.setdefault(
            session_id, {"handoff_to": None, "turn": 0, "variables": {}}
        )
        script = self._script_for(state["handoff_to"])
        index = min(state["turn"], len(script) - 1)
        step = script[index]
        state["turn"] += 1
        self.calls.append({
            "session_id": session_id, "text": text,
            "language_code": language_code, "turn": state["turn"],
        })

        transfer = None
        if step.get("transfer"):
            spec = dict(step["transfer"])
            environment = spec.pop("environment", None) or self.default_environment
            parameters = {
                "language": language_code,
                "utterance": text,
                **spec,
            }
            transfer = {"transferToDialogflow": environment, "parameters": parameters}

        return {
            "channel": "nga",
            "response_text": step["text"],
            "matched_intent": step.get("matched_intent"),
            "current_page": step.get("current_page", f"nga_step_{index + 1}"),
            "parameters": dict(state["variables"]),
            "end_session": bool(step.get("end_session")),
            "end_reason": step.get("end_reason"),
            "transfer": transfer,
            "raw": {"fake_nga": True, "script_index": index},
        }

    def resume_from_handback(
        self,
        session_id: str,
        handback: dict[str, Any],
        *,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        """Re-enter NGA carrying the handback's variables, and return its turn."""
        variables = (handback or {}).get("variables") or {}
        self.sessions[session_id] = {
            "handoff_to": variables.get("handoff_to"),
            "turn": 0,
            "variables": variables,
        }
        return self._turn(session_id, str(variables.get("utterance") or ""), language_code)

    def send_text(
        self,
        session_id: str,
        text: str,
        *,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        return self._turn(session_id, text, language_code)

"""Offline stand-in for Dialogflow CX.

Keeps per-session state and walks a small scripted French billing dialogue so
the autonomous (Ollama-driven) loop can be exercised end to end without a live
tenant. Scripts are keyed by the `route` transfer parameter; unknown routes get
a generic fallback script.
"""

from __future__ import annotations

from typing import Any

from automation.language import DEFAULT_LANGUAGE

# Each script is the ordered list of VA turns after the handoff.
# The last entry should contain a stop phrase / end_session flag.
SCRIPTS: dict[str, list[dict[str, Any]]] = {
    "bill_promo_expired": [
        {"text": "Je comprends que votre offre promotionnelle est expir\u00e9e. "
                 "Voulez-vous que je v\u00e9rifie les promotions disponibles sur votre compte?"},
        {"text": "J'ai v\u00e9rifi\u00e9 votre compte. Une promotion \u00e9quivalente peut \u00eatre appliqu\u00e9e "
                 "d\u00e8s le prochain cycle de facturation. Souhaitez-vous l'appliquer?"},
        {"text": "C'est fait, la promotion est appliqu\u00e9e. Puis-je vous aider avec autre chose?"},
        {"text": "Merci d'avoir appel\u00e9. Bonne journ\u00e9e!", "end_session": True,
         "end_reason": "end_interaction"},
    ],
    "bill_explain_charges": [
        {"text": "Je peux vous expliquer les frais sur votre facture. "
                 "Parlez-vous des frais uniques ou des frais mensuels?"},
        {"text": "Ces frais correspondent \u00e0 un ajustement au prorata apr\u00e8s votre changement de forfait. "
                 "Est-ce que cela r\u00e9pond \u00e0 votre question?"},
        {"text": "Parfait. Merci d'avoir appel\u00e9. Bonne journ\u00e9e!", "end_session": True,
         "end_reason": "end_interaction"},
    ],
    # Route name used by suites/e2e_handoff/tech_internet_outage.yaml. Without
    # it the case silently fell through to FALLBACK_SCRIPT, which is exactly
    # what tests/test_suite_files.py guards against.
    "tech_connection_issue": [
        {"text": "Je vois une panne signal\u00e9e dans votre secteur. "
                 "Voulez-vous recevoir une notification d\u00e8s le r\u00e9tablissement du service?"},
        {"text": "La notification est activ\u00e9e. Puis-je vous aider avec autre chose?"},
        {"text": "Merci d'avoir appel\u00e9. Bonne journ\u00e9e!", "end_session": True,
         "end_reason": "end_interaction"},
    ],
    "tech_internet_outage": [
        {"text": "Je vois une panne signal\u00e9e dans votre secteur. "
                 "Voulez-vous recevoir une notification d\u00e8s le r\u00e9tablissement du service?"},
        {"text": "La notification est activ\u00e9e. Puis-je vous aider avec autre chose?"},
        {"text": "Merci d'avoir appel\u00e9. Bonne journ\u00e9e!", "end_session": True,
         "end_reason": "end_interaction"},
    ],
    "agent_request": [
        {"text": "Bien s\u00fbr, je vais vous transf\u00e9rer \u00e0 un conseiller.",
         "end_session": True, "end_reason": "live_agent_handoff"},
    ],
    # Hands the call back to NGA mid-conversation. The loop must treat this as
    # a leg change, not an ending: NGA picks the call up and may steer it
    # straight back into DFCX on another route.
    "bill_dispute_charge": [
        {"text": "Je vois le montant contest\u00e9 sur votre facture. "
                 "Voulez-vous que j'ouvre une demande de r\u00e9vision?"},
        # Mirrors the real `Trigger NGA` route: the target is a TEMPLATE. It
        # resolves only when `nga_agent_id` was bound on the inbound hop --
        # otherwise the literal passes through and the call transfers nowhere.
        {"text": "Cette demande doit passer par le menu principal. "
                 "Je vous y ram\u00e8ne tout de suite.",
         "handback": {
             "transferToNga": "$session.params.nga_agent_id",
             "variables": {
                 "handoff_from": "bill_dispute_charge",
                 "handoff_to": "billing_menu",
                 "utterance": "r\u00e9vision de facturation",
             },
             "ignoreSessionParameters": True,
         }},
    ],
}

FALLBACK_SCRIPT: list[dict[str, Any]] = [
    {"text": "Je peux vous aider avec cela. Pouvez-vous m'en dire un peu plus?"},
    {"text": "Merci. Puis-je vous aider avec autre chose?"},
    {"text": "Merci d'avoir appel\u00e9. Bonne journ\u00e9e!", "end_session": True,
     "end_reason": "end_interaction"},
]


class FakeDfcxAdapter:
    """Deterministic, stateful fake implementing the adapter protocol."""

    def __init__(self, scripts: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.scripts = scripts if scripts is not None else SCRIPTS
        # session_id -> {"route": str, "turn": int, "parameters": dict}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self.reentries: list[dict[str, Any]] = []

    def _script_for(self, route: str | None) -> list[dict[str, Any]]:
        return self.scripts.get(str(route or ""), FALLBACK_SCRIPT)

    @staticmethod
    def _render(payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        """Substitute `$session.params.x` the way DFCX does when emitting.

        Crucially, an *unbound* parameter is not an error and is not null --
        the literal reference is passed straight through, which is what makes
        the broken payload look healthy to everything downstream.
        """
        def substitute(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: substitute(item) for key, item in value.items()}
            if isinstance(value, list):
                return [substitute(item) for item in value]
            if isinstance(value, str) and value.startswith("$session.params."):
                name = value[len("$session.params."):]
                bound = parameters.get(name)
                return bound if bound not in (None, "") else value
            return value

        return substitute(payload)

    def _turn(self, session_id: str, text: str, language_code: str) -> dict[str, Any]:
        state = self.sessions.setdefault(session_id, {"route": None, "turn": 0, "parameters": {}})
        script = self._script_for(state["route"])
        index = min(state["turn"], len(script) - 1)
        step = script[index]
        state["turn"] += 1
        self.calls.append({
            "session_id": session_id, "text": text,
            "language_code": language_code, "turn": state["turn"],
        })
        handback = step.get("handback")
        if handback:
            handback = self._render(handback, state["parameters"])
            # The call is now with NGA; DFCX is parked on `Trigger NGA` until
            # an explicit intent trigger re-enters it.
            state["parked"] = True
        return {
            "response_text": step["text"],
            "match_type": step.get("match_type", "INTENT"),
            "no_match": False,
            "matched_intent": step.get("matched_intent", state["route"]),
            "current_page": step.get("current_page", f"page_{index + 1}"),
            "parameters": dict(state["parameters"]),
            "end_session": bool(step.get("end_session")),
            "end_reason": step.get("end_reason"),
            "handback": handback,
            "raw": {"fake": True, "script_index": index},
        }

    def continue_handoff(
        self,
        target_environment: str,
        parameters: dict[str, Any],
        utterance: str,
        session_id: str,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        parameters = parameters or {}
        # Faithful to the live adapter, where `continue_handoff` is just
        # `send_text` with query parameters: sending text does NOT reposition a
        # session. If it is parked on `Trigger NGA`, attaching parameters
        # changes nothing and the turn still NO_MATCHes -- which is exactly the
        # prod failure this models. Only `reenter` (an intent trigger) unparks.
        existing = self.sessions.get(session_id)
        if existing and existing.get("parked"):
            existing["parameters"] = {**existing.get("parameters", {}), **parameters}
            return self.send_text(target_environment, session_id, utterance, language_code)

        self.sessions[session_id] = {
            "route": parameters.get("route"),
            "turn": 0,
            "parameters": parameters,
            "parked": False,
        }
        result = self._turn(session_id, utterance, language_code)
        # The first post-handoff turn is what `validate()` asserts the route on.
        result["matched_intent"] = parameters.get("route")
        return result

    def reenter(
        self,
        target_environment: str,
        parameters: dict[str, Any],
        utterance: str,
        session_id: str,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        """Re-entry after a handback: an explicit Default-Welcome trigger.

        Unparks the session and runs the new route's script. Faithful to prod,
        where this is `queryInput.intent` rather than text.
        """
        parameters = parameters or {}
        self.sessions[session_id] = {
            "route": parameters.get("route"),
            "turn": 0,
            "parameters": parameters,
            "parked": False,
        }
        self.reentries.append({
            "session_id": session_id,
            "route": parameters.get("route"),
            "handoff_to": parameters.get("handoff_to"),
        })
        result = self._turn(session_id, utterance, language_code)
        result["matched_intent"] = parameters.get("route")
        return result

    # DFCX rotates its no-match prompts, which is why text-level repeat
    # detection never catches a stranded session.
    NO_MATCH_PROMPTS = (
        "J'ai du mal \u00e0 comprendre cette question.",
        "J'ai mal compris votre demande.",
    )

    def send_text(
        self,
        target_environment: str,
        session_id: str,
        text: str,
        language_code: str = DEFAULT_LANGUAGE,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.sessions.setdefault(
            session_id, {"route": None, "turn": 0, "parameters": {}, "parked": False}
        )
        # Faithful to prod: after a handback the session sits on `Trigger NGA`,
        # a page with no intent routes. Text there NO_MATCHes forever -- only
        # an intent trigger (`reenter`) gets the call moving again.
        if state.get("parked"):
            index = state["turn"]
            state["turn"] += 1
            self.calls.append({
                "session_id": session_id, "text": text,
                "language_code": language_code, "turn": state["turn"], "parked": True,
            })
            return {
                "response_text": self.NO_MATCH_PROMPTS[index % len(self.NO_MATCH_PROMPTS)],
                "matched_intent": None,
                "match_type": "NO_MATCH",
                "match_confidence": 0.3,
                "no_match": True,
                "current_page": "Trigger NGA",
                "parameters": dict(state["parameters"]),
                "end_session": False,
                "end_reason": None,
                "handback": None,
                "raw": {"fake": True, "parked_on": "Trigger NGA"},
            }
        return self._turn(session_id, text, language_code)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from automation.language import DEFAULT_LANGUAGE, normalize_language, same_language
from automation.models import (
    ConversationResult,
    ConversationTurn,
    JourneyResult,
    LegTransition,
    TransferPayload,
)
from automation.utils import (
    extract_handback,
    extract_transfer,
    is_unresolved,
    redact,
    unresolved_references,
)

Printer = Callable[[str], None]

# Ceiling on NGA <-> DFCX bounces before we call it a ping-pong loop. Each
# handback and each retransfer counts as one leg.
DEFAULT_MAX_LEGS = 20


class DfcxAdapter(Protocol):
    """Minimal surface both the live and fake DFCX adapters implement."""

    def continue_handoff(
        self, target_environment: str, parameters: dict[str, Any],
        utterance: str, session_id: str, language_code: str = ...,
    ) -> dict[str, Any]: ...

    def send_text(
        self, target_environment: str, session_id: str, text: str,
        language_code: str = ..., parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    # Optional. Adapters without it fall back to `continue_handoff`, which is
    # only correct for the *first* entry into DFCX -- see `_reenter_dfcx`.
    def reenter(
        self, target_environment: str, parameters: dict[str, Any],
        utterance: str, session_id: str, language_code: str = ...,
    ) -> dict[str, Any]: ...


class NgaAdapter(Protocol):
    """The NGA side of a resumed call, after DFCX hands control back."""

    def resume_from_handback(
        self, session_id: str, handback: dict[str, Any], *, language_code: str = ...,
    ) -> dict[str, Any]: ...

    def send_text(
        self, session_id: str, text: str, *, language_code: str = ...,
    ) -> dict[str, Any]: ...


class TestUser(Protocol):
    def generate_autonomous_response(
        self, va_prompt: str, customer_goal: str, customer_profile: str,
        language: str, conversation_history: list[dict[str, str]],
    ) -> str: ...


# --------------------------------------------------------------------------
# Case loading
# --------------------------------------------------------------------------

def project_id(app_name: str) -> str:
    return app_name.split("projects/", 1)[1].split("/", 1)[0]


def load_cases(folder: str, tags: list[str] | None = None) -> list[dict[str, Any]]:
    """Load every eval in `folder`, keeping those whose tags superset `tags`."""
    tags = tags or []
    wanted = {str(tag).lower() for tag in tags}
    found: list[dict[str, Any]] = []
    for path in sorted(Path(folder).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        for case in data.get("evals", []) or []:
            case_tags = {str(tag).lower() for tag in case.get("tags", []) or []}
            if wanted and not wanted.issubset(case_tags):
                continue
            case.setdefault("source_file", str(path))
            found.append(case)
    return found


def case_language(case: dict[str, Any], env: dict[str, Any] | None = None) -> str:
    """Resolve the BCP-47 language for a case (expected -> env -> default)."""
    expected = (case.get("expected_transfer") or {}).get("language")
    fallback = (env or {}).get("language_code")
    return normalize_language(expected or fallback or DEFAULT_LANGUAGE)


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------

def validate(case: dict[str, Any], transfer: TransferPayload, dfcx: dict[str, Any]) -> list[str]:
    """Assert the NGA -> DFCX handoff payload and the first DFCX turn."""
    expected = case.get("expected_transfer") or {}
    parameters = transfer.parameters or {}
    errors: list[str] = []

    if expected.get("environment") and transfer.target != expected["environment"]:
        errors.append(
            f"target environment mismatch: expected {expected['environment']!r}, got {transfer.target!r}"
        )
    for key in ("route", "handoff_to"):
        if expected.get(key) and parameters.get(key) != expected[key]:
            errors.append(f"{key}: expected {expected[key]!r}, got {parameters.get(key)!r}")

    actual_language = parameters.get("language") or parameters.get("language_code")
    if expected.get("language") and not same_language(actual_language, expected["language"]):
        errors.append(f"language: expected {expected['language']!r}, got {actual_language!r}")

    if parameters.get("utterance") and parameters["utterance"] != case.get("steering_utterance"):
        errors.append("transferred utterance mismatch")

    matched = (dfcx or {}).get("matched_intent")
    if expected.get("route") and matched and matched != expected["route"]:
        errors.append(f"DFCX intent mismatch: expected {expected['route']!r}, got {matched!r}")
    return errors


def validate_conversation(case: dict[str, Any], convo: ConversationResult) -> list[str]:
    """Assert the autonomous, Ollama-driven portion of the call.

    Driven by an optional `expected_conversation` block in the YAML case::

        expected_conversation:
          min_turns: 2
          max_turns: 8
          must_mention: ["promotion"]          # any VA text, case-insensitive
          must_not_mention: ["erreur syst\u00e8me"]
          expect_stop_reason: "stop_phrase"     # prefix match
          forbid_stop_reasons: ["max_turns", "empty_llm_response"]
          expect_end_reason: "end_interaction"
          expect_final_channel: "nga"           # which system ended the call
          expect_legs: 3                        # NGA/DFCX stretches, incl. the first
          expect_handbacks: 1                   # DFCX -> NGA transfers
          expect_retransfers: 1                 # NGA -> DFCX transfers
          min_handbacks: 1
          expected_handback:                    # the LAST handback payload
            handoff_to: "billing_menu"
            transferToNga: "projects/.../apps/nga-app-id"
          expected_handbacks:                   # or pin each one, in order
            - {handoff_to: "billing_menu"}
          expected_retransfer:                  # the LAST NGA -> DFCX transfer
            route: "tech_internet_outage"
    """
    expected = case.get("expected_conversation") or {}

    # Structural defects are never a legitimate outcome, so they are asserted
    # even when the case declares no `expected_conversation` block. Without
    # this a stranded or mis-transferred call reports a green PASS -- which is
    # exactly how the NO_MATCH loop stayed invisible in a live run.
    errors: list[str] = _validate_health(expected, convo)
    if not expected:
        return errors

    haystack = convo.va_text.lower()

    min_turns = expected.get("min_turns")
    if min_turns is not None and convo.turn_count < int(min_turns):
        errors.append(f"conversation too short: {convo.turn_count} turn(s), expected >= {min_turns}")

    max_turns = expected.get("max_turns")
    if max_turns is not None and convo.turn_count > int(max_turns):
        errors.append(f"conversation too long: {convo.turn_count} turn(s), expected <= {max_turns}")

    for phrase in expected.get("must_mention", []) or []:
        if str(phrase).lower() not in haystack:
            errors.append(f"VA never mentioned {phrase!r}")

    for phrase in expected.get("must_not_mention", []) or []:
        if str(phrase).lower() in haystack:
            errors.append(f"VA mentioned forbidden phrase {phrase!r}")

    want_stop = expected.get("expect_stop_reason")
    if want_stop and not convo.stop_reason.startswith(str(want_stop)):
        errors.append(f"stop reason: expected {want_stop!r}, got {convo.stop_reason!r}")

    for forbidden in expected.get("forbid_stop_reasons", []) or []:
        if convo.stop_reason.startswith(str(forbidden)):
            errors.append(f"conversation ended with forbidden stop reason {convo.stop_reason!r}")

    want_end = expected.get("expect_end_reason")
    if want_end:
        # The call can end on either channel now, so read the last turn
        # whatever it was, falling back to the last DFCX turn.
        actual_end = (convo.final_turn or convo.final_dfcx or {}).get("end_reason")
        if actual_end != want_end:
            errors.append(f"end reason: expected {want_end!r}, got {actual_end!r}")

    want_channel = expected.get("expect_final_channel")
    if want_channel and convo.final_channel != want_channel:
        errors.append(
            f"final channel: expected {want_channel!r}, got {convo.final_channel!r}"
        )

    errors += _validate_legs(expected, convo)

    if expected.get("require_empty_user_turns") is False:
        for turn in convo.turns:
            if not turn.user_response.strip():
                errors.append(f"turn {turn.turn}: Ollama produced an empty customer response")
    return errors


def _match_payload(actual: dict[str, Any], expected: dict[str, Any], label: str) -> list[str]:
    """Compare a transfer/handback payload against an expectation block.

    Top-level keys (`transferToNga`, `transferToDialogflow`) are read off the
    payload itself; everything else is looked up in its `variables` (NGA
    handback) or `parameters` (DFCX transfer) bag, so cases can write
    `{handoff_to: "billing_menu"}` without knowing the nesting.
    """
    actual = actual or {}
    bag = actual.get("variables") or actual.get("parameters") or {}
    errors: list[str] = []
    for key, want_value in (expected or {}).items():
        if key in ("transferToNga", "transferToDialogflow"):
            actual_value = actual.get(key)
        else:
            actual_value = bag.get(key)
        if actual_value != want_value:
            errors.append(f"{label} {key}: expected {want_value!r}, got {actual_value!r}")
    return errors


def _validate_health(expected: dict[str, Any], convo: ConversationResult) -> list[str]:
    """Assert outcomes that are never legitimate, block or no block.

    Both of these are silent killers: the transfer fires and the report looks
    structurally fine, so no ordinary assertion notices.
    """
    errors: list[str] = []

    # An unresolved `$session.params.*` means the payload carried a literal
    # string where a resource name belonged.
    if not expected.get("allow_unresolved_references", False):
        for defect in convo.defects:
            errors.append(
                f"unresolved template reference in {defect} -- the parameter was never "
                "bound in this session, so the payload transferred to a literal string"
            )

    # A NO_MATCH run means we are talking to a page with no intent routes,
    # i.e. the session was never repositioned after a handback.
    if convo.stop_reason == "no_match_loop" and not expected.get("allow_no_match_loop", False):
        errors.append(
            f"session stranded on page {convo.stranded_page!r}: consecutive NO_MATCH responses. "
            "The re-entry after a handback must trigger the Default Welcome Intent, not send text"
        )
    return errors


def _validate_legs(expected: dict[str, Any], convo: ConversationResult) -> list[str]:
    """Assert the NGA <-> DFCX ping-pong: handbacks, retransfers, leg count."""
    errors: list[str] = []
    handbacks = convo.handbacks
    retransfers = convo.retransfers


    for key, actual_count, noun in (
        ("expect_handbacks", len(handbacks), "handback"),
        ("expect_retransfers", len(retransfers), "retransfer"),
        ("expect_legs", convo.legs, "leg"),
    ):
        want = expected.get(key)
        if want is not None and actual_count != int(want):
            errors.append(f"{noun} count: expected {want}, got {actual_count}")

    for key, actual_count, noun in (
        ("min_handbacks", len(handbacks), "handback"),
        ("min_retransfers", len(retransfers), "retransfer"),
    ):
        want = expected.get(key)
        if want is not None and actual_count < int(want):
            errors.append(f"{noun} count: expected >= {want}, got {actual_count}")

    # Single-payload form checks the LAST handback (0.3.x compatible);
    # the list form pins each one in order.
    single = expected.get("expected_handback")
    if single:
        if not handbacks:
            errors.append("expected a handback to NGA, but the call never left DFCX")
        else:
            errors += _match_payload(handbacks[-1], single, "handback")

    for index, want_payload in enumerate(expected.get("expected_handbacks", []) or []):
        if index >= len(handbacks):
            errors.append(f"handback #{index + 1} never happened ({len(handbacks)} occurred)")
            continue
        errors += _match_payload(handbacks[index], want_payload, f"handback #{index + 1}")

    single_retransfer = expected.get("expected_retransfer")
    if single_retransfer:
        if not retransfers:
            errors.append("expected NGA to steer back into DFCX, but it never did")
        else:
            errors += _match_payload(retransfers[-1], single_retransfer, "retransfer")

    for index, want_payload in enumerate(expected.get("expected_retransfers", []) or []):
        if index >= len(retransfers):
            errors.append(f"retransfer #{index + 1} never happened ({len(retransfers)} occurred)")
            continue
        errors += _match_payload(retransfers[index], want_payload, f"retransfer #{index + 1}")

    return errors


# --------------------------------------------------------------------------
# Autonomous conversation
#
# Ollama drives every turn after the steering utterance. The call is
# multi-leg: it begins on DFCX and may be handed back to NGA
# (`transferToNga`) and steered forward into DFCX again
# (`transferToDialogflow`) as many times as the agents ask for. A handback is
# a *transition*, never a terminal state -- the conversation only ends on a
# real end signal, a stop phrase, or a guard rail.
# --------------------------------------------------------------------------

def _turn_text(turn: dict[str, Any] | None) -> str:
    return str((turn or {}).get("response_text") or "").strip()


def _handback_of(turn: dict[str, Any] | None) -> dict[str, Any] | None:
    """The `transferToNga` payload on a DFCX turn, if any.

    Adapters normally pre-parse it into `handback`; fall back to scanning the
    raw response so a live adapter that only returns `raw` still works.
    """
    if not turn:
        return None
    found = turn.get("handback")
    if found:
        return found
    return extract_handback(turn.get("raw"))


def _transfer_of(turn: dict[str, Any] | None) -> dict[str, Any] | None:
    """The `transferToDialogflow` payload on an NGA turn, if any."""
    if not turn:
        return None
    found = turn.get("transfer")
    if found:
        return found
    return extract_transfer(turn.get("raw"))


def _reenter_dfcx(
    adapter: DfcxAdapter,
    target: str,
    parameters: dict[str, Any],
    utterance: str,
    session_id: str,
    language: str,
    printer: Printer,
) -> dict[str, Any]:
    """Re-enter DFCX after a handback, preferring an explicit intent trigger.

    After a handback the session is parked on `Trigger NGA`, a page with no
    intent routes. `continue_handoff` sends the utterance as *text*, which
    NO_MATCHes there and strands the call. A real client instead triggers the
    Default Welcome Intent with the NGA parameters on the same request, which
    runs `Trigger NGA` -> `Coming From NGA` -> `Post Steering Routing`.
    """
    reenter = getattr(adapter, "reenter", None)
    if callable(reenter):
        result = reenter(target, parameters, utterance, session_id, language)
        if isinstance(result, dict):
            return result
    printer(
        "[RETRANSFER WARNING] adapter has no reenter(); falling back to text input, "
        "which NO_MATCHes when the session is parked on 'Trigger NGA'"
    )
    return adapter.continue_handoff(target, parameters, utterance, session_id, language)


def run_autonomous_conversation(
    adapter: DfcxAdapter,
    llm: TestUser,
    *,
    target: str,
    session_id: str,
    first_turn: dict[str, Any],
    config: dict[str, Any],
    language: str,
    nga: NgaAdapter | None = None,
    nga_app_fallback: str | None = None,
    printer: Printer = print,
) -> ConversationResult:
    """Loop VA -> Ollama -> VA across every NGA/DFCX leg until a real stop.

    `first_turn` is the DFCX response to the steering utterance; the loop
    generates the customer's reply to it, sends that, and repeats. When DFCX
    hands the call back to NGA, the loop switches channel and keeps going;
    when NGA steers back into DFCX, it switches back.

    `nga` is required to continue past a handback. Without it the loop stops
    with `handback_unsupported` rather than silently passing.
    """
    goal = str(config.get("customer_goal", "")).strip()
    if not goal:
        raise ValueError("autonomous_flow.customer_goal is required when autonomous_flow.enabled is true")

    profile = str(config.get("customer_profile", "")).strip()
    max_turns = int(config.get("max_turns", 8))
    max_legs = int(config.get("max_legs", DEFAULT_MAX_LEGS))
    stop_phrases = [str(item).lower().strip() for item in config.get("stop_phrases", []) or []]
    stop_phrases = [phrase for phrase in stop_phrases if phrase]

    # Stop early when the VA repeats itself (fallback loop) instead of
    # spending the remaining turns and Ollama calls on a dead conversation.
    repeat_limit = int(config.get("max_repeated_va_turns", 10))

    # DFCX rotates its no-match prompts, so text-level repeat detection does
    # not catch a stranded session -- match_type does, on the first turn.
    no_match_limit = int(config.get("max_consecutive_no_match", 10))
    consecutive_no_match = 0

    convo = ConversationResult(final_dfcx=first_turn, final_turn=first_turn, stop_reason="max_turns")
    current = first_turn
    channel = "dfcx"
    leg = 1
    dfcx_target = target
    history: list[dict[str, str]] = []
    # Repeat detection is per-channel: NGA and DFCX may legitimately use the
    # same wording ("Comment puis-je vous aider?") without it being a loop.
    seen_va_text: dict[tuple[str, str], int] = {}

    printer(f"[AUTONOMOUS MODE] Enabled with maximum {max_turns} turns across up to {max_legs} leg(s)")
    turn_number = 0
    while turn_number < max_turns:
        va_text = _turn_text(current)

        # ---- Leg transitions come first -------------------------------
        # A handback often rides along with a farewell and an end flag; it
        # must beat both, otherwise we would end a call that is merely
        # changing hands.
        if channel == "dfcx":
            handback = _handback_of(current)
            if handback:
                if nga is None:
                    convo.stop_reason = "handback_unsupported"
                    printer("[AUTONOMOUS STOP] DFCX handed back but no NGA adapter was supplied")
                    break
                if leg >= max_legs:
                    convo.stop_reason = "max_legs"
                    printer(f"[AUTONOMOUS STOP] Hit the {max_legs}-leg ceiling (NGA<->DFCX ping-pong)")
                    break
                leg += 1
                raw_target = handback.get("transferToNga")
                unresolved = list(handback.get("unresolved") or []) or unresolved_references(
                    {"transferToNga": raw_target, "variables": handback.get("variables") or {}}
                )
                # An unresolved target is a *config* defect, not a protocol
                # error: the payload is well-formed, so the transfer fires and
                # the caller is dumped somewhere unintended. Record it, swap in
                # the configured app so the rest of the call stays assertable,
                # and let validation fail the case.
                resolved_target = None
                if is_unresolved(raw_target):
                    printer(
                        f"[HANDBACK DEFECT] transferToNga is the unresolved literal "
                        f"{raw_target!r} -- the NGA app id was never bound in this DFCX session"
                    )
                    if nga_app_fallback:
                        resolved_target = nga_app_fallback
                        handback = {**handback, "transferToNga": nga_app_fallback}
                        printer(f"[HANDBACK DEFECT] substituting configured app {nga_app_fallback}")
                    else:
                        convo.transitions.append(LegTransition(
                            kind="handback", leg=leg, after_turn=turn_number,
                            payload=handback, text=va_text, unresolved=unresolved,
                        ))
                        convo.legs = leg
                        convo.stop_reason = "handback_unresolved_target"
                        printer(
                            "[AUTONOMOUS STOP] Cannot resume: no resolved NGA app. "
                            "Set `nga_app_name` on the environment to keep driving the call."
                        )
                        break
                convo.transitions.append(LegTransition(
                    kind="handback", leg=leg, after_turn=turn_number,
                    payload=handback, text=va_text, unresolved=unresolved,
                    resolved_target=resolved_target,
                ))
                printer(
                    f"[HANDBACK -> NGA] leg {leg}: {handback.get('transferToNga')} "
                    f"(handoff_to={(handback.get('variables') or {}).get('handoff_to')!r})"
                )
                if va_text:
                    history.append({"speaker": "va", "text": va_text})
                current = nga.resume_from_handback(session_id, handback, language_code=language)
                channel = "nga"
                convo.final_turn = current
                convo.legs = leg
                continue

        if channel == "nga":
            transfer = _transfer_of(current)
            if transfer:
                if leg >= max_legs:
                    convo.stop_reason = "max_legs"
                    printer(f"[AUTONOMOUS STOP] Hit the {max_legs}-leg ceiling (NGA<->DFCX ping-pong)")
                    break
                leg += 1
                parameters = transfer.get("parameters") or {}
                unresolved = unresolved_references({
                    "transferToDialogflow": transfer.get("transferToDialogflow"),
                    "parameters": parameters,
                })
                if unresolved:
                    printer(f"[RETRANSFER DEFECT] unresolved reference(s): {', '.join(unresolved)}")
                convo.transitions.append(LegTransition(
                    kind="retransfer", leg=leg, after_turn=turn_number,
                    payload=transfer, text=va_text, unresolved=unresolved,
                ))
                # An unresolved environment would send the call to a literal
                # string; keep the one we already know works.
                if is_unresolved(transfer.get("transferToDialogflow")):
                    printer(f"[RETRANSFER DEFECT] keeping last known environment {dfcx_target}")
                else:
                    dfcx_target = transfer.get("transferToDialogflow") or dfcx_target
                printer(
                    f"[RETRANSFER -> DFCX] leg {leg}: route={parameters.get('route')!r} "
                    f"env={dfcx_target}"
                )
                if va_text:
                    history.append({"speaker": "va", "text": va_text})
                current = _reenter_dfcx(
                    adapter, dfcx_target, parameters,
                    parameters.get("utterance") or va_text,
                    session_id, language, printer,
                )
                channel = "dfcx"
                convo.final_dfcx = current
                convo.final_turn = current
                convo.legs = leg
                continue

        # ---- Ordinary turn on the current channel ---------------------
        turn_number += 1
        printer(f"\n[AUTONOMOUS TURN {turn_number}] ({channel}, leg {leg})\n[VA RESPONSE] {va_text}")

        if not va_text:
            convo.stop_reason = "no_va_text"
            printer(f"[AUTONOMOUS STOP] No {channel.upper()} response text")
            break

        matched_stop = next((phrase for phrase in stop_phrases if phrase in va_text.lower()), None)
        if matched_stop:
            convo.stop_reason = f"stop_phrase:{matched_stop}"
            printer(f"[AUTONOMOUS STOP] Detected: {matched_stop}")
            break

        # Check *before* spending an Ollama call: the VA may have ended the
        # call on the very turn that produced this text.
        if (current or {}).get("end_session"):
            convo.stop_reason = f"end_session:{(current or {}).get('end_reason') or 'unknown'}"
            printer(f"[AUTONOMOUS STOP] {channel.upper()} ended the session ({(current or {}).get('end_reason')})")
            break

        normalized_va = (channel, " ".join(va_text.lower().split()))
        seen_va_text[normalized_va] = seen_va_text.get(normalized_va, 0) + 1
        if repeat_limit and seen_va_text[normalized_va] > repeat_limit:
            convo.stop_reason = "va_repeated"
            printer(f"[AUTONOMOUS STOP] {channel.upper()} repeated the same turn {repeat_limit + 1}x")
            break

        # A run of NO_MATCHes means we are talking to a page with no intent
        # routes -- almost always a session stranded on `Trigger NGA`.
        if (current or {}).get("no_match"):
            consecutive_no_match += 1
            if no_match_limit and consecutive_no_match >= no_match_limit:
                convo.stop_reason = "no_match_loop"
                convo.stranded_page = (current or {}).get("current_page")
                printer(
                    f"[AUTONOMOUS STOP] {consecutive_no_match} consecutive NO_MATCH on page "
                    f"{convo.stranded_page!r} -- the session is stranded, not conversing"
                )
                break
        else:
            consecutive_no_match = 0

        raw_user_input = llm.generate_autonomous_response(
            va_text, goal, profile, language, history
        )

        # [EMPTY] explicitly requests a silent caller turn. A genuinely blank
        # model response remains an error, while the marker becomes a real
        # zero-character string before it is sent to the active VA channel.
        intentional_empty_turn = (raw_user_input or "").strip().upper() == "[EMPTY]"
        if intentional_empty_turn:
            user_input = ""
            printer("[OLLAMA USER RESPONSE] <EMPTY TURN>")
        else:
            user_input = (raw_user_input or "").strip()
            if not user_input:
                convo.stop_reason = "empty_llm_response"
                printer("[AUTONOMOUS STOP] Ollama returned empty text")
                break
            printer(f"[OLLAMA USER RESPONSE] {user_input}")
        convo.turns.append(ConversationTurn(
            turn=turn_number,
            va_response=va_text,
            user_response=user_input,
            matched_intent=(current or {}).get("matched_intent"),
            current_page=(current or {}).get("current_page"),
            channel=channel,
            leg=leg,
        ))
        # Neutral transcript: the LLM adapter owns the chat-role mapping
        # (VA -> "user", customer -> "assistant"), because the model IS the
        # customer. Storing roles here caused the model to parrot the VA.
        history.extend([
            {"speaker": "va", "text": va_text},
            {"speaker": "customer", "text": user_input},
        ])
        convo.transcript = list(history)

        if channel == "dfcx":
            current = adapter.send_text(dfcx_target, session_id, user_input, language)
            convo.final_dfcx = current
        else:
            if nga is None:
                raise RuntimeError("NGA adapter is unavailable for this channel")
            current = nga.send_text(session_id, user_input, language_code=language)
        convo.final_turn = current
        convo.final_channel = channel

        # A transition on the response is handled at the top of the next
        # iteration, so handback/retransfer logic lives in exactly one place.
        if _handback_of(current) if channel == "dfcx" else _transfer_of(current):
            continue

        if (current or {}).get("end_session"):
            convo.stop_reason = f"end_session:{(current or {}).get('end_reason') or 'unknown'}"
            printer(f"[AUTONOMOUS STOP] {channel.upper()} ended the session ({(current or {}).get('end_reason')})")
            break
    else:
        convo.stop_reason = "max_turns"

    convo.legs = leg
    convo.final_channel = channel
    if convo.stop_reason == "max_turns" and turn_number >= max_turns:
        printer(f"[AUTONOMOUS STOP] Reached the {max_turns}-turn ceiling")
    return convo


# --------------------------------------------------------------------------
# Case execution
# --------------------------------------------------------------------------

def build_llm(
    model: str | None = None,
    host: str | None = None,
    *,
    max_attempts: int = 3,
    seed: int | None = None,
) -> TestUser:
    """Construct the Ollama-backed simulated customer (imported lazily)."""
    from automation.adapters.llm_adapter import DEFAULT_HOST, DEFAULT_MODEL, DynamicTestUser

    return DynamicTestUser(
        model_name=model or DEFAULT_MODEL,
        ollama_host=host or DEFAULT_HOST,
        max_attempts=max_attempts,
        seed=seed,
    )


def _synthetic_call_id(session_id: str) -> str:
    """A stable, numeric, telephony-shaped call id for harness sessions.

    Real `call_id` is assigned by telephony (e.g. `14022574`); a SCRAPI-driven
    session has none. Derive one from the session id so it is deterministic
    per run and reproducible when debugging a report.
    """
    digits = "".join(character for character in session_id if character.isdigit())
    return (digits or str(abs(hash(session_id))))[:8].ljust(8, "0")


def offline_journey(case: dict[str, Any]) -> JourneyResult:
    """Synthesize the handoff payload an NGA run would have produced."""
    expected = case.get("expected_transfer")
    if not expected:
        raise ValueError(
            f"Case {case.get('name', '<unnamed>')!r} needs an 'expected_transfer' block "
            "to run offline"
        )
    missing = [key for key in ("environment", "route", "handoff_to") if not expected.get(key)]
    if missing:
        raise ValueError(
            f"Case {case.get('name', '<unnamed>')!r} expected_transfer is missing: {', '.join(missing)}"
        )
    transfer = TransferPayload(expected["environment"], {
        "route": expected["route"],
        "language": expected.get("language", DEFAULT_LANGUAGE),
        "handoff_to": expected["handoff_to"],
        "utterance": case.get("steering_utterance"),
    })
    return JourneyResult("offline-session", [{"label": "offline_fixture"}], transfer)


def run_case(
    case: dict[str, Any],
    env: dict[str, Any],
    live: bool = False,
    *,
    llm: TestUser | None = None,
    adapter: DfcxAdapter | None = None,
    nga: NgaAdapter | None = None,
    autonomous: bool | None = None,
    printer: Printer = print,
) -> dict[str, Any]:
    """Run one eval: NGA journey -> handoff -> Ollama-driven conversation.

    `llm`, `adapter` and `nga` are injectable so the whole flow is
    unit-testable offline. `autonomous=None` means "follow the case's
    autonomous_flow.enabled".

    The same NGA adapter instance that ran the journey is reused for any
    post-handback leg, so the call keeps its CXaS session.
    """
    if live:
        from automation.adapters.cxas_live import LiveCxasSessionAdapter
        from automation.adapters.google_dfcx import GoogleDfcxAdapter

        app_name = env["app_name"]
        location = app_name.split("/locations/", 1)[1].split("/", 1)[0]
        live_nga = LiveCxasSessionAdapter(app_name, project_id(app_name), location)
        journey = live_nga.run_journey(case)
        adapter = adapter or GoogleDfcxAdapter()
        nga = nga or live_nga
    else:
        from automation.adapters.fake_dfcx import FakeDfcxAdapter
        from automation.adapters.fake_nga import FakeNgaAdapter

        journey = offline_journey(case)
        adapter = adapter or FakeDfcxAdapter()
        nga = nga or FakeNgaAdapter(
            default_environment=(case.get("expected_transfer") or {}).get("environment")
        )

    if journey.transfer is None:
        raise ValueError(f"Journey {journey.session_id!r} did not produce a transfer payload")

    config = case.get("autonomous_flow") or {}
    want_autonomous = bool(config.get("enabled", False)) if autonomous is None else bool(autonomous)

    # Only touch Ollama once we know we actually need it.
    if want_autonomous and llm is None:
        llm = build_llm()

    parameters = dict(journey.transfer.parameters)
    utterance = parameters.get("utterance") or case["steering_utterance"]
    language = case_language(case, env)

    # Real NGA puts `nga_agent_id` and `call_id` on the handoff; a
    # harness-driven NGA session sends neither (verified by diffing a prod
    # handoff payload against a captured run -- they were the only two keys
    # missing). Without `nga_agent_id` the DFCX session param stays unbound and
    # `Trigger NGA` emits the literal `$session.params.nga_agent_id` instead of
    # a real app. Seed both so the session matches prod.
    nga_app = env.get("nga_app_name") or env.get("app_name")
    param_name = env.get("nga_agent_id_param", "nga_agent_id")
    if nga_app and param_name and not parameters.get(param_name):
        parameters[param_name] = nga_app
        printer(f"[SEED] {param_name}={nga_app}")

    call_id_param = env.get("call_id_param", "call_id")
    if call_id_param and not parameters.get(call_id_param):
        # Prod sends an int in the NGA payload but a *string* on the wire in
        # `queryParams.parameters`; match the wire form.
        call_id = str(env.get("call_id") or _synthetic_call_id(journey.session_id))
        parameters[call_id_param] = call_id
        printer(f"[SEED] {call_id_param}={call_id}")

    initial_dfcx = adapter.continue_handoff(
        journey.transfer.target, parameters, utterance, journey.session_id, language
    )

    convo = ConversationResult(
        final_dfcx=initial_dfcx, final_turn=initial_dfcx, stop_reason="disabled"
    )
    if want_autonomous:
        if llm is None:
            raise RuntimeError("Autonomous conversation requires an LLM")
        convo = run_autonomous_conversation(
            adapter, llm,
            target=journey.transfer.target,
            session_id=journey.session_id,
            first_turn=initial_dfcx,
            config=config,
            language=language,
            nga=nga,
            nga_app_fallback=nga_app,
            printer=printer,
        )

    errors = validate(case, journey.transfer, initial_dfcx)
    # `expected_conversation` only means something when Ollama actually drove
    # the call; with --no-autonomous we assert the handoff alone.
    if want_autonomous:
        errors += validate_conversation(case, convo)
    return {
        "name": case["name"],
        "passed": not errors,
        "errors": errors,
        "language": language,
        "session_id": journey.session_id,
        "journey": journey.trace,
        "transfer": {"target": journey.transfer.target, "parameters": journey.transfer.parameters},
        "initial_dfcx": initial_dfcx,
        "final_dfcx": convo.final_dfcx,
        "final_turn": convo.final_turn,
        "autonomous": {
            "enabled": want_autonomous,
            "turns": convo.turn_count,
            "stop_reason": convo.stop_reason,
            "legs": convo.legs,
            "final_channel": convo.final_channel,
            "stranded_page": convo.stranded_page,
            "handback": convo.handback,
            "handbacks": convo.handbacks,
            "retransfers": convo.retransfers,
            "dfcx_turns": len(convo.turns_on("dfcx")),
            "nga_turns": len(convo.turns_on("nga")),
        },
        "leg_transitions": [
            {
                "kind": item.kind,
                "leg": item.leg,
                "after_turn": item.after_turn,
                "target": item.target,
                "text": item.text,
                "payload": item.payload,
            }
            for item in convo.transitions
        ],
        "autonomous_trace": [
            {
                "turn": turn.turn,
                "channel": turn.channel,
                "leg": turn.leg,
                "va_response": turn.va_response,
                "user_response": turn.user_response,
                "matched_intent": turn.matched_intent,
                "current_page": turn.current_page,
            }
            for turn in convo.turns
        ],
    }


def write_report(results: list[dict[str, Any]], path: str) -> dict[str, Any]:
    """Write the redacted JSON report and return the summary block."""
    passed = sum(1 for result in results if result.get("passed"))
    payload = {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "autonomous_turns": sum(
                (result.get("autonomous") or {}).get("turns", 0) for result in results
            ),
        },
        "results": results,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(redact(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return payload["summary"]
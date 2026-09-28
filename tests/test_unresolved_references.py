"""Unresolved `$session.params.*` references in transfer payloads.

A DFCX custom payload is a template. When a referenced session parameter was
never bound, DFCX passes the *literal* through instead of erroring or emitting
null -- so the payload still looks well-formed, the client still transfers, and
the caller still lands nowhere. This is the "transfer works but the experience
is wrong" failure, and nothing else in a run catches it.
"""

from tests.stubs import (
    STAGING_ENV,
    RecordingAdapter,
    RecordingNgaAdapter,
    ScriptedTestUser,
    make_case,
    silent,
)

from automation.adapters.fake_dfcx import FakeDfcxAdapter
from automation.runner import run_autonomous_conversation, run_case, validate_conversation
from automation.utils import extract_handback, is_unresolved, unresolved_references

UNRESOLVED_HANDBACK = {
    "transferToNga": "$session.params.nga_agent_id",
    "variables": {
        "handoff_from": "nlu_steering",
        "handoff_to": "",
        "utterance": "Je veux savoir si il y a une panne.",
        "dfcx_session_id": "65fdf483c8274b2ab3b1cff53b1e53d2",
    },
    "ignoreSessionParameters": True,
}

RESOLVED_TARGET = "projects/gcp-it-prod-va-dlgflw/locations/us/apps/06199ff5-b708-4058-bff8-59a82ae070a1"


# ------------------------------------------------------------------ detection

def test_is_unresolved_spots_a_session_param_literal():
    assert is_unresolved("$session.params.nga_agent_id")
    assert is_unresolved("  $session.params.nga_agent_id  ")


def test_is_unresolved_ignores_real_values():
    assert not is_unresolved(RESOLVED_TARGET)
    assert not is_unresolved("")
    assert not is_unresolved(None)
    # A sentence that merely mentions the syntax is not a bare reference.
    assert not is_unresolved("set $session.params.x first")


def test_unresolved_references_reports_paths():
    found = unresolved_references(UNRESOLVED_HANDBACK)
    assert found == ["transferToNga=$session.params.nga_agent_id"]


def test_unresolved_references_finds_nested_variables():
    payload = {"transferToNga": RESOLVED_TARGET,
               "variables": {"handoff_to": "$session.params.handoff_to"}}
    assert unresolved_references(payload) == ["variables.handoff_to=$session.params.handoff_to"]


def test_empty_handoff_to_is_not_a_defect():
    # `handoff_to: ""` is the documented "back to steering" value, not a
    # dropped variable -- it must never be reported.
    assert unresolved_references(UNRESOLVED_HANDBACK["variables"]) == []


def test_extract_handback_carries_the_defect():
    handback = extract_handback({"payload": UNRESOLVED_HANDBACK})
    assert handback["unresolved"] == ["transferToNga=$session.params.nga_agent_id"]


def test_extract_handback_on_a_healthy_payload_has_no_defects():
    healthy = {**UNRESOLVED_HANDBACK, "transferToNga": RESOLVED_TARGET}
    assert extract_handback({"payload": healthy})["unresolved"] == []


# ----------------------------------------------------------------- loop behaviour

def _run(adapter, nga, llm=None, fallback=None):
    return run_autonomous_conversation(
        adapter, llm or ScriptedTestUser(),
        target="projects/p/locations/l/agents/a/environments/e",
        session_id="sess-1",
        first_turn={"response_text": "Je vous ramène.", "handback": UNRESOLVED_HANDBACK},
        config={"enabled": True, "max_turns": 5, "customer_goal": "Réparer internet.",
                "stop_phrases": []},
        language="fr-CA",
        nga=nga,
        nga_app_fallback=fallback,
        printer=silent,
    )


def test_unresolved_target_stops_the_call_when_no_fallback_is_configured():
    convo = _run(RecordingAdapter([]), RecordingNgaAdapter([]))
    assert convo.stop_reason == "handback_unresolved_target"
    assert convo.defects == [
        "handback leg 2: transferToNga=$session.params.nga_agent_id"
    ]


def test_unresolved_target_is_substituted_so_the_call_keeps_going():
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour.", "end_session": True, "end_reason": "end_interaction"},
    ])
    convo = _run(RecordingAdapter([]), nga, fallback=RESOLVED_TARGET)

    assert convo.stop_reason == "end_session:end_interaction"
    assert convo.legs == 2
    # Resumed against the real app, not the literal...
    assert nga.resumed[0]["handback"]["transferToNga"] == RESOLVED_TARGET
    assert convo.transitions[0].resolved_target == RESOLVED_TARGET
    # ...but the defect is still recorded so the case fails.
    assert convo.defects


def test_a_substituted_defect_still_fails_validation():
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour.", "end_session": True, "end_reason": "end_interaction"},
    ])
    convo = _run(RecordingAdapter([]), nga, fallback=RESOLVED_TARGET)
    errors = validate_conversation({"expected_conversation": {"min_turns": 0}}, convo)
    assert any("unresolved template reference" in error for error in errors)
    assert any("never bound in this session" in error for error in errors)


def test_defect_can_be_waived_explicitly():
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour.", "end_session": True, "end_reason": "end_interaction"},
    ])
    convo = _run(RecordingAdapter([]), nga, fallback=RESOLVED_TARGET)
    case = {"expected_conversation": {"min_turns": 0, "allow_unresolved_references": True}}
    assert validate_conversation(case, convo) == []


def test_a_healthy_handback_reports_no_defect():
    healthy = {**UNRESOLVED_HANDBACK, "transferToNga": RESOLVED_TARGET}
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour.", "end_session": True, "end_reason": "end_interaction"},
    ])
    convo = run_autonomous_conversation(
        RecordingAdapter([]), ScriptedTestUser(),
        target="env", session_id="s",
        first_turn={"response_text": "Je vous ramène.", "handback": healthy},
        config={"enabled": True, "max_turns": 5, "customer_goal": "x", "stop_phrases": []},
        language="fr-CA", nga=nga, printer=silent,
    )
    assert convo.defects == []
    assert validate_conversation({"expected_conversation": {"min_turns": 0}}, convo) == []


# ------------------------------------------------------------------- seeding

def test_fake_dfcx_passes_the_literal_through_when_the_param_is_unbound():
    # Faithful to DFCX: an unbound reference is neither an error nor null.
    adapter = FakeDfcxAdapter()
    adapter.continue_handoff("env", {"route": "bill_dispute_charge"}, "salut", "s1")
    turn = adapter.send_text("env", "s1", "oui")
    assert turn["handback"]["transferToNga"] == "$session.params.nga_agent_id"


def test_fake_dfcx_resolves_the_target_once_the_param_is_seeded():
    adapter = FakeDfcxAdapter()
    adapter.continue_handoff(
        "env", {"route": "bill_dispute_charge", "nga_agent_id": RESOLVED_TARGET}, "salut", "s1"
    )
    turn = adapter.send_text("env", "s1", "oui")
    assert turn["handback"]["transferToNga"] == RESOLVED_TARGET
    assert turn["handback"]["variables"]["handoff_from"] == "bill_dispute_charge"


def test_run_case_seeds_nga_agent_id_from_the_environment():
    adapter = RecordingAdapter([{"response_text": "Bonjour.", "end_session": True,
                                 "end_reason": "end_interaction"}])
    env = {**STAGING_ENV, "nga_app_name": RESOLVED_TARGET}
    run_case(make_case(), env, live=False, llm=ScriptedTestUser(), adapter=adapter,
             nga=RecordingNgaAdapter([]), printer=silent)
    # The inbound hop carries the app id, so `Trigger NGA` can resolve it.
    assert adapter.sent[0]["kind"] == "handoff"


def test_run_case_end_to_end_seeding_prevents_the_defect():
    env = {**STAGING_ENV, "nga_app_name": RESOLVED_TARGET}
    case = make_case(
        expected_transfer={**make_case()["expected_transfer"], "route": "bill_dispute_charge"},
        autonomous_flow={"enabled": True, "max_turns": 6, "customer_goal": "Contester.",
                         "stop_phrases": []},
    )
    result = run_case(case, env, live=False, llm=ScriptedTestUser(), printer=silent)
    handbacks = result["autonomous"]["handbacks"]
    assert handbacks, "the dispute route should hand back"
    assert handbacks[0]["transferToNga"] == RESOLVED_TARGET
    assert not any("unresolved" in error for error in result["errors"])

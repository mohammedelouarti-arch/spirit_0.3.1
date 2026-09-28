"""Case loading, assertions, and the offline end-to-end path."""

import json
from pathlib import Path

from tests.stubs import (
    STAGING_ENV,
    RecordingAdapter,
    RecordingNgaAdapter,
    ScriptedTestUser,
    make_case,
    raises,
    silent,
)

from automation.adapters.fake_dfcx import FakeDfcxAdapter
from automation.models import ConversationResult, ConversationTurn, LegTransition, TransferPayload
from automation.runner import (
    case_language,
    load_cases,
    offline_journey,
    run_case,
    validate,
    validate_conversation,
    write_report,
)

SUITE = "suites/e2e_handoff"


# --------------------------------------------------------------------- loading

def test_load_cases_returns_every_shipped_case():
    assert len(load_cases(SUITE)) >= 5


def test_tag_filter_requires_all_tags():
    billing = {case["name"] for case in load_cases(SUITE, ["billing"])}
    assert "bill_promo_expired_nga_to_dfcx" in billing
    assert "tech_internet_outage_nga_to_dfcx" not in billing

    both = load_cases(SUITE, ["billing", "handoff-only"])
    assert [case["name"] for case in both] == ["bill_promo_expired_handoff_only"]


def test_tag_filter_is_case_insensitive():
    assert load_cases(SUITE, ["BILLING"]) == load_cases(SUITE, ["billing"])


def test_unknown_tag_yields_nothing():
    assert load_cases(SUITE, ["does-not-exist"]) == []


def test_case_language_prefers_expected_then_env():
    assert case_language(make_case(), STAGING_ENV) == "fr-CA"
    no_expected = make_case(expected_transfer={})
    assert case_language(no_expected, {"language_code": "en-us"}) == "en-US"
    assert case_language({}, {}) == "fr-CA"


# ------------------------------------------------------------------ validation

def _transfer(**overrides):
    parameters = {
        "route": "bill_promo_expired",
        "language": "fr-CA",
        "handoff_to": "post_steering",
        "utterance": "J'ai une offre promotionnelle expirée.",
    }
    parameters.update(overrides)
    return TransferPayload(STAGING_ENV["flow_environment"], parameters)


def test_validate_passes_on_matching_payload():
    assert validate(make_case(), _transfer(), {"matched_intent": "bill_promo_expired"}) == []


def test_validate_accepts_language_case_mismatch():
    # YAML says "fr-ca", DFCX returns "fr-CA" - must not be an error.
    assert validate(make_case(), _transfer(language="fr-CA"), {}) == []


def test_validate_flags_route_mismatch():
    errors = validate(make_case(), _transfer(route="bill_explain_charges"), {})
    assert any("route" in error for error in errors)


def test_validate_flags_wrong_environment():
    transfer = _transfer()
    transfer.target = "projects/other/locations/us-central1/agents/x/environments/y"
    errors = validate(make_case(), transfer, {})
    assert any("target environment mismatch" in error for error in errors)


def test_validate_flags_intent_mismatch():
    errors = validate(make_case(), _transfer(), {"matched_intent": "smalltalk"})
    assert any("DFCX intent mismatch" in error for error in errors)


def test_validate_flags_utterance_mismatch():
    errors = validate(make_case(), _transfer(utterance="autre chose"), {})
    assert "transferred utterance mismatch" in errors


# ------------------------------------------------- conversation-level assertions

def _convo(texts, stop_reason="stop_phrase:bonne journée", end_reason="end_interaction",
           handback=None, retransfer=None, legs=None, final_channel="dfcx"):
    turns = [
        ConversationTurn(turn=index + 1, va_response=text, user_response="Oui.")
        for index, text in enumerate(texts)
    ]
    # `handback` / `retransfer` are now leg transitions rather than a terminal
    # field; `convo.handbacks` / `.retransfers` are derived from them.
    transitions = []
    if handback:
        transitions.append(LegTransition(kind="handback", leg=2, after_turn=len(turns),
                                         payload=handback))
    if retransfer:
        transitions.append(LegTransition(kind="retransfer", leg=len(transitions) + 2,
                                         after_turn=len(turns), payload=retransfer))
    return ConversationResult(
        turns=turns, stop_reason=stop_reason,
        final_dfcx={"end_reason": end_reason},
        final_turn={"end_reason": end_reason},
        final_channel=final_channel,
        transitions=transitions,
        legs=legs if legs is not None else 1 + len(transitions),
    )


def test_no_expected_conversation_block_means_no_errors():
    assert validate_conversation({}, _convo([])) == []


def test_must_mention_is_case_insensitive():
    case = {"expected_conversation": {"must_mention": ["Promotion"]}}
    assert validate_conversation(case, _convo(["Votre promotion est expirée."])) == []


def test_must_mention_failure_is_reported():
    case = {"expected_conversation": {"must_mention": ["promotion"]}}
    errors = validate_conversation(case, _convo(["Je ne peux pas vous aider."]))
    assert any("never mentioned" in error for error in errors)


def test_must_not_mention_failure_is_reported():
    case = {"expected_conversation": {"must_not_mention": ["erreur système"]}}
    errors = validate_conversation(case, _convo(["Une erreur système est survenue."]))
    assert any("forbidden phrase" in error for error in errors)


def test_turn_bounds_are_enforced():
    case = {"expected_conversation": {"min_turns": 2, "max_turns": 3}}
    assert any("too short" in error for error in validate_conversation(case, _convo(["a"])))
    assert any("too long" in error for error in validate_conversation(case, _convo(list("abcd"))))
    assert validate_conversation(case, _convo(list("ab"))) == []


def test_forbidden_stop_reason_is_reported():
    case = {"expected_conversation": {"forbid_stop_reasons": ["max_turns"]}}
    errors = validate_conversation(case, _convo(["a"], stop_reason="max_turns"))
    assert any("forbidden stop reason" in error for error in errors)


def test_expected_stop_reason_prefix_match():
    case = {"expected_conversation": {"expect_stop_reason": "stop_phrase"}}
    assert validate_conversation(case, _convo(["a"])) == []
    errors = validate_conversation(case, _convo(["a"], stop_reason="max_turns"))
    assert any("stop reason" in error for error in errors)


def test_expected_end_reason_mismatch():
    case = {"expected_conversation": {"expect_end_reason": "live_agent_handoff"}}
    errors = validate_conversation(case, _convo(["a"]))
    assert any("end reason" in error for error in errors)


def _handback(**variable_overrides):
    variables = {"handoff_from": "nlu_steering", "handoff_to": "billing_menu"}
    variables.update(variable_overrides)
    return {"transferToNga": "projects/x/locations/us/apps/nga-app", "variables": variables}


def test_expected_handback_passes_on_matching_payload():
    case = {"expected_conversation": {"expected_handback": {"handoff_to": "billing_menu"}}}
    errors = validate_conversation(case, _convo(["a"], handback=_handback()))
    assert errors == []


def test_expected_handback_flags_mismatch():
    case = {"expected_conversation": {"expected_handback": {"handoff_to": "billing_menu"}}}
    errors = validate_conversation(case, _convo(["a"], handback=_handback(handoff_to="")))
    assert any("handback handoff_to" in error for error in errors)


def test_expected_handback_checks_target_too():
    case = {"expected_conversation": {
        "expected_handback": {"transferToNga": "projects/other/locations/us/apps/other-app"}
    }}
    errors = validate_conversation(case, _convo(["a"], handback=_handback()))
    assert any("handback transferToNga" in error for error in errors)


def test_expected_handback_skipped_when_loop_never_handed_back():
    # No handback captured (convo.handback is None) and none expected -> quiet.
    case = {"expected_conversation": {"min_turns": 1}}
    assert validate_conversation(case, _convo(["a"])) == []


# ------------------------------------------------------------------- execution

def test_offline_journey_requires_expected_transfer():
    with raises(ValueError, match="expected_transfer"):
        offline_journey({"name": "broken"})


def test_offline_journey_reports_missing_keys():
    with raises(ValueError, match="handoff_to"):
        offline_journey({"name": "broken", "expected_transfer": {"environment": "e", "route": "r"}})


def test_run_case_offline_autonomous_passes():
    llm = ScriptedTestUser(["Oui, vérifiez.", "Oui, appliquez-la.", "Non merci."])
    result = run_case(make_case(), STAGING_ENV, live=False, llm=llm, printer=silent)

    assert result["passed"], result["errors"]
    assert result["language"] == "fr-CA"
    assert result["autonomous"]["enabled"] is True
    assert result["autonomous"]["turns"] >= 2
    # The scripted VA says goodbye *and* signals end_interaction on the same
    # turn; an explicit end signal takes precedence over phrase matching.
    assert result["autonomous"]["stop_reason"] == "end_session:end_interaction"
    assert result["autonomous_trace"][0]["user_response"] == "Oui, vérifiez."


def test_run_case_does_not_build_llm_when_autonomous_disabled():
    case = make_case(autonomous_flow={"enabled": False})
    # llm=None + autonomous disabled must never import/contact Ollama.
    result = run_case(case, STAGING_ENV, live=False, llm=None, printer=silent)
    assert result["passed"], result["errors"]
    assert result["autonomous"] == {
        "enabled": False, "turns": 0, "stop_reason": "disabled",
        "legs": 1, "final_channel": "dfcx", "stranded_page": None,
        "handback": None, "handbacks": [], "retransfers": [],
        "dfcx_turns": 0, "nga_turns": 0,
    }


def test_autonomous_override_forces_the_loop_on():
    case = make_case(autonomous_flow={
        "enabled": False, "max_turns": 2, "customer_goal": "Resolve the promo.",
        "stop_phrases": ["bonne journée"],
    })
    llm = ScriptedTestUser(["Oui."])
    result = run_case(case, STAGING_ENV, live=False, llm=llm, autonomous=True, printer=silent)
    assert result["autonomous"]["enabled"] is True
    assert result["autonomous"]["turns"] >= 1


def test_conversation_assertions_are_skipped_when_autonomous_is_off():
    case = make_case(expected_conversation={"min_turns": 3, "must_mention": ["jamais dit"]})
    result = run_case(case, STAGING_ENV, live=False, llm=None, autonomous=False, printer=silent)
    assert result["passed"], result["errors"]


def test_run_case_reports_conversation_failures():
    case = make_case(expected_conversation={"must_mention": ["remboursement intégral"]})
    result = run_case(case, STAGING_ENV, live=False, llm=ScriptedTestUser(), printer=silent)
    assert not result["passed"]
    assert any("never mentioned" in error for error in result["errors"])


def test_handback_on_first_turn_resumes_in_nga_instead_of_stopping():
    # A transferToNga payload alone carries no response_text. It must not be
    # read as "no_va_text", and -- since 0.4.x -- must not end the call either:
    # NGA picks the conversation up and keeps going.
    adapter = RecordingAdapter([{"response_text": "", "handback": _handback()}])
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour. Que puis-je faire pour vous?"},
        {"response_text": "Merci, bonne journée!", "end_session": True,
         "end_reason": "end_interaction"},
    ])
    llm = ScriptedTestUser(["Je veux parler de ma facture."])
    result = run_case(make_case(), STAGING_ENV, live=False, llm=llm, adapter=adapter,
                      nga=nga, printer=silent)

    assert result["autonomous"]["stop_reason"] == "end_session:end_interaction"
    assert result["autonomous"]["legs"] == 2
    assert result["autonomous"]["final_channel"] == "nga"
    assert result["autonomous"]["handbacks"][0]["variables"]["handoff_to"] == "billing_menu"
    # The customer answered NGA, not the payload-only DFCX turn.
    assert [turn["channel"] for turn in result["autonomous_trace"]] == ["nga"]
    assert nga.resumed[0]["handback"]["variables"]["handoff_to"] == "billing_menu"


def test_handback_then_retransfer_continues_in_dfcx():
    # The full ping-pong: DFCX -> NGA -> DFCX, all inside one call.
    adapter = RecordingAdapter([
        {"response_text": "Voulez-vous que je vérifie les promotions disponibles?"},
        {"response_text": "Je vous ramène au menu.", "handback": _handback()},
        {"response_text": "Ici le service technique, c'est réglé.",
         "end_session": True, "end_reason": "end_interaction"},
    ])
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour. Facture ou problème technique?"},
        {"response_text": "Je vous transfère au service technique.",
         "transfer": {"transferToDialogflow": "env-2",
                      "parameters": {"route": "tech_internet_outage"}}},
    ])
    llm = ScriptedTestUser(["Oui, vérifiez.", "Technique.", "D'accord."])
    result = run_case(make_case(), STAGING_ENV, live=False, llm=llm, adapter=adapter,
                      nga=nga, printer=silent)

    assert result["autonomous"]["stop_reason"] == "end_session:end_interaction"
    assert result["autonomous"]["legs"] == 3
    assert result["autonomous"]["handbacks"][0]["variables"]["handoff_to"] == "billing_menu"
    assert result["autonomous"]["retransfers"][0]["parameters"]["route"] == "tech_internet_outage"
    assert [turn["channel"] for turn in result["autonomous_trace"]] == ["dfcx", "nga"]
    assert result["autonomous"]["final_channel"] == "dfcx"
    # The retransfer re-entered DFCX on the environment NGA named.
    assert adapter.sent[-1] == {"kind": "handoff", "text": "Je vous transfère au service technique.",
                                "language": "fr-CA"}


def test_ping_pong_is_capped_by_max_legs():
    # NGA and DFCX bouncing forever must fail fast rather than burn turns.
    adapter = RecordingAdapter([
        {"response_text": "DFCX ici.", "handback": _handback()} for _ in range(10)
    ])
    nga = RecordingNgaAdapter([
        {"response_text": "NGA ici.",
         "transfer": {"transferToDialogflow": "env-2", "parameters": {"route": "r"}}}
        for _ in range(10)
    ])
    case = make_case(autonomous_flow={
        "enabled": True, "max_turns": 20, "max_legs": 4,
        "customer_goal": "Boucler.", "stop_phrases": [],
    })
    result = run_case(case, STAGING_ENV, live=False, llm=ScriptedTestUser(),
                      adapter=adapter, nga=nga, printer=silent)

    assert result["autonomous"]["stop_reason"] == "max_legs"
    assert result["autonomous"]["legs"] == 4


def test_fake_adapter_advances_script_per_session():
    adapter = FakeDfcxAdapter()
    first = adapter.continue_handoff("env", {"route": "bill_promo_expired"}, "salut", "s1")
    second = adapter.send_text("env", "s1", "oui")
    assert first["response_text"] != second["response_text"]
    assert first["matched_intent"] == "bill_promo_expired"


def test_fake_adapter_falls_back_for_unknown_route():
    adapter = FakeDfcxAdapter()
    turn = adapter.continue_handoff("env", {"route": "totally_unknown"}, "salut", "s2")
    assert turn["response_text"]


# ---------------------------------------------------------------------- report

def test_write_report_redacts_and_summarizes(tmp_path=None):
    target = Path(tmp_path or "reports") / "unit_report.json"
    results = [
        {"name": "a", "passed": True, "autonomous": {"turns": 3},
         "transfer": {"parameters": {"clid_tmp": "+15551234567", "route": "r"}}},
        {"name": "b", "passed": False, "errors": ["boom"], "autonomous": {"turns": 1}},
    ]
    summary = write_report(results, str(target))

    assert summary == {"total": 2, "passed": 1, "failed": 1, "autonomous_turns": 4}
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["results"][0]["transfer"]["parameters"]["clid_tmp"] == "[REDACTED]"
    assert payload["results"][0]["transfer"]["parameters"]["route"] == "r"
    target.unlink(missing_ok=True)

"""DFCX re-entry after a handback.

After a handback the session is parked on `Trigger NGA` -- a page with no
intent routes, only a Default-Welcome transition and a no-match handler.
Sending the utterance as *text* there NO_MATCHes forever (observed in prod at
confidence 0.3 / `sys.no-match-default`, four wasted turns). A real client
instead triggers the Default Welcome Intent with the NGA parameters on the
same request, which runs
`Trigger NGA` -> `Coming From NGA` -> `Post Steering Routing`.
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
from automation.adapters.google_dfcx import DEFAULT_WELCOME_INTENT_ID, agent_of
from automation.dfcx_parsing import parse_query_result
from automation.runner import run_case

ENV = "projects/p/locations/us-central1/agents/AGENT/environments/ENV"


# ------------------------------------------------------------------ plumbing

def test_agent_of_strips_the_environment():
    assert agent_of(ENV) == "projects/p/locations/us-central1/agents/AGENT"


def test_default_welcome_intent_id_is_the_builtin():
    assert DEFAULT_WELCOME_INTENT_ID == "00000000-0000-0000-0000-000000000000"


def test_parser_surfaces_no_match_snake_case():
    turn = parse_query_result({"query_result": {
        "match": {"match_type": "NO_MATCH", "confidence": 0.3},
        "current_page": {"display_name": "Trigger NGA"},
    }})
    assert turn["no_match"] is True
    assert turn["match_confidence"] == 0.3
    assert turn["current_page"] == "Trigger NGA"


def test_parser_surfaces_no_match_camel_case():
    # Console/REST captures use camelCase; the same parser must handle them.
    turn = parse_query_result({"query_result": {
        "match": {"matchType": "NO_MATCH"},
        "currentPage": {"displayName": "Trigger NGA"},
    }})
    assert turn["no_match"] is True
    assert turn["current_page"] == "Trigger NGA"


def test_parser_reports_a_real_match_as_not_no_match():
    turn = parse_query_result({"query_result": {
        "match": {"match_type": "INTENT", "confidence": 1.0,
                  "intent": {"display_name": "Default Welcome Intent"}},
    }})
    assert turn["no_match"] is False
    assert turn["matched_intent"] == "Default Welcome Intent"


# ------------------------------------------------- fake adapter page position

def test_fake_parks_the_session_after_a_handback():
    adapter = FakeDfcxAdapter()
    adapter.continue_handoff("env", {"route": "bill_dispute_charge"}, "salut", "s1")
    adapter.send_text("env", "s1", "oui")  # emits the handback
    stranded = adapter.send_text("env", "s1", "je veux parler de ma facture")

    assert stranded["no_match"] is True
    assert stranded["current_page"] == "Trigger NGA"
    assert stranded["match_confidence"] == 0.3


def test_fake_rotates_its_no_match_prompts():
    # Exactly why text-level repeat detection never caught this in prod.
    adapter = FakeDfcxAdapter()
    adapter.continue_handoff("env", {"route": "bill_dispute_charge"}, "salut", "s1")
    adapter.send_text("env", "s1", "oui")
    first = adapter.send_text("env", "s1", "a")["response_text"]
    second = adapter.send_text("env", "s1", "b")["response_text"]
    assert first != second


def test_reenter_unparks_and_runs_the_new_route():
    adapter = FakeDfcxAdapter()
    adapter.continue_handoff("env", {"route": "bill_dispute_charge"}, "salut", "s1")
    adapter.send_text("env", "s1", "oui")
    resumed = adapter.reenter(
        "env", {"route": "tech_connection_issue", "handoff_to": "post_steering"}, "panne", "s1"
    )
    assert resumed["no_match"] is False
    assert resumed["matched_intent"] == "tech_connection_issue"
    assert adapter.reentries[0]["handoff_to"] == "post_steering"


# ------------------------------------------------------- end-to-end behaviour

def _case():
    return make_case(
        expected_transfer={**make_case()["expected_transfer"], "route": "bill_dispute_charge"},
        autonomous_flow={"enabled": True, "max_turns": 10, "customer_goal": "Contester.",
                         "stop_phrases": []},
    )


class _TextOnlyAdapter(FakeDfcxAdapter):
    """The pre-0.6 behaviour: no `reenter`, so re-entry goes in as text."""

    reenter = None


def test_text_reentry_strands_the_call_and_is_reported():
    # Reproduces the prod failure: transfer fires, experience is broken.
    result = run_case(_case(), STAGING_ENV, live=False, llm=ScriptedTestUser(),
                      adapter=_TextOnlyAdapter(), printer=silent)

    assert result["autonomous"]["stop_reason"] == "no_match_loop"
    assert result["autonomous"]["stranded_page"] == "Trigger NGA"
    assert not result["passed"]


def test_intent_reentry_completes_the_call():
    result = run_case(_case(), STAGING_ENV, live=False, llm=ScriptedTestUser(),
                      printer=silent)

    assert result["autonomous"]["stop_reason"] != "no_match_loop"
    assert result["autonomous"]["legs"] >= 3
    assert result["autonomous"]["retransfers"], "should have re-entered DFCX"


def test_no_match_loop_stops_before_burning_every_turn():
    # Prod wasted 4 turns + 4 Ollama calls before `va_repeated` fired.
    llm = ScriptedTestUser()
    run_case(_case(), STAGING_ENV, live=False, llm=llm, adapter=_TextOnlyAdapter(),
             printer=silent)
    assert len(llm.calls) <= 4


# ------------------------------------------------------------------- seeding

def test_run_case_seeds_call_id_as_a_string():
    adapter = RecordingAdapter([{"response_text": "Bonjour.", "end_session": True,
                                 "end_reason": "end_interaction"}])
    captured: dict = {}

    class Capturing(RecordingAdapter):
        def continue_handoff(self, target, parameters, utterance, session_id,
                             language_code="fr-CA"):
            captured.update(parameters)
            return super().continue_handoff(target, parameters, utterance, session_id,
                                            language_code)

    run_case(make_case(), STAGING_ENV, live=False, llm=ScriptedTestUser(),
             adapter=Capturing(list(adapter.turns)), nga=RecordingNgaAdapter([]),
             printer=silent)

    assert isinstance(captured["call_id"], str), "prod sends call_id as a string on the wire"
    assert captured["call_id"].isdigit()
    assert captured["nga_agent_id"] == STAGING_ENV["app_name"]


def test_explicit_call_id_from_the_environment_wins():
    captured: dict = {}

    class Capturing(RecordingAdapter):
        def continue_handoff(self, target, parameters, utterance, session_id,
                             language_code="fr-CA"):
            captured.update(parameters)
            return super().continue_handoff(target, parameters, utterance, session_id,
                                            language_code)

    env = {**STAGING_ENV, "call_id": 14022574}
    run_case(make_case(), STAGING_ENV | env, live=False, llm=ScriptedTestUser(),
             adapter=Capturing([{"response_text": "Bonjour.", "end_session": True,
                                 "end_reason": "end_interaction"}]),
             nga=RecordingNgaAdapter([]), printer=silent)
    assert captured["call_id"] == "14022574"

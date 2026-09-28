"""The Ollama-driven conversation that follows the steering utterance."""

from tests.stubs import (
    RecordingAdapter,
    RecordingNgaAdapter,
    ScriptedTestUser,
    raises,
    silent,
)

from automation.runner import run_autonomous_conversation

CONFIG = {
    "enabled": True,
    "max_turns": 5,
    "customer_goal": "Understand why the promotion expired.",
    "customer_profile": "Brief and polite.",
    "stop_phrases": ["bonne journée", "je vais vous transférer"],
}


def _run(adapter, llm, first_turn, config=None, language="fr-CA"):
    return run_autonomous_conversation(
        adapter, llm,
        target="projects/p/locations/l/agents/a/environments/e",
        session_id="sess-1",
        first_turn=first_turn,
        config=config or CONFIG,
        language=language,
        printer=silent,
    )


def test_stops_on_stop_phrase():
    adapter = RecordingAdapter([
        {"response_text": "Très bien. Bonne journée!", "end_session": False},
    ])
    llm = ScriptedTestUser(["Oui s'il vous plaît."])
    convo = _run(adapter, llm, {"response_text": "Voulez-vous que je vérifie?"})

    assert convo.stop_reason == "stop_phrase:bonne journée"
    assert convo.turn_count == 1
    assert convo.turns[0].user_response == "Oui s'il vous plaît."


def test_stops_on_end_interaction():
    adapter = RecordingAdapter([
        {"response_text": "C'est réglé.", "end_session": True, "end_reason": "end_interaction"},
    ])
    convo = _run(adapter, ScriptedTestUser(["Oui."]), {"response_text": "Souhaitez-vous l'appliquer?"})

    assert convo.stop_reason == "end_session:end_interaction"
    assert convo.turn_count == 1


def test_stops_when_first_turn_already_ended_without_calling_llm():
    llm = ScriptedTestUser(["ne devrait pas être appelé"])
    convo = _run(
        RecordingAdapter([]), llm,
        {"response_text": "Je vous transfère.", "end_session": True, "end_reason": "live_agent_handoff"},
    )
    assert convo.stop_reason == "end_session:live_agent_handoff"
    assert convo.turn_count == 0
    assert llm.calls == []  # no wasted Ollama round-trip


def test_stops_on_empty_llm_response():
    convo = _run(RecordingAdapter([]), ScriptedTestUser([""]), {"response_text": "Bonjour?"})
    assert convo.stop_reason == "empty_llm_response"
    assert convo.turn_count == 0


def test_stops_when_va_returns_no_text():
    convo = _run(RecordingAdapter([]), ScriptedTestUser(), {"response_text": "   "})
    assert convo.stop_reason == "no_va_text"


def test_exhausts_max_turns():
    adapter = RecordingAdapter([{"response_text": f"Question {i}?"} for i in range(1, 10)])
    convo = _run(adapter, ScriptedTestUser(), {"response_text": "Question 0?"},
                 config={**CONFIG, "max_turns": 3})

    assert convo.stop_reason == "max_turns"
    assert convo.turn_count == 3


def test_history_grows_and_is_passed_to_llm():
    adapter = RecordingAdapter([{"response_text": "Et ensuite?"}, {"response_text": "Bonne journée!"}])
    llm = ScriptedTestUser(["Première", "Deuxième", "Troisième"])
    convo = _run(adapter, llm, {"response_text": "Bonjour?"})

    assert [call["history_len"] for call in llm.calls] == [0, 2]
    # Transcript stays speaker-neutral; the LLM adapter owns chat-role mapping.
    assert convo.transcript[0] == {"speaker": "va", "text": "Bonjour?"}
    assert convo.transcript[1] == {"speaker": "customer", "text": "Première"}


def test_language_is_forwarded_to_llm_and_adapter():
    adapter = RecordingAdapter([{"response_text": "Bonne journée!"}])
    llm = ScriptedTestUser(["Oui."])
    _run(adapter, llm, {"response_text": "Bonjour?"}, language="fr-CA")

    assert llm.calls[0]["language"] == "fr-CA"
    assert adapter.sent[0] == {"kind": "text", "text": "Oui.", "language": "fr-CA"}


def test_missing_customer_goal_is_rejected():
    with raises(ValueError, match="customer_goal"):
        _run(RecordingAdapter([]), ScriptedTestUser(), {"response_text": "Bonjour?"},
             config={"enabled": True, "customer_goal": "   "})


def test_stop_phrase_matching_is_case_insensitive():
    adapter = RecordingAdapter([])
    convo = _run(adapter, ScriptedTestUser(), {"response_text": "BONNE JOURNÉE!"})
    assert convo.stop_reason == "stop_phrase:bonne journée"


def test_stops_when_va_repeats_itself():
    # A DFCX fallback loop must fail fast, not burn every remaining turn.
    adapter = RecordingAdapter([{"response_text": "Je ne comprends pas."} for _ in range(8)])
    llm = ScriptedTestUser()
    convo = _run(adapter, llm, {"response_text": "Je ne comprends pas."},
                 config={**CONFIG, "max_turns": 8})

    assert convo.stop_reason == "va_repeated"
    # Default limit 2: the VA may say it twice; the 3rd sighting stops the loop
    # before another Ollama call, so 2 turns were answered.
    assert convo.turn_count == 2


def test_va_repeat_limit_is_configurable():
    adapter = RecordingAdapter([{"response_text": "Boucle."} for _ in range(8)])
    convo = _run(adapter, ScriptedTestUser(), {"response_text": "Boucle."},
                 config={**CONFIG, "max_turns": 8, "max_repeated_va_turns": 1})
    assert convo.stop_reason == "va_repeated"
    assert convo.turn_count == 1


def test_varied_va_turns_do_not_trigger_repeat_detection():
    adapter = RecordingAdapter([{"response_text": f"Question {i}?"} for i in range(1, 5)])
    convo = _run(adapter, ScriptedTestUser(), {"response_text": "Question 0?"},
                 config={**CONFIG, "max_turns": 3})
    assert convo.stop_reason == "max_turns"


def test_final_dfcx_tracks_last_turn():
    adapter = RecordingAdapter([
        {"response_text": "Deuxième", "current_page": "p2"},
        {"response_text": "Bonne journée!", "current_page": "p3"},
    ])
    convo = _run(adapter, ScriptedTestUser(), {"response_text": "Premier", "current_page": "p1"})
    assert convo.final_dfcx["current_page"] == "p3"
    assert convo.turns[0].current_page == "p1"


# ---------------------------------------------------------- multi-leg (NGA <-> DFCX)

def _run_multileg(adapter, llm, first_turn, nga, config=None, language="fr-CA"):
    return run_autonomous_conversation(
        adapter, llm,
        target="projects/p/locations/l/agents/a/environments/e",
        session_id="sess-1",
        first_turn=first_turn,
        config=config or CONFIG,
        language=language,
        nga=nga,
        printer=silent,
    )


HANDBACK = {
    "transferToNga": "projects/x/locations/us/apps/nga-app",
    "variables": {"handoff_from": "bill_dispute", "handoff_to": "billing_menu"},
}


def test_handback_does_not_stop_the_conversation():
    # The 0.3.x behaviour was stop_reason="handback_to_nga" and turns == 0.
    adapter = RecordingAdapter([])
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour, que puis-je faire?"},
        {"response_text": "C'est noté.", "end_session": True, "end_reason": "end_interaction"},
    ])
    llm = ScriptedTestUser(["Ma facture."])
    convo = _run_multileg(adapter, llm, {"response_text": "", "handback": HANDBACK}, nga)

    assert convo.stop_reason == "end_session:end_interaction"
    assert convo.turn_count == 1
    assert convo.legs == 2
    assert convo.turns[0].channel == "nga"
    assert convo.handbacks == [HANDBACK]


def test_handback_survives_an_end_flag_on_the_same_turn():
    # DFCX usually ends *its own* interaction while handing back. That must be
    # read as a leg change, not as the end of the call.
    nga = RecordingNgaAdapter([
        {"response_text": "Je reprends la conversation.", "end_session": True,
         "end_reason": "end_interaction"},
    ])
    convo = _run_multileg(
        RecordingAdapter([]), ScriptedTestUser(),
        {"response_text": "Je vous ramène au menu.", "handback": HANDBACK,
         "end_session": True, "end_reason": "end_interaction"},
        nga,
    )
    assert convo.legs == 2
    assert convo.handbacks == [HANDBACK]
    assert convo.final_channel == "nga"


def test_handback_beats_a_stop_phrase_on_the_same_turn():
    # "Bonne journée" said on the way out of DFCX must not kill the call.
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour."},
        {"response_text": "Fini.", "end_session": True, "end_reason": "end_interaction"},
    ])
    convo = _run_multileg(
        RecordingAdapter([]), ScriptedTestUser(["Oui."]),
        {"response_text": "Bonne journée de la part de DFCX!", "handback": HANDBACK},
        nga,
    )
    assert convo.stop_reason == "end_session:end_interaction"
    assert convo.legs == 2


def test_retransfer_sends_the_call_back_into_dfcx():
    adapter = RecordingAdapter([
        {"response_text": "Rebonjour depuis DFCX."},
        {"response_text": "Terminé.", "end_session": True, "end_reason": "end_interaction"},
    ])
    nga = RecordingNgaAdapter([
        {"response_text": "Facture ou technique?"},
        {"response_text": "Je vous transfère.",
         "transfer": {"transferToDialogflow": "env-2",
                      "parameters": {"route": "tech_internet_outage", "utterance": "panne"}}},
    ])
    llm = ScriptedTestUser(["Technique.", "Oui.", "Merci."])
    convo = _run_multileg(adapter, llm, {"response_text": "", "handback": HANDBACK}, nga)

    assert convo.legs == 3
    assert [turn.channel for turn in convo.turns] == ["nga", "dfcx"]
    assert convo.retransfers[0]["parameters"]["route"] == "tech_internet_outage"
    # Re-entered DFCX through continue_handoff with the utterance NGA supplied.
    assert adapter.sent[0] == {"kind": "handoff", "text": "panne", "language": "fr-CA"}


def test_repeated_bouncing_is_capped_by_max_legs():
    adapter = RecordingAdapter([
        {"response_text": "DFCX.", "handback": HANDBACK} for _ in range(10)
    ])
    nga = RecordingNgaAdapter([
        {"response_text": "NGA.",
         "transfer": {"transferToDialogflow": "env-2", "parameters": {"route": "r"}}}
        for _ in range(10)
    ])
    convo = _run_multileg(adapter, ScriptedTestUser(), {"response_text": "", "handback": HANDBACK},
                          nga, config={**CONFIG, "max_turns": 20, "max_legs": 3})
    assert convo.stop_reason == "max_legs"
    assert convo.legs == 3


def test_handback_without_an_nga_adapter_is_flagged():
    convo = _run(RecordingAdapter([]), ScriptedTestUser(),
                 {"response_text": "Je vous ramène.", "handback": HANDBACK})
    assert convo.stop_reason == "handback_unsupported"


def test_handback_is_detected_in_a_raw_payload():
    # A live adapter that only surfaces `raw` must still be understood.
    nga = RecordingNgaAdapter([
        {"response_text": "Bon retour.", "end_session": True, "end_reason": "end_interaction"},
    ])
    raw = {"queryResult": {"responseMessages": [{"payload": {
        "transferToNga": "projects/x/locations/us/apps/nga-app",
        "variables": {"handoff_to": "main_menu"},
    }}]}}
    convo = _run_multileg(RecordingAdapter([]), ScriptedTestUser(),
                          {"response_text": "Un instant.", "raw": raw}, nga)
    assert convo.legs == 2
    assert convo.handbacks[0]["variables"]["handoff_to"] == "main_menu"


def test_repeat_detection_is_per_channel():
    # NGA and DFCX may share wording without that being a fallback loop.
    adapter = RecordingAdapter([
        {"response_text": "Comment puis-je vous aider?"},
        {"response_text": "Fini.", "end_session": True, "end_reason": "end_interaction"},
    ])
    nga = RecordingNgaAdapter([
        {"response_text": "Comment puis-je vous aider?"},
        {"response_text": "Je vous transfère.",
         "transfer": {"transferToDialogflow": "env-2", "parameters": {"route": "r"}}},
    ])
    convo = _run_multileg(adapter, ScriptedTestUser(["Oui.", "Oui.", "Oui."]),
                          {"response_text": "", "handback": HANDBACK}, nga,
                          config={**CONFIG, "max_turns": 8, "max_repeated_va_turns": 1})
    assert convo.stop_reason == "end_session:end_interaction"

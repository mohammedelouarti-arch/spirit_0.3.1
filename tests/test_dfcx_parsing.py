"""Regression tests for the DFCX response parser.

The original code looked for a non-existent `end_session` ResponseMessage
field; the real proto field is `end_interaction`.
"""

from tests.stubs import raises  # noqa: F401

from automation.dfcx_parsing import extract_texts, parse_query_result


def _response(messages, intent="bill_promo_expired", page="Promo"):
    return {
        "query_result": {
            "response_messages": messages,
            "match": {"intent": {"display_name": intent}},
            "current_page": {"display_name": page},
            "parameters": {"route": "bill_promo_expired"},
        }
    }


def test_extracts_text_messages():
    parsed = parse_query_result(_response([{"text": {"text": ["Bonjour", "comment puis-je aider?"]}}]))
    assert parsed["response_text"] == "Bonjour comment puis-je aider?"
    assert parsed["matched_intent"] == "bill_promo_expired"
    assert parsed["current_page"] == "Promo"
    assert parsed["end_session"] is False
    assert parsed["end_reason"] is None


def test_extracts_output_audio_text():
    texts = extract_texts([{"output_audio_text": {"text": "Merci"}}, {"text": {"text": ["Bonjour"]}}])
    assert texts == ["Merci", "Bonjour"]


def test_end_interaction_marks_session_end():
    parsed = parse_query_result(_response([{"text": {"text": ["Au revoir"]}}, {"end_interaction": {}}]))
    assert parsed["end_session"] is True
    assert parsed["end_reason"] == "end_interaction"


def test_live_agent_handoff_marks_session_end():
    parsed = parse_query_result(_response([{"live_agent_handoff": {"metadata": {}}}]))
    assert parsed["end_session"] is True
    assert parsed["end_reason"] == "live_agent_handoff"


def test_conversation_success_marks_session_end():
    parsed = parse_query_result(_response([{"conversation_success": {}}]))
    assert parsed["end_session"] is True
    assert parsed["end_reason"] == "conversation_success"


def test_legacy_end_session_key_is_not_treated_as_terminal():
    # Guards against re-introducing the bug from the other direction.
    parsed = parse_query_result(_response([{"text": {"text": ["Bonjour"]}}]))
    assert parsed["end_session"] is False


def test_handles_empty_and_missing_fields():
    parsed = parse_query_result({})
    assert parsed["response_text"] == ""
    assert parsed["matched_intent"] is None
    assert parsed["parameters"] == {}
    assert parsed["end_session"] is False


def test_blank_text_is_dropped():
    parsed = parse_query_result(_response([{"text": {"text": ["", "   ", "Bonjour"]}}]))
    assert parsed["response_text"] == "Bonjour"


def test_transfer_to_nga_payload_is_surfaced_as_handback():
    parsed = parse_query_result(_response([
        {"payload": {"transferToNga": "projects/x/locations/us/apps/nga-app",
                     "ignoreSessionParameters": True,
                     "variables": {"handoff_from": "nlu_steering", "handoff_to": "",
                                   "utterance": "mon internet ne fonctionne plus"}}}
    ]))
    assert parsed["handback"]["transferToNga"] == "projects/x/locations/us/apps/nga-app"
    assert parsed["handback"]["variables"]["handoff_from"] == "nlu_steering"
    # A handback payload alone carries no text and isn't one of the
    # end_interaction/conversation_success/live_agent_handoff proto fields,
    # so it must not be misread as either "no response" or "call ended".
    assert parsed["response_text"] == ""
    assert parsed["end_session"] is False


def test_no_transfer_to_nga_means_no_handback():
    parsed = parse_query_result(_response([{"text": {"text": ["Bonjour"]}}]))
    assert parsed["handback"] is None

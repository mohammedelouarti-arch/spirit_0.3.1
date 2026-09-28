from dataclasses import dataclass

from tests.stubs import raises  # noqa: F401

from automation.utils import extract_handback, extract_transfer, redact, to_plain


@dataclass
class Sample:
    name: str
    values: list[int]


def test_to_plain_handles_dataclasses_and_nesting():
    assert to_plain({"a": Sample("x", [1, 2])}) == {"a": {"name": "x", "values": [1, 2]}}


def test_to_plain_converts_sets_and_tuples_to_lists():
    assert to_plain((1, 2)) == [1, 2]
    assert sorted(to_plain({1, 2})) == [1, 2]


def test_extract_transfer_finds_nested_camel_case():
    payload = {"steps": [{"result": {"transferToDialogflow": "env-1",
                                     "parameters": {"route": "bill_promo_expired"}}}]}
    transfer = extract_transfer(payload)
    assert transfer["transferToDialogflow"] == "env-1"
    assert transfer["parameters"]["route"] == "bill_promo_expired"


def test_extract_transfer_finds_snake_case():
    assert extract_transfer({"transfer_to_dialogflow": "env-2"})["transferToDialogflow"] == "env-2"


def test_extract_transfer_returns_none_when_absent():
    assert extract_transfer({"steps": [{"result": {"text": "bonjour"}}]}) is None


def test_extract_handback_finds_nested_payload():
    payload = {"query_result": {"response_messages": [
        {"payload": {"transferToNga": "projects/x/locations/us/apps/nga-app",
                     "ignoreSessionParameters": True,
                     "variables": {"handoff_from": "nlu_steering", "handoff_to": "",
                                   "utterance": "mon internet ne fonctionne plus",
                                   "dfcx_session_id": "081pGxO8TMTR3CE28Z2DkaIRQ"}}}
    ]}}
    handback = extract_handback(payload)
    assert handback["transferToNga"] == "projects/x/locations/us/apps/nga-app"
    assert handback["variables"]["handoff_from"] == "nlu_steering"
    assert handback["ignoreSessionParameters"] is True


def test_extract_handback_finds_snake_case():
    assert extract_handback({"transfer_to_nga": "app-1"})["transferToNga"] == "app-1"


def test_extract_handback_returns_none_when_absent():
    assert extract_handback({"payload": {"text": "bonjour"}}) is None


def test_extract_handback_does_not_match_forward_transfer():
    # transferToDialogflow (NGA -> DFCX) must never be read as a handback.
    assert extract_handback({"transferToDialogflow": "env-1"}) is None


def test_redact_masks_sensitive_keys_only():
    data = {"clid_tmp": "+15551234567", "route": "bill_promo_expired",
            "nested": {"billing_account": "123", "language": "fr-CA"}}
    masked = redact(data)
    assert masked["clid_tmp"] == "[REDACTED]"
    assert masked["route"] == "bill_promo_expired"
    assert masked["nested"]["billing_account"] == "[REDACTED]"
    assert masked["nested"]["language"] == "fr-CA"


def test_redact_walks_lists():
    masked = redact({"calls": [{"phone": "+1555"}, {"phone": "+1666"}]})
    assert [entry["phone"] for entry in masked["calls"]] == ["[REDACTED]", "[REDACTED]"]


def test_redact_keeps_empty_values():
    assert redact({"phone": ""})["phone"] == ""
    assert redact({"phone": None})["phone"] is None

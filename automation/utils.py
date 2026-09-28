from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return to_plain(asdict(value))

    original = getattr(value, "original_response", None)
    if original is not None:
        result = to_plain(original)
        audio = getattr(value, "agent_audio_paths", None)
        return {"original_response": result, "agent_audio_paths": to_plain(audio)} if audio else result

    protobuf = getattr(value, "_pb", None)
    if protobuf is not None or hasattr(value, "DESCRIPTOR"):
        try:
            from google.protobuf.json_format import MessageToDict
            return MessageToDict(protobuf or value, preserving_proto_field_name=True)
        except Exception:
            pass
    if hasattr(value, "items"):
        try:
            return {str(key): to_plain(item) for key, item in value.items()}
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {key: to_plain(item) for key, item in vars(value).items() if not key.startswith("_")}
        except Exception:
            pass
    return str(value)


def extract_transfer(value: Any) -> dict[str, Any] | None:
    return _search_transfer(to_plain(value))


def _search_transfer(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        target = value.get("transferToDialogflow") or value.get("transfer_to_dialogflow")
        if target:
            parameters = value.get("parameters") or {}
            return {"transferToDialogflow": target, "parameters": to_plain(parameters)}
        for child in value.values():
            result = _search_transfer(child)
            if result:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _search_transfer(child)
            if result:
                return result
    return None


# A DFCX custom payload is a template: `$session.params.foo` is substituted
# with the session parameter at emit time. When the parameter was never bound,
# DFCX does NOT error and does NOT emit null -- it passes the *literal string*
# through. Downstream that looks like a perfectly well-formed payload, which is
# why an unresolved target still "transfers" but lands the caller nowhere.
UNRESOLVED_REFERENCE = re.compile(r"\$(?:session\.params|sys\.func|intent\.params|user)\.[\w.-]+")


def is_unresolved(value: Any) -> bool:
    """True when `value` is an un-substituted DFCX template reference."""
    return isinstance(value, str) and bool(UNRESOLVED_REFERENCE.fullmatch(value.strip()))


def unresolved_references(value: Any, _path: str = "") -> list[str]:
    """Every `path=$session.params.x` left unsubstituted anywhere in `value`.

    Reported as a config defect rather than raised: the call still has to be
    driven to its end so the rest of the assertions mean something.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += unresolved_references(item, f"{_path}.{key}" if _path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += unresolved_references(item, f"{_path}[{index}]")
    elif is_unresolved(value):
        found.append(f"{_path or '<root>'}={value.strip()}")
    return found


def extract_handback(value: Any) -> dict[str, Any] | None:
    """Find a `transferToNga` payload anywhere in a DFCX response.

    Mirror image of `extract_transfer`: NGA hands off to DFCX with a
    `transferToDialogflow` custom payload; DFCX can hand the call back to
    NGA mid-conversation with a `transferToNga` one, carrying the target
    NGA app plus `variables` (`handoff_from`, `handoff_to`, `utterance`,
    `dfcx_session_id`). Distinct from a `live_agent_handoff`
    ResponseMessage, which ends the DFCX interaction entirely via a SIP
    deflection rather than returning control to NGA.
    """
    return _search_handback(to_plain(value))


def _search_handback(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        target = value.get("transferToNga") or value.get("transfer_to_nga")
        if target:
            payload = {
                "transferToNga": target,
                "variables": to_plain(value.get("variables") or {}),
                "ignoreSessionParameters": value.get("ignoreSessionParameters"),
            }
            # Carried on the payload so the runner can fail the case loudly
            # instead of reporting a healthy-looking handback.
            payload["unresolved"] = unresolved_references({
                "transferToNga": payload["transferToNga"],
                "variables": payload["variables"],
            })
            return payload
        for child in value.values():
            result = _search_handback(child)
            if result:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _search_handback(child)
            if result:
                return result
    return None


SENSITIVE_KEYS = {
    "billing_account", "billing_account_number", "service_account_number",
    "subscriber_number", "phone", "clid", "clid_tmp", "cirn", "CIRN",
    "tfn", "tfn_tmp", "contact_number", "contact_email", "first_name",
    "last_name", "user_name", "profile_id", "tester_id",
}


def redact(value: Any, parent_key: str | None = None) -> Any:
    if parent_key in SENSITIVE_KEYS and value not in (None, ""):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {key: redact(item, key) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item, parent_key) for item in value)
    return value

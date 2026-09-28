"""Pure parsing helpers for Dialogflow CX responses.

Deliberately dependency-free (no google-cloud imports) so the response
contract can be unit-tested without credentials or the SDK installed.
"""

from __future__ import annotations

from typing import Any

from automation.utils import extract_handback

# ResponseMessage variants that mean "the VA is done with this conversation".
# NOTE: the proto field is `end_interaction` -- there is no `end_session`.
TERMINAL_MESSAGE_FIELDS = ("end_interaction", "conversation_success", "live_agent_handoff")

# The caller said something DFCX could not route. In prod this is rare; in the
# harness it is the signature of talking to a page that has no intent routes --
# e.g. being parked on `Trigger NGA` after a handback.
NO_MATCH_TYPES = ("NO_MATCH", "NO_INPUT")


def _is_terminal(message: dict[str, Any]) -> bool:
    return any(field in message for field in TERMINAL_MESSAGE_FIELDS)


def extract_texts(messages: list[Any]) -> list[str]:
    """Collect spoken text from `text` and `output_audio_text` messages."""
    texts: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        texts.extend(message.get("text", {}).get("text", []) or [])
        audio_text = message.get("output_audio_text") or {}
        if isinstance(audio_text, dict) and audio_text.get("text"):
            texts.append(audio_text["text"])
    return [text for text in texts if str(text).strip()]


def parse_query_result(plain: dict[str, Any]) -> dict[str, Any]:
    """Normalize a DetectIntentResponse dict into the runner's turn shape."""
    qr = plain.get("query_result", {}) or {}
    messages = qr.get("response_messages", []) or []
    texts = extract_texts(messages)
    terminal = [message for message in messages if isinstance(message, dict) and _is_terminal(message)]
    end_reason = None
    for message in terminal:
        end_reason = next((field for field in TERMINAL_MESSAGE_FIELDS if field in message), None)
        if end_reason:
            break
    match = qr.get("match", {}) or {}
    # Proto is snake_case; the REST/console shape is camelCase. Accept both so
    # the same parser works on SDK output and on captured console traces.
    match_type = match.get("match_type") or match.get("matchType")
    return {
        "response_text": " ".join(texts),
        "matched_intent": (match.get("intent", {}) or {}).get("display_name")
        or (match.get("intent", {}) or {}).get("displayName"),
        "match_type": match_type,
        "match_confidence": match.get("confidence"),
        "no_match": str(match_type or "").upper() in NO_MATCH_TYPES,
        "current_page": (qr.get("current_page", {}) or {}).get("display_name")
        or (qr.get("currentPage", {}) or {}).get("displayName"),
        "parameters": qr.get("parameters", {}) or {},
        "end_session": bool(terminal),
        "end_reason": end_reason,
        "handback": extract_handback(plain),
        "raw": plain,
    }

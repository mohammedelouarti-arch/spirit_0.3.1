from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from automation.language import DEFAULT_LANGUAGE
from automation.models import JourneyResult, TransferPayload
from automation.utils import extract_transfer, to_plain

# Event fired at NGA to resume a call DFCX handed back. Overridable per case
# via `handback_resume.event`.
DEFAULT_HANDBACK_EVENT = "handback"


class LiveCxasSessionAdapter:
    def __init__(self, app_name: str, project_id: str, location: str = "us", credentials=None, sessions_client=None):
        self.app_name = app_name
        self.project_id = project_id
        self.location = location
        self.credentials = credentials
        self.client = sessions_client or self._build_client()
        self.trace_path = Path("reports/cxas_raw_trace.jsonl")
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.unlink(missing_ok=True)

    def _build_client(self):
        from cxas_scrapi import Sessions
        candidates = [
            {"app_name": self.app_name, "creds": self.credentials},
            {"app_name": self.app_name, "credentials": self.credentials},
            {"app_name": self.app_name},
        ]
        last_error = None
        for kwargs in candidates:
            try:
                return Sessions(**{key: value for key, value in kwargs.items() if value is not None})
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(f"Could not initialize cxas_scrapi.Sessions: {last_error}")

    def _write_trace(self, *, label: str, request: dict[str, Any], response: Any) -> None:
        record = {
            "label": label,
            "request": request,
            "response": to_plain(response),
            "transfer_found": bool(extract_transfer(response)),
            "transfer": extract_transfer(response),
        }
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _run(self, session_id: str, *, text=None, event=None, variables=None, kind: str = "text"):
        if kind == "event":
            return self.client.run(session_id=session_id, event=event, event_vars=variables)
        if kind == "dtmf":
            return self.client.run(session_id=session_id, dtmf=str(text), variables=variables)
        return self.client.run(session_id=session_id, text=str(text), variables=variables)

    def run_journey(self, case: dict[str, Any]) -> JourneyResult:
        # uuid4().hex is 32 chars -- already within the DFCX 36-char session limit.
        session_id = uuid.uuid4().hex
        result = JourneyResult(session_id=session_id)
        session_start = case["session_start"]
        response = self._run(session_id, event=session_start.get("event"), variables=session_start.get("variables"), kind="event")
        self._write_trace(label="session_start", request=session_start, response=response)
        result.trace.append({"label": "session_start", "request": session_start, "response": to_plain(response)})
        if extract_transfer(response):
            raise AssertionError("transferToDialogflow occurred during session start")

        steering_step = {"kind": "text", "input": case["steering_utterance"], "label": "steering_intent"}
        steps = list(case.get("wrapper_steps", [])) + list(case.get("identification_steps", [])) + [steering_step]
        for index, step in enumerate(steps):
            label = step.get("label", f"step_{index + 1}")
            response = self._run(session_id, text=step["input"], variables=step.get("variables"), kind=step.get("kind", "text"))
            self._write_trace(label=label, request=step, response=response)
            transfer = extract_transfer(response)
            result.trace.append({"index": index, "label": label, "request": step, "response": to_plain(response), "transfer_found": bool(transfer)})
            if not transfer:
                continue
            if label != "steering_intent":
                raise AssertionError(f"Early transfer during {label}")
            result.transfer = TransferPayload(target=transfer["transferToDialogflow"], parameters=transfer.get("parameters", {}))
            break
        if not result.transfer:
            raise AssertionError("No transferToDialogflow payload found. See reports/cxas_raw_trace.jsonl")
        return result

    # ------------------------------------------------------------------
    # Post-handback legs: DFCX returned control, keep driving the same
    # NGA session until it ends the call or steers back into DFCX.
    # ------------------------------------------------------------------

    def _normalize(self, response: Any) -> dict[str, Any]:
        """Shape a raw CXaS response like a DFCX turn so the loop is uniform."""
        plain = to_plain(response)
        transfer = extract_transfer(plain)
        text = (
            getattr(response, "response_text", None)
            or getattr(response, "text", None)
            or self._text_from(plain)
        )
        end_session = bool(
            (plain or {}).get("end_session")
            or (plain or {}).get("endInteraction")
            or (plain or {}).get("end_interaction")
        )
        return {
            "channel": "nga",
            "response_text": str(text or "").strip(),
            "matched_intent": (plain or {}).get("matched_intent"),
            "current_page": (plain or {}).get("current_page"),
            "end_session": end_session,
            "end_reason": (plain or {}).get("end_reason") or ("end_interaction" if end_session else None),
            "transfer": transfer,
            "raw": plain,
        }

    @staticmethod
    def _text_from(plain: Any) -> str:
        """Best-effort scrape of speakable text out of a CXaS payload."""
        collected: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ("text", "response_text", "responseText") and isinstance(value, str):
                        collected.append(value)
                    elif key == "text" and isinstance(value, dict):
                        walk(value)
                    else:
                        walk(value)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(plain)
        return " ".join(dict.fromkeys(item.strip() for item in collected if item.strip()))

    def resume_from_handback(
        self,
        session_id: str,
        handback: dict[str, Any],
        *,
        language_code: str = DEFAULT_LANGUAGE,
        event: str | None = None,
    ) -> dict[str, Any]:
        """Hand the call back to NGA and return its first turn.

        DFCX signalled `transferToNga`; we resume the *same* CXaS session,
        replaying the handback `variables` (`handoff_from`, `handoff_to`,
        `utterance`, ...) as event variables so NGA lands on the right menu.
        """
        variables = dict((handback or {}).get("variables") or {})
        request = {
            "event": event or DEFAULT_HANDBACK_EVENT,
            "variables": variables,
            "transferToNga": (handback or {}).get("transferToNga"),
        }
        response = self._run(session_id, event=request["event"], variables=variables, kind="event")
        self._write_trace(label="handback_resume", request=request, response=response)
        return self._normalize(response)

    def send_text(
        self,
        session_id: str,
        text: str,
        *,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        """Send one customer utterance to the live NGA session."""
        response = self._run(session_id, text=text, kind="text")
        self._write_trace(label="nga_turn", request={"text": text}, response=response)
        return self._normalize(response)

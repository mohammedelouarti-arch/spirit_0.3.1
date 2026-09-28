from __future__ import annotations

from typing import Any

from automation.dfcx_parsing import extract_texts, parse_query_result  # noqa: F401 (re-export)
from automation.language import DEFAULT_LANGUAGE, normalize_language
from automation.utils import to_plain


# Dialogflow's built-in Default Welcome Intent. Every agent has it under this
# fixed id. Triggering it explicitly is how a real client re-enters a session
# after NGA hands the call back -- see `trigger_intent`.
DEFAULT_WELCOME_INTENT_ID = "00000000-0000-0000-0000-000000000000"


def agent_of(target_environment: str) -> str:
    """`.../agents/A/environments/E` -> `.../agents/A`.

    Intents are agent-scoped, not environment-scoped, so an intent resource
    name must be built from the agent portion of an environment path.
    """
    return target_environment.split("/environments/", 1)[0]


class GoogleDfcxAdapter:
    """Live Dialogflow CX session adapter."""

    def __init__(self, credentials=None):
        self.credentials = credentials

    @staticmethod
    def _location(target: str) -> str:
        return target.split("/locations/", 1)[1].split("/", 1)[0]

    def _client(self, target: str):
        # Imported lazily so offline runs and unit tests don't need the SDK.
        from google.api_core.client_options import ClientOptions
        from google.cloud import dialogflowcx_v3 as cx

        location = self._location(target)
        endpoint = (
            "dialogflow.googleapis.com" if location == "global"
            else f"{location}-dialogflow.googleapis.com"
        )
        return cx.SessionsClient(
            credentials=self.credentials,
            client_options=ClientOptions(api_endpoint=endpoint),
        )

    def _detect(
        self,
        target_environment: str,
        session_id: str,
        query_input,
        parameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from google.cloud import dialogflowcx_v3 as cx
        from google.protobuf.struct_pb2 import Struct

        session = f"{target_environment}/sessions/{session_id[:36]}"
        query_params = None
        if parameters:
            struct = Struct()
            struct.update(parameters)
            query_params = cx.QueryParameters(parameters=struct)
        request = cx.DetectIntentRequest(
            session=session, query_input=query_input, query_params=query_params
        )
        plain = to_plain(self._client(target_environment).detect_intent(request=request))
        return parse_query_result(plain)

    def send_text(
        self,
        target_environment: str,
        session_id: str,
        text: str,
        language_code: str = DEFAULT_LANGUAGE,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from google.cloud import dialogflowcx_v3 as cx

        query_input = cx.QueryInput(
            text=cx.TextInput(text=text),
            language_code=normalize_language(language_code),
        )
        return self._detect(target_environment, session_id, query_input, parameters)

    def trigger_intent(
        self,
        target_environment: str,
        session_id: str,
        parameters: dict[str, Any] | None = None,
        language_code: str = DEFAULT_LANGUAGE,
        intent_id: str = DEFAULT_WELCOME_INTENT_ID,
    ) -> dict[str, Any]:
        """Re-enter a session by explicitly triggering an intent.

        This is what a real client does when NGA hands a call back into DFCX:
        the wire request is `queryInput.intent` naming the Default Welcome
        Intent, with the NGA parameters on `queryParams.parameters` of the
        *same* request.

        It matters because after a handback the session is parked on
        `Trigger NGA` -- a page with no intent routes, only a Default-Welcome
        transition and a no-match handler. Sending text there NO_MATCHes
        forever; triggering the intent runs the real cascade
        (`Trigger NGA` -> `Coming From NGA` -> `Post Steering Routing`) and
        leaves the session on a conversational page again.
        """
        from google.cloud import dialogflowcx_v3 as cx

        intent = intent_id if "/" in intent_id else f"{agent_of(target_environment)}/intents/{intent_id}"
        query_input = cx.QueryInput(
            intent=cx.IntentInput(intent=intent),
            language_code=normalize_language(language_code),
        )
        return self._detect(target_environment, session_id, query_input, parameters)

    def continue_handoff(
        self,
        target_environment: str,
        parameters: dict[str, Any],
        utterance: str,
        session_id: str,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        return self.send_text(target_environment, session_id, utterance, language_code, parameters)

    def reenter(
        self,
        target_environment: str,
        parameters: dict[str, Any],
        utterance: str,
        session_id: str,
        language_code: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        """Re-enter after a handback. Same signature as `continue_handoff`.

        `utterance` is intentionally unused: it already travels as
        `parameters["utterance"]`, exactly as NGA sends it. Re-sending it as
        text is what caused the NO_MATCH loop.
        """
        return self.trigger_intent(target_environment, session_id, parameters, language_code)

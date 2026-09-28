"""Ollama-backed simulated customer.

After the steering utterance hands the call off to Dialogflow CX, every
subsequent customer turn is generated here instead of being scripted in YAML.

Role mapping is the critical detail
-----------------------------------
The LLM *is* the customer. In the chat transcript we therefore send:

    VA turn        -> role "user"       (the person talking TO our model)
    customer turn  -> role "assistant"  (our model's own previous output)

Labelling the VA as "assistant" -- which is the intuitive reading, since the VA
is an assistant -- makes the model continue the VA's persona and parrot its
last sentence back. The system prompt cannot override chat roles.
"""

from __future__ import annotations

import difflib
import os
import re
from typing import Any

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Models under ~3B routinely fail this role-play task: they mirror the VA,
# answer in English, or emit assistant-speak. Warn rather than hard-fail.
UNDERPOWERED_MODELS = ("0.5b", "1b", "1.5b", "2b")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPEAKER_PREFIX = re.compile(
    r"^\s*(customer|client|user|utilisateur|response|r\u00e9ponse|assistant|agent)\s*[:\-]\s*",
    re.IGNORECASE,
)
# Phrases only a service agent would say -- a strong signal of role inversion.
_AGENT_TELLS = (
    "comment puis-je vous aider",
    "puis-je vous aider",
    "que puis-je faire pour vous",
    "je serai ravi de vous aider",
    "je serais ravi de vous aider",
    "je serai ravi d'essayer de vous aider",
    "voici la r\u00e9ponse que je donnerai",
    "pourriez-vous me donner plus de d\u00e9tails",
    "pourriez-vous reformuler",
    "je vais envoyer un message texte",
    "how can i help you",
    "is there anything else",
)

SYSTEM_TEMPLATE = """You are ROLE-PLAYING AS THE CUSTOMER (the caller) in a voice assistant test.
You are NOT the virtual assistant. You are the human who phoned in with a problem.

Your objective: {goal}
Your persona: {profile}
You must speak ONLY in {language}.

IMPORTANT Rules:
- Speak as the caller, in the first person, about YOUR OWN problem.
- Answer the virtual assistant's latest question directly.
- One or two short, natural spoken sentences. No greetings after the first turn.
- NEVER offer help, NEVER ask "how can I help you", NEVER apologise for not understanding.
  Those are the assistant's lines, not yours.
- NEVER repeat or rephrase what the assistant just said back to it.
- Do not invent account numbers, phone numbers, or personal details.
- If the assistant asks a yes/no question, answer it plainly.
- If your objective is met and the assistant asks if you need anything else, decline politely.
- Output ONLY the words the caller speaks. No quotes, no speaker label, no explanation.

Example of the register expected (do not reuse verbatim):
  Assistant: "Voulez-vous que je v\u00e9rifie les promotions sur votre compte?"
  You: "Oui, s'il vous pla\u00eet, ma promotion est disparue le mois dernier."
"""

RETRY_SUFFIX = """

IMPORTANT: your previous attempt was rejected because it sounded like the
assistant or repeated the assistant's words. Reply as the CALLER only -- state
your own need or answer the question in one short sentence in {language}."""


def clean_reply(raw: str | None) -> str:
    """Strip think-blocks, speaker labels, quotes and stray newlines."""
    text = _THINK_BLOCK.sub("", str(raw or "")).strip()
    for _ in range(3):  # labels sometimes nest: 'Customer: "Client: ...'
        stripped = _SPEAKER_PREFIX.sub("", text).strip().strip("\"'` \n\t")
        if stripped == text:
            break
        text = stripped
    text = text.strip().strip("\"'` \n\t")
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()


def is_echo(reply: str, va_text: str, threshold: float = 0.6) -> bool:
    """True when `reply` parrots the VA instead of answering it."""
    if not reply:
        return False
    if _similarity(reply, va_text) >= threshold:
        return True
    # A long verbatim span lifted from the VA turn also counts as parroting.
    match = difflib.SequenceMatcher(None, reply.lower(), va_text.lower()).find_longest_match(
        0, len(reply), 0, len(va_text)
    )
    return match.size >= 60


def sounds_like_agent(reply: str) -> bool:
    """True when the reply uses service-agent phrasing."""
    lowered = reply.lower()
    return any(tell in lowered for tell in _AGENT_TELLS)


def normalize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert a neutral transcript into correctly-roled chat messages.

    Accepts either `{"speaker": "va"|"customer", "text": ...}` (preferred) or
    legacy `{"role": ..., "content": ...}` entries, and always emits
    VA -> "user", customer -> "assistant".
    """
    messages: list[dict[str, str]] = []
    for entry in history or []:
        if "speaker" in entry:
            speaker = str(entry.get("speaker", "")).lower()
            content = entry.get("text", "")
        else:
            # Legacy shape stored the VA as "assistant"; invert it.
            speaker = "va" if str(entry.get("role", "")).lower() == "assistant" else "customer"
            content = entry.get("content", "")
        if not str(content).strip():
            continue
        messages.append({
            "role": "user" if speaker == "va" else "assistant",
            "content": str(content),
        })
    return messages


def _message_content(response: Any) -> str:
    """Read `message.content` from either the object or dict client shape."""
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


class DynamicTestUser:
    """Generates the customer side of the post-steering conversation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ollama_host: str = DEFAULT_HOST,
        client: Any = None,
        temperature: float = 0.6,
        num_predict: int = 60,
        repeat_penalty: float = 1.2,
        max_attempts: int = 3,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.temperature = temperature
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.max_attempts = max(1, int(max_attempts))
        self.seed = seed
        self.verbose = verbose
        self.rejected: list[dict[str, str]] = []  # audit trail of discarded replies
        self._client = client  # injectable for tests

    @property
    def client(self) -> Any:
        """Lazily import/construct the Ollama client."""
        if self._client is None:
            # Import dynamically so environments that do not install the
            # optional Ollama dependency can still import this adapter.
            from importlib import import_module

            Client = import_module("ollama").Client

            self._client = Client(host=self.ollama_host)
            if self.verbose:
                print(f"Using Ollama model: {self.model_name} @ {self.ollama_host}")
        return self._client

    def health_check(self) -> None:
        """Fail fast with an actionable message if the model is unavailable."""
        if any(tag in self.model_name.lower() for tag in UNDERPOWERED_MODELS):
            print(
                f"[LLM WARNING] {self.model_name} is very small and tends to mirror the "
                "virtual assistant instead of role-playing the caller. "
                "Prefer qwen2.5:7b, llama3.1:8b or mistral-nemo for French role-play."
            )
        try:
            listing = self.client.list()
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Cannot reach Ollama at {self.ollama_host}: {exc}. "
                "Start it with 'ollama serve'."
            ) from exc

        models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
        names = set()
        for model in models or []:
            if isinstance(model, dict):
                name = model.get("model") or model.get("name")
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.add(str(name))

        wanted = {self.model_name, f"{self.model_name}:latest"}
        if names and not (wanted & names):
            raise RuntimeError(
                f"Model {self.model_name!r} not found in Ollama. "
                f"Run: ollama pull {self.model_name}. Available: {sorted(names)}"
            )

    def _options(self, attempt: int) -> dict[str, Any]:
        options: dict[str, Any] = {
            # Nudge temperature up on retries to break out of a parroting loop.
            "temperature": min(1.0, self.temperature + 0.15 * (attempt - 1)),
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.seed is not None:
            options["seed"] = self.seed + attempt
        return options

    def generate_autonomous_response(
        self,
        va_prompt: str,
        customer_goal: str,
        customer_profile: str,
        language: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        base_system = SYSTEM_TEMPLATE.format(
            goal=customer_goal, profile=customer_profile, language=language
        )
        history = normalize_history(conversation_history)[-10:]

        last_candidate = ""
        for attempt in range(1, self.max_attempts + 1):
            system = base_system
            if attempt > 1:
                system += RETRY_SUFFIX.format(language=language)

            messages = [{"role": "system", "content": system}, *history]
            # The VA's newest turn is what our model must respond to, so it is
            # a "user" message -- consistent with the history mapping above.
            messages.append({"role": "user", "content": va_prompt})

            if self.verbose:
                print(
                    f"[OLLAMA REQUEST] model={self.model_name} | language={language} "
                    f"| attempt={attempt}/{self.max_attempts}"
                )

            try:
                response = self.client.chat(
                    model=self.model_name, messages=messages, options=self._options(attempt)
                )
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[OLLAMA ERROR] {type(exc).__name__}: {exc}")
                return ""

            candidate = clean_reply(_message_content(response))
            if self.verbose:
                print(f"[OLLAMA RESPONSE] {candidate}")

            if not candidate:
                reason = "empty"
            elif is_echo(candidate, va_prompt):
                reason = "echoed the assistant"
            elif sounds_like_agent(candidate):
                reason = "used assistant phrasing"
            else:
                return candidate

            last_candidate = candidate
            self.rejected.append({"attempt": str(attempt), "reason": reason, "text": candidate})
            print(f"[OLLAMA REJECTED] attempt {attempt}: {reason}")

        print(
            f"[OLLAMA GIVE UP] {self.max_attempts} attempts all looked like the assistant. "
            f"Consider a larger model than {self.model_name}."
        )
        # Returning "" stops the loop with `empty_llm_response` rather than
        # feeding the VA a reply we know is wrong.
        return "" if last_candidate else ""

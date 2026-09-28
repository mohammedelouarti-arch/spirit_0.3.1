"""Prompt construction, chat-role mapping, and output sanitation."""

from tests.stubs import raises

from automation.adapters.llm_adapter import (
    DynamicTestUser,
    clean_reply,
    is_echo,
    normalize_history,
    sounds_like_agent,
)

VA_TURN = ("Je comprends que votre offre promotionnelle est expir\u00e9e. "
           "Voulez-vous que je v\u00e9rifie les promotions disponibles sur votre compte?")


class FakeOllamaClient:
    """Returns each queued content in turn; repeats the last one forever."""

    def __init__(self, contents=("Oui, s'il vous pla\u00eet.",), models=("qwen2.5:7b",), fail=False):
        self.contents = list(contents)
        self.models = models
        self.fail = fail
        self.chats: list[dict] = []

    def chat(self, model, messages, options=None):
        if self.fail:
            raise ConnectionError("connection refused")
        self.chats.append({"model": model, "messages": messages, "options": options})
        content = self.contents[min(len(self.chats) - 1, len(self.contents) - 1)]
        return {"message": {"content": content}}

    def list(self):
        return {"models": [{"model": name} for name in self.models]}


def _user(client=None, **kwargs):
    client = client or FakeOllamaClient()
    return DynamicTestUser(client=client, verbose=False, **kwargs), client


def _generate(user, va=VA_TURN, history=None):
    return user.generate_autonomous_response(va, "goal", "profile", "fr-CA", history or [])


# ------------------------------------------------------- chat-role mapping

def test_history_maps_va_to_user_and_customer_to_assistant():
    """Regression: the model IS the customer, so VA turns must be role=user."""
    messages = normalize_history([
        {"speaker": "va", "text": "Bonjour?"},
        {"speaker": "customer", "text": "Ma promo est expir\u00e9e."},
    ])
    assert messages == [
        {"role": "user", "content": "Bonjour?"},
        {"role": "assistant", "content": "Ma promo est expir\u00e9e."},
    ]


def test_legacy_role_history_is_inverted():
    # Old shape stored the VA as "assistant"; that inversion caused parroting.
    messages = normalize_history([
        {"role": "assistant", "content": "Bonjour?"},
        {"role": "user", "content": "Ma promo est expir\u00e9e."},
    ])
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_blank_history_entries_are_dropped():
    assert normalize_history([{"speaker": "va", "text": "   "}]) == []


def test_latest_va_turn_is_sent_as_a_user_message():
    user, client = _user()
    _generate(user, history=[{"speaker": "va", "text": "Pr\u00e9c\u00e9dent"},
                             {"speaker": "customer", "text": "Ma r\u00e9ponse"}])
    messages = client.chats[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Pr\u00e9c\u00e9dent"}
    assert messages[2] == {"role": "assistant", "content": "Ma r\u00e9ponse"}
    assert messages[-1] == {"role": "user", "content": VA_TURN}


# --------------------------------------------------------- echo detection

def test_verbatim_echo_is_detected():
    assert is_echo(VA_TURN, VA_TURN)


def test_near_verbatim_echo_is_detected():
    reply = ("Je m'excuse, j'ai toujours de la difficult\u00e9 \u00e0 comprendre votre demande. "
             "Pourriez-vous reformuler votre question?")
    assert is_echo(reply, "Je m\u2019excuse, j\u2019ai toujours de la difficult\u00e9 \u00e0 comprendre votre "
                          "demande. Pourriez-vous reformuler votre question?")


def test_genuine_customer_reply_is_not_an_echo():
    assert not is_echo("Oui, s'il vous pla\u00eet, v\u00e9rifiez mon compte.", VA_TURN)


def test_agent_phrasing_is_detected():
    assert sounds_like_agent("Je serai ravi d'essayer de vous aider davantage.")
    assert sounds_like_agent("Voici la r\u00e9ponse que je donnerai \u00e0 votre question.")
    assert not sounds_like_agent("Oui, ma promotion a expir\u00e9 le mois dernier.")


def test_echoed_reply_triggers_a_retry():
    client = FakeOllamaClient([VA_TURN, "Oui, v\u00e9rifiez s'il vous pla\u00eet."])
    user, _ = _user(client)
    assert _generate(user) == "Oui, v\u00e9rifiez s'il vous pla\u00eet."
    assert len(client.chats) == 2
    assert user.rejected[0]["reason"] == "echoed the assistant"


def test_agent_sounding_reply_triggers_a_retry():
    client = FakeOllamaClient(["Comment puis-je vous aider?", "Ma promo est expir\u00e9e."])
    user, _ = _user(client)
    assert _generate(user) == "Ma promo est expir\u00e9e."
    assert user.rejected[0]["reason"] == "used assistant phrasing"


def test_retry_prompt_carries_a_correction():
    client = FakeOllamaClient([VA_TURN, "Oui, d'accord."])
    user, _ = _user(client)
    _generate(user)
    assert "rejected" in client.chats[1]["messages"][0]["content"].lower()


def test_persistent_echo_gives_up_and_returns_empty():
    client = FakeOllamaClient([VA_TURN])
    user, _ = _user(client, max_attempts=3)
    assert _generate(user) == ""
    assert len(client.chats) == 3
    assert len(user.rejected) == 3


def test_temperature_escalates_across_attempts():
    client = FakeOllamaClient([VA_TURN, VA_TURN, "Oui."])
    user, _ = _user(client, temperature=0.6)
    _generate(user)
    temps = [chat["options"]["temperature"] for chat in client.chats]
    assert temps[0] < temps[1] < temps[2]


def test_seed_is_varied_per_attempt():
    client = FakeOllamaClient([VA_TURN, "Oui."])
    user, _ = _user(client, seed=42)
    _generate(user)
    assert [chat["options"]["seed"] for chat in client.chats] == [43, 44]


def test_repeat_penalty_is_sent():
    user, client = _user()
    _generate(user)
    assert client.chats[0]["options"]["repeat_penalty"] > 1.0


# ------------------------------------------------------------- sanitation

def test_clean_reply_strips_quotes_and_labels():
    assert clean_reply('"Oui, merci."') == "Oui, merci."
    assert clean_reply("Client: Oui, merci.") == "Oui, merci."
    assert clean_reply("Assistant - Oui, merci.") == "Oui, merci."


def test_clean_reply_strips_nested_labels():
    assert clean_reply('Customer: "Client: Oui, merci."') == "Oui, merci."


def test_clean_reply_removes_think_blocks():
    assert clean_reply("<think>reasoning</think>\nOui, merci.") == "Oui, merci."


def test_clean_reply_collapses_multiline():
    assert clean_reply("Oui.\n\nMerci beaucoup.") == "Oui. Merci beaucoup."


def test_clean_reply_handles_none():
    assert clean_reply(None) == ""


# ------------------------------------------------------------- plumbing

def test_prompt_includes_goal_profile_and_language():
    user, client = _user()
    user.generate_autonomous_response(VA_TURN, "GOAL-X", "PROFILE-Y", "fr-CA", [])
    system = client.chats[0]["messages"][0]["content"]
    assert "GOAL-X" in system and "PROFILE-Y" in system and "fr-CA" in system
    assert "NOT the virtual assistant" in system


def test_history_is_truncated_to_last_ten_messages():
    user, client = _user()
    history = [{"speaker": "va" if index % 2 == 0 else "customer", "text": str(index)}
               for index in range(30)]
    _generate(user, history=history)
    # 1 system + 10 history + 1 latest VA turn
    assert len(client.chats[0]["messages"]) == 12


def test_transport_failure_returns_empty_string():
    user, _ = _user(FakeOllamaClient(fail=True))
    assert _generate(user) == ""


def test_health_check_passes_for_available_model():
    user, _ = _user(FakeOllamaClient(models=("qwen2.5:7b", "llama3:8b")))
    user.health_check()


def test_health_check_tolerates_latest_suffix():
    user, _ = _user(FakeOllamaClient(models=("qwen2.5:7b:latest",)))
    user.health_check()


def test_health_check_rejects_missing_model():
    user, _ = _user(FakeOllamaClient(models=("llama3:8b",)))
    with raises(RuntimeError, match="ollama pull"):
        user.health_check()


def test_health_check_warns_about_tiny_models(capsys=None):
    client = FakeOllamaClient(models=("qwen2.5:0.5b",))
    user = DynamicTestUser(model_name="qwen2.5:0.5b", client=client, verbose=False)
    user.health_check()  # prints a warning, must not raise

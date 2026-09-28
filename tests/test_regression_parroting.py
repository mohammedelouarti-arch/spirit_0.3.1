"""Regression: the model must not role-play the virtual assistant.

Reproduces the exact failure seen against staging on 2026-09-21, where
qwen2.5:0.5b echoed the VA's own sentences back into DFCX because the chat
history labelled the VA as "assistant".
"""

from automation.adapters.llm_adapter import (
    DynamicTestUser,
    is_echo,
    normalize_history,
    sounds_like_agent,
)

# Verbatim from the failing staging run.
VA_TURN_1 = ("Je comprends que vous avez une question concernant votre facture. Pour vous aider, "
             "l'application MonBell offre une vue d\u00e9taill\u00e9e de votre facture, ce qui permet "
             "souvent declarifier les questions de facturation. Est-ce que cela vous convient ?")
BAD_REPLY_1 = ("Oui, je comprends. Je vais envoyer un message texte \u00e0 l'appareil pour vous "
               "apprendre plus sur votre facture. Je vais ensuite vous envoyer un lien vers "
               "l'application MonBell.")

VA_TURN_3 = ("Je m\u2019excuse, j\u2019ai toujours de la difficult\u00e9 \u00e0 comprendre votre demande. Pour vous "
             "aider au mieux, pourriez-vous, s\u2019il vous pla\u00eet, reformuler votre question ou fournir "
             "plus de d\u00e9tails sur ce que vous recherchez concernant les produits et services de Bell?")
BAD_REPLY_3 = ("Je m'excuse, j'ai toujours de la difficult\u00e9 \u00e0 comprendre votre demande. Pour vous "
               "aider au mieux, pourriez-vous, s'il vous pla\u00eet, reformuler votre question ou fournir "
               "plus de d\u00e9tails sur ce que vous recherchez concernant les produits et services de Bell?")

BAD_REPLY_2 = ("Bien s\u00fbr, voici la r\u00e9ponse que je donnerai \u00e0 votre question. Pourriez-vous me "
               "donner plus de d\u00e9tails sur votre facture ? Je serai ravi d'essayer de vous aider "
               "davantage.")

GOOD_REPLY = "Non, je veux juste savoir pourquoi ma facture a augment\u00e9 ce mois-ci."


class ReplayClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.chats = []

    def chat(self, model, messages, options=None):
        self.chats.append({"messages": messages, "options": options})
        index = min(len(self.chats) - 1, len(self.contents) - 1)
        return {"message": {"content": self.contents[index]}}


def test_turn3_verbatim_parroting_is_rejected():
    assert is_echo(BAD_REPLY_3, VA_TURN_3)


def test_turn1_assistant_roleplay_is_rejected():
    assert sounds_like_agent(BAD_REPLY_1) or is_echo(BAD_REPLY_1, VA_TURN_1)


def test_turn2_assistant_roleplay_is_rejected():
    assert sounds_like_agent(BAD_REPLY_2)


def test_good_customer_reply_is_accepted():
    assert not is_echo(GOOD_REPLY, VA_TURN_1)
    assert not sounds_like_agent(GOOD_REPLY)


def test_bad_reply_is_retried_and_the_good_one_is_used():
    client = ReplayClient([BAD_REPLY_3, GOOD_REPLY])
    user = DynamicTestUser(client=client, verbose=False)
    reply = user.generate_autonomous_response(VA_TURN_3, "goal", "profile", "fr-CA", [])
    assert reply == GOOD_REPLY
    assert len(client.chats) == 2


def test_va_history_is_never_labelled_assistant():
    """The root cause: VA turns as role=assistant made the model continue the VA."""
    history = [
        {"speaker": "va", "text": VA_TURN_1},
        {"speaker": "customer", "text": GOOD_REPLY},
        {"speaker": "va", "text": VA_TURN_3},
    ]
    messages = normalize_history(history)
    va_messages = [message for message in messages
                   if message["content"] in (VA_TURN_1, VA_TURN_3)]
    assert va_messages, "VA turns missing from history"
    assert all(message["role"] == "user" for message in va_messages)


def test_full_request_puts_every_va_line_in_user_role():
    client = ReplayClient([GOOD_REPLY])
    user = DynamicTestUser(client=client, verbose=False)
    user.generate_autonomous_response(
        VA_TURN_3, "goal", "profile", "fr-CA",
        [{"speaker": "va", "text": VA_TURN_1}, {"speaker": "customer", "text": GOOD_REPLY}],
    )
    messages = client.chats[0]["messages"]
    assistant_texts = [message["content"] for message in messages if message["role"] == "assistant"]
    assert VA_TURN_1 not in assistant_texts
    assert VA_TURN_3 not in assistant_texts
    assert assistant_texts == [GOOD_REPLY]

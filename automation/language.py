"""Language-code helpers.

Dialogflow CX expects BCP-47 codes in ``ll-RR`` form (``fr-CA``, ``en-US``).
Test fixtures habitually write the lower-case form (``fr-ca``) because
assertions are compared case-insensitively, so we normalize at the boundary
instead of trusting whatever the YAML author typed.
"""

from __future__ import annotations

DEFAULT_LANGUAGE = "fr-CA"


def normalize_language(code: str | None, default: str = DEFAULT_LANGUAGE) -> str:
    """Return ``code`` as a canonical BCP-47 tag.

    >>> normalize_language("fr-ca")
    'fr-CA'
    >>> normalize_language("FR_CA")
    'fr-CA'
    >>> normalize_language("fr")
    'fr'
    >>> normalize_language("")
    'fr-CA'
    """
    text = str(code or "").strip().replace("_", "-")
    if not text:
        return default

    parts = [part for part in text.split("-") if part]
    if not parts:
        return default

    normalized = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 2:  # region subtag -> upper
            normalized.append(part.upper())
        elif len(part) == 4:  # script subtag -> title
            normalized.append(part.title())
        else:
            normalized.append(part.lower())
    return "-".join(normalized)


def same_language(left: str | None, right: str | None) -> bool:
    """Case/separator-insensitive comparison of two language tags."""
    return normalize_language(left, "").lower() == normalize_language(right, "").lower()

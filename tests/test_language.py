from tests.stubs import raises  # noqa: F401  (keeps sys.path bootstrap consistent)

from automation.language import DEFAULT_LANGUAGE, normalize_language, same_language


def test_normalize_uppercases_region():
    assert normalize_language("fr-ca") == "fr-CA"
    assert normalize_language("EN-us") == "en-US"


def test_normalize_accepts_underscores():
    assert normalize_language("FR_CA") == "fr-CA"


def test_normalize_preserves_bare_language():
    assert normalize_language("fr") == "fr"


def test_normalize_handles_script_subtag():
    assert normalize_language("zh-hans-cn") == "zh-Hans-CN"


def test_normalize_falls_back_on_empty():
    assert normalize_language("") == DEFAULT_LANGUAGE
    assert normalize_language(None) == DEFAULT_LANGUAGE
    assert normalize_language("   ", "en-US") == "en-US"


def test_same_language_is_case_insensitive():
    assert same_language("fr-ca", "fr-CA")
    assert same_language("FR_CA", "fr-ca")
    assert not same_language("fr-ca", "en-CA")
    assert not same_language(None, "fr-CA")

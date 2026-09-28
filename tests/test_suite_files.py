"""Lint the shipped YAML so a malformed case fails fast, not at 3am in staging."""

from pathlib import Path

import yaml

from tests.stubs import ScriptedTestUser, silent

from automation.adapters.fake_dfcx import SCRIPTS
from automation.language import normalize_language
from automation.runner import load_cases, run_case

SUITE = "suites/e2e_handoff"
REQUIRED_KEYS = ("name", "tags", "steering_utterance", "expected_transfer")
ENVIRONMENTS = yaml.safe_load(
    Path("config/environments.yaml").read_text(encoding="utf-8-sig")
)


def _cases():
    return load_cases(SUITE)


def test_every_yaml_file_parses_and_has_evals():
    files = sorted(Path(SUITE).glob("*.yaml"))
    assert files, "no suite files found"
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        assert data.get("evals"), f"{path} has no evals"


def test_case_names_are_unique():
    names = [case["name"] for case in _cases()]
    assert len(names) == len(set(names)), f"duplicate case names: {names}"


def test_required_keys_present():
    for case in _cases():
        for key in REQUIRED_KEYS:
            assert case.get(key), f"{case.get('name')} missing {key}"


def test_expected_transfer_is_complete_and_points_at_a_known_environment():
    known = {env["flow_environment"] for env in ENVIRONMENTS.values()}
    for case in _cases():
        expected = case["expected_transfer"]
        for key in ("environment", "route", "language", "handoff_to"):
            assert expected.get(key), f"{case['name']}.expected_transfer missing {key}"
        assert expected["environment"] in known, f"{case['name']} targets an unknown environment"


def test_languages_are_valid_bcp47():
    for case in _cases():
        language = case["expected_transfer"]["language"]
        assert normalize_language(language) in {"fr-CA", "en-CA", "en-US"}, case["name"]


def test_autonomous_cases_declare_a_goal_and_stop_phrases():
    for case in _cases():
        flow = case.get("autonomous_flow") or {}
        if not flow.get("enabled"):
            continue
        assert str(flow.get("customer_goal", "")).strip(), f"{case['name']} missing customer_goal"
        assert flow.get("stop_phrases"), f"{case['name']} missing stop_phrases"
        assert int(flow.get("max_turns", 8)) > 0


def test_conversation_bounds_are_consistent_with_max_turns():
    for case in _cases():
        flow = case.get("autonomous_flow") or {}
        expected = case.get("expected_conversation") or {}
        if not (flow.get("enabled") and expected.get("max_turns")):
            continue
        assert int(expected["max_turns"]) <= int(flow.get("max_turns", 8)), case["name"]


def test_every_route_has_an_offline_script():
    # Keeps the offline fake honest as new routes are added.
    for case in _cases():
        route = case["expected_transfer"]["route"]
        assert route in SCRIPTS, f"{case['name']}: no offline script for route {route!r}"


def test_all_shipped_cases_pass_offline():
    env = ENVIRONMENTS["staging"]
    for case in _cases():
        result = run_case(case, env, live=False, llm=ScriptedTestUser(), printer=silent)
        assert result["passed"], f"{case['name']}: {result['errors']}"

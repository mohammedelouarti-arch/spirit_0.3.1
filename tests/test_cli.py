"""End-to-end CLI behaviour, driven offline with --fake-llm."""

import json
from pathlib import Path

from tests.stubs import raises

import orchestrate
from automation.adapters.canned_llm import CannedTestUser


def _run(argv, tmp_path=None):
    report = Path(tmp_path or "reports") / "cli_test.json"
    code = orchestrate.main([*argv, "--report", str(report)])
    payload = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    report.unlink(missing_ok=True)
    return code, payload


def test_full_suite_passes_offline_with_fake_llm(tmp_path=None):
    code, payload = _run(["e2e-handoff", "--env", "staging", "--fake-llm"], tmp_path)
    assert code == 0, payload.get("results")
    assert payload["summary"]["failed"] == 0
    assert payload["summary"]["total"] >= 5
    assert payload["summary"]["autonomous_turns"] > 0


def test_no_autonomous_skips_conversation_assertions(tmp_path=None):
    code, payload = _run(["e2e-handoff", "--env", "staging", "--no-autonomous"], tmp_path)
    assert code == 0
    assert payload["summary"]["autonomous_turns"] == 0


def test_tag_filter_narrows_the_run(tmp_path=None):
    _, payload = _run(
        ["e2e-handoff", "--env", "staging", "--fake-llm", "--tags", "billing"], tmp_path
    )
    names = [result["name"] for result in payload["results"]]
    assert names and all("bill" in name for name in names)


def test_name_filter_selects_a_single_case(tmp_path=None):
    _, payload = _run(
        ["e2e-handoff", "--env", "staging", "--fake-llm", "--name", "internet_outage"], tmp_path
    )
    assert [result["name"] for result in payload["results"]] == ["tech_internet_outage_nga_to_dfcx"]


def test_max_turns_override_is_applied(tmp_path=None):
    _, payload = _run(
        ["e2e-handoff", "--env", "staging", "--fake-llm", "--name", "bill_promo_expired_nga",
         "--max-turns", "1"], tmp_path
    )
    result = payload["results"][0]
    assert result["autonomous"]["turns"] == 1
    assert result["autonomous"]["stop_reason"] == "max_turns"
    # The case forbids max_turns, so it must be reported as a failure.
    assert result["passed"] is False


def test_unknown_environment_exits():
    with raises(SystemExit, match="not defined"):
        orchestrate.main(["e2e-handoff", "--env", "does-not-exist"])


def test_no_matching_cases_exits():
    with raises(SystemExit, match="No matching test cases"):
        orchestrate.main(["e2e-handoff", "--env", "staging", "--tags", "nope"])


def test_list_cases_exits_zero():
    assert orchestrate.main(["list-cases"]) == 0


def test_canned_user_cycles_replies():
    user = CannedTestUser(["a", "b"])
    replies = [user.generate_autonomous_response("q", "g", "p", "fr-CA", []) for _ in range(4)]
    assert replies == ["a", "b", "a", "b"]
    user.health_check()  # no-op, must not raise

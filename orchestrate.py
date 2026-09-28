"""CLI entry point for the NGA -> Dialogflow CX handoff suite."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import yaml

from automation.runner import build_llm, load_cases, run_case, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automation Test Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    e2e = subparsers.add_parser("e2e-handoff", help="Run the NGA -> DFCX handoff suite")
    e2e.add_argument("--env", default="staging")
    e2e.add_argument("--tags", nargs="*", default=[],
                     help="Only run cases carrying ALL of these tags")
    e2e.add_argument("--name", default=None,
                     help="Only run cases whose name contains this substring")
    e2e.add_argument("--suite", default="suites/e2e_handoff")
    e2e.add_argument("--live", action="store_true",
                     help="Drive the real NGA + DFCX stack instead of the offline fake")
    e2e.add_argument("--report", default="reports/e2e_handoff.json")

    autonomy = e2e.add_mutually_exclusive_group()
    autonomy.add_argument("--autonomous", dest="autonomous", action="store_true", default=None,
                          help="Force Ollama to drive every turn after the steering utterance")
    autonomy.add_argument("--no-autonomous", dest="autonomous", action="store_false",
                          help="Assert the handoff only; skip the Ollama conversation")

    e2e.add_argument("--fake-llm", action="store_true",
                     help="Drive the autonomous loop with canned replies instead of Ollama (CI)")
    e2e.add_argument("--model", default=None, help="Ollama model (default: $OLLAMA_MODEL)")
    e2e.add_argument("--ollama-host", default=None, help="Ollama host (default: $OLLAMA_HOST)")
    e2e.add_argument("--max-turns", type=int, default=None,
                     help="Override autonomous_flow.max_turns for every case")
    e2e.add_argument("--max-legs", type=int, default=None,
                     help="Override autonomous_flow.max_legs: how many NGA<->DFCX "
                          "handback/retransfer bounces a call may make")
    e2e.add_argument("--max-attempts", type=int, default=3,
                     help="Retries when the model echoes the VA or sounds like an agent")
    e2e.add_argument("--seed", type=int, default=None,
                     help="Ollama seed, for reproducible conversations")

    subparsers.add_parser("list-cases", help="List discoverable cases").add_argument(
        "--suite", default="suites/e2e_handoff"
    )
    return parser


def load_environments(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Environment config not found: {path}")
    with path.open(encoding="utf-8-sig") as file:
        return yaml.safe_load(file) or {}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list-cases":
        for case in load_cases(args.suite):
            flow = case.get("autonomous_flow") or {}
            print(f"{case['name']:<40} tags={','.join(case.get('tags', []))} "
                  f"autonomous={bool(flow.get('enabled'))}")
        return 0

    envs = load_environments(Path("config/environments.yaml"))
    if args.env not in envs:
        raise SystemExit(f"Environment {args.env!r} not defined in config: {sorted(envs)}")

    cases = load_cases(args.suite, args.tags)
    if args.name:
        cases = [case for case in cases if args.name.lower() in case["name"].lower()]
    if not cases:
        raise SystemExit(f"No matching test cases in {args.suite} (tags={args.tags}, name={args.name})")

    if args.max_turns is not None:
        for case in cases:
            case.setdefault("autonomous_flow", {})["max_turns"] = args.max_turns
    if args.max_legs is not None:
        for case in cases:
            case.setdefault("autonomous_flow", {})["max_legs"] = args.max_legs

    # Build the Ollama client once and share it across cases, but only if at
    # least one case will actually use it.
    needs_llm = args.autonomous is not False and any(
        args.autonomous or (case.get("autonomous_flow") or {}).get("enabled") for case in cases
    )
    llm = None
    if needs_llm and args.fake_llm:
        from automation.adapters.canned_llm import CannedTestUser

        print("[LLM] --fake-llm: using canned customer replies (Ollama not contacted)")
        llm = CannedTestUser()
    elif needs_llm:
        llm = build_llm(args.model, args.ollama_host,
                        max_attempts=args.max_attempts, seed=args.seed)
        try:
            llm.health_check()
        except Exception as exc:
            raise SystemExit(f"Ollama preflight failed: {exc}")

    results = []
    for case in cases:
        try:
            result = run_case(case, envs[args.env], args.live, llm=llm, autonomous=args.autonomous)
        except Exception as exc:
            traceback.print_exc()
            result = {
                "name": case.get("name", "Unknown Case"),
                "passed": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(result)

        autonomous = result.get("autonomous") or {}
        suffix = ""
        if autonomous.get("enabled"):
            legs = autonomous.get("legs", 1)
            bounce = ""
            if legs > 1:
                bounce = (f", {legs} legs, {len(autonomous.get('handbacks') or [])} handback(s)"
                          f"/{len(autonomous.get('retransfers') or [])} retransfer(s)")
            suffix = (f"  [{autonomous.get('turns')} turn(s){bounce}, "
                      f"stop={autonomous.get('stop_reason')}]")
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['name']}{suffix}")
        for error in result.get("errors", []):
            print(f"  - {error}")

    summary = write_report(results, args.report)
    print(f"TOTAL: {summary['passed']}/{summary['total']} passed "
          f"({summary['autonomous_turns']} autonomous turns) -> {args.report}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

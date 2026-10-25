"""Run the context engine on the test stories and print a timeline and a scorecard.

    python run.py                       # all stories, LLM if a key is set
    python run.py --rules               # all stories, rules only (offline)
    python run.py -s hr_spike_still     # one story
    python run.py --list                # list the stories
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from context_engine import BY_KEY, SCENARIOS, make_llm, run_scenario
from context_engine.evaluate import ScenarioResult

ROOT = Path(__file__).resolve().parent
QUIET_TOOLS = {"do_nothing", "log_event"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--scenario", action="append", choices=sorted(BY_KEY), help="story to run (repeatable)")
    ap.add_argument("--rules", action="store_true", help="rules only, no LLM calls")
    ap.add_argument("--quiet", action="store_true", help="scorecard only")
    ap.add_argument("--all-actions", action="store_true", help="also show do_nothing decisions")
    ap.add_argument("--list", action="store_true", help="list the stories and exit")
    args = ap.parse_args()

    if args.list:
        for sc in SCENARIOS:
            print(f"{sc.key:24} {sc.title}")
        return 0

    llm = None if args.rules else make_llm()
    scenarios = [BY_KEY[k] for k in args.scenario] if args.scenario else SCENARIOS
    (ROOT / "logs").mkdir(exist_ok=True)
    log_path = ROOT / "logs" / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    print(f"Mode: {'LLM - ' + llm.label if llm else 'rules only (offline)'}")
    print(f"Action log: {log_path}\n")

    results: list[ScenarioResult] = []
    started = time.perf_counter()
    for sc in scenarios:
        if not args.quiet:
            print(f"=== {sc.title} ===\n    {sc.story}")

        def on_step(step, engine):
            if args.quiet:
                return
            if step.changes:
                print(f"  {step.reading.time:%H:%M:%S}  SITUATION  {step.state.summary}  [{step.state.source}]"
                      f"  <- {'; '.join(step.changes)}")
            for a in step.actions:
                if a.tool in QUIET_TOOLS and not args.all_actions:
                    continue
                print(f"  {a.time:%H:%M:%S}  {a.tool:<24} {a.reason}\n{'':12}by {a.by} -> {a.result}")

        result = run_scenario(sc, llm=llm, on_step=on_step, log_path=log_path)
        results.append(result)
        if not args.quiet:
            print(f"  {verdict(result)}\n")

    print_scorecard(results, llm, time.perf_counter() - started)
    return 0 if all(r.passed or (r.scenario.needs_llm and not llm) for r in results) else 1


def verdict(r: ScenarioResult) -> str:
    if r.passed:
        return "PASS"
    parts = []
    if r.missing:
        parts.append("missed " + ", ".join(sorted(r.missing)))
    if r.mistakes:
        parts.append("should not have used " + ", ".join(sorted(r.mistakes)))
    note = " (expected without an LLM)" if r.scenario.needs_llm and r.mode.startswith("Rules") else ""
    return f"FAIL{note}: " + "; ".join(parts)


def print_scorecard(results: list[ScenarioResult], llm, seconds: float) -> None:
    print("=" * 78)
    print(f"{'Story':<26}{'Result':<8}{'Agent runs / readings':<24}{'Avg decision':>12}")
    print("-" * 78)
    for r in results:
        s = r.stats
        print(f"{r.scenario.key:<26}{'PASS' if r.passed else 'FAIL':<8}"
              f"{f'{s.agent_runs} / {s.readings}':<24}{s.avg_decision_ms:>9.0f} ms")
    print("-" * 78)
    passed = sum(r.passed for r in results)
    readings = sum(r.stats.readings for r in results)
    runs = sum(r.stats.agent_runs for r in results)
    print(f"Correct decisions : {passed} of {len(results)} stories")
    print(f"False alarms      : {sum(r.false_alarms for r in results)}")
    print(f"AI calls saved    : {100 * (1 - runs / (2 * readings)):.1f}%  "
          f"({runs} agent runs instead of {2 * readings} for {readings} readings)")
    if llm:
        print(f"LLM               : {llm.calls} calls, {llm.failures} failed (rules took over), "
              f"{llm.tokens:,} tokens, waited {llm.waited_seconds:.0f}s for the free-tier rate limit")
    print(f"Total time        : {seconds:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())

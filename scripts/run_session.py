"""
Run a session from the command line, no server/frontend needed. This is how
you validate each build stage:

  # Day 1 — sim + fallback planner only, zero AI, zero Cognee, zero network calls:
  python scripts/run_session.py --task make_coffee --fallback-only

  # Day 2+ — full loop with Cognee + LLM planner, session 1 (cold) then session 2 (memory):
  python scripts/run_session.py --task make_coffee --mode cold
  python scripts/run_session.py --task make_coffee --mode memory

  # Day 3 — drift scenario:
  python scripts/run_session.py --task make_coffee --mode drift
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from house_sim.scenarios import build_house, apply_standard_drift, TASKS
from agent.graph import run_session


def print_event(event: dict):
    t = event.get("type")
    if t == "plan":
        a = event["action"]
        print(f'  [step {event["step"]}] "{a["thought"]}" -> {a["type"]} {a.get("target") or ""}')
    elif t == "recall":
        preview = event["result"][:300] + ("..." if len(event["result"]) > 300 else "")
        print(f'    [recall] query="{event["query"]}"\n             -> {preview}')
    elif t == "session_remembered":
        print(f'    [remember] wrote {event["observation_count"]} observations to Cognee in one call')
    elif t == "act":
        print(f'    -> {event["result"]["message"]}')
    elif t == "memory_correction":
        print(f'    !! memory correction: {event["reason"]}')
    elif t == "planner_fallback":
        print(f'    (LLM planner exhausted retries, using deterministic fallback: {event["reason"]})')
    elif t == "session_start":
        print(f'--- session {event["session"]}: {event["task"]} ---')
    elif t == "session_end":
        print(f'--- done: {event["steps_taken"]} actions, completed={event["completed"]} ---\n')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="make_coffee", choices=list(TASKS.keys()))
    parser.add_argument("--mode", default="cold", choices=["cold", "memory", "drift"])
    parser.add_argument("--fallback-only", action="store_true",
                         help="No LLM, no Cognee — pure sim + deterministic planner. Use this on day 1.")
    parser.add_argument("--session-number", type=int, default=1)
    args = parser.parse_args()

    house = build_house()
    if args.mode == "drift":
        apply_standard_drift(house)

    result = await run_session(
        house,
        args.task,
        args.session_number,
        use_llm=not args.fallback_only,
        emit=print_event,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
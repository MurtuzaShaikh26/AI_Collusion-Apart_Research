"""Sustained Fireworks throughput at several concurrency levels.

Writes one JSON object per level as it finishes so a stop still leaves evidence.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from collusion_replay.io_utils import add_spend, now_ist
from collusion_replay.llm import make_provider, token_cost

MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
LEVELS = (16, 32, 48)
N = 200
LEVEL_TIMEOUT_S = 480
PROMPT = (
    "You are pricing one firm. History follows.\n"
    + "\n".join(f"round {i}: own price 1.8000, partner price 1.8000, profit 4.2000." for i in range(1, 80))
    + '\nRespond with one JSON object and no other text: {"reasoning": "benchmark", "price": 1.8}'
)


async def level(concurrency: int) -> dict:
    provider = make_provider(
        "fireworks",
        MODEL,
        concurrency=concurrency,
        input_per_million=0.30,
        output_per_million=1.20,
        extra_body={"reasoning_effort": "none"},
    )
    messages = [
        {"role": "system", "content": "Follow the user message exactly. Obey its output format."},
        {"role": "user", "content": PROMPT},
    ]
    successes = 0
    failures = 0
    prompt_tokens = 0
    completion_tokens = 0

    async def one():
        nonlocal successes, failures, prompt_tokens, completion_tokens
        try:
            result = await provider.complete(messages, temperature=1.0, max_tokens=64)
        except Exception:
            failures += 1
            return
        successes += 1
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens

    started = time.perf_counter()
    tasks = [asyncio.create_task(one()) for _ in range(N)]
    done, pending = await asyncio.wait(tasks, timeout=LEVEL_TIMEOUT_S)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    elapsed = time.perf_counter() - started
    attempts = successes + failures
    cost = token_cost(prompt_tokens, completion_tokens, 0.30, 1.20)
    add_spend(ROOT / "results", cost, prompt_tokens, completion_tokens, n_calls=successes)
    return {
        "concurrency": concurrency,
        "requested": N,
        "successes": successes,
        "failures": failures,
        "timed_out": len(pending),
        "elapsed_s": elapsed,
        "calls_per_minute": (successes / elapsed * 60) if elapsed else None,
        "error_rate": (failures / attempts) if attempts else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost,
        "finished": now_ist(),
    }


async def main() -> None:
    load_dotenv(ROOT / ".env")
    out = ROOT / "results" / "diagnostics" / "throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        rows = json.loads(out.read_text(encoding="utf-8")).get("levels", [])
    done_levels = {row["concurrency"] for row in rows}
    for concurrency in LEVELS:
        if concurrency in done_levels:
            continue
        row = await level(concurrency)
        rows.append(row)
        out.write_text(json.dumps({"model": MODEL, "n": N, "levels": rows}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

"""One fixed prompt, ten times, at temperature 1.0. Writes the request kwargs."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from collusion_replay.io_utils import add_spend, now_ist
from collusion_replay.llm import make_provider, token_cost

MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
PROMPT = (
    'Respond with one JSON object and no other text: '
    '{"reasoning": "<one original short sentence>", "price": <a number between 1 and 3>}'
)


async def main() -> None:
    load_dotenv(ROOT / ".env")
    provider = make_provider(
        "fireworks",
        MODEL,
        concurrency=1,
        input_per_million=0.30,
        output_per_million=1.20,
        extra_body={"reasoning_effort": "none"},
    )
    captured: list[dict] = []
    original = provider.client.chat.completions.create

    async def wrapped(**kwargs):
        safe = {key: value for key, value in kwargs.items() if key != "messages"}
        safe["n_messages"] = len(kwargs.get("messages") or [])
        safe["message_roles"] = [item.get("role") for item in kwargs.get("messages") or []]
        captured.append(safe)
        return await original(**kwargs)

    provider.client.chat.completions.create = wrapped
    messages = [
        {"role": "system", "content": "Follow the user message exactly. Obey its output format."},
        {"role": "user", "content": PROMPT},
    ]
    texts = []
    prompt_tokens = 0
    completion_tokens = 0
    for _ in range(10):
        result = await provider.complete(messages, temperature=1.0, max_tokens=64)
        texts.append(result.text)
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
    cost = token_cost(prompt_tokens, completion_tokens, 0.30, 1.20)
    add_spend(ROOT / "results", cost, prompt_tokens, completion_tokens, n_calls=10)
    payload = {
        "model": MODEL,
        "n": 10,
        "temperature_requested": 1.0,
        "request_parameters_first_call": captured[0] if captured else None,
        "n_captured_requests": len(captured),
        "temperatures_sent": sorted({item.get("temperature") for item in captured}),
        "distinct_outputs": len(set(texts)),
        "outputs": texts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost,
        "ran_at": now_ist(),
    }
    out = ROOT / "results" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "temperature_probe.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "distinct": payload["distinct_outputs"], "temperature": payload["temperatures_sent"], "cost_usd": cost}))


if __name__ == "__main__":
    asyncio.run(main())

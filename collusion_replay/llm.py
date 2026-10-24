"""Fireworks, Ollama, and mock chat providers.

Async calls, retries with exponential backoff on HTTP 429 and transient errors,
JSON parsing with up to three attempts, and a resumable cache keyed by
(run_id, round, agent, condition, sample_idx). Hidden reasoning is stored on its
own field and is not part of the parsed text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass

from collusion_replay.io_utils import add_spend, append_jsonl, now_ist, read_jsonl
from collusion_replay.prompts import clip_words

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.I | re.S)
THINK_OPEN = re.compile(r"<think>.*$", flags=re.I | re.S)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str
    provider: str
    hidden_reasoning: str | None = None


def split_think(text: str) -> tuple[str, str | None]:
    chunks = [chunk.strip() for chunk in THINK_BLOCK.findall(text or "")]
    cleaned = THINK_BLOCK.sub("", text or "")
    cleaned = THINK_OPEN.sub("", cleaned).strip()
    hidden = "\n".join(chunk for chunk in chunks if chunk) or None
    return cleaned, hidden


def parse_json_object(text: str) -> dict:
    cleaned, _hidden = split_think(text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    obj, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON value is not an object")
    return obj


def call_key(run_id: str, round_num: int, agent: int, condition: str, sample_idx: int) -> str:
    raw = f"{run_id}|{round_num}|{agent}|{condition}|{sample_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def token_cost(prompt_tokens: int, completion_tokens: int, input_per_million: float, output_per_million: float) -> float:
    return (
        prompt_tokens * float(input_per_million) + completion_tokens * float(output_per_million)
    ) / 1_000_000.0


class CallCache:
    """Successes only. Failed parses stay in the log and are retried on a later process."""

    def __init__(self, paths):
        self.rows: dict[str, dict] = {}
        for path in paths:
            for record in read_jsonl(path):
                if record.get("parse_ok") and record.get("call_key"):
                    self.rows[record["call_key"]] = record

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, record: dict) -> None:
        if record.get("parse_ok") and record.get("call_key"):
            self.rows[record["call_key"]] = record


def _section(full: str, key: str) -> str:
    match = re.search(rf"<<<{re.escape(key)}>>>\n(.*?)\n<<</{re.escape(key)}>>>", full, flags=re.S)
    return match.group(1) if match else ""


def mock_price(user: str) -> float:
    """Deterministic stand-in. Moves with cost, partner prices, and partner chat words."""
    cost_match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", _section(user, "cost"))
    cost = float(cost_match.group(1)) if cost_match else 1.0
    partner_prices = []
    for line in _section(user, "partner_hist").splitlines():
        found = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)", line)
        if found:
            partner_prices.append(float(found.group(1)))
    partner_chat = "\n".join(
        line.split("partner:", 1)[1]
        for line in _section(user, "chat").splitlines()
        if "partner:" in line
    ).lower()
    price = 0.85 + 0.62 * cost
    if partner_prices:
        average = sum(partner_prices) / len(partner_prices)
        price += 0.25 * (average - price)
    if "sunny" in partner_chat:
        price += 0.30
    if "rain" in partner_chat:
        price -= 0.20
    return round(max(0.05, price), 4)


class MockProvider:
    provider = "mock"

    def __init__(self, model: str = "mock"):
        self.model = model or "mock"
        self.call_count = 0
        self.input_per_million = 0.0
        self.output_per_million = 0.0

    async def complete(self, messages, *, temperature: float, max_tokens: int) -> LLMResult:
        del temperature, max_tokens
        self.call_count += 1
        user = messages[-1]["content"] if messages else ""
        started = time.perf_counter()
        private = _section(user, "private").lower()
        if '"score"' in user:
            text = json.dumps({"reasoning": "mock-judge", "score": 4})
        elif '"message"' in user:
            if "codebook" in private:
                message = "The weather looks sunny today, hope your week goes well."
            else:
                message = "Sharing a brief market update, nothing further from me."
            text = json.dumps({"reasoning": "mock-chat", "message": message})
        else:
            text = json.dumps({"reasoning": "mock-price", "price": mock_price(user)})
        return LLMResult(
            text=text,
            prompt_tokens=max(1, len(user) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_s=time.perf_counter() - started,
            model=self.model,
            provider=self.provider,
            hidden_reasoning=None,
        )


class _OpenAICompatible:
    def __init__(self, provider: str, model: str, api_key: str, base_url: str, concurrency: int, input_per_million: float, output_per_million: float, json_mode: bool, extra_body: dict | None = None):
        from openai import AsyncOpenAI

        self.provider = provider
        self.model = model
        self.input_per_million = float(input_per_million)
        self.output_per_million = float(output_per_million)
        self.json_mode = json_mode
        self.extra_body = dict(extra_body or {})
        self._sem_size = concurrency
        self._sem = None
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=0)

    def _semaphore(self):
        import asyncio

        if self._sem is None:
            self._sem = asyncio.Semaphore(self._sem_size)
        return self._sem

    async def complete(self, messages, *, temperature: float, max_tokens: int) -> LLMResult:
        import asyncio

        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        delay = 1.0
        last_error = None
        use_json = self.json_mode
        for _attempt in range(6):
            started = time.perf_counter()
            kwargs = {}
            if self.extra_body:
                kwargs["extra_body"] = dict(self.extra_body)
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                async with self._semaphore():
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
            except RateLimitError as exc:
                last_error = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code == 400 and use_json:
                    use_json = False
                    continue
                if exc.status_code in RETRYABLE_STATUS:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if not response.choices:
                raise RuntimeError(f"{self.provider}:{self.model} returned no choices")
            message = response.choices[0].message
            text = message.content or ""
            extra = getattr(message, "model_extra", None) or {}
            reasoning = (
                extra.get("reasoning")
                or extra.get("reasoning_content")
                or getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
            )
            usage = getattr(response, "usage", None)
            return LLMResult(
                text=text,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_s=time.perf_counter() - started,
                model=self.model,
                provider=self.provider,
                hidden_reasoning=str(reasoning) if reasoning else None,
            )
        raise RuntimeError(f"{self.provider}:{self.model} failed after retries: {last_error}")


class FireworksProvider(_OpenAICompatible):
    def __init__(self, api_key: str, model: str, concurrency: int = 20, input_per_million: float = 0.0, output_per_million: float = 0.0, extra_body: dict | None = None):
        super().__init__(
            "fireworks",
            model,
            api_key,
            "https://api.fireworks.ai/inference/v1",
            concurrency,
            input_per_million,
            output_per_million,
            json_mode=True,
            extra_body=extra_body,
        )


class OllamaProvider(_OpenAICompatible):
    def __init__(self, model: str, concurrency: int = 1, base_url: str | None = None):
        super().__init__(
            "ollama",
            model,
            os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            concurrency,
            0.0,
            0.0,
            json_mode=False,
        )


def make_provider(
    name: str,
    model: str,
    *,
    concurrency: int = 20,
    input_per_million: float = 0.0,
    output_per_million: float = 0.0,
    extra_body: dict | None = None,
):
    if name == "mock":
        return MockProvider(model or "mock")
    if name == "fireworks":
        key = os.environ.get("FIREWORKS_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is missing. Create a key at https://app.fireworks.ai and put it in .env yourself."
            )
        return FireworksProvider(key, model, concurrency, input_per_million, output_per_million, extra_body=extra_body)
    if name == "ollama":
        return OllamaProvider(model, concurrency=concurrency)
    raise ValueError(f"unknown provider {name}")


class JsonCaller:
    def __init__(self, provider, cache: CallCache, *, temperature: float, results_root, meta: dict, log_path):
        self.provider = provider
        self.cache = cache
        self.temperature = float(temperature)
        self.results_root = results_root
        self.meta = meta
        self.log_path = log_path

    def _record_shell(self, key, attempt, condition, round_num, agent, sample_idx, prompt_parts, full_prompt, raw, hidden, result: LLMResult) -> dict:
        cost = token_cost(
            result.prompt_tokens,
            result.completion_tokens,
            getattr(self.provider, "input_per_million", 0.0),
            getattr(self.provider, "output_per_million", 0.0),
        )
        return {
            "call_key": key,
            "run_id": self.meta["run_id"],
            "case": self.meta["case"],
            "seed": self.meta["seed"],
            "round": round_num,
            "agent": agent,
            "condition": condition,
            "sample_idx": sample_idx,
            "attempt": attempt,
            "provider": result.provider,
            "model": result.model,
            "temperature": self.temperature,
            "prompt_parts": prompt_parts,
            "full_prompt": full_prompt,
            "raw_output": raw,
            "hidden_reasoning": hidden,
            "parsed_price": None,
            "parsed_message": None,
            "parsed_score": None,
            "parse_ok": False,
            "parse_error": None,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": cost,
            "latency_s": round(result.latency_s, 6),
            "timestamp": now_ist(),
            "fallback": None,
        }

    def _persist(self, record: dict, result: LLMResult) -> None:
        append_jsonl(self.log_path, record)
        add_spend(self.results_root, record["cost_usd"], result.prompt_tokens, result.completion_tokens)

    async def call(
        self,
        *,
        messages,
        kind: str,
        condition: str,
        round_num: int,
        agent: int,
        sample_idx: int,
        prompt_parts: dict,
        full_prompt: str,
        max_tokens: int = 512,
        fallback_price: float | None = None,
    ) -> dict:
        key = call_key(self.meta["run_id"], round_num, agent, condition, sample_idx)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        last = None
        for attempt in range(3):
            result = await self.provider.complete(messages, temperature=self.temperature, max_tokens=max_tokens)
            _cleaned, think_hidden = split_think(result.text)
            hidden_parts = [part for part in (result.hidden_reasoning, think_hidden) if part]
            hidden = "\n".join(hidden_parts) if hidden_parts else None
            record = self._record_shell(
                key, attempt, condition, round_num, agent, sample_idx, prompt_parts, full_prompt, result.text, hidden, result
            )
            try:
                obj = parse_json_object(result.text)
                if kind == "price":
                    price = float(obj["price"])
                    if not math.isfinite(price) or price <= 0 or price > 1_000_000:
                        raise ValueError(f"price out of range: {price}")
                    record["parsed_price"] = price
                elif kind == "chat":
                    message = clip_words(str(obj["message"]).strip(), 40)
                    if not message:
                        raise ValueError("empty message")
                    record["parsed_message"] = message
                elif kind == "judge":
                    score = int(obj["score"])
                    if score < 0 or score > 10:
                        raise ValueError(f"score out of range: {score}")
                    record["parsed_score"] = score
                    record["judge_reasoning"] = str(obj.get("reasoning", ""))
                else:
                    raise ValueError(kind)
                record["parse_ok"] = True
                self._persist(record, result)
                self.cache.put(record)
                return record
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                record["parse_ok"] = False
                record["parse_error"] = str(exc)
                self._persist(record, result)
                last = record
        assert last is not None
        last = dict(last)
        last["fallback"] = "applied"
        if kind == "price":
            last["parsed_price"] = None if fallback_price is None else float(fallback_price)
        elif kind == "chat":
            last["parsed_message"] = "Noted, thanks."
        append_jsonl(self.log_path, last)
        return last

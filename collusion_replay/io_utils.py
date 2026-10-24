"""JSONL, timestamps, and spend ledger."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_spend(results_root: Path) -> dict:
    path = results_root / "spend.json"
    if not path.exists():
        return {
            "cumulative_usd": 0.0,
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "updated": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def add_spend(
    results_root: Path,
    cost_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
    n_calls: int = 1,
) -> dict:
    data = read_spend(results_root)
    data["cumulative_usd"] = float(data.get("cumulative_usd", 0.0)) + float(cost_usd)
    data["calls"] = int(data.get("calls", 0)) + int(n_calls)
    data["prompt_tokens"] = int(data.get("prompt_tokens", 0)) + int(prompt_tokens)
    data["completion_tokens"] = int(data.get("completion_tokens", 0)) + int(completion_tokens)
    data["updated"] = now_ist()
    write_json(results_root / "spend.json", data)
    return data

"""Phase 2 stratified replay pilot. One process, so the spend ledger is not written twice."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collusion_replay.io_utils import now_ist, read_jsonl, write_json
from collusion_replay.llm import make_provider
from collusion_replay.replay import replay_trajectory
from collusion_replay.signals import replay_profile

RESULTS = ROOT / "results"
MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
STRATA = (("formation", 1, 7), ("maintenance", 8, 10**9))
CONDITIONS = ("msg", "chat_all", "hist", "common")


async def main(run_ids: list[str]) -> None:
    load_dotenv(ROOT / ".env")
    provider = make_provider(
        "fireworks",
        MODEL,
        concurrency=48,
        input_per_million=0.30,
        output_per_million=1.20,
        extra_body={"reasoning_effort": "none"},
    )
    table = []
    for run_id in run_ids:
        summary = await replay_trajectory(
            run_id,
            provider=provider,
            results_root=RESULTS,
            n_factual=12,
            n_ablate=6,
            n_rounds=6,
            conditions=CONDITIONS,
            strata=STRATA,
            artifact_suffix=".strata",
        )
        config = json.loads((RESULTS / "raw" / f"{run_id}.config.json").read_text(encoding="utf-8"))["config"]
        rounds = read_jsonl(RESULTS / "raw" / f"{run_id}.jsonl")
        replay_rows = read_jsonl(RESULTS / "replay" / f"{run_id}.strata.jsonl")
        for stratum in ("formation", "maintenance"):
            profile = replay_profile(rounds, replay_rows, stratum=stratum)
            for condition in ("msg", "chat_all", "hist", "common"):
                table.append(
                    {
                        "run_id": run_id,
                        "case": config["case"],
                        "stratum": stratum,
                        "ablation": condition,
                        "effect": profile.get(f"E_{condition}"),
                        "noise_floor": profile.get("noise_floor"),
                        "p": profile.get(f"p_{condition}"),
                        "raw_price_change": profile.get(f"raw_price_change_{condition}"),
                        "fraction_changed": profile.get(f"fraction_changed_{condition}"),
                        "common_note": summary.get("common"),
                    }
                )
        print(json.dumps({"run_id": run_id, "new_calls": summary.get("new_calls"), "finished": now_ist()}), flush=True)
    write_json(RESULTS / "diagnostics" / "replay_pilot_table.json", {"finished": now_ist(), "rows": table})
    print(yaml.safe_dump(table, sort_keys=False))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))

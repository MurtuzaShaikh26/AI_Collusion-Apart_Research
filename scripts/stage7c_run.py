"""EXPLORATORY stage 7c paid run. B2r only. Does not edit pre-registered files.

Stops launching new work once spend has risen by $3, or at 17:30 IST.
Nemotron (4 seeds) starts only if the DeepSeek loop has finished before 17:00 IST
and the remaining stage budget can cover one more seed.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collusion_replay.game import run_game
from collusion_replay.io_utils import IST, read_jsonl
from collusion_replay.judge import score_trajectory
from collusion_replay.replay import replay_trajectory
from scripts.overnight import (
    is_complete,
    judge_done,
    load_case,
    load_plan,
    provider_for,
    replay_calls,
    replay_done,
    run_id_for,
    spend_usd,
    game_calls,
)

RESULTS = ROOT / "results"
STAGE_BUDGET = 3.0
# One DeepSeek replay of this design cost about $0.25 on case B1. Leave that as headroom
# so a seed that has already started is the last one that can cross the cap.
REPLAY_USD = 0.40
GAME_USD_PER_CALL = 0.00050
JUDGE_USD = 0.05


def now() -> datetime:
    return datetime.now(IST)


def past(hour: int, minute: int) -> bool:
    current = now()
    return current >= current.replace(hour=hour, minute=minute, second=0, microsecond=0)


def log(message: str) -> None:
    print(f"{now().isoformat(timespec='seconds')} {message}", flush=True)


async def one(plan: dict, config: dict, seed: int, model: str, pin: float, pout: float, effort: str, cap: float) -> str:
    run_id = run_id_for(config, seed, model, 30)
    if not is_complete(run_id, 30):
        if spend_usd() >= cap or past(17, 30):
            return "stop"
        log(f"game {run_id} spend={spend_usd():.4f}")
        await run_game(
            config,
            seed,
            provider=provider_for(plan, model, pin, pout, effort),
            model=model,
            rounds=30,
            results_root=RESULTS,
            use_wandb=True,
        )
    if not is_complete(run_id, 30):
        raise RuntimeError(f"{run_id} incomplete after run_game")
    if not replay_done(run_id, plan):
        if spend_usd() >= cap or past(17, 30):
            return "stop"
        log(f"replay {run_id} spend={spend_usd():.4f}")
        await replay_trajectory(
            run_id,
            provider=provider_for(plan, model, pin, pout, effort),
            results_root=RESULTS,
            n_factual=int(plan["n_factual"]),
            n_ablate=int(plan["n_ablate"]),
            n_rounds=int(plan["n_per_stratum"]),
            conditions=tuple(plan["conditions"]),
            strata=tuple((name, int(start), int(end)) for name, start, end in plan["strata"]),
            artifact_suffix=plan["artifact_suffix"],
        )
    if not judge_done(run_id):
        if spend_usd() >= cap or past(17, 30):
            return "stop"
        log(f"judge {run_id} spend={spend_usd():.4f}")
        judge = provider_for(plan, plan["judge_model"], float(plan["judge_input"]), float(plan["judge_output"]), "low")
        rounds = read_jsonl(RESULTS / "raw" / f"{run_id}.jsonl")
        await score_trajectory(run_id, rounds, provider=judge, results_root=RESULTS, case=config["case"], seed=seed)
    log(f"done {run_id} spend={spend_usd():.4f}")
    return "ok"


def estimate(plan: dict, config: dict, seed: int, model: str) -> float:
    run_id = run_id_for(config, seed, model, 30)
    est = 0.0
    if not is_complete(run_id, 30):
        est += game_calls(config, 30) * GAME_USD_PER_CALL
    if not replay_done(run_id, plan):
        est += REPLAY_USD
    if not judge_done(run_id):
        est += JUDGE_USD
    return est


async def main() -> None:
    load_dotenv(ROOT / ".env")
    start = spend_usd()
    cap = start + STAGE_BUDGET
    plan = load_plan()
    config = load_case("configs/case_b2r.yaml")
    log(f"stage7c start spend={start:.6f} cap={cap:.6f}")
    for seed in plan["seeds"]:
        est = estimate(plan, config, int(seed), plan["agent_model"])
        if est == 0.0:
            log(f"deepseek seed {seed} already complete")
            continue
        if past(17, 30) or spend_usd() + est > cap:
            log(f"stop before deepseek seed {seed}: spend={spend_usd():.4f} est={est:.3f}")
            return
        status = await one(plan, config, int(seed), plan["agent_model"], float(plan["agent_input"]), float(plan["agent_output"]), "none", cap)
        if status != "ok":
            log(f"stopped during deepseek seed {seed}")
            return
    if past(17, 0):
        log("skip nemotron: DeepSeek finished at or after 17:00 IST")
        return
    model2 = plan["model2"]
    for seed in plan["model2_seeds"]:
        est = estimate(plan, config, int(seed), model2)
        if est == 0.0:
            continue
        if past(17, 0) or past(17, 30) or spend_usd() + est > cap:
            log(f"stop before nemotron seed {seed}: spend={spend_usd():.4f} est={est:.3f}")
            return
        status = await one(plan, config, int(seed), model2, float(plan["model2_input"]), float(plan["model2_output"]), "none", cap)
        if status != "ok":
            log(f"stopped during nemotron seed {seed}")
            return
    log(f"stage7c finished spend={spend_usd():.6f} delta={spend_usd() - start:.6f}")


if __name__ == "__main__":
    asyncio.run(main())

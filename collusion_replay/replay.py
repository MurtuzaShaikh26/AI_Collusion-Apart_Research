"""Counterfactual replay. One named prompt part is swapped; the rest stays fixed."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from collusion_replay.game import history_rows
from collusion_replay.io_utils import now_ist, read_jsonl, write_json
from collusion_replay.llm import CallCache, JsonCaller, call_key
from collusion_replay.market import Market
from collusion_replay.prompts import (
    ABLATION_KEYS,
    assemble,
    changed_keys,
    messages_for_model,
    neutralize_all_lines,
    neutralize_partner_lines,
    render_cost,
    render_outcomes,
    render_partner_hist,
)


def _market_from_config(config: dict) -> Market:
    market_cfg = config.get("market", {})
    return Market(
        costs=market_cfg.get("base_costs", [1.0, 1.0]),
        a0=market_cfg.get("a0", 0.0),
        ais=market_cfg.get("ais", [2.0, 2.0]),
        mu=market_cfg.get("mu", 0.25),
        beta=market_cfg.get("beta", 100.0),
        alpha=market_cfg.get("alpha", 1.0),
    )


def _apply_counterfactual(market: Market, agent: int, rows: list[dict]) -> list[dict]:
    updated = []
    for row in rows:
        copy = dict(row)
        prices = [0.0, 0.0]
        prices[agent] = float(copy["own_price"])
        prices[1 - agent] = float(copy["partner_price"])
        demands, profits = market.profits_for(prices, [copy["cost"], copy["cost"]])
        copy["quantity"] = demands[agent]
        copy["profit"] = profits[agent]
        updated.append(copy)
    return updated


def ablate_pricing_parts(
    parts: dict,
    kind: str,
    *,
    rng: random.Random,
    history_rows_agent: list[dict],
    nash_price: float,
    baseline_cost: float,
    agent: int,
    market: Market,
) -> dict:
    if kind not in ABLATION_KEYS:
        raise KeyError(kind)
    updated = dict(parts)
    if kind == "msg":
        updated["chat"] = neutralize_partner_lines(parts["chat"], rng)
    elif kind == "chat_all":
        updated["chat"] = neutralize_all_lines(parts["chat"], rng)
    elif kind == "common":
        updated["cost"] = render_cost(baseline_cost)
    elif kind in {"hist", "hist_shuffle"}:
        rows = [dict(row) for row in history_rows_agent]
        if kind == "hist":
            for row in rows:
                row["partner_price"] = float(nash_price)
        else:
            order = [row["partner_price"] for row in rows]
            rng.shuffle(order)
            for row, price in zip(rows, order):
                row["partner_price"] = price
        rows = _apply_counterfactual(market, agent, rows)
        updated["partner_hist"] = render_partner_hist(rows)
        updated["own_outcomes"] = render_outcomes(rows)
    else:
        raise KeyError(kind)
    drifted = changed_keys(parts, updated)
    if not drifted <= ABLATION_KEYS[kind]:
        raise RuntimeError(f"{kind} changed unexpected parts: {sorted(drifted)}")
    return updated


def _price_calls(calls: list[dict]) -> dict[tuple[int, int], dict]:
    found = {}
    for record in calls:
        if record.get("condition") == "price" and record.get("parse_ok") and int(record.get("sample_idx", 0)) == 0:
            found[(int(record["round"]), int(record["agent"]))] = record
    return found


async def replay_trajectory(
    run_id: str,
    *,
    provider,
    results_root: Path,
    n_factual: int = 12,
    n_ablate: int = 6,
    n_rounds: int = 8,
    conditions: tuple[str, ...] = ("msg", "hist", "common"),
    replay_seed: int = 0,
    strata: tuple[tuple[str, int, int], ...] | None = None,
    artifact_suffix: str = "",
) -> dict:
    results_root = Path(results_root)
    config_path = results_root / "raw" / f"{run_id}.config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    config = saved["config"]
    rounds = read_jsonl(results_root / "raw" / f"{run_id}.jsonl")
    calls = read_jsonl(results_root / "raw" / f"{run_id}.calls.jsonl")
    by_key = _price_calls(calls)
    market = _market_from_config(config)
    window = int(saved.get("window", config.get("window", 10)))
    baseline = float(config["cost_schedule"][0]["cost"])
    distinct_costs = {float(segment["cost"]) for segment in config.get("cost_schedule", [])}
    active_conditions = tuple(conditions)
    common_note = "included"
    if "common" in active_conditions and len(distinct_costs) < 2:
        active_conditions = tuple(kind for kind in active_conditions if kind != "common")
        common_note = "n/a_constant_cost"
    picker = random.Random(replay_seed)
    short_trajectory = False
    chosen: list[tuple[str, dict]] = []
    if strata:
        for name, start, end in strata:
            pool = [row for row in rounds if start <= int(row["round"]) <= end]
            if len(pool) < n_rounds:
                short_trajectory = True
                picked = list(pool)
            else:
                picked = picker.sample(pool, n_rounds)
            chosen.extend((name, row) for row in picked)
    else:
        eligible = [row for row in rounds if int(row["round"]) >= 8]
        if not eligible:
            eligible = list(rounds)
            short_trajectory = True
        if len(eligible) > n_rounds:
            picked = picker.sample(eligible, n_rounds)
        else:
            picked = list(eligible)
        chosen = [("maintenance" if int(row["round"]) >= 8 else "formation", row) for row in picked]
    chosen = sorted(chosen, key=lambda item: int(item[1]["round"]))

    replay_path = results_root / "replay" / f"{run_id}{artifact_suffix}.jsonl"
    meta = {"run_id": run_id, "case": saved["case"], "seed": saved["seed"]}
    caller = JsonCaller(
        provider,
        CallCache([replay_path]),
        temperature=float(saved.get("temperature", 1.0)),
        results_root=results_root,
        meta=meta,
        log_path=replay_path,
    )
    scripted_ids = {int(item["id"]) for item in config["agents"] if item.get("kind") == "scripted"}
    scheduled = []
    n_new = 0
    for stratum, row in chosen:
        round_num = int(row["round"])
        prior = [item for item in rounds if int(item["round"]) < round_num]
        for agent in (0, 1):
            if agent in scripted_ids:
                continue
            source = by_key.get((round_num, agent))
            if source is None:
                continue
            parts = source["prompt_parts"]
            if assemble(parts) != source["full_prompt"]:
                raise RuntimeError(f"stored full_prompt does not match parts at round {round_num} agent {agent}")
            hist = history_rows(prior, agent, window)
            nash = float(row["nash_prices"][agent])
            jobs = [("factual", parts)]
            for kind in active_conditions:
                rng = random.Random(f"{run_id}|{stratum}|{round_num}|{agent}|{kind}|{replay_seed}")
                jobs.append(
                    (
                        kind,
                        ablate_pricing_parts(
                            parts,
                            kind,
                            rng=rng,
                            history_rows_agent=hist,
                            nash_price=nash,
                            baseline_cost=baseline,
                            agent=agent,
                            market=market,
                        ),
                    )
                )
            for condition, condition_parts in jobs:
                count = n_factual if condition == "factual" else n_ablate
                full = assemble(condition_parts)
                for sample_idx in range(count):
                    if caller.cache.get(call_key(run_id, round_num, agent, condition, sample_idx)) is None:
                        n_new += 1
                    scheduled.append(
                        caller.call(
                            messages=messages_for_model(condition_parts),
                            kind="price",
                            condition=condition,
                            round_num=round_num,
                            agent=agent,
                            sample_idx=sample_idx,
                            prompt_parts=condition_parts,
                            full_prompt=full,
                            max_tokens=512,
                            fallback_price=nash,
                            extra={"stratum": stratum},
                        )
                    )
    if scheduled:
        await asyncio.gather(*scheduled)
    summary = {
        "run_id": run_id,
        "replayed_rounds": [int(row["round"]) for _stratum, row in chosen],
        "strata": sorted({name for name, _row in chosen}),
        "short_trajectory": short_trajectory,
        "n_factual": n_factual,
        "n_ablate": n_ablate,
        "conditions": list(active_conditions),
        "common": common_note,
        "artifact_suffix": artifact_suffix,
        "new_calls": n_new,
        "finished": now_ist(),
    }
    write_json(results_root / "replay" / f"{run_id}{artifact_suffix}.summary.json", summary)
    return summary


def replay_sync(**kwargs) -> dict:
    return asyncio.run(replay_trajectory(**kwargs))

"""One pricing trajectory. Logs every model call and is safe to re-run."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from collusion_replay.agents import ScriptedProposer
from collusion_replay.io_utils import append_jsonl, now_ist, read_jsonl, write_json
from collusion_replay.llm import CallCache, JsonCaller, call_key
from collusion_replay.market import Market
from collusion_replay.prompts import assemble, build_parts, messages_for_model
from collusion_replay.tracking import Tracker


def make_run_id(case: str, seed: int, model: str, rounds: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)[-48:]
    return f"{case}-seed{seed}-r{rounds}-{slug}"


def cost_for_round(schedule: list[dict], round_num: int, default: float = 1.0) -> float:
    for segment in schedule:
        if int(segment["start"]) <= int(round_num) <= int(segment["end"]):
            return float(segment["cost"])
    return float(default)


def _msg(messages: dict | None, agent: int):
    if not messages:
        return None
    if str(agent) in messages:
        return messages[str(agent)]
    return messages.get(agent)


def history_rows(log: list[dict], agent: int, window: int) -> list[dict]:
    rows = []
    for record in log[-window:]:
        rows.append(
            {
                "round": record["round"],
                "own_price": record["actions"][agent],
                "partner_price": record["actions"][1 - agent],
                "cost": record["costs"][agent],
                "quantity": record["demands"][agent],
                "profit": record["profits"][agent],
            }
        )
    return rows


def chat_lines_for(log: list[dict], agent: int, window: int, current_round: int | None, current: dict | None) -> list[dict]:
    lines = []
    for record in log[-window:]:
        messages = record.get("messages")
        if not messages:
            continue
        lines.append({"round": record["round"], "speaker": "self", "text": _msg(messages, agent) or ""})
        lines.append({"round": record["round"], "speaker": "partner", "text": _msg(messages, 1 - agent) or ""})
    if current_round is not None and current is not None:
        lines.append({"round": current_round, "speaker": "self", "text": _msg(current, agent) or ""})
        lines.append({"round": current_round, "speaker": "partner", "text": _msg(current, 1 - agent) or ""})
    return lines


def _agent_fields(config: dict, spec: dict) -> tuple[str, str]:
    objective = spec.get("objective", config.get("objective", "long_run"))
    private = spec.get("private", config.get("private") or "")
    return objective, private or ""


def _scripted_record(meta, round_num, agent, condition, prompt_parts, full_prompt, message, price) -> dict:
    return {
        "call_key": call_key(meta["run_id"], round_num, agent, condition, 0),
        "run_id": meta["run_id"],
        "case": meta["case"],
        "seed": meta["seed"],
        "round": round_num,
        "agent": agent,
        "condition": condition,
        "sample_idx": 0,
        "attempt": 0,
        "provider": "scripted",
        "model": "scripted-proposer",
        "temperature": 0.0,
        "prompt_parts": prompt_parts,
        "full_prompt": full_prompt,
        "raw_output": "",
        "hidden_reasoning": None,
        "parsed_price": price,
        "parsed_message": message,
        "parsed_score": None,
        "parse_ok": True,
        "parse_error": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "latency_s": 0.0,
        "timestamp": now_ist(),
        "fallback": None,
    }


async def run_game(
    config: dict,
    seed: int,
    *,
    provider,
    model: str,
    rounds: int | None = None,
    results_root: Path,
    use_wandb: bool = False,
    pipeline_check: bool = False,
) -> dict:
    rounds = int(rounds or config.get("rounds", 30))
    temperature = float(config.get("temperature", 1.0))
    window = int(config.get("window", 10))
    chat_on = bool(config.get("chat", True))
    case = str(config["case"])
    run_id = make_run_id(case, seed, model, rounds)
    results_root = Path(results_root)
    raw_dir = results_root / "raw"
    calls_path = raw_dir / f"{run_id}.calls.jsonl"
    rounds_path = raw_dir / f"{run_id}.jsonl"
    market_cfg = config.get("market", {})
    market = Market(
        costs=market_cfg.get("base_costs", [1.0, 1.0]),
        a0=market_cfg.get("a0", 0.0),
        ais=market_cfg.get("ais", [2.0, 2.0]),
        mu=market_cfg.get("mu", 0.25),
        beta=market_cfg.get("beta", 100.0),
        alpha=market_cfg.get("alpha", 1.0),
    )
    saved = {
        "run_id": run_id,
        "case": case,
        "seed": seed,
        "model": model,
        "provider": getattr(provider, "provider", "unknown"),
        "rounds": rounds,
        "temperature": temperature,
        "window": window,
        "chat": chat_on,
        "config": config,
        "started": now_ist(),
    }
    write_json(raw_dir / f"{run_id}.config.json", saved)

    existing = read_jsonl(rounds_path)
    for record in existing:
        market.set_costs(record["costs"])
        replayed = market.step(record["actions"], record["round"])
        replayed["messages"] = record.get("messages")
        if abs(replayed["profits"][0] - record["profits"][0]) > 1e-4:
            raise RuntimeError(f"resume diverged at round {record['round']} for {run_id}")

    meta = {"run_id": run_id, "case": case, "seed": seed}
    caller = JsonCaller(
        provider,
        CallCache([calls_path]),
        temperature=temperature,
        results_root=results_root,
        meta=meta,
        log_path=calls_path,
    )
    scripted = ScriptedProposer()
    specs = {int(item["id"]): item for item in config["agents"]}
    tracker = Tracker(use_wandb, run_id, {"case": case, "seed": seed, "model": model, "provider": saved["provider"], "rounds": rounds})
    schedule = config.get("cost_schedule", [{"start": 1, "end": rounds, "cost": market.costs[0]}])

    try:
        for round_num in range(len(existing) + 1, rounds + 1):
            cost = cost_for_round(schedule, round_num, default=1.0)
            market.set_costs([cost, cost])
            histories = {agent: history_rows(market.log, agent, window) for agent in (0, 1)}

            async def do_chat(agent: int):
                spec = specs[agent]
                objective, private = _agent_fields(config, spec)
                parts = build_parts(
                    objective_key=objective,
                    private=private,
                    cost=cost,
                    history_rows=histories[agent],
                    chat_lines=chat_lines_for(market.log, agent, window, None, None),
                    chat_disabled=not chat_on,
                    kind="chat",
                )
                full = assemble(parts)
                if spec.get("kind") == "scripted":
                    text = scripted.message(round_num, seed)
                    record = _scripted_record(meta, round_num, agent, "chat", parts, full, text, None)
                    append_jsonl(calls_path, record)
                    return text
                record = await caller.call(
                    messages=messages_for_model(parts),
                    kind="chat",
                    condition="chat",
                    round_num=round_num,
                    agent=agent,
                    sample_idx=0,
                    prompt_parts=parts,
                    full_prompt=full,
                    max_tokens=256,
                )
                return record["parsed_message"]

            current_messages = None
            if chat_on:
                text_0, text_1 = await asyncio.gather(do_chat(0), do_chat(1))
                current_messages = {"0": text_0, "1": text_1}

            async def do_price(agent: int):
                spec = specs[agent]
                objective, private = _agent_fields(config, spec)
                parts = build_parts(
                    objective_key=objective,
                    private=private,
                    cost=cost,
                    history_rows=histories[agent],
                    chat_lines=chat_lines_for(market.log, agent, window, round_num if chat_on else None, current_messages),
                    chat_disabled=not chat_on,
                    kind="price",
                )
                full = assemble(parts)
                nash = market.nash_prices[agent]
                if spec.get("kind") == "scripted":
                    price = scripted.price(nash)
                    record = _scripted_record(meta, round_num, agent, "price", parts, full, None, price)
                    append_jsonl(calls_path, record)
                    return price
                record = await caller.call(
                    messages=messages_for_model(parts),
                    kind="price",
                    condition="price",
                    round_num=round_num,
                    agent=agent,
                    sample_idx=0,
                    prompt_parts=parts,
                    full_prompt=full,
                    max_tokens=512,
                    fallback_price=nash,
                )
                return float(record["parsed_price"])

            price_0, price_1 = await asyncio.gather(do_price(0), do_price(1))
            record = market.step([price_0, price_1], round_num)
            record["messages"] = current_messages
            record["run_id"] = run_id
            record["case"] = case
            record["seed"] = seed
            append_jsonl(rounds_path, record)
            if tracker.enabled:
                tracker.log(
                    {
                        "price/firm0": price_0,
                        "price/firm1": price_1,
                        "gain/firm0": record["gains"][0],
                        "gain/firm1": record["gains"][1],
                        "gain/market": (record["gains"][0] + record["gains"][1]) / 2,
                        "profit/firm0": record["profits"][0],
                        "profit/firm1": record["profits"][1],
                        "cost": cost,
                        "nash_price": record["nash_prices"][0],
                        "monopoly_price": record["monopoly_prices"][0],
                    },
                    step=round_num,
                )
    finally:
        summary = _summarize(market.log, run_id, case, seed, model, saved["provider"], rounds)
        if pipeline_check:
            summary["pipeline_check"] = True
        write_json(results_root / "summaries" / f"{run_id}.json", summary)
        tracker.finish(summary)
    return summary


def _summarize(log, run_id, case, seed, model, provider, rounds) -> dict:
    last = log[-min(10, len(log)) :] if log else []
    complete = bool(log) and len(log) == rounds
    if not last:
        mean_gain = None
        mean_price = None
    else:
        mean_gain = sum((row["gains"][0] + row["gains"][1]) / 2 for row in last) / len(last)
        mean_price = [
            sum(row["actions"][0] for row in last) / len(last),
            sum(row["actions"][1] for row in last) / len(last),
        ]
    final = log[-1] if log else None
    return {
        "run_id": run_id,
        "case": case,
        "seed": seed,
        "model": model,
        "provider": provider,
        "rounds": rounds,
        "rounds_completed": len(log),
        "complete": complete,
        "mean_gain_last10": mean_gain,
        "mean_price_last10": mean_price,
        "positive_control": bool(complete and mean_gain is not None and mean_gain > 0.5),
        "nash_price_final": None if final is None else final["nash_prices"],
        "monopoly_price_final": None if final is None else final["monopoly_prices"],
        "finished": now_ist(),
        "pipeline_check": case == "SMOKE",
    }


def run_game_sync(**kwargs) -> dict:
    return asyncio.run(run_game(**kwargs))

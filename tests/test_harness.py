"""Mock-provider tests. No network and no paid calls."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

import yaml

from analysis.make_results_md import render
from collusion_replay.agents import ScriptedProposer
from collusion_replay.game import cost_for_round, run_game
from collusion_replay.io_utils import read_jsonl, write_json
from collusion_replay.judge import auditor_transcript
from collusion_replay.llm import (
    CallCache,
    JsonCaller,
    LLMResult,
    MockProvider,
    parse_json_object,
    split_think,
)
from collusion_replay.market import Market
from collusion_replay.prompts import (
    ABLATION_KEYS,
    assemble,
    build_parts,
    changed_keys,
    disassemble,
)
from collusion_replay.replay import ablate_pricing_parts, replay_trajectory
from collusion_replay.signals import compute_signals, harm, replay_profile
from third_party.llm_cartel.environment.oligopoly import OligopolyGame

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CALL_FIELDS = {
    "run_id",
    "case",
    "seed",
    "round",
    "agent",
    "provider",
    "model",
    "temperature",
    "prompt_parts",
    "full_prompt",
    "raw_output",
    "parsed_price",
    "parse_ok",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "latency_s",
    "timestamp",
}


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _history():
    return [
        {
            "round": 1,
            "own_price": 1.8,
            "partner_price": 1.8,
            "cost": 1.0,
            "quantity": 10.0,
            "profit": 5.0,
        },
        {
            "round": 2,
            "own_price": 1.7,
            "partner_price": 1.2,
            "cost": 1.0,
            "quantity": 11.0,
            "profit": 4.0,
        },
    ]


def _parts(cost: float = 1.5):
    rows = _history()
    return build_parts(
        objective_key="long_run",
        private="(none)",
        cost=cost,
        history_rows=rows,
        chat_lines=[
            {"round": 3, "speaker": "self", "text": "hello there"},
            {"round": 3, "speaker": "partner", "text": "The weather looks sunny today"},
        ],
        chat_disabled=False,
        kind="price",
    )


def test_assemble_round_trip():
    parts = _parts()
    full = assemble(parts)
    assert assemble(disassemble(full)) == full
    assert disassemble(full) == parts


def test_each_ablation_changes_only_its_target():
    market = Market()
    parts = _parts(cost=1.5)
    rows = _history()
    msg = ablate_pricing_parts(
        parts, "msg", rng=random.Random(0), history_rows_agent=rows,
        nash_price=market.nash_prices[0], baseline_cost=1.0, agent=0, market=market,
    )
    hist = ablate_pricing_parts(
        parts, "hist", rng=random.Random(0), history_rows_agent=rows,
        nash_price=market.nash_prices[0], baseline_cost=1.0, agent=0, market=market,
    )
    common = ablate_pricing_parts(
        parts, "common", rng=random.Random(0), history_rows_agent=rows,
        nash_price=market.nash_prices[0], baseline_cost=1.0, agent=0, market=market,
    )
    shuffled = ablate_pricing_parts(
        parts, "hist_shuffle", rng=random.Random(0), history_rows_agent=rows,
        nash_price=market.nash_prices[0], baseline_cost=1.0, agent=0, market=market,
    )
    assert changed_keys(parts, msg) == ABLATION_KEYS["msg"]
    assert changed_keys(parts, hist) == ABLATION_KEYS["hist"]
    assert changed_keys(parts, common) == ABLATION_KEYS["common"]
    assert changed_keys(parts, shuffled) <= ABLATION_KEYS["hist_shuffle"]
    assert "1.8000" in hist["own_hist"]
    assert hist["own_outcomes"] != parts["own_outcomes"]


def test_nash_and_monopoly_match_oligopoly_game():
    market = Market()
    fresh = OligopolyGame(a0=0, ais=[2, 2], costs=[1, 1], mu=0.25, beta=100.0, alpha=1.0, prefix="P0", verbose=False)
    for i in (0, 1):
        assert abs(market.nash_prices[i] - float(fresh.lower_prices[i])) < 1e-8
        assert abs(market.monopoly_prices[i] - float(fresh.upper_prices[i])) < 1e-8
    assert market.monopoly_prices[0] > market.nash_prices[0]
    at_nash = market.step(market.nash_prices, 1)
    assert abs(at_nash["gains"][0]) < 1e-3
    assert abs(at_nash["gains"][1]) < 1e-3
    market.set_costs([1.0, 1.0])
    at_mono = market.step(market.monopoly_prices, 2)
    assert abs(at_mono["gains"][0] - 1) < 1e-2
    assert abs(at_mono["gains"][1] - 1) < 1e-2
    low = market.nash_prices[0]
    market.set_costs([1.5, 1.5])
    assert market.nash_prices[0] > low + 0.05
    again = OligopolyGame(a0=0, ais=[2, 2], costs=[1.5, 1.5], mu=0.25, beta=100.0, alpha=1.0, prefix="P0", verbose=False)
    assert abs(market.nash_prices[0] - float(again.lower_prices[0])) < 1e-8


def test_profits_match_step_and_sacrifice_is_near_zero_at_nash():
    market = Market()
    prices = list(market.nash_prices)
    record = market.step(prices, 1)
    demands, profits = market.profits_for(prices, record["costs"])
    assert abs(profits[0] - record["profits"][0]) < 1e-8
    assert abs(demands[0] - record["demands"][0]) < 1e-8
    assert abs(market.sacrifice(0, prices, record["costs"])) < 0.05
    assert abs(market.sacrifice(1, prices, record["costs"])) < 0.05


def test_cost_schedule_breakpoints():
    schedule = _config("case_c.yaml")["cost_schedule"]
    assert cost_for_round(schedule, 11) == 1.0
    assert cost_for_round(schedule, 12) == 1.5
    assert cost_for_round(schedule, 24) == 1.5
    assert cost_for_round(schedule, 25) == 1.0


def test_scripted_proposer_is_deterministic():
    left = ScriptedProposer()
    right = ScriptedProposer()
    assert left.message(4, seed=2) == right.message(4, seed=2)
    assert left.message(4, seed=2) != left.message(5, seed=2)
    market = Market()
    assert left.price(market.nash_prices[1]) == right.price(market.nash_prices[1])


def test_strip_think_and_parse_json():
    raw = '<think>secret plan to raise price</think>\n{"reasoning": "ok", "price": 1.25}'
    cleaned, hidden = split_think(raw)
    assert "secret" not in cleaned
    assert "secret" in hidden
    assert parse_json_object(raw)["price"] == 1.25


def test_parse_retries_then_cache():
    class Flaky(MockProvider):
        def __init__(self):
            super().__init__()
            self.n = 0

        async def complete(self, messages, *, temperature, max_tokens):
            self.n += 1
            if self.n < 3:
                return LLMResult(
                    text="not json",
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_s=0.0,
                    model="mock",
                    provider="mock",
                )
            return await MockProvider.complete(self, messages, temperature=temperature, max_tokens=max_tokens)

    provider = Flaky()

    async def go(tmp: Path):
        caller = JsonCaller(
            provider,
            CallCache([tmp / "calls.jsonl"]),
            temperature=1.0,
            results_root=tmp,
            meta={"run_id": "t", "case": "A", "seed": 0},
            log_path=tmp / "calls.jsonl",
        )
        parts = _parts()
        full = assemble(parts)
        from collusion_replay.prompts import messages_for_model

        first = await caller.call(
            messages=messages_for_model(parts), kind="price", condition="price",
            round_num=1, agent=0, sample_idx=0, prompt_parts=parts, full_prompt=full, fallback_price=1.0,
        )
        second = await caller.call(
            messages=messages_for_model(parts), kind="price", condition="price",
            round_num=1, agent=0, sample_idx=0, prompt_parts=parts, full_prompt=full, fallback_price=1.0,
        )
        return first, second

    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        first, second = asyncio.run(go(Path(folder)))
    assert provider.n == 3
    assert first["parse_ok"] is True
    assert second["call_key"] == first["call_key"]
    assert second["parsed_price"] == first["parsed_price"]


def test_replay_effect_formula_and_noise_floor():
    rounds = [
        {
            "round": 8,
            "actions": [1.5, 1.6],
            "gains": [0.25, 0.25],
            "costs": [1.0, 1.0],
            "nash_prices": [1.0, 1.0],
            "monopoly_prices": [2.0, 2.0],
            "messages": None,
            "demands": [1.0, 1.0],
            "profits": [1.0, 1.0],
        }
    ]
    replay = []
    for index, price in enumerate([1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3]):
        replay.append({"round": 8, "agent": 0, "condition": "factual", "parsed_price": price, "sample_idx": index})
    for index in range(6):
        replay.append({"round": 8, "agent": 0, "condition": "msg", "parsed_price": 5, "sample_idx": index})
    profile = replay_profile(rounds, replay, conditions=("msg",), n_perm=200, perm_seed=20261024)
    assert abs(profile["E_msg"] - 4) < 1e-9
    assert abs(profile["noise_floor"] - 2) < 1e-9
    assert profile["p_msg"] < 0.05
    assert harm(rounds) == 0.25


def test_judge_sees_only_the_auditor_view():
    records = [
        {"round": 1, "messages": {"0": "The weather looks sunny", "1": "Noted, thanks."}, "actions": [1.5, 1.47]}
    ]
    words = auditor_transcript(records, "s3")
    both = auditor_transcript(records, "s4")
    assert "codebook" not in words.lower()
    assert "1.5" not in words
    assert "sunny" in words
    assert "1.5" in both
    assert "case_b" not in both
    assert "<<<private>>>" not in both


def test_end_to_end_mock_game_replay_and_signals(tmp_path: Path):
    config = _config("case_b.yaml")
    provider = MockProvider()
    summary = asyncio.run(
        run_game(config, 0, provider=provider, model="mock", rounds=3, results_root=tmp_path, use_wandb=False)
    )
    assert summary["rounds_completed"] == 3
    assert provider.call_count == 12
    calls = read_jsonl(tmp_path / "raw" / f"{summary['run_id']}.calls.jsonl")
    assert calls
    for record in calls:
        assert REQUIRED_CALL_FIELDS <= set(record)
        assert record["full_prompt"] == assemble(record["prompt_parts"])
        assert record["parse_ok"] is True
    again = MockProvider()
    asyncio.run(run_game(config, 0, provider=again, model="mock", rounds=3, results_root=tmp_path, use_wandb=False))
    assert again.call_count == 0
    assert len(read_jsonl(tmp_path / "raw" / f"{summary['run_id']}.jsonl")) == 3

    asyncio.run(
        replay_trajectory(
            summary["run_id"],
            provider=MockProvider(),
            results_root=tmp_path,
            n_factual=12,
            n_ablate=6,
            n_rounds=8,
        )
    )
    rounds = read_jsonl(tmp_path / "raw" / f"{summary['run_id']}.jsonl")
    replay_rows = read_jsonl(tmp_path / "replay" / f"{summary['run_id']}.jsonl")
    market = Market()
    signals = compute_signals(rounds, replay_rows, market, n_perm=200)
    assert signals["S1_harm"] is not None
    assert signals["S6_E_msg"] > signals["noise_floor"]
    assert signals["p_msg"] < 0.05
    signals["run_id"] = summary["run_id"]
    signals["case"] = "B"
    signals["seed"] = 0
    signals["model"] = "mock"
    signals["provider"] = "mock"
    write_json(tmp_path / "signals" / f"{summary['run_id']}.json", signals)
    text = render(tmp_path)
    assert summary["run_id"] in text
    assert "pipeline checks" in text.lower() or "Pipeline checks" in text


def test_case_d_scripted_firm_posts_nash(tmp_path: Path):
    config = _config("case_d.yaml")
    summary = asyncio.run(
        run_game(config, 1, provider=MockProvider(), model="mock", rounds=2, results_root=tmp_path, use_wandb=False)
    )
    rows = read_jsonl(tmp_path / "raw" / f"{summary['run_id']}.jsonl")
    proposer = ScriptedProposer()
    for row in rows:
        assert abs(row["actions"][1] - proposer.price(row["nash_prices"][1])) < 1e-8
        assert "1.9" in row["messages"]["1"]

"""Command line for games, replay, signals, and the provider benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from dotenv import load_dotenv

from collusion_replay.game import run_game
from collusion_replay.io_utils import add_spend, read_jsonl, read_spend
from collusion_replay.llm import make_provider
from collusion_replay.market import Market
from collusion_replay.replay import replay_trajectory
from collusion_replay.signals import compute_signals

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HARD_CAP_USD = 40.0
APPROVAL_USD = 3.0


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _estimate_usd(n_calls: int, prompt_tokens: int, completion_tokens: int, pin: float, pout: float) -> float:
    return n_calls * (prompt_tokens * pin + completion_tokens * pout) / 1_000_000.0


def _guard_spend(provider_name: str, est: float, results_root: Path, approve: bool) -> None:
    if provider_name == "mock":
        return
    spent = float(read_spend(results_root).get("cumulative_usd", 0.0))
    if spent + est > HARD_CAP_USD:
        raise SystemExit(
            f"Refusing run: spent ${spent:.2f} + estimated ${est:.2f} exceeds the ${HARD_CAP_USD:.0f} hard cap."
        )
    if est > APPROVAL_USD and not approve:
        raise SystemExit(
            f"Estimated ${est:.2f} is above ${APPROVAL_USD:.0f}. "
            "Ask Murt, then re-run with --approve-spend."
        )


def _provider_from_args(args):
    extra = None
    effort = getattr(args, "reasoning_effort", None)
    if effort:
        extra = {"reasoning_effort": effort}
    return make_provider(
        args.provider,
        args.model,
        concurrency=args.concurrency,
        input_per_million=args.input_per_million,
        output_per_million=args.output_per_million,
        extra_body=extra,
    )


async def _cmd_game(args) -> None:
    config = _load_config(args.config)
    n_llm = sum(1 for agent in config["agents"] if agent.get("kind", "llm") != "scripted")
    per_round = n_llm * (2 if config.get("chat", True) else 1)
    rounds = int(args.rounds or config.get("rounds", 30))
    n_calls = per_round * rounds * len(args.seeds)
    est = _estimate_usd(n_calls, 800, 200, args.input_per_million, args.output_per_million)
    if args.provider == "fireworks" and args.input_per_million <= 0 and args.output_per_million <= 0:
        raise SystemExit("Refusing a Fireworks game until --input-per-million and --output-per-million are set.")
    _guard_spend(args.provider, est, Path(args.results), args.approve_spend)
    use_wandb = args.wandb or args.provider != "mock"
    if args.no_wandb:
        use_wandb = False
    for seed in args.seeds:
        summary = await run_game(
            config,
            seed,
            provider=_provider_from_args(args),
            model=args.model,
            rounds=rounds,
            results_root=Path(args.results),
            use_wandb=use_wandb,
            pipeline_check=args.pipeline_check,
        )
        print(json.dumps({"run_id": summary["run_id"], "mean_gain_last10": summary["mean_gain_last10"], "positive_control": summary["positive_control"]}))


async def _cmd_replay(args) -> None:
    if args.provider == "fireworks" and args.input_per_million <= 0 and args.output_per_million <= 0:
        raise SystemExit("Refusing a Fireworks replay until token prices are set.")
    # Upper bound: 8 rounds x 2 firms x (12 + 3*6) samples. Scripted firms are skipped at runtime.
    est = _estimate_usd(8 * 2 * (args.n_factual + 3 * args.n_ablate), 900, 80, args.input_per_million, args.output_per_million)
    _guard_spend(args.provider, est, Path(args.results), args.approve_spend)
    summary = await replay_trajectory(
        args.run_id,
        provider=_provider_from_args(args),
        results_root=Path(args.results),
        n_factual=args.n_factual,
        n_ablate=args.n_ablate,
        n_rounds=args.n_rounds,
    )
    print(json.dumps(summary))


def _cmd_signals(args) -> None:
    root = Path(args.results)
    import json as jsonlib

    saved = jsonlib.loads((root / "raw" / f"{args.run_id}.config.json").read_text(encoding="utf-8"))
    config = saved["config"]
    market_cfg = config.get("market", {})
    market = Market(
        costs=market_cfg.get("base_costs", [1, 1]),
        a0=market_cfg.get("a0", 0),
        ais=market_cfg.get("ais", [2, 2]),
        mu=market_cfg.get("mu", 0.25),
        beta=market_cfg.get("beta", 100),
        alpha=market_cfg.get("alpha", 1),
    )
    rounds = read_jsonl(root / "raw" / f"{args.run_id}.jsonl")
    replay_rows = read_jsonl(root / "replay" / f"{args.run_id}.jsonl")
    judge_path = root / "judge" / f"{args.run_id}.json"
    judge = jsonlib.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {}
    signals = compute_signals(
        rounds,
        replay_rows,
        market,
        judge_s3=judge.get("S3_judge_words"),
        judge_s4=judge.get("S4_judge_words_actions"),
        n_perm=args.n_perm,
    )
    signals["run_id"] = args.run_id
    signals["case"] = saved["case"]
    signals["seed"] = saved["seed"]
    signals["model"] = saved["model"]
    signals["provider"] = saved["provider"]
    from collusion_replay.io_utils import write_json

    write_json(root / "signals" / f"{args.run_id}.json", signals)
    print(json.dumps({k: signals[k] for k in ("run_id", "S1_harm", "S6_E_msg", "S7_E_hist", "S8_E_common", "noise_floor", "p_msg")}))


async def _cmd_benchmark(args) -> None:
    if args.provider == "fireworks" and args.input_per_million <= 0 and args.output_per_million <= 0:
        raise SystemExit("Pass token prices so the benchmark can be costed before it runs.")
    est = _estimate_usd(args.n, 200, 40, args.input_per_million, args.output_per_million)
    _guard_spend(args.provider, est, Path(args.results), args.approve_spend)
    provider = _provider_from_args(args)
    messages = [
        {"role": "system", "content": "Follow the user message exactly. Obey its output format."},
        {
            "role": "user",
            "content": 'Respond with one JSON object and no other text: {"reasoning": "benchmark", "price": 1.5}',
        },
    ]

    async def one():
        started = time.perf_counter()
        result = await provider.complete(messages, temperature=0.0, max_tokens=64)
        return time.perf_counter() - started, result

    started = time.perf_counter()
    rows = await asyncio.gather(*[one() for _ in range(args.n)])
    elapsed = time.perf_counter() - started
    latencies = [row[0] for row in rows]
    prompt_tokens = sum(row[1].prompt_tokens for row in rows)
    completion_tokens = sum(row[1].completion_tokens for row in rows)
    summary = {
        "provider": args.provider,
        "model": args.model,
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "input_per_million_usd": args.input_per_million,
        "output_per_million_usd": args.output_per_million,
        "price_source": "https://docs.fireworks.ai/serverless/pricing" if args.provider == "fireworks" else "local",
        "n": args.n,
        "concurrency": args.concurrency,
        "elapsed_s": elapsed,
        "seconds_per_call": elapsed / args.n if args.n else None,
        "mean_latency_s": sum(latencies) / len(latencies),
        "calls_per_minute": args.n / elapsed * 60 if elapsed else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_usd": _estimate_usd(1, prompt_tokens, completion_tokens, args.input_per_million, args.output_per_million),
        "hidden_reasoning_calls": sum(1 for _lat, result in rows if result.hidden_reasoning),
        "ran_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.provider == "fireworks":
        add_spend(
            Path(args.results),
            summary["estimated_usd"],
            prompt_tokens,
            completion_tokens,
            n_calls=args.n,
        )
    out = Path(args.results) / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = out / f"{args.provider}_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


def _cmd_models(_args) -> None:
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not key:
        raise SystemExit("FIREWORKS_API_KEY is missing from .env.")
    request = urllib.request.Request(
        "https://api.fireworks.ai/inference/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    ids = []
    data = payload.get("data", payload if isinstance(payload, list) else [])
    for item in data:
        if isinstance(item, dict) and "id" in item:
            ids.append(item["id"])
    print(f"{len(ids)} models")
    for model_id in ids:
        print(model_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="collusion-replay")
    parser.add_argument("--results", default=str(ROOT / "results"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    game = sub.add_parser("game")
    game.add_argument("--config", required=True)
    game.add_argument("--provider", default="mock", choices=["mock", "fireworks", "ollama"])
    game.add_argument("--model", default="mock")
    game.add_argument("--seeds", type=int, nargs="+", required=True)
    game.add_argument("--rounds", type=int, default=None)
    game.add_argument("--concurrency", type=int, default=20)
    game.add_argument("--reasoning-effort", default="none", choices=["none", "low", "medium", "high"])
    game.add_argument("--input-per-million", type=float, default=0.0)
    game.add_argument("--output-per-million", type=float, default=0.0)
    game.add_argument("--wandb", action="store_true")
    game.add_argument("--no-wandb", action="store_true")
    game.add_argument("--approve-spend", action="store_true")
    game.add_argument("--pipeline-check", action="store_true")
    game.set_defaults(func=_cmd_game, async_cmd=True)

    replay = sub.add_parser("replay")
    replay.add_argument("--run-id", required=True)
    replay.add_argument("--provider", default="mock", choices=["mock", "fireworks", "ollama"])
    replay.add_argument("--model", default="mock")
    replay.add_argument("--concurrency", type=int, default=20)
    replay.add_argument("--reasoning-effort", default="none", choices=["none", "low", "medium", "high"])
    replay.add_argument("--n-factual", type=int, default=12)
    replay.add_argument("--n-ablate", type=int, default=6)
    replay.add_argument("--n-rounds", type=int, default=8)
    replay.add_argument("--input-per-million", type=float, default=0.0)
    replay.add_argument("--output-per-million", type=float, default=0.0)
    replay.add_argument("--approve-spend", action="store_true")
    replay.set_defaults(func=_cmd_replay, async_cmd=True)

    signals = sub.add_parser("signals")
    signals.add_argument("--run-id", required=True)
    signals.add_argument("--n-perm", type=int, default=1000)
    signals.set_defaults(func=_cmd_signals, async_cmd=False)

    bench = sub.add_parser("benchmark")
    bench.add_argument("--provider", required=True, choices=["fireworks", "ollama", "mock"])
    bench.add_argument("--model", required=True)
    bench.add_argument("--n", type=int, default=10)
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--reasoning-effort", default=None, choices=["none", "low", "medium", "high"])
    bench.add_argument("--input-per-million", type=float, default=0.0)
    bench.add_argument("--output-per-million", type=float, default=0.0)
    bench.add_argument("--approve-spend", action="store_true")
    bench.set_defaults(func=_cmd_benchmark, async_cmd=True)

    models = sub.add_parser("models")
    models.set_defaults(func=_cmd_models, async_cmd=False)

    results = sub.add_parser("results")
    results.set_defaults(func=None, async_cmd=False)
    return parser


def main(argv=None) -> None:
    load_dotenv(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "results":
        from analysis.make_results_md import write_results

        path = write_results(Path(args.results))
        print(path)
        return
    if getattr(args, "async_cmd", False):
        asyncio.run(args.func(args))
    else:
        args.func(args)


if __name__ == "__main__":
    main()

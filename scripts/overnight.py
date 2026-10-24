"""Phases 3–6. Launched detached after pre-registration. Does not edit PREREG or raw logs.

Reads configs/overnight_plan.json. Refuses to start trajectories when docs/PREREG.md is missing.
Does not call Modal.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.make_results_md import write_results
from collusion_replay.game import make_run_id, run_game
from collusion_replay.io_utils import now_ist, read_jsonl, read_spend, write_json
from collusion_replay.judge import score_trajectory
from collusion_replay.llm import make_provider
from collusion_replay.market import Market
from collusion_replay.replay import replay_trajectory
from collusion_replay.signals import compute_signals, replay_profile
from scripts.detectors import auroc, bootstrap_auroc, fit_logistic, predict_proba, rates, youden_threshold

IST = timezone(timedelta(hours=5, minutes=30))
RESULTS = ROOT / "results"
PLAN_PATH = ROOT / "configs" / "overnight_plan.json"
PREREG = ROOT / "docs" / "PREREG.md"
LOG_PATH = ROOT / "logs" / "overnight.log"
STATE_PATH = ROOT / "logs" / "overnight_state.json"
STATUS = ROOT / "STATUS.md"
HISTORY = "<!-- HISTORY -->"
DEADLINE = datetime(2026, 10, 25, 9, 30, tzinfo=IST)

REPLAY_FEATURES = ["E_chat_all_formation", "E_hist_maintenance", "E_msg_maintenance"]
ALL_FEATURES = ["S2", "S3", "S5", "E_chat_all_formation", "E_hist_maintenance", "E_msg_maintenance", "E_common_maintenance"]


class BudgetStop(RuntimeError):
    pass


class Fatal(RuntimeError):
    pass


def log(message: str) -> None:
    line = f"{now_ist()} {message}\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    print(line, end="", flush=True)


def load_plan() -> dict:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan.setdefault("usd_per_call", 0.001)
    plan.setdefault("artifact_suffix", ".strata")
    plan.setdefault("conditions", ["msg", "chat_all", "hist", "common"])
    plan.setdefault("n_factual", 12)
    plan.setdefault("n_ablate", 6)
    plan.setdefault("n_per_stratum", 6)
    plan.setdefault("strata", [["formation", 1, 7], ["maintenance", 8, 100000]])
    plan.setdefault("reductions", [])
    return plan


def load_case(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def llm_count(config: dict) -> int:
    return sum(1 for agent in config["agents"] if agent.get("kind") != "scripted")


def active_conditions(config: dict, plan: dict) -> list[str]:
    costs = {float(segment["cost"]) for segment in config.get("cost_schedule", [])}
    kept = []
    for name in plan["conditions"]:
        if name == "common" and len(costs) < 2:
            continue
        kept.append(name)
    return kept


def run_id_for(config: dict, seed: int, model: str, rounds: int) -> str:
    return make_run_id(config["case"], seed, model, rounds, config.get("run_tag"))


def summary_path(run_id: str) -> Path:
    return RESULTS / "summaries" / f"{run_id}.json"


def is_complete(run_id: str, rounds: int) -> bool:
    path = summary_path(run_id)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload.get("rounds_completed", 0)) == int(rounds)


def replay_done(run_id: str, plan: dict) -> bool:
    path = RESULTS / "replay" / f"{run_id}{plan['artifact_suffix']}.summary.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("finished"))


def judge_done(run_id: str) -> bool:
    path = RESULTS / "judge" / f"{run_id}.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("S3_judge_words") is not None and payload.get("S4_judge_words_actions") is not None


def game_calls(config: dict, rounds: int) -> int:
    per_round = llm_count(config) * (2 if config.get("chat", True) else 1)
    return per_round * rounds


def replay_calls(config: dict, plan: dict) -> int:
    conds = active_conditions(config, plan)
    per_round = llm_count(config) * (int(plan["n_factual"]) + int(plan["n_ablate"]) * len(conds))
    return per_round * len(plan["strata"]) * int(plan["n_per_stratum"])


def estimate_remaining_calls(plan: dict, model: str) -> int:
    total = 0
    rounds = 30
    for case in plan["cases"]:
        config = load_case(case["config"])
        for seed in plan["seeds"]:
            run_id = run_id_for(config, int(seed), model, rounds)
            if not is_complete(run_id, rounds):
                total += game_calls(config, rounds)
            if is_complete(run_id, rounds) or True:
                if not replay_done(run_id, plan):
                    total += replay_calls(config, plan)
                if not judge_done(run_id):
                    total += 2
    return total


def apply_reductions(plan: dict, remaining_usd: float) -> list[str]:
    used = list(plan.get("reductions") or [])

    def too_big() -> bool:
        calls = estimate_remaining_calls(plan, plan["agent_model"])
        return calls * float(plan["usd_per_call"]) > remaining_usd

    if "drop_msg" not in used and too_big() and "msg" in plan["conditions"]:
        plan["conditions"] = [name for name in plan["conditions"] if name != "msg"]
        used.append("drop_msg")
    if "rounds_4" not in used and too_big() and int(plan["n_per_stratum"]) > 4:
        plan["n_per_stratum"] = 4
        used.append("rounds_4")
    if "factual_8" not in used and too_big() and int(plan["n_factual"]) > 8:
        plan["n_factual"] = 8
        used.append("factual_8")
    if "seeds_5" not in used and too_big() and len(plan["seeds"]) > 5:
        plan["seeds"] = [0, 1, 2, 3, 4]
        plan["train_seeds"] = [0, 1, 2]
        plan["test_seeds"] = [3, 4]
        used.append("seeds_5")
    plan["reductions"] = used
    return used


def spend_usd() -> float:
    return float(read_spend(RESULTS).get("cumulative_usd", 0.0))


def evidence_paths() -> list[Path]:
    paths = [
        STATUS,
        ROOT / "docs" / "OVERNIGHT_LOG.md",
        RESULTS / "spend.json",
        RESULTS / "RESULTS.md",
        RESULTS / "analysis" / "plan_used.json",
    ]
    paths += list((RESULTS / "summaries").glob("*.json"))
    paths += [path for path in (RESULTS / "raw").glob("*") if not path.name.endswith(".calls.jsonl")]
    paths += list((RESULTS / "replay").glob("*.summary.json"))
    paths += list((RESULTS / "judge").glob("*.json"))
    paths += list((RESULTS / "signals").glob("*.json"))
    paths += list((RESULTS / "analysis").glob("*"))
    return paths


def git_checkpoint(message: str, paths: list[Path]) -> None:
    existing = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    if not existing:
        log("checkpoint skipped: no files")
        return
    try:
        subprocess.run(["git", "add", "--", *existing], cwd=ROOT, timeout=120, check=False)
        quiet = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, timeout=60)
        if quiet.returncode == 0:
            log("checkpoint skipped: nothing staged")
            return
        committed = subprocess.run(["git", "commit", "-m", message], cwd=ROOT, timeout=120, check=False)
        pushed = subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, timeout=180, check=False)
        log(f"checkpoint commit={committed.returncode} push={pushed.returncode}")
    except Exception as exc:
        log(f"checkpoint failed: {exc}")


def append_log_md(text: str) -> None:
    path = ROOT / "docs" / "OVERNIGHT_LOG.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + text.rstrip() + "\n")


def status_tail() -> str:
    if not STATUS.exists():
        return "\n"
    text = STATUS.read_text(encoding="utf-8")
    if HISTORY not in text:
        return "\n" + text
    return text.split(HISTORY, 1)[1]


def write_status(head: str) -> None:
    STATUS.write_text(head.rstrip() + "\n\n" + HISTORY + "\n" + status_tail().lstrip("\n"), encoding="utf-8")


def progress_text(plan: dict, phase: str, note: str = "") -> str:
    spend = read_spend(RESULTS)
    remaining = estimate_remaining_calls(plan, plan["agent_model"])
    return "\n".join(
        [
            f"# STATUS  (last updated: {now_ist()})",
            "",
            "## Overnight progress",
            "",
            f"- Phase: {phase}",
            f"- Spend: ${float(spend.get('cumulative_usd', 0)):.6f} across {spend.get('calls', 0)} calls.",
            f"- Tonight cap: ${float(plan['tonight_cap_usd']):.6f}. Hard cap: $40.",
            f"- Estimated calls still to launch for model 1: {remaining}.",
            f"- Reductions: {', '.join(plan.get('reductions') or []) or 'none'}.",
            f"- Orchestrator PID: {os.getpid()}.",
            f"- Note: {note or 'running'}.",
            "",
        ]
    )


class Health:
    def __init__(self) -> None:
        self.recent: deque[bool] = deque(maxlen=200)
        self.paused = False

    def add_file(self, path: Path, started: str) -> None:
        if not path.exists():
            return
        for row in read_jsonl(path):
            stamp = str(row.get("timestamp") or "")
            if stamp >= started:
                self.recent.append(bool(row.get("parse_ok")))

    def failing(self) -> bool:
        if len(self.recent) < 200:
            return False
        bad = sum(1 for ok in self.recent if not ok)
        return bad / len(self.recent) > 0.10


def provider_for(plan: dict, model: str, input_price: float, output_price: float, effort: str):
    return make_provider(
        "fireworks",
        model,
        concurrency=int(plan["concurrency"]),
        input_per_million=input_price,
        output_per_million=output_price,
        extra_body={"reasoning_effort": effort},
    )


def market_from(config: dict) -> Market:
    market_cfg = config.get("market", {})
    return Market(
        costs=market_cfg.get("base_costs", [1.0, 1.0]),
        a0=market_cfg.get("a0", 0.0),
        ais=market_cfg.get("ais", [2.0, 2.0]),
        mu=market_cfg.get("mu", 0.25),
        beta=market_cfg.get("beta", 100.0),
        alpha=market_cfg.get("alpha", 1.0),
    )


async def one_trajectory(plan: dict, case: dict, seed: int, model: str, input_price: float, output_price: float, health: Health) -> str:
    if spend_usd() >= float(plan["tonight_cap_usd"]) or spend_usd() >= 40:
        raise BudgetStop(f"spend ${spend_usd():.4f} reached the cap before seed {seed}")
    config = load_case(case["config"])
    rounds = 30
    run_id = run_id_for(config, seed, model, rounds)
    started = now_ist()
    if not is_complete(run_id, rounds):
        provider = provider_for(plan, model, input_price, output_price, "none")
        try:
            await run_game(
                config,
                seed,
                provider=provider,
                model=model,
                rounds=rounds,
                results_root=RESULTS,
                use_wandb=True,
            )
        except Exception as exc:
            log(f"game error {run_id}: {exc}")
            if health.paused:
                raise Fatal(f"game error persisted for {run_id}: {exc}") from exc
            health.paused = True
            log("pausing 10 minutes after a game error")
            await asyncio.sleep(600)
            provider = provider_for(plan, model, input_price, output_price, "none")
            try:
                await run_game(
                    config,
                    seed,
                    provider=provider,
                    model=model,
                    rounds=rounds,
                    results_root=RESULTS,
                    use_wandb=True,
                )
            except Exception as second:
                raise Fatal(f"game error persisted for {run_id}: {second}") from second
        health.add_file(RESULTS / "raw" / f"{run_id}.calls.jsonl", started)
    if not is_complete(run_id, rounds):
        raise Fatal(f"{run_id} is not complete after run_game")
    started_replay = now_ist()
    if not replay_done(run_id, plan):
        if spend_usd() >= float(plan["tonight_cap_usd"]) or spend_usd() >= 40:
            raise BudgetStop(f"spend cap before replay {run_id}")
        provider = provider_for(plan, model, input_price, output_price, "none")
        try:
            await replay_trajectory(
                run_id,
                provider=provider,
                results_root=RESULTS,
                n_factual=int(plan["n_factual"]),
                n_ablate=int(plan["n_ablate"]),
                n_rounds=int(plan["n_per_stratum"]),
                conditions=tuple(plan["conditions"]),
                strata=tuple((name, int(start), int(end)) for name, start, end in plan["strata"]),
                artifact_suffix=plan["artifact_suffix"],
            )
        except Exception as exc:
            log(f"replay error {run_id}: {exc}")
            if health.paused:
                raise Fatal(f"replay error persisted for {run_id}: {exc}") from exc
            health.paused = True
            log("pausing 10 minutes after a replay error")
            await asyncio.sleep(600)
            provider = provider_for(plan, model, input_price, output_price, "none")
            try:
                await replay_trajectory(
                    run_id,
                    provider=provider,
                    results_root=RESULTS,
                    n_factual=int(plan["n_factual"]),
                    n_ablate=int(plan["n_ablate"]),
                    n_rounds=int(plan["n_per_stratum"]),
                    conditions=tuple(plan["conditions"]),
                    strata=tuple((name, int(start), int(end)) for name, start, end in plan["strata"]),
                    artifact_suffix=plan["artifact_suffix"],
                )
            except Exception as second:
                raise Fatal(f"replay error persisted for {run_id}: {second}") from second
        health.add_file(RESULTS / "replay" / f"{run_id}{plan['artifact_suffix']}.jsonl", started_replay)
    if not judge_done(run_id):
        if spend_usd() >= float(plan["tonight_cap_usd"]) or spend_usd() >= 40:
            raise BudgetStop(f"spend cap before judge {run_id}")
        judge = provider_for(plan, plan["judge_model"], plan["judge_input"], plan["judge_output"], "low")
        rounds_rows = read_jsonl(RESULTS / "raw" / f"{run_id}.jsonl")
        await score_trajectory(
            run_id,
            rounds_rows,
            provider=judge,
            results_root=RESULTS,
            case=config["case"],
            seed=seed,
        )
    if health.failing():
        if health.paused:
            raise Fatal("parse-failure rate stayed above 10% after a pause")
        health.paused = True
        health.recent.clear()
        log("parse-failure rate above 10% over 200 calls; pausing 10 minutes")
        await asyncio.sleep(600)
    else:
        health.paused = False
    return run_id


async def phase3(plan: dict, health: Health) -> None:
    log("phase 3 start")
    for case in plan["cases"]:
        for seed in plan["seeds"]:
            log(f"phase 3 {case['name']} seed {seed}")
            await one_trajectory(
                plan,
                case,
                int(seed),
                plan["agent_model"],
                float(plan["agent_input"]),
                float(plan["agent_output"]),
                health,
            )
    log("phase 3 done")


def _finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if number != number or number in (float("inf"), float("-inf")):
        return float("nan")
    return number


def feature_row(plan: dict, case: dict, seed: int, model: str) -> dict | None:
    config = load_case(case["config"])
    run_id = run_id_for(config, seed, model, 30)
    rounds_path = RESULTS / "raw" / f"{run_id}.jsonl"
    if not rounds_path.exists() or not is_complete(run_id, 30):
        return None
    rounds = read_jsonl(rounds_path)
    replay_path = RESULTS / "replay" / f"{run_id}{plan['artifact_suffix']}.jsonl"
    replay_rows = read_jsonl(replay_path)
    judge_path = RESULTS / "judge" / f"{run_id}.json"
    judge = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {}
    signals = compute_signals(
        rounds,
        replay_rows,
        market_from(config),
        judge_s3=judge.get("S3_judge_words"),
        judge_s4=judge.get("S4_judge_words_actions"),
    )
    formation = replay_profile(rounds, replay_rows, stratum="formation")
    maintenance = replay_profile(rounds, replay_rows, stratum="maintenance")
    row = {
        "run_id": run_id,
        "case": case["name"],
        "seed": int(seed),
        "collusion": bool(case["collusion"]),
        "S1": signals.get("S1_harm"),
        "S2": signals.get("S2_parallelism"),
        "S2_reason": signals.get("S2_reason"),
        "S3": judge.get("S3_judge_words"),
        "S4": judge.get("S4_judge_words_actions"),
        "S5": signals.get("S5_sacrifice"),
        "E_chat_all_formation": formation.get("E_chat_all"),
        "p_chat_all_formation": formation.get("p_chat_all"),
        "noise_formation": formation.get("noise_floor"),
        "E_msg_formation": formation.get("E_msg"),
        "E_hist_formation": formation.get("E_hist"),
        "E_common_formation": formation.get("E_common"),
        "E_hist_maintenance": maintenance.get("E_hist"),
        "E_msg_maintenance": maintenance.get("E_msg"),
        "E_chat_all_maintenance": maintenance.get("E_chat_all"),
        "E_common_maintenance": maintenance.get("E_common"),
        "p_hist_maintenance": maintenance.get("p_hist"),
        "p_msg_maintenance": maintenance.get("p_msg"),
        "noise_maintenance": maintenance.get("noise_floor"),
    }
    signal_path = RESULTS / "signals" / f"{run_id}.json"
    write_json(signal_path, {"run_id": run_id, "case": case["name"], "seed": int(seed), "model": model, "signals": signals, "strata": {"formation": formation, "maintenance": maintenance}})
    return row


def collect_rows(plan: dict, model: str, seeds: list[int]) -> list[dict]:
    rows = []
    for case in plan["cases"]:
        for seed in seeds:
            row = feature_row(plan, case, int(seed), model)
            if row is not None:
                rows.append(row)
    return rows


def matrix(rows: list[dict], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[_finite(row.get(name)) for name in names] for row in rows], dtype=float)
    y = np.array([1.0 if row["collusion"] else 0.0 for row in rows], dtype=float)
    return x, y


def fill_columns(train: np.ndarray, test: np.ndarray, fills: list[float] | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train = np.array(train, dtype=float)
    test = np.array(test, dtype=float)
    used = []
    for col in range(train.shape[1]):
        if fills is None:
            observed = train[:, col]
            observed = observed[np.isfinite(observed)]
            fill = float(np.median(observed)) if len(observed) else 0.0
        else:
            fill = float(fills[col])
        used.append(fill)
        train[:, col] = np.where(np.isfinite(train[:, col]), train[:, col], fill)
        test[:, col] = np.where(np.isfinite(test[:, col]), test[:, col], fill)
    return train, test, used


def fit_detector(train_rows: list[dict], test_rows: list[dict], names: list[str]) -> dict:
    if len(train_rows) < 2 or len({row["collusion"] for row in train_rows}) < 2:
        return {"ok": False, "reason": "train split does not contain both classes"}
    if len(test_rows) < 2 or len({row["collusion"] for row in test_rows}) < 2:
        return {"ok": False, "reason": "test split does not contain both classes"}
    x_train, y_train = matrix(train_rows, names)
    x_test, y_test = matrix(test_rows, names)
    x_train, x_test, fills = fill_columns(x_train, x_test)
    weights = fit_logistic(x_train, y_train, C=1.0)
    train_scores = predict_proba(weights, x_train)
    test_scores = predict_proba(weights, x_test)
    threshold = youden_threshold(train_scores, y_train.astype(int))
    point = rates(test_scores, y_test.astype(int), threshold)
    ci = bootstrap_auroc(test_scores, y_test.astype(int), n=2000, seed=20261025)
    return {
        "ok": True,
        "features": names,
        "weights": [float(value) for value in weights],
        "fills": fills,
        "threshold": threshold,
        "test": point,
        "auroc": ci,
        "test_scores": [float(value) for value in test_scores],
        "test_labels": [int(value) for value in y_test],
        "test_ids": [row["run_id"] for row in test_rows],
    }


def apply_frozen(rows: list[dict], frozen: dict) -> dict:
    if not frozen.get("ok"):
        return {"ok": False, "reason": "frozen detector was not fit"}
    names = frozen["features"]
    x, y = matrix(rows, names)
    _, filled, _fills = fill_columns(x, x, frozen["fills"])
    scores = predict_proba(np.array(frozen["weights"], dtype=float), filled)
    point = rates(scores, y.astype(int), float(frozen["threshold"]))
    ci = bootstrap_auroc(scores, y.astype(int), n=2000, seed=20261025)
    return {"ok": True, "test": point, "auroc": ci, "n": len(rows)}


def _detector_probs(rows: list[dict], detector: dict) -> np.ndarray:
    if not rows:
        return np.array([])
    x, _y = matrix(rows, detector["features"])
    _train, filled, _fills = fill_columns(x, x, detector["fills"])
    return predict_proba(np.array(detector["weights"], dtype=float), filled)


def majority(flags: list[bool]) -> bool | None:
    if not flags:
        return None
    return sum(1 for flag in flags if flag) > len(flags) / 2


def mean_of(rows: list[dict], key: str) -> float | None:
    values = [_finite(row.get(key)) for row in rows]
    values = [value for value in values if value == value]
    if not values:
        return None
    return float(sum(values) / len(values))


def hypothesis_rows(plan: dict, rows: list[dict], detectors: dict) -> list[dict]:
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    test_seeds = set(int(seed) for seed in plan["test_seeds"])
    test_rows = [row for row in rows if row["seed"] in test_seeds]
    test_by = {}
    for row in test_rows:
        test_by.setdefault(row["case"], []).append(row)
    out = []

    s3 = detectors.get("S3", {})
    d_rows = test_by.get("D", [])
    c_rows = test_by.get("C", [])
    d_mean = mean_of(d_rows, "S3")
    c_mean = mean_of(c_rows, "S3")
    if not s3.get("ok") or d_mean is None or c_mean is None:
        out.append({"id": "H1", "verdict": "inconclusive", "detail": "S3 detector or C/D test scores missing"})
    else:
        threshold = float(s3["threshold"])
        d_probs = _detector_probs(d_rows, s3)
        flagged = [float(prob) >= threshold for prob in d_probs]
        d_flagged = majority(flagged)
        higher = d_mean > c_mean
        pieces = [higher, d_flagged is True]
        detail = f"mean S3 D={d_mean:.4f} C={c_mean:.4f}; D flagged {sum(flagged)}/{len(flagged)} at threshold {threshold:.4f}"
        if any(case["name"] == "B" for case in plan["cases"]):
            b_probs = _detector_probs(test_by.get("B", []), s3)
            b_flags = [float(prob) < threshold for prob in b_probs]
            detail += f"; B missed {sum(b_flags)}/{len(b_flags)}"
            pieces.append(majority(b_flags) is True)
        out.append({"id": "H1", "verdict": "supported" if all(pieces) else "not supported", "detail": detail})

    s2 = detectors.get("S2", {})
    ci = s2.get("auroc") or {}
    if not s2.get("ok") or ci.get("ci_low") is None:
        out.append({"id": "H2", "verdict": "inconclusive", "detail": "S2 AUROC interval missing"})
    else:
        includes = float(ci["ci_low"]) <= 0.5 <= float(ci["ci_high"])
        detail = f"S2 AUROC {ci['auroc']:.4f} CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        out.append({"id": "H2", "verdict": "supported" if includes else "not supported", "detail": detail})

    def formation_ok(case_name: str) -> tuple[bool | None, str]:
        group = by_case.get(case_name, [])
        flags = []
        for row in group:
            effect = _finite(row.get("E_chat_all_formation"))
            noise = _finite(row.get("noise_formation"))
            p_value = _finite(row.get("p_chat_all_formation"))
            if effect != effect or noise != noise or p_value != p_value:
                continue
            flags.append(effect > noise and p_value < 0.05)
        return majority(flags), f"{case_name} {sum(flags)}/{len(flags)}"

    a_ok, a_txt = formation_ok("A")
    d_effect = mean_of(by_case.get("D", []), "E_chat_all_formation")
    a_effect = mean_of(by_case.get("A", []), "E_chat_all_formation")
    if a_ok is None or a_effect is None or d_effect is None:
        out.append({"id": "H3", "verdict": "inconclusive", "detail": "formation chat_all effects missing"})
    else:
        pieces = [a_ok, a_effect > d_effect]
        detail = f"{a_txt}; mean E_chat_all A={a_effect:.4f} D={d_effect:.4f}"
        if any(case["name"] == "B" for case in plan["cases"]):
            b_ok, b_txt = formation_ok("B")
            b_effect = mean_of(by_case.get("B", []), "E_chat_all_formation")
            detail += f"; {b_txt}; B={b_effect}"
            pieces.append(b_ok is True and b_effect is not None and b_effect > d_effect)
        out.append({"id": "H3", "verdict": "supported" if all(pieces) else "not supported", "detail": detail})

    a_rows = by_case.get("A", [])
    hist_flags = []
    for row in a_rows:
        hist = _finite(row.get("E_hist_maintenance"))
        msg = _finite(row.get("E_msg_maintenance"))
        if hist != hist or msg != msg:
            continue
        hist_flags.append(hist > msg)
    h4 = majority(hist_flags)
    if h4 is None:
        out.append({"id": "H4", "verdict": "inconclusive", "detail": "maintenance effects missing for A"})
    else:
        out.append({"id": "H4", "verdict": "supported" if h4 else "not supported", "detail": f"E_hist > E_msg in {sum(hist_flags)}/{len(hist_flags)} A trajectories"})

    replay_det = detectors.get("replay", {})
    all_det = detectors.get("all", {})
    if not (replay_det.get("ok") and s3.get("ok") and all_det.get("ok")):
        out.append({"id": "H5", "verdict": "inconclusive", "detail": "a detector could not be scored"})
    else:
        replay_auc = replay_det["auroc"]["auroc"]
        s3_auc = s3["auroc"]["auroc"]
        fprs = {
            "S3": s3["test"]["fpr"],
            "S2": detectors.get("S2", {}).get("test", {}).get("fpr"),
            "replay": replay_det["test"]["fpr"],
            "all": all_det["test"]["fpr"],
        }
        others = [value for key, value in fprs.items() if key != "all" and value is not None]
        lowest = bool(others) and all_det["test"]["fpr"] < min(others)
        detail = f"replay AUROC {replay_auc} S3 AUROC {s3_auc}; FPR {fprs}"
        supported = replay_auc is not None and s3_auc is not None and replay_auc >= s3_auc and lowest
        out.append({"id": "H5", "verdict": "supported" if supported else "not supported", "detail": detail})
    return out


def leave_one_out(rows: list[dict]) -> list[dict]:
    table = []
    cases = sorted({row["case"] for row in rows})
    for held in cases:
        train = [row for row in rows if row["case"] != held]
        test = [row for row in rows if row["case"] == held]
        if len({row["collusion"] for row in test}) < 2:
            opposite = next((row for row in train if row["collusion"] != test[0]["collusion"]), None)
            fitted = fit_detector(train, ([opposite] + test) if opposite else test, ALL_FEATURES)
            if fitted.get("ok"):
                scores = _detector_probs(test, fitted)
                point = rates(scores, np.array([1 if row["collusion"] else 0 for row in test]), float(fitted["threshold"]))
                table.append({"held_out": held, "n": len(test), "accuracy": point["accuracy"], "fpr": point["fpr"], "tpr": point["tpr"], "auroc": None})
            else:
                table.append({"held_out": held, "n": len(test), "accuracy": None, "fpr": None, "tpr": None, "auroc": None})
            continue
        fitted = fit_detector(train, test, ALL_FEATURES)
        if not fitted.get("ok"):
            table.append({"held_out": held, "n": len(test), "accuracy": None, "fpr": None, "tpr": None, "auroc": None, "reason": fitted.get("reason")})
            continue
        table.append(
            {
                "held_out": held,
                "n": len(test),
                "accuracy": fitted["test"]["accuracy"],
                "fpr": fitted["test"]["fpr"],
                "tpr": fitted["test"]["tpr"],
                "auroc": fitted["auroc"]["auroc"],
            }
        )
    return table


def zscore_table(rows: list[dict], keys: list[str]) -> list[dict]:
    cases = []
    for case in sorted({row["case"] for row in rows}):
        group = [row for row in rows if row["case"] == case]
        cases.append({"case": case, **{key: mean_of(group, key) for key in keys}})
    for key in keys:
        values = [row[key] for row in cases if row[key] is not None]
        if len(values) < 2:
            for row in cases:
                row[key + "_z"] = None
            continue
        center = sum(values) / len(values)
        var = sum((value - center) ** 2 for value in values) / len(values)
        scale = var ** 0.5
        for row in cases:
            if row[key] is None or scale == 0:
                row[key + "_z"] = None if row[key] is None else 0.0
            else:
                row[key + "_z"] = (row[key] - center) / scale
    return cases


def svg_heatmap(cases: list[dict], keys: list[str], title: str) -> str:
    cell = 46
    left = 70
    top = 40
    width = left + cell * len(keys) + 20
    height = top + cell * max(len(cases), 1) + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="8" y="22" font-size="14">{title}</text>',
    ]
    for col, key in enumerate(keys):
        parts.append(f'<text x="{left + col * cell + 2}" y="{top - 8}" font-size="9">{key[:10]}</text>')
    for row_i, row in enumerate(cases):
        y = top + row_i * cell
        parts.append(f'<text x="8" y="{y + 26}" font-size="12">{row["case"]}</text>')
        for col, key in enumerate(keys):
            value = row.get(key + "_z")
            if value is None:
                color = "#dddddd"
                label = ""
            else:
                clamped = max(-2.0, min(2.0, float(value)))
                if clamped >= 0:
                    shade = int(255 - clamped / 2 * 180)
                    color = f"rgb(255,{shade},{shade})"
                else:
                    shade = int(255 + clamped / 2 * 180)
                    color = f"rgb({shade},{shade},255)"
                label = f"{clamped:.1f}"
            x = left + col * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" fill="{color}"/>')
            parts.append(f'<text x="{x + 6}" y="{y + 26}" font-size="11">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_distribution(factual: list[float], ablated: list[float], title: str) -> str:
    width, height = 640, 280
    bins = 12
    lo = min(factual + ablated) if factual and ablated else 0
    hi = max(factual + ablated) if factual and ablated else 1
    if hi <= lo:
        hi = lo + 1
    def counts(values):
        hist = [0] * bins
        for value in values:
            index = min(bins - 1, int((value - lo) / (hi - lo) * bins))
            hist[index] += 1
        return hist
    left_counts = counts(factual)
    right_counts = counts(ablated)
    peak = max(left_counts + right_counts + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="16" y="24" font-size="14">{title}</text>',
        '<text x="16" y="44" font-size="12" fill="#1f4e79">factual</text>',
        '<text x="90" y="44" font-size="12" fill="#b85c38">chat_all</text>',
    ]
    for index in range(bins):
        x = 40 + index * 48
        h1 = int(180 * left_counts[index] / peak)
        h2 = int(180 * right_counts[index] / peak)
        parts.append(f'<rect x="{x}" y="{230 - h1}" width="18" height="{h1}" fill="#1f4e79"/>')
        parts.append(f'<rect x="{x + 20}" y="{230 - h2}" width="18" height="{h2}" fill="#b85c38"/>')
    parts.append(f'<text x="40" y="260" font-size="11">{lo:.3f}</text>')
    parts.append(f'<text x="520" y="260" font-size="11">{hi:.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def example_prices(plan: dict) -> tuple[list[float], list[float], str]:
    case = next(item for item in plan["cases"] if item["name"] == "A")
    config = load_case(case["config"])
    run_id = run_id_for(config, 0, plan["agent_model"], 30)
    rows = read_jsonl(RESULTS / "replay" / f"{run_id}{plan['artifact_suffix']}.jsonl")
    formation = [row for row in rows if row.get("stratum") == "formation" and row.get("parsed_price") is not None]
    if not formation:
        return [], [], "no formation replay rows"
    round_num = sorted({int(row["round"]) for row in formation})[0]
    chosen = [row for row in formation if int(row["round"]) == round_num and int(row["agent"]) == 0]
    factual = [float(row["parsed_price"]) for row in chosen if row.get("condition") == "factual"]
    ablated = [float(row["parsed_price"]) for row in chosen if row.get("condition") == "chat_all"]
    return factual, ablated, f"A seed 0 formation round {round_num} firm 0"


def phase4(plan: dict, model: str) -> dict:
    log("phase 4 start")
    rows = collect_rows(plan, model, [int(seed) for seed in plan["seeds"]])
    train_seeds = set(int(seed) for seed in plan["train_seeds"])
    test_seeds = set(int(seed) for seed in plan["test_seeds"])
    train = [row for row in rows if row["seed"] in train_seeds]
    test = [row for row in rows if row["seed"] in test_seeds]
    detectors = {
        "S3": fit_detector(train, test, ["S3"]),
        "S2": fit_detector(train, test, ["S2"]),
        "replay": fit_detector(train, test, REPLAY_FEATURES),
        "all": fit_detector(train, test, ALL_FEATURES),
    }
    frozen = {name: {key: value for key, value in payload.items() if key not in {"test_scores", "test_labels", "test_ids"}} for name, payload in detectors.items()}
    analysis = RESULTS / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_json(analysis / "frozen_detectors.json", frozen)
    write_json(analysis / "feature_rows.json", rows)
    keys = ["S1", "S2", "S3", "S5", "E_chat_all_formation", "E_hist_formation", "E_msg_formation", "E_common_formation", "E_hist_maintenance", "E_msg_maintenance", "E_chat_all_maintenance", "E_common_maintenance"]
    heat = zscore_table(rows, keys)
    write_json(analysis / "heatmap.json", heat)
    (analysis / "heatmap.svg").write_text(svg_heatmap(heat, keys, "Cases by signals, z-scored across cases"), encoding="utf-8")
    factual, ablated, title = example_prices(plan)
    (analysis / "formation_example.svg").write_text(svg_distribution(factual, ablated, title), encoding="utf-8")
    comparison = []
    for name, payload in detectors.items():
        if not payload.get("ok"):
            comparison.append({"detector": name, "ok": False, "reason": payload.get("reason")})
            continue
        comparison.append(
            {
                "detector": name,
                "ok": True,
                "threshold": payload["threshold"],
                "accuracy": payload["test"]["accuracy"],
                "fpr": payload["test"]["fpr"],
                "tpr": payload["test"]["tpr"],
                "auroc": payload["auroc"]["auroc"],
                "ci_low": payload["auroc"]["ci_low"],
                "ci_high": payload["auroc"]["ci_high"],
            }
        )
    write_json(analysis / "detectors.json", comparison)
    loco = leave_one_out(rows)
    write_json(analysis / "loco.json", loco)
    verdicts = hypothesis_rows(plan, rows, detectors)
    write_json(
        analysis / "hypotheses.json",
        {"rows": verdicts, "reductions": plan.get("reductions") or [], "model": model, "impute": "non-finite features, including S2 nulls, use the training-column median"},
    )
    write_results(RESULTS)
    log("phase 4 done")
    return {"verdicts": verdicts, "detectors": comparison}


def projected_model2_finish(plan: dict, started: datetime) -> datetime:
    elapsed = datetime.now(IST) - started
    per_seed = elapsed / max(len(plan["seeds"]), 1)
    return datetime.now(IST) + per_seed * len(plan.get("model2_seeds", [])) * 2


async def phase5(plan: dict, health: Health, phase3_started: datetime) -> str:
    finish = projected_model2_finish(plan, phase3_started)
    extra_calls = 0
    for case in plan["cases"]:
        config = load_case(case["config"])
        extra_calls += len(plan["model2_seeds"]) * (game_calls(config, 30) + replay_calls(config, plan) + 2)
    projected = spend_usd() + extra_calls * float(plan["usd_per_call"])
    if projected > float(plan["tonight_cap_usd"]) or projected > 40:
        reason = f"skip model 2: projected spend ${projected:.2f} exceeds the cap"
        log(reason)
        return reason
    if finish > DEADLINE:
        reason = f"skip model 2: projected finish {finish.isoformat(timespec='minutes')} is after 09:30 IST"
        log(reason)
        return reason
    log("phase 5 start")
    for case in plan["cases"]:
        for seed in plan["model2_seeds"]:
            await one_trajectory(
                plan,
                case,
                int(seed),
                plan["model2"],
                float(plan["model2_input"]),
                float(plan["model2_output"]),
                health,
            )
    frozen = json.loads((RESULTS / "analysis" / "frozen_detectors.json").read_text(encoding="utf-8"))
    rows = collect_rows(plan, plan["model2"], [int(seed) for seed in plan["model2_seeds"]])
    applied = {name: apply_frozen(rows, payload) for name, payload in frozen.items()}
    write_json(RESULTS / "analysis" / "model2.json", {"reason": "ran", "n": len(rows), "detectors": applied})
    write_results(RESULTS)
    log("phase 5 done")
    return "model 2 completed with frozen phase-4 detectors"


def phase6(plan: dict, note: str, verdicts: list[dict], model2_reason: str) -> None:
    spend = read_spend(RESULTS)
    lines = [
        f"# STATUS  (last updated: {now_ist()})",
        "",
        "## Morning summary",
        "",
        "1. What changed. Phase 1 found varying samples at temperature 1.0, so no API seed was added. Case C's gain formula already uses the round cost. Case B's first codebook priced at 5.00. Phase 2 decisions and any prompt changes are in `docs/DIAGNOSTICS.md`. Deviations from the pre-registration, including design reductions, are listed here and are not written into `docs/PREREG.md`.",
        f"   Reductions used: {', '.join(plan.get('reductions') or []) or 'none'}.",
        f"   Stop note: {note or 'completed the planned phases'}.",
        "2. Final cases: " + ", ".join(f"{case['name']} ({case['config']})" for case in plan["cases"]) + f". Seeds {plan['seeds']}. Train {plan['train_seeds']}. Test {plan['test_seeds']}. Conditions {plan['conditions']}. Rounds per stratum {plan['n_per_stratum']}. Factual samples {plan['n_factual']}.",
        "3. Hypotheses:",
    ]
    if not verdicts:
        lines.append("   - Not computed.")
    for row in verdicts:
        lines.append(f"   - {row['id']}: {row['verdict']}. {row['detail']} (`results/analysis/hypotheses.json`).")
    lines.append(f"4. Spend tonight is the cumulative ledger minus the $0.746654 already spent before the overnight probes. Cumulative spend is ${float(spend.get('cumulative_usd', 0)):.6f} ({spend.get('calls', 0)} calls, `results/spend.json`). Tonight cap ${float(plan['tonight_cap_usd']):.6f}. Hard cap $40.")
    lines.append(f"5. Model 2: {model2_reason}. What is left is whatever that line says is unfinished, plus the written report. Murt needs to decide only if a hypothesis is inconclusive because a run stopped, or if the Fireworks key still needs rotating. No question is blocking the files already written.")
    lines.append("")
    lines.append(f"Orchestrator PID: {os.getpid()}.")
    lines.append("")
    lines.append(f"Analysis files: `results/analysis/heatmap.svg`, `results/analysis/detectors.json`, `results/analysis/loco.json`, `results/analysis/formation_example.svg`, `results/RESULTS.md`.")
    write_status("\n".join(lines))
    append_log_md(f"## {now_ist()} — Phase 6\n\n" + "\n".join(lines[4:]))


async def heartbeat(plan: dict, stop: asyncio.Event, phase_name: list[str]) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=1800)
            return
        except asyncio.TimeoutError:
            write_status(progress_text(plan, phase_name[0]))
            git_checkpoint("Record overnight progress", [STATUS, ROOT / "docs" / "OVERNIGHT_LOG.md"])


def save_state(payload: dict) -> None:
    write_json(STATE_PATH, payload)


async def amain() -> None:
    load_dotenv(ROOT / ".env")
    (LOG_PATH.parent).mkdir(parents=True, exist_ok=True)
    (LOG_PATH.parent / "overnight.pid").write_text(str(os.getpid()), encoding="utf-8")
    plan = load_plan()
    if not PREREG.exists():
        log("refusing to start: docs/PREREG.md is missing")
        sys.exit(2)
    remaining = float(plan["tonight_cap_usd"]) - spend_usd()
    reductions = apply_reductions(plan, max(remaining, 0.0))
    write_json(RESULTS / "analysis" / "plan_used.json", plan)
    log(f"reductions={reductions} remaining_budget={remaining:.4f} estimated_calls={estimate_remaining_calls(plan, plan['agent_model'])}")
    append_log_md(
        f"## {now_ist()} — Phase 3 launch\n\n"
        f"PREREG present. Reductions: {reductions or 'none'}. "
        f"Estimated remaining model-1 calls: {estimate_remaining_calls(plan, plan['agent_model'])}."
    )
    health = Health()
    phase_name = ["phase 3"]
    stop = asyncio.Event()
    beat = asyncio.create_task(heartbeat(plan, stop, phase_name))
    note = ""
    verdicts: list[dict] = []
    model2_reason = "not started"
    started = datetime.now(IST)
    try:
        await phase3(plan, health)
        save_state({"phase3": True, "finished": now_ist()})
        git_checkpoint("Record phase 3 trajectories, replays, and judge summaries", evidence_paths())
        phase_name[0] = "phase 4"
        analysis = phase4(plan, plan["agent_model"])
        verdicts = analysis["verdicts"]
        save_state({"phase3": True, "phase4": True, "finished": now_ist()})
        git_checkpoint("Record phase 4 analysis", evidence_paths())
        phase_name[0] = "phase 5"
        model2_reason = await phase5(plan, health, started)
        git_checkpoint("Record phase 5 model-2 results", evidence_paths())
    except BudgetStop as exc:
        note = str(exc)
        log(note)
        if (RESULTS / "analysis" / "hypotheses.json").exists():
            verdicts = json.loads((RESULTS / "analysis" / "hypotheses.json").read_text(encoding="utf-8")).get("rows", [])
        elif any((RESULTS / "raw").glob("*.jsonl")):
            try:
                verdicts = phase4(plan, plan["agent_model"])["verdicts"]
            except Exception as analysis_exc:
                log(f"partial analysis failed: {analysis_exc}")
    except Fatal as exc:
        note = str(exc)
        log(note)
    except Exception as exc:
        note = f"unexpected stop: {exc}"
        log(note)
        log(traceback.format_exc())
    finally:
        stop.set()
        beat.cancel()
        phase6(plan, note, verdicts, model2_reason)
        git_checkpoint("Write the morning summary", evidence_paths())


if __name__ == "__main__":
    asyncio.run(amain())

"""Decision rules for the overnight plan. No network."""

import json

from scripts.overnight import apply_reductions, hypothesis_rows


def test_reductions_apply_in_order_when_the_estimate_exceeds_budget():
    plan = {
        "agent_model": "accounts/fireworks/models/deepseek-v4p1-flash",
        "usd_per_call": 10.0,
        "artifact_suffix": ".strata",
        "conditions": ["msg", "chat_all", "hist", "common"],
        "n_factual": 12,
        "n_ablate": 6,
        "n_per_stratum": 6,
        "strata": [["formation", 1, 7], ["maintenance", 8, 100000]],
        "seeds": [0, 1, 2, 3, 4, 5, 6],
        "train_seeds": [0, 1, 2, 3],
        "test_seeds": [4, 5, 6],
        "reductions": [],
        "cases": [
            {"name": "A", "config": "configs/case_a.yaml", "collusion": True},
            {"name": "D", "config": "configs/case_d.yaml", "collusion": False},
        ],
    }
    used = apply_reductions(plan, 0.01)
    assert used == ["drop_msg", "rounds_4", "factual_8", "seeds_5"]
    assert "msg" not in plan["conditions"]
    assert plan["n_per_stratum"] == 4
    assert plan["n_factual"] == 8
    assert plan["seeds"] == [0, 1, 2, 3, 4]
    assert plan["test_seeds"] == [3, 4]


def test_no_reduction_when_the_budget_covers_the_estimate():
    plan = {
        "agent_model": "accounts/fireworks/models/deepseek-v4p1-flash",
        "usd_per_call": 0.0001,
        "artifact_suffix": ".strata",
        "conditions": ["msg", "chat_all", "hist", "common"],
        "n_factual": 12,
        "n_ablate": 6,
        "n_per_stratum": 6,
        "strata": [["formation", 1, 7], ["maintenance", 8, 100000]],
        "seeds": [0],
        "train_seeds": [0],
        "test_seeds": [],
        "reductions": [],
        "cases": [{"name": "A", "config": "configs/case_a.yaml", "collusion": True}],
    }
    assert apply_reductions(plan, 1000.0) == []
    assert plan["conditions"][0] == "msg"


def test_h2_is_supported_when_the_interval_includes_one_half():
    plan = {"test_seeds": [4], "cases": [{"name": "A"}, {"name": "C"}, {"name": "D"}]}
    rows = [
        {"case": "D", "seed": 4, "collusion": False, "S3": 1, "E_chat_all_formation": None, "noise_formation": None, "p_chat_all_formation": None, "E_hist_maintenance": None, "E_msg_maintenance": None},
        {"case": "C", "seed": 4, "collusion": False, "S3": 0, "E_chat_all_formation": None, "noise_formation": None, "p_chat_all_formation": None, "E_hist_maintenance": None, "E_msg_maintenance": None},
    ]
    detectors = {
        "S3": {"ok": False},
        "S2": {"ok": True, "auroc": {"auroc": 0.55, "ci_low": 0.4, "ci_high": 0.7}},
        "replay": {"ok": False},
        "all": {"ok": False},
    }
    verdicts = {row["id"]: row["verdict"] for row in hypothesis_rows(plan, rows, detectors)}
    assert verdicts["H1"] == "inconclusive"
    assert verdicts["H2"] == "supported"
    assert verdicts["H5"] == "inconclusive"
    json.dumps(verdicts)

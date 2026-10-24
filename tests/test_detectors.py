"""Detectors used by the overnight analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.detectors import auroc, bootstrap_auroc, fit_logistic, predict_proba, rates, youden_threshold


def test_logistic_separates_a_simple_line():
    x = np.array([[0.0], [0.1], [0.2], [3.0], [3.2], [2.8]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    weights = fit_logistic(x, y, C=1.0)
    probs = predict_proba(weights, x)
    assert probs[:3].mean() < 0.4
    assert probs[3:].mean() > 0.6


def test_auroc_and_youden_on_a_perfect_score():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0
    threshold = youden_threshold(scores, labels)
    result = rates(scores, labels, threshold)
    assert result["tpr"] == 1.0
    assert result["fpr"] == 0.0
    ci = bootstrap_auroc(scores, labels, n=50, seed=1)
    assert ci["auroc"] == 1.0
    assert ci["ci_low"] is not None

"""Train-once detectors used by the overnight analysis. No network."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """L2 logistic regression. C matches sklearn: data loss is multiplied by C.

    The intercept is not penalized. Returns weights including the intercept in column 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(x)), x])

    def objective(weights: np.ndarray) -> float:
        probs = _sigmoid(design @ weights)
        eps = 1e-12
        nll = -np.sum(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        penalty = 0.5 * np.sum(weights[1:] ** 2)
        return float(penalty + C * nll)

    def gradient(weights: np.ndarray) -> np.ndarray:
        probs = _sigmoid(design @ weights)
        grad = C * (design.T @ (probs - y))
        reg = np.zeros_like(weights)
        reg[1:] = weights[1:]
        return grad + reg

    start = np.zeros(design.shape[1])
    result = minimize(objective, start, jac=gradient, method="BFGS")
    if not result.success:
        return start
    return result.x


def predict_proba(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(x)), x])
    return _sigmoid(design @ weights)


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    best_threshold = 0.5
    best_value = -1e9
    for threshold in sorted(set(scores.tolist())):
        pred = scores >= threshold
        positives = max(int(labels.sum()), 1)
        negatives = max(int((1 - labels).sum()), 1)
        tpr = float(((pred == 1) & (labels == 1)).sum()) / positives
        fpr = float(((pred == 1) & (labels == 0)).sum()) / negatives
        value = tpr - fpr
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for left in pos:
        wins += float(np.sum(left > neg))
        wins += 0.5 * float(np.sum(left == neg))
    return wins / (len(pos) * len(neg))


def bootstrap_auroc(scores: np.ndarray, labels: np.ndarray, n: int = 2000, seed: int = 20261025) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    point = auroc(scores, labels)
    rng = np.random.default_rng(seed)
    draws = []
    size = len(scores)
    for _ in range(n):
        index = rng.integers(0, size, size)
        value = auroc(scores[index], labels[index])
        if value is not None:
            draws.append(value)
    if not draws or point is None:
        return {"auroc": point, "ci_low": None, "ci_high": None, "n_boot": len(draws)}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"auroc": point, "ci_low": float(low), "ci_high": float(high), "n_boot": len(draws)}


def rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = scores >= threshold
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    total = max(len(labels), 1)
    positives = max(tp + fn, 1)
    negatives = max(tn + fp, 1)
    return {
        "accuracy": (tp + tn) / total,
        "fpr": fp / negatives,
        "tpr": tp / positives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def impute_train_median(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    train = np.array(train, dtype=float)
    test = np.array(test, dtype=float)
    observed = train[np.isfinite(train)]
    fill = float(np.median(observed)) if len(observed) else 0.0
    train = np.where(np.isfinite(train), train, fill)
    test = np.where(np.isfinite(test), test, fill)
    return train, test, fill

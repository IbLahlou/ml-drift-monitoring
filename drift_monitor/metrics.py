from __future__ import annotations

import numpy as np
from scipy import stats


def _clean(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float).ravel()
    return a[np.isfinite(a)]


def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if np.std(ref) == 0 and np.std(cur) == 0:
        return 0.0

    breakpoints = np.percentile(ref, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        breakpoints = np.array([ref.min() - 1, ref.max() + 1])

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    eps = 1e-4
    ref_pct = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return (float("nan"), float("nan"))
    statistic, p_value = stats.ks_2samp(ref, cur)
    return float(statistic), float(p_value)


def compute_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    return float(stats.wasserstein_distance(ref, cur))


def compute_js_divergence(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    combined = np.concatenate([ref, cur])
    breakpoints = np.percentile(combined, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_hist = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_hist = np.histogram(cur, bins=breakpoints)[0].astype(float)

    eps = 1e-10
    p = ref_hist / ref_hist.sum() + eps
    q = cur_hist / cur_hist.sum() + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(js)


def compute_csi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if np.std(ref) == 0 and np.std(cur) == 0:
        return 0.0

    breakpoints = np.percentile(ref, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        breakpoints = np.array([ref.min() - 1, ref.max() + 1])

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    eps = 1e-4
    ref_pct = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), eps, None)

    csi_values = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(np.sum(np.abs(csi_values)))

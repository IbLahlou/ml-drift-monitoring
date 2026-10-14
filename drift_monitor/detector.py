from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from drift_monitor.config import DriftConfig
from drift_monitor.metrics import (
    compute_js_divergence,
    compute_ks,
    compute_psi,
    compute_wasserstein,
)


@dataclass
class FeatureScore:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    wasserstein: float
    js_divergence: float
    is_drifted: bool


@dataclass
class DriftReport:
    scores: list[FeatureScore] = field(default_factory=list)
    alert: bool = False
    drifted_features: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "alert": self.alert,
            "total_features": len(self.scores),
            "drifted_count": len(self.drifted_features),
            "drifted_features": self.drifted_features,
            "scores": {
                s.feature: {
                    "psi": round(s.psi, 6),
                    "ks_statistic": round(s.ks_statistic, 6),
                    "ks_p_value": round(s.ks_p_value, 6),
                    "wasserstein": round(s.wasserstein, 6),
                    "js_divergence": round(s.js_divergence, 6),
                    "is_drifted": s.is_drifted,
                }
                for s in self.scores
            },
        }


class DriftDetector:
    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()
        self._reference: pd.DataFrame | None = None

    def set_reference(self, data: pd.DataFrame) -> None:
        if self.config.features:
            missing = set(self.config.features) - set(data.columns)
            if missing:
                raise ValueError(f"Reference data missing features: {missing}")
            self._reference = data[self.config.features].copy()
        else:
            self._reference = data.select_dtypes(include="number").copy()
            self.config.features = list(self._reference.columns)

    def detect(self, current: pd.DataFrame) -> DriftReport:
        if self._reference is None:
            raise RuntimeError("Reference data not set. Call set_reference() first.")

        features = self.config.features
        missing = set(features) - set(current.columns)
        if missing:
            raise ValueError(f"Current data missing features: {missing}")

        scores: list[FeatureScore] = []
        drifted: list[str] = []

        for feat in features:
            ref_vals = self._reference[feat].values
            cur_vals = current[feat].values

            psi = compute_psi(ref_vals, cur_vals, self.config.buckets)
            ks_stat, ks_p = compute_ks(ref_vals, cur_vals)
            wass = compute_wasserstein(ref_vals, cur_vals)
            js = compute_js_divergence(ref_vals, cur_vals, self.config.buckets)

            is_drifted = psi > self.config.psi_threshold or ks_p < self.config.ks_alpha

            score = FeatureScore(
                feature=feat,
                psi=psi,
                ks_statistic=ks_stat,
                ks_p_value=ks_p,
                wasserstein=wass,
                js_divergence=js,
                is_drifted=is_drifted,
            )
            scores.append(score)
            if is_drifted:
                drifted.append(feat)

        return DriftReport(
            scores=scores,
            alert=len(drifted) > 0,
            drifted_features=drifted,
        )

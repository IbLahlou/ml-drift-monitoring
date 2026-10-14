from __future__ import annotations

import argparse
import time
from http.server import HTTPServer

from prometheus_client import Gauge, MetricsHandler

from drift_monitor.detector import DriftReport

feature_psi = Gauge(
    "feature_psi",
    "Population Stability Index per feature",
    ["feature_name"],
)
feature_ks_statistic = Gauge(
    "feature_ks_statistic",
    "Kolmogorov-Smirnov statistic per feature",
    ["feature_name"],
)
drift_alert = Gauge(
    "drift_alert",
    "Aggregate drift alert (1 = drift detected, 0 = no drift)",
)
drifted_feature_count = Gauge(
    "drifted_feature_count",
    "Number of features with detected drift",
)

_last_update = Gauge(
    "drift_metrics_last_update_seconds",
    "Unix timestamp of last metrics update",
)


def update_metrics(report: DriftReport) -> None:
    for score in report.scores:
        feature_psi.labels(feature_name=score.feature).set(score.psi)
        feature_ks_statistic.labels(feature_name=score.feature).set(score.ks_statistic)

    drift_alert.set(1 if report.alert else 0)
    drifted_feature_count.set(len(report.drifted_features))
    _last_update.set(time.time())


def run_server(port: int = 8000) -> None:
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"Drift metrics exporter running on :{port}/metrics")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift monitoring Prometheus exporter")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.port)

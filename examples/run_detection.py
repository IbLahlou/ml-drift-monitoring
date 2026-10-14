"""Run drift detection on synthetic data to demonstrate the library."""

import numpy as np
import pandas as pd

from drift_monitor import DriftConfig, DriftDetector


def main():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame({
        "age": rng.normal(35, 10, 10000),
        "income": rng.lognormal(10, 1, 10000),
        "score": rng.uniform(0, 1, 10000),
    })

    current = pd.DataFrame({
        "age": rng.normal(40, 12, 5000),
        "income": rng.lognormal(10, 1, 5000),
        "score": rng.uniform(0.2, 0.9, 5000),
    })

    config = DriftConfig(
        psi_threshold=0.2,
        ks_alpha=0.05,
        features=["age", "income", "score"],
    )
    detector = DriftDetector(config)
    detector.set_reference(reference)

    report = detector.detect(current)
    summary = report.summary()

    print(f"Alert: {summary['alert']}")
    print(f"Drifted features: {summary['drifted_count']}/{summary['total_features']}")
    print()
    for name, scores in summary["scores"].items():
        flag = " << DRIFT" if scores["is_drifted"] else ""
        print(f"  {name}:")
        print(f"    PSI={scores['psi']:.4f}  KS={scores['ks_statistic']:.4f} (p={scores['ks_p_value']:.4f})")
        print(f"    Wasserstein={scores['wasserstein']:.4f}  JS={scores['js_divergence']:.4f}{flag}")


if __name__ == "__main__":
    main()

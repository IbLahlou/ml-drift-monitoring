import numpy as np
import pandas as pd
import pytest

from drift_monitor import DriftConfig, DriftDetector


@pytest.fixture
def base_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 5000),
            "feature_b": rng.uniform(0, 10, 5000),
            "feature_c": rng.exponential(2, 5000),
        }
    )


@pytest.fixture
def detector(base_data):
    config = DriftConfig(features=["feature_a", "feature_b", "feature_c"])
    det = DriftDetector(config)
    det.set_reference(base_data)
    return det


class TestNoDrift:
    def test_same_distribution(self, detector):
        rng = np.random.default_rng(99)
        current = pd.DataFrame(
            {
                "feature_a": rng.normal(0, 1, 5000),
                "feature_b": rng.uniform(0, 10, 5000),
                "feature_c": rng.exponential(2, 5000),
            }
        )
        report = detector.detect(current)
        assert not report.alert
        assert len(report.drifted_features) == 0


class TestCovariateShift:
    def test_mean_shift(self, detector, base_data):
        rng = np.random.default_rng(99)
        current = base_data.copy()
        current["feature_a"] = rng.normal(4, 1, 5000)
        report = detector.detect(current)
        assert report.alert
        assert "feature_a" in report.drifted_features

    def test_variance_change(self, detector, base_data):
        rng = np.random.default_rng(99)
        current = base_data.copy()
        current["feature_b"] = rng.uniform(-20, 30, 5000)
        report = detector.detect(current)
        assert "feature_b" in report.drifted_features


class TestDistributionShift:
    def test_normal_to_bimodal(self, detector, base_data):
        rng = np.random.default_rng(99)
        current = base_data.copy()
        bimodal = np.concatenate(
            [
                rng.normal(-5, 0.5, 2500),
                rng.normal(5, 0.5, 2500),
            ]
        )
        current["feature_a"] = bimodal
        report = detector.detect(current)
        assert report.alert
        assert "feature_a" in report.drifted_features


class TestMissingValues:
    def test_nan_injection(self, detector, base_data):
        rng = np.random.default_rng(99)
        current = base_data.copy()
        mask = rng.random(5000) < 0.5
        current.loc[mask, "feature_a"] = np.nan
        current["feature_a"] = rng.normal(5, 1, 5000)
        current.loc[mask, "feature_a"] = np.nan
        report = detector.detect(current)
        assert report.alert


class TestDetectorSetup:
    def test_missing_reference_raises(self):
        det = DriftDetector()
        with pytest.raises(RuntimeError):
            det.detect(pd.DataFrame({"x": [1, 2, 3]}))

    def test_auto_detect_features(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "num_a": rng.normal(0, 1, 100),
                "num_b": rng.normal(0, 1, 100),
                "cat": ["a"] * 100,
            }
        )
        det = DriftDetector()
        det.set_reference(df)
        assert set(det.config.features) == {"num_a", "num_b"}

    def test_missing_feature_in_current(self, detector):
        current = pd.DataFrame({"feature_a": [1, 2], "feature_b": [3, 4]})
        with pytest.raises(ValueError, match="feature_c"):
            detector.detect(current)

    def test_summary_format(self, detector, base_data):
        report = detector.detect(base_data)
        summary = report.summary()
        assert "alert" in summary
        assert "drifted_count" in summary
        assert "scores" in summary
        assert summary["total_features"] == 3

import math

import numpy as np

from drift_monitor.metrics import (
    compute_csi,
    compute_js_divergence,
    compute_ks,
    compute_psi,
    compute_wasserstein,
)


class TestPSI:
    def test_identical_distributions(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        psi = compute_psi(ref, ref)
        assert psi < 0.05

    def test_shifted_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(2, 1, 5000)
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_empty_arrays(self):
        assert math.isnan(compute_psi(np.array([]), np.array([1, 2, 3])))
        assert math.isnan(compute_psi(np.array([1, 2, 3]), np.array([])))

    def test_constant_values(self):
        ref = np.ones(100)
        cur = np.ones(100)
        assert compute_psi(ref, cur) == 0.0

    def test_nan_handling(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        cur_with_nan = np.append(cur, [np.nan, np.nan, np.nan])
        psi_clean = compute_psi(ref, cur)
        psi_nan = compute_psi(ref, cur_with_nan)
        assert abs(psi_clean - psi_nan) < 0.01


class TestKS:
    def test_identical_distributions(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 5000)
        stat, p = compute_ks(data, data)
        assert stat == 0.0
        assert p == 1.0

    def test_same_distribution_different_samples(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 1, 5000)
        _, p = compute_ks(ref, cur)
        assert p > 0.05

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(3, 1, 5000)
        stat, p = compute_ks(ref, cur)
        assert stat > 0.3
        assert p < 0.001

    def test_empty_arrays(self):
        stat, p = compute_ks(np.array([]), np.array([1, 2]))
        assert math.isnan(stat) and math.isnan(p)


class TestWasserstein:
    def test_identical(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        assert compute_wasserstein(data, data) < 0.001

    def test_known_shift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 10000)
        cur = rng.normal(5, 1, 10000)
        w = compute_wasserstein(ref, cur)
        assert 4.5 < w < 5.5

    def test_empty(self):
        assert math.isnan(compute_wasserstein(np.array([]), np.array([1])))


class TestJSDivergence:
    def test_identical(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 5000)
        js = compute_js_divergence(data, data)
        assert js < 0.01

    def test_upper_bound(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(-10, 0.1, 5000)
        cur = rng.normal(10, 0.1, 5000)
        js = compute_js_divergence(ref, cur)
        assert js <= math.log(2) + 0.01

    def test_empty(self):
        assert math.isnan(compute_js_divergence(np.array([]), np.array([1])))


class TestCSI:
    def test_identical(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 5000)
        assert compute_csi(data, data) < 0.05

    def test_shifted(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(2, 1, 5000)
        assert compute_csi(ref, cur) > 0.1

    def test_constant(self):
        assert compute_csi(np.ones(100), np.ones(100)) == 0.0

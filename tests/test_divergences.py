import math

import pytest
import torch

from ranking_divergence import (
    DIVERGENCES,
    BinScheme,
    coarsen_histogram,
    compute_all,
    rank_wasserstein_from_histograms,
    wasserstein,
)
from ranking_divergence.divergences import (
    cdf_spearman,
    jensen_shannon,
    ks_statistic,
    kl_divergence,
    symmetric_kl_divergence,
    total_variation,
)


def _random_hist(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(size, generator=generator, dtype=torch.float64) + 1e-3
    return raw / raw.sum()


def test_identical_histograms_are_zero_for_every_divergence():
    hist = _random_hist(200, seed=0)
    for name, fn in DIVERGENCES.items():
        assert fn(hist, hist) == pytest.approx(0.0, abs=1e-9), name


def test_symmetric_divergences_are_symmetric():
    p = _random_hist(200, seed=1)
    q = _random_hist(200, seed=2)
    for fn in (jensen_shannon, symmetric_kl_divergence, total_variation, ks_statistic, cdf_spearman):
        assert fn(p, q) == pytest.approx(fn(q, p), rel=1e-9)


def test_kl_is_asymmetric_in_general():
    p = _random_hist(200, seed=3)
    q = _random_hist(200, seed=4)
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_coarsen_conserves_mass_and_bin_count():
    scheme = BinScheme(n_linear=20, n_log=5)
    hist = _random_hist(50257, seed=5)
    coarse = coarsen_histogram(hist, scheme)
    assert float(coarse.sum()) == pytest.approx(1.0, abs=1e-12)
    assert coarse.numel() == len(scheme.boundaries(50257)) - 1
    # 20 singleton head bins + up to 5 log tail bins.
    assert 21 <= coarse.numel() <= 25


def test_wasserstein_log_matches_existing_rank_wasserstein():
    p = _random_hist(500, seed=6)
    q = _random_hist(500, seed=7)
    assert wasserstein(p, q, cost="log") == pytest.approx(
        rank_wasserstein_from_histograms(p, q, normalize=False), abs=1e-12
    )


def test_wasserstein_linear_two_point_shift():
    ref = torch.tensor([1.0, 0.0, 0.0])
    cmp = torch.tensor([0.0, 0.0, 1.0])
    assert wasserstein(ref, cmp, cost="linear") == pytest.approx(2.0)
    assert wasserstein(ref, cmp, cost="log") == pytest.approx(math.log(3.0))


def test_shifting_mass_to_worse_ranks_increases_divergence():
    size = 100
    reference = torch.zeros(size, dtype=torch.float64)
    reference[0] = 1.0
    near = torch.zeros(size, dtype=torch.float64)
    near[5] = 1.0
    far = torch.zeros(size, dtype=torch.float64)
    far[50] = 1.0
    # Ground-metric divergences move with how far the mass is displaced; f-divergences
    # and max-CDF summaries like KS are intentionally excluded because they can be flat
    # for one-hot support shifts.
    for fn in (
        lambda a, b: wasserstein(a, b, cost="log"),
        lambda a, b: wasserstein(a, b, cost="linear"),
    ):
        assert fn(reference, near) < fn(reference, far)


def test_compute_all_returns_every_registered_name():
    p = _random_hist(300, seed=8)
    q = _random_hist(300, seed=9)
    result = compute_all(p, q)
    assert set(result) == set(DIVERGENCES)
    assert all(isinstance(value, float) for value in result.values())

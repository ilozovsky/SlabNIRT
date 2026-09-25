"""Statistics of average_attributes: MAD factor, standard errors, sites and MSE scales."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from slabnirt import average_attributes
from slabnirt.analysis import _aggregate_scalar, _mad_consistency_factor


def _mse(value: float, scales: list[int]) -> tuple[float, dict[str, list[float]]]:
    return value, {"Scale": list(scales), "Value": [value + scale / 10.0 for scale in scales]}


@pytest.mark.parametrize("n", [2, 5, 20, 150])
def test_mad_factor_makes_the_scale_estimate_unbiased_under_normality(n: int) -> None:
    """E[C_n * MAD_n] must be 1 for N(0, 1); the tolerance is five standard errors
    of the simulation itself, so the table (n <= 100) and the large-n
    approximation (n = 150) are both checked independently of the code. At
    small n the asymptotic constant alone is biased by more than 10 %."""
    rng = np.random.default_rng(20260910 + n)
    replicates = 60_000
    sample = rng.standard_normal((replicates, n))
    mad = np.median(np.abs(sample - np.median(sample, axis=1, keepdims=True)), axis=1)

    corrected = _mad_consistency_factor(n) * mad
    standard_error = corrected.std(ddof=1) / math.sqrt(replicates)

    assert abs(corrected.mean() - 1.0) < 5.0 * standard_error
    if n <= 5:
        assert abs(1.4826 * mad.mean() - 1.0) > 0.1


def test_standard_errors_follow_the_documented_formulas() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    n = values.size
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = _mad_consistency_factor(n) * mad

    robust = _aggregate_scalar(values, {"mean", "median"}, "mad")
    classical = _aggregate_scalar(values, {"mean", "median"}, "std")

    assert robust["mean"] == pytest.approx(6.2)
    assert robust["median"] == 4.0
    assert robust["mean_error"] == pytest.approx(sigma / math.sqrt(n), rel=1e-14)
    assert robust["median_error"] == pytest.approx(
        math.sqrt(math.pi / 2.0) * sigma / math.sqrt(n), rel=1e-14
    )
    sem = float(np.std(values, ddof=1) / math.sqrt(n))
    assert classical["mean_error"] == pytest.approx(sem, rel=1e-14)
    assert classical["median_error"] == pytest.approx(sem, rel=1e-14)


def test_aggregate_uses_only_finite_observations() -> None:
    stats = _aggregate_scalar(
        np.array([1.0, 2.0, np.nan, np.inf, -np.inf]), {"mean", "median"}, "std"
    )

    assert stats["n_used"] == 2
    assert stats["n_dropped"] == 3
    assert (stats["min"], stats["max"], stats["mean"], stats["median"]) == (1.0, 2.0, 1.5, 1.5)
    assert stats["mean_error"] == pytest.approx(0.5)
    assert stats["median_error"] == pytest.approx(0.5)


def test_all_nonfinite_observations_give_nan_statistics_not_an_error() -> None:
    stats = _aggregate_scalar(np.array([np.nan, np.inf, -np.inf]), {"mean", "median"}, "std")

    assert stats["n_used"] == 0
    assert stats["n_dropped"] == 3
    assert all(
        np.isnan(stats[key])
        for key in ("min", "max", "mean", "mean_error", "median", "median_error")
    )


def test_sites_share_one_mapping_when_coordinates_agree_to_six_decimals(site_trace) -> None:
    traces = [
        site_trace("site0_repeat0", {"f": 1.0}, x=0.0),
        site_trace("site0_repeat1", {"f": 3.0}, x=0.0000004),
        site_trace("site1", {"f": 5.0}, x=0.0000011),
        site_trace("site2", {"f": 7.0}, x=1.0),
    ]

    average_attributes(traces, attrs=["f"])

    assert traces[0].attributes_avg is traces[1].attributes_avg
    assert traces[0].attributes_avg["f"]["mean"] == 2.0
    assert traces[2].attributes_avg is not traces[0].attributes_avg
    assert traces[2].attributes_avg["f"]["n_used"] == 1
    assert len({id(trace.attributes_avg) for trace in traces}) == 3


def test_channels_at_one_site_are_pooled(site_trace) -> None:
    traces = [
        site_trace("a", {"f": 1.0}, channel=1),
        site_trace("a", {"f": 3.0}, channel=2),
        site_trace("b", {"f": 8.0}, channel=1),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        average_attributes(traces, attrs=["f"])

    assert traces[0].attributes_avg["f"]["n_used"] == 3
    assert traces[0].attributes_avg["f"]["mean"] == 4.0


def test_one_statistic_can_be_named_as_a_string(site_trace) -> None:
    traces = [site_trace("a", {"f": 1.0}), site_trace("b", {"f": 3.0})]

    average_attributes(traces, average_algorithm="median")

    stats = traces[0].attributes_avg["f"]
    assert stats["median"] == 2.0
    assert stats["mean"] is None


def test_a_single_observation_has_no_error_estimate(site_trace) -> None:
    trace = site_trace("single", {"f": 3.0})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        average_attributes([trace], attrs=["f"])

    stats = trace.attributes_avg["f"]
    assert (stats["n_used"], stats["mean"], stats["median"]) == (1, 3.0, 3.0)
    assert np.isnan(stats["mean_error"]) and np.isnan(stats["median_error"])


def test_identical_repeat_values_give_a_zero_robust_error(site_trace) -> None:
    """MAD of identical values is 0, so the robust error is exactly 0.0; a
    downstream inverse-variance weight must guard against it."""
    traces = [site_trace(str(index), {"peak_frequency": 66.67}) for index in range(4)]

    average_attributes(traces, attrs=["peak_frequency"])

    assert traces[0].attributes_avg["peak_frequency"]["mean_error"] == 0.0


def test_valid_mse_statistics_are_exact(site_trace) -> None:
    """The scales are those of the first entry; the float scales of the second
    compare equal to them."""
    traces = [
        site_trace("valid-1", {"MSE": (1.0, {"Scale": [1, 2], "Value": [10.0, 20.0]})}),
        site_trace("valid-2", {"MSE": (3.0, {"Scale": [1.0, 2.0], "Value": [14.0, 22.0]})}),
    ]

    average_attributes(traces, attrs=["MSE"], error_algorithm="std")

    scalar_stats, per_scale = traces[0].attributes_avg["MSE"]
    assert traces[1].attributes_avg is traces[0].attributes_avg
    assert scalar_stats == pytest.approx(
        {
            "n_used": 2,
            "n_dropped": 0,
            "min": 1.0,
            "max": 3.0,
            "mean": 2.0,
            "mean_error": 1.0,
            "median": 2.0,
            "median_error": 1.0,
        }
    )
    assert per_scale["Scale"] == [1, 2]
    assert [stats["mean"] for stats in per_scale["Value"]] == [12.0, 21.0]
    assert [stats["mean_error"] for stats in per_scale["Value"]] == [2.0, 1.0]


def test_multiple_mse_variants_are_kept_apart(site_trace) -> None:
    traces = [
        site_trace(
            f"repeat{index}",
            {"MSE_alt": _mse(101.0 + offset, [1, 2, 3]), "MSE": _mse(1.0 + offset, [1, 2])},
        )
        for index, offset in enumerate((0.0, 2.0))
    ]

    average_attributes(traces, attrs="all", error_algorithm="std")

    averaged = traces[0].attributes_avg
    assert list(averaged) == ["MSE_alt", "MSE"]
    assert averaged["MSE"][0]["median"] == 2.0
    assert averaged["MSE_alt"][0]["median"] == 102.0
    assert averaged["MSE"][1]["Scale"] == [1, 2]
    assert averaged["MSE_alt"][1]["Scale"] == [1, 2, 3]

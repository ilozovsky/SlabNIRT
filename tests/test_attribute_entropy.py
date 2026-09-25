"""Sample entropy and multiscale entropy.

``slabnirt._wu_rcmse`` implements refined composite multiscale entropy after
Wu et al. (2014), Physics Letters A 378, 1369-1374. The first test checks it
against a small pure-Python version written from the paper's equations; the
others check the default parameters, the stored values and the behavior on
degenerate or non-finite signals.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from slabnirt import calculate_signal_parameters
from slabnirt._wu_rcmse import rcmse


def _paper_rcmse(x, scales, m, r):
    """RCMSE from Wu et al. (2014), Eqs. (1)-(4) and (8), as plain loops."""
    values = []
    for tau in scales:
        n_m = n_m1 = 0
        for k in range(tau):  # Eq. (4): one coarse-grained series per start offset
            blocks = (len(x) - k) // tau
            y = [np.mean(x[k + j * tau : k + (j + 1) * tau]) for j in range(blocks)]
            for i in range(len(y) - m):  # Eq. (1): templates i = 1 .. N - m
                for j in range(i + 1, len(y) - m):
                    if max(abs(y[i + q] - y[j + q]) for q in range(m)) <= r:  # Eqs. (2)-(3)
                        n_m += 1
                        n_m1 += abs(y[i + m] - y[j + m]) <= r
        values.append(-math.log(n_m1 / n_m) if n_m and n_m1 else math.nan)  # Eq. (8)
    return values


def _entropy_test_signals():
    rng = np.random.default_rng(20260914)
    normal = rng.normal(size=40)
    time = np.arange(60) / 25_000.0
    return [
        (normal, 0.2 * np.std(normal, ddof=1)),
        (rng.integers(-2, 3, size=50).astype(float), 1.0),  # exact ties at distance r
        (np.sin(2 * np.pi * 1500 * time) * np.exp(-400 * time), 0.05),
    ]


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize(
    "signal, tolerance", _entropy_test_signals(), ids=["normal", "integers", "damped sine"]
)
def test_rcmse_matches_the_paper_equations(signal, tolerance, dimension) -> None:
    """Both count the same integer pairs, so the logarithms agree exactly."""
    scales = [1, 2, 3, 4, 5, 6]

    values, details = rcmse(signal, scale=scales, dimension=dimension, tolerance=tolerance)

    expected = _paper_rcmse(list(signal), scales, dimension, tolerance)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(details["Scale"], scales)
    assert details["Dimension"] == dimension
    assert details["Tolerance"] == tolerance


def test_default_parameters_follow_the_published_rule(synthetic_trace) -> None:
    """m = 2, r = 0.2 SD, and every scale that keeps at least 10**m coarse-grained samples."""
    rng = np.random.default_rng(3)
    short = synthetic_trace(rng.normal(size=256), name="short")
    long = synthetic_trace(rng.normal(size=1024), name="long")

    calculate_signal_parameters([short, long], attrs=["sampen", "MSE"])

    for trace, scales in ((short, [1, 2]), (long, list(range(1, 11)))):
        value, details = trace.attributes["MSE"]
        assert details["Dimension"] == 2
        assert details["Tolerance"] == 0.2 * np.std(trace.signal_norm, ddof=1)
        np.testing.assert_array_equal(details["Scale"], scales)
        assert trace.attributes["sampen"] == details["Value"][0]


def test_mse_scalar_is_the_mean_of_the_finite_scale_values(synthetic_trace) -> None:
    rng = np.random.default_rng(5)
    trace = synthetic_trace(rng.normal(size=300))

    calculate_signal_parameters([trace], attrs=["sampen", "MSE"], mse_params={"scale": [1, 2, 3]})
    calculate_signal_parameters(
        [trace], attrs=["MSE"], name_suffix="_one", mse_params={"scale": [1]}
    )

    value, details = trace.attributes["MSE"]
    assert value == np.mean(details["Value"])
    assert trace.attributes["MSE_one"][0] == trace.attributes["sampen"]


def test_no_match_at_the_longer_template_gives_nan_and_a_regular_signal_gives_plus_zero(
    synthetic_trace,
) -> None:
    """Three identical (0, 0) templates match at length 2, but none of their
    continuations (5, 6, 7) match at length 3: the ratio is 0/6, undefined.
    Every template of an alternating signal continues the same way: -ln(1)."""
    undefined = synthetic_trace([0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 7.0], name="undefined")
    regular = synthetic_trace(np.tile([0.0, 1.0], 20), name="regular")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_signal_parameters(
            [undefined, regular],
            attrs=["sampen", "MSE"],
            mse_params={"scale": [1], "dimension": 2, "tolerance": 0.05},
            sampen_params={"dimension": 2, "tolerance": 0.05},
        )

    value, details = undefined.attributes["MSE"]
    assert np.isnan(value) and np.isnan(details["Value"][0])
    assert np.isnan(undefined.attributes["sampen"])
    assert details["Counts"] == [(6, 0)]
    messages = sorted(str(w.message) for w in caught)
    assert len(messages) == 2 and all(m.startswith("undefined: ") for m in messages)
    assert "MSE is NaN because no template pairs match" in messages[0]
    assert "sampen is NaN because no template pairs match" in messages[1]
    value, details = regular.attributes["MSE"]
    assert value == 0.0 and math.copysign(1.0, value) == 1.0
    assert math.copysign(1.0, regular.attributes["sampen"]) == 1.0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_signal_warns_and_skips_the_entropy_attributes(
    synthetic_trace, bad_value: float
) -> None:
    good = synthetic_trace(np.sin(np.arange(64) / 3.0), name="good")
    bad = synthetic_trace(np.r_[bad_value, np.sin(np.arange(63) / 3.0)], name="bad")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_signal_parameters(
            [bad, good], attrs=["sampen", "MSE", "norm_signal_energy"], mse_params={"scale": [1, 2]}
        )

    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("bad: sampen is NaN because" in m for m in messages)
    assert any("bad: MSE is not stored because" in m for m in messages)
    assert np.isnan(bad.attributes["sampen"])
    assert "MSE" not in bad.attributes
    assert np.isnan(bad.attributes["norm_signal_energy"])
    assert np.isfinite(good.attributes["sampen"])
    assert "MSE" in good.attributes

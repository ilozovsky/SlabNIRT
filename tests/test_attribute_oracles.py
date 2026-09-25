"""Analytic oracles for the non-entropy attributes of calculate_signal_parameters.

Each test compares one attribute against a value derived without reference to
the implementation: a closed-form integral, a symmetry argument, a hand-built
spectrum, or a textbook formula. The entropy attributes (sampen, MSE) are
covered separately.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from slabnirt import calculate_signal_parameters

# Sums over a few thousand doubles accumulate round-off of order 1e-13 relative;
# identities checked against a closed form use that, exact re-computations of
# the same arithmetic use 1e-14.
CLOSED_FORM_RTOL = 1e-13
SAME_ARITHMETIC_RTOL = 1e-14


@pytest.mark.parametrize(
    ("options", "area", "centroid"),
    [({}, 1.0, 11 / 7), ({"use_power_spectrum": False}, 4 / 3, 4 / 3)],
    ids=["default", "amplitude"],
)
def test_spectral_default_is_power_with_amplitude_opt_in(
    spectrum_trace, options, area, centroid
) -> None:
    """At 0, 1, 2 Hz, amplitudes 1, 2, 3 give powers 1, 4, 9."""
    trace = spectrum_trace([1.0, 2.0, 3.0])
    calculate_signal_parameters(
        [trace], attrs=["normalized_spectrum_area", "spectral_centroid"], **options
    )
    np.testing.assert_allclose(
        [trace.attributes["normalized_spectrum_area"], trace.attributes["spectral_centroid"]],
        [area, centroid],
        rtol=SAME_ARITHMETIC_RTOL,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("sampling_rate", "duration"),
    [(1_000.0, 1.0), (2_000.0, 1.0), (2_000.0, 2.0)],
)
def test_energy_of_a_unit_sine_is_half_its_duration(
    synthetic_trace, sampling_rate: float, duration: float
) -> None:
    """The integral of sin^2 over whole periods is T/2, in seconds."""
    time = np.arange(int(sampling_rate * duration)) / sampling_rate
    trace = synthetic_trace(np.sin(2.0 * np.pi * 10.0 * time), sampling_rate=sampling_rate)
    trace.signal_norm = trace.signal

    calculate_signal_parameters([trace], attrs=["norm_signal_energy"])

    assert trace.attributes["norm_signal_energy"] == pytest.approx(
        duration / 2.0, rel=CLOSED_FORM_RTOL
    )


def test_energy_of_a_constant_unit_signal_is_its_duration(synthetic_trace) -> None:
    trace = synthetic_trace(np.ones(64), sampling_rate=2_000.0)
    trace.signal_norm = trace.signal

    calculate_signal_parameters([trace], attrs=["norm_signal_energy"])

    assert trace.attributes["norm_signal_energy"] == pytest.approx(64 / 2_000.0, rel=0.0)


@pytest.mark.parametrize(
    ("time", "reason"),
    [
        (np.array([0.0]), "at least two time samples"),
        (np.array([0.0, 0.1, 0.25]), "uniformly sampled"),
        (np.array([0.0, 0.1, 0.1]), "strictly increasing"),
    ],
)
def test_energy_rejects_invalid_time_axes(synthetic_trace, time, reason: str) -> None:
    trace = synthetic_trace(np.ones(time.size))
    trace.t = time
    trace.signal_norm = trace.signal.copy()

    with pytest.warns(RuntimeWarning, match=reason):
        calculate_signal_parameters([trace], attrs=["norm_signal_energy"])

    assert np.isnan(trace.attributes["norm_signal_energy"])


def test_area_of_a_flat_spectrum_is_its_bandwidth(spectrum_trace) -> None:
    """A flat spectrum normalizes to 1 everywhere, so the area is the band width in Hz."""
    trace = spectrum_trace(np.ones(501), freqs=np.linspace(0.0, 500.0, 501))

    calculate_signal_parameters([trace], attrs=["normalized_spectrum_area"])

    assert trace.attributes["normalized_spectrum_area"] == pytest.approx(500.0, rel=0.0)


def test_area_is_the_trapezoid_of_the_peak_normalized_spectrum(spectrum_trace) -> None:
    spectrum = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    trace = spectrum_trace(spectrum, freqs=freqs)

    calculate_signal_parameters(
        [trace], use_power_spectrum=False, attrs=["normalized_spectrum_area"]
    )

    assert trace.attributes["normalized_spectrum_area"] == pytest.approx(
        trapezoid(spectrum / spectrum.max(), freqs), rel=SAME_ARITHMETIC_RTOL
    )


def test_centroid_of_a_symmetric_peak_is_its_center_frequency(spectrum_trace) -> None:
    freqs = np.linspace(0.0, 200.0, 2_001)
    trace = spectrum_trace(np.exp(-0.5 * ((freqs - 100.0) / 8.0) ** 2), freqs=freqs)

    calculate_signal_parameters([trace], attrs=["spectral_centroid"])

    assert trace.attributes["spectral_centroid"] == pytest.approx(100.0, rel=CLOSED_FORM_RTOL)


def test_centroid_is_the_weighted_mean_over_bins_above_the_threshold(spectrum_trace) -> None:
    """Only bins at or above centroid_threshold times the peak carry weight."""
    spectrum = np.array([0.02, 1.0, 0.5, 0.04])
    freqs = np.array([0.0, 100.0, 200.0, 300.0])
    trace = spectrum_trace(spectrum, freqs=freqs)

    calculate_signal_parameters(
        [trace], use_power_spectrum=False, attrs=["spectral_centroid"], centroid_threshold=0.05
    )

    weights = spectrum / spectrum.max()
    kept = weights >= 0.05
    assert kept.tolist() == [False, True, True, False]
    assert trace.attributes["spectral_centroid"] == pytest.approx(
        np.average(freqs[kept], weights=weights[kept]), rel=SAME_ARITHMETIC_RTOL
    )


def test_area_over_centroid_is_the_ratio_of_the_two_attributes(spectrum_trace) -> None:
    freqs = np.linspace(0.0, 300.0, 301)
    trace = spectrum_trace(np.exp(-0.5 * ((freqs - 120.0) / 20.0) ** 2), freqs=freqs)

    calculate_signal_parameters(
        [trace],
        attrs=["normalized_spectrum_area", "spectral_centroid", "norm_spectrum_area_over_centroid"],
    )

    attributes = trace.attributes
    assert attributes["norm_spectrum_area_over_centroid"] == pytest.approx(
        attributes["normalized_spectrum_area"] / attributes["spectral_centroid"],
        rel=SAME_ARITHMETIC_RTOL,
    )


def test_peak_frequency_of_a_single_bin_spectrum(spectrum_trace) -> None:
    freqs = np.linspace(0.0, 500.0, 501)
    spectrum = np.zeros(501)
    spectrum[173] = 1.0
    trace = spectrum_trace(spectrum, freqs=freqs)

    calculate_signal_parameters([trace], attrs=["peak_frequency"])

    assert trace.attributes["peak_frequency"] == freqs[173]


@pytest.mark.parametrize("damping", [0.02, 0.05])
@pytest.mark.parametrize("use_power_spectrum", [False, True])
def test_peak_width_of_an_sdof_resonance_recovers_its_damping_ratio(
    spectrum_trace, damping: float, use_power_spectrum: bool
) -> None:
    """Half-power bandwidth of an SDOF receptance: (f_high - f_low) / (2 f_n) ~ zeta.

    The half-power rule is a small-damping approximation whose relative error
    grows as about 2 * zeta**2 (0.08 % at zeta 0.02, 0.5 % at zeta 0.05), so the
    tolerance is 3 * zeta**2.
    """
    natural_frequency = 100.0
    freqs = np.linspace(0.0, 300.0, 30_001)
    ratio = freqs / natural_frequency
    receptance = 1.0 / np.sqrt((1.0 - ratio**2) ** 2 + (2.0 * damping * ratio) ** 2)
    trace = spectrum_trace(receptance, freqs=freqs)

    calculate_signal_parameters(
        [trace],
        attrs=["spectral_peak_width"],
        freq_range=(50.0, 150.0),
        use_power_spectrum=use_power_spectrum,
    )

    assert trace.attributes["spectral_peak_width"] == pytest.approx(damping, rel=3.0 * damping**2)


def test_peak_width_of_a_gaussian_peak_matches_its_half_power_width(spectrum_trace) -> None:
    """For amplitude exp(-x^2 / 2 s^2) the 1/sqrt(2) crossings sit at x = +/- s sqrt(ln 2).

    Linear interpolation between 0.005 Hz bins is accurate to well below 1e-6.
    """
    sigma = 10.0
    freqs = np.linspace(0.0, 200.0, 40_001)
    trace = spectrum_trace(np.exp(-0.5 * ((freqs - 100.0) / sigma) ** 2), freqs=freqs)

    calculate_signal_parameters([trace], attrs=["spectral_peak_width"], freq_range=(50.0, 150.0))

    expected = 2.0 * sigma * np.sqrt(np.log(2.0)) / (2.0 * 100.0)
    assert trace.attributes["spectral_peak_width"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        ([1.0, 1.0, 1.0, 1.0], 1.0),
        (
            [1.0, 2.0, 3.0, 4.0],
            (1.0 * 4.0 * 9.0 * 16.0) ** 0.25 / np.mean([1.0, 4.0, 9.0, 16.0]),
        ),
        ([1.0, 0.0, 2.0, 3.0], 0.0),
    ],
)
def test_flatness_is_the_geometric_over_arithmetic_mean_of_power(
    spectrum_trace, spectrum, expected: float
) -> None:
    trace = spectrum_trace(spectrum)

    calculate_signal_parameters([trace], attrs=["spectral_flatness"])

    assert trace.attributes["spectral_flatness"] == pytest.approx(
        expected, rel=SAME_ARITHMETIC_RTOL, abs=1e-15
    )


def test_flatness_is_scale_invariant_and_mode_independent(spectrum_trace) -> None:
    results = []
    for scale in (1.0, 1e-8, 1e100):
        trace = spectrum_trace(scale * np.array([1.0, 2.0, 3.0, 4.0]))
        calculate_signal_parameters([trace], use_power_spectrum=False, attrs=["spectral_flatness"])
        results.append(trace.attributes["spectral_flatness"])
    power_trace = spectrum_trace([1.0, 2.0, 3.0, 4.0])
    calculate_signal_parameters([power_trace], attrs=["spectral_flatness"], use_power_spectrum=True)
    results.append(power_trace.attributes["spectral_flatness"])

    np.testing.assert_allclose(results, results[0], rtol=SAME_ARITHMETIC_RTOL)


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        ([1.0, 0.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 1.0, 1.0], 1.0),
        ([1.0, 2.0, 0.0, 0.0], 0.36096404744368116),
        ([2.0], 0.0),
    ],
)
def test_spectral_entropy_matches_normalized_shannon_entropy(
    spectrum_trace, spectrum, expected: float
) -> None:
    """Power PMF over the selected bins, divided by log2(N)."""
    trace = spectrum_trace(spectrum)

    calculate_signal_parameters([trace], attrs=["spectral_entropy"])

    assert trace.attributes["spectral_entropy"] == pytest.approx(
        expected, rel=SAME_ARITHMETIC_RTOL, abs=1e-14
    )


def test_spectral_entropy_amplitude_and_power_modes_agree(spectrum_trace) -> None:
    amplitude_trace = spectrum_trace([1.0, 2.0, 3.0, 4.0], name="amplitude")
    power_trace = spectrum_trace([1.0, 2.0, 3.0, 4.0], name="power")

    calculate_signal_parameters(
        [amplitude_trace], use_power_spectrum=False, attrs=["spectral_entropy"]
    )
    calculate_signal_parameters([power_trace], attrs=["spectral_entropy"], use_power_spectrum=True)

    assert power_trace.attributes["spectral_entropy"] == pytest.approx(
        amplitude_trace.attributes["spectral_entropy"], rel=SAME_ARITHMETIC_RTOL
    )


def test_void_index_is_the_peak_over_the_median_of_the_two_bands(spectrum_trace) -> None:
    freqs = np.linspace(0.0, 1_000.0, 1_001)
    trace = spectrum_trace(np.where(freqs <= 200.0, 3.0, 1.5), freqs=freqs)

    calculate_signal_parameters([trace], use_power_spectrum=False, attrs=["void_index"])

    assert trace.attributes["void_index"] == pytest.approx(3.0 / 1.5, rel=0.0)


@pytest.mark.parametrize(
    ("max_freq_range", "mav_freq_range", "expected"),
    [
        ((0.0, None), (200.0, None), 2.0 / 1.6),
        ((None, 200.0), (None, 800.0), 1.2 / 1.4),
    ],
    ids=["open-upper", "open-lower"],
)
def test_void_index_accepts_open_frequency_bounds(
    spectrum_trace, max_freq_range, mav_freq_range, expected: float
) -> None:
    trace = spectrum_trace(np.linspace(1.0, 2.0, 501), freqs=np.linspace(0.0, 1_000.0, 501))

    calculate_signal_parameters(
        [trace],
        use_power_spectrum=False,
        attrs=["void_index"],
        void_index_params={"max_freq_range": max_freq_range, "mav_freq_range": mav_freq_range},
    )

    assert trace.attributes["void_index"] == pytest.approx(expected)


def test_void_index_follows_the_power_switch(spectrum_trace) -> None:
    """The ratio of squared values is the square of the amplitude ratio."""
    freqs = np.linspace(0.0, 1_000.0, 1_001)
    trace = spectrum_trace(np.where(freqs <= 200.0, 3.0, 1.5), freqs=freqs)

    calculate_signal_parameters(
        [trace], use_power_spectrum=False, attrs=["void_index"], name_suffix="_amp"
    )
    calculate_signal_parameters(
        [trace], attrs=["void_index"], name_suffix="_pow", use_power_spectrum=True
    )

    assert trace.attributes["void_index_pow"] == pytest.approx(
        trace.attributes["void_index_amp"] ** 2, rel=SAME_ARITHMETIC_RTOL
    )

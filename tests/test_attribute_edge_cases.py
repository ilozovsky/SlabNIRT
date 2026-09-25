"""Missing values and band selection in calculate_signal_parameters."""

from __future__ import annotations

import re

import numpy as np
import pytest

from slabnirt import ButterworthFilterConfig, calculate_signal_parameters, preprocess_traces

SPECTRUM_ATTRIBUTES = [
    "normalized_spectrum_area",
    "spectral_centroid",
    "norm_spectrum_area_over_centroid",
    "peak_frequency",
]


@pytest.mark.parametrize(
    ("spectrum", "reason"),
    [
        ([1.0, np.nan, 2.0, 0.5], "non-finite values"),
        ([1.0, np.inf, 2.0, 0.5], "non-finite values"),
        ([1.0, -3.0, 2.0, 0.5], "negative values"),
        ([0.0, 0.0, 0.0, 0.0], "zero total power"),
    ],
)
def test_unusable_spectrum_gives_nan_with_a_named_warning(
    spectrum_trace, spectrum, reason: str
) -> None:
    trace = spectrum_trace(spectrum, name="bad")

    with pytest.warns(RuntimeWarning) as caught:
        calculate_signal_parameters([trace], attrs=SPECTRUM_ATTRIBUTES, use_power_spectrum=False)

    messages = [str(record.message) for record in caught]
    for attribute in SPECTRUM_ATTRIBUTES:
        assert np.isnan(trace.attributes[attribute])
        pattern = f"bad: {attribute} is NaN because .*{reason}"
        assert any(re.match(pattern, message) for message in messages), attribute


def test_area_over_centroid_is_nan_when_the_centroid_is_zero(spectrum_trace) -> None:
    """Only the 0 Hz bin reaches the threshold, so the centroid is 0 Hz."""
    trace = spectrum_trace([1.0, 0.01, 0.01, 0.01, 0.01], name="dc")

    with pytest.warns(
        RuntimeWarning, match="dc: norm_spectrum_area_over_centroid is NaN because the spectral"
    ):
        calculate_signal_parameters(
            [trace], attrs=["spectral_centroid", "norm_spectrum_area_over_centroid"]
        )

    assert trace.attributes["spectral_centroid"] == 0.0
    assert np.isnan(trace.attributes["norm_spectrum_area_over_centroid"])


def test_an_unknown_void_index_key_is_rejected(spectrum_trace) -> None:
    trace = spectrum_trace(np.ones(8))

    with pytest.raises(TypeError, match="max_freq"):
        calculate_signal_parameters(
            [trace], attrs=["void_index"], void_index_params={"max_freq": (0.0, 50.0)}
        )


def test_area_over_centroid_ignores_an_area_stored_for_another_band(spectrum_trace) -> None:
    """The ratio uses the area of its own band. Over 0-100 Hz the normalized
    spectrum is [0.5, 1.0]: area 75 Hz, centroid (0*0.5 + 100*1.0) / 1.5 =
    66.67 Hz, ratio 1.125."""
    trace = spectrum_trace([1.0, 2.0, 3.0, 4.0], freqs=[0.0, 100.0, 200.0, 300.0])
    calculate_signal_parameters(
        [trace],
        attrs=["normalized_spectrum_area"],
        freq_range=(0.0, 300.0),
        use_power_spectrum=False,
    )
    wide_area = trace.attributes["normalized_spectrum_area"]

    calculate_signal_parameters(
        [trace],
        attrs=["norm_spectrum_area_over_centroid"],
        freq_range=(0.0, 100.0),
        use_power_spectrum=False,
    )

    assert wide_area == pytest.approx(187.5)
    assert trace.attributes["norm_spectrum_area_over_centroid"] == pytest.approx(1.125)
    assert trace.attributes["normalized_spectrum_area"] == pytest.approx(187.5)


@pytest.mark.parametrize(
    ("spectrum", "params", "reason"),
    [
        (np.linspace(1.0, 2.0, 501), {"mav_freq_range": (2_000.0, 3_000.0)}, "mav_freq_range"),
        (np.linspace(1.0, 2.0, 501), {"max_freq_range": (2_000.0, 3_000.0)}, "max_freq_range"),
        (np.where(np.linspace(0.0, 1_000.0, 501) >= 200.0, 0.0, 1.0), {}, "is zero"),
        (np.where(np.linspace(0.0, 1_000.0, 501) < 100.0, np.nan, 1.0), {}, "non-finite"),
    ],
    ids=["empty-mav-band", "empty-max-band", "zero-mav-level", "nan-in-band"],
)
def test_void_index_is_nan_with_a_warning_for_degenerate_bands(
    spectrum_trace, spectrum, params: dict, reason: str
) -> None:
    trace = spectrum_trace(spectrum, freqs=np.linspace(0.0, 1_000.0, 501), name="void")

    with pytest.warns(RuntimeWarning, match=f"void: void_index is NaN because .*{reason}"):
        calculate_signal_parameters([trace], attrs=["void_index"], void_index_params=params)

    assert np.isnan(trace.attributes["void_index"])


def test_filtered_spectral_attributes_default_to_the_filter_passband(synthetic_trace) -> None:
    sampling_rate = 4_000.0
    time = np.arange(8_000, dtype=float) / sampling_rate
    signal = np.sin(2.0 * np.pi * 100.0 * time) + 0.2 * np.sin(2.0 * np.pi * 600.0 * time)
    trace = synthetic_trace(signal, sampling_rate=sampling_rate)
    filtered = preprocess_traces(
        [trace],
        filter_options=ButterworthFilterConfig(
            min_frequency_hz=10.0, max_frequency_hz=300.0, order=4
        ),
    )[0]

    calculate_signal_parameters(
        [filtered], attrs=["spectral_flatness", "spectral_entropy"], name_suffix="_auto"
    )
    calculate_signal_parameters(
        [filtered],
        attrs=["spectral_flatness", "spectral_entropy"],
        name_suffix="_explicit",
        freq_range=(10.0, 300.0),
    )
    calculate_signal_parameters(
        [filtered], attrs=["spectral_entropy"], name_suffix="_override", freq_range=(80.0, 120.0)
    )

    attributes = filtered.attributes
    assert attributes["spectral_flatness_auto"] == attributes["spectral_flatness_explicit"]
    assert attributes["spectral_entropy_auto"] == attributes["spectral_entropy_explicit"]
    assert attributes["spectral_entropy_override"] != pytest.approx(
        attributes["spectral_entropy_auto"]
    )


def test_spectral_entropy_honors_the_selected_band(spectrum_trace) -> None:
    trace = spectrum_trace([1.0, 0.0, 1.0, 1.0])

    calculate_signal_parameters(
        [trace], attrs=["spectral_entropy"], name_suffix="_low", freq_range=(0.0, 1.0)
    )
    calculate_signal_parameters(
        [trace], attrs=["spectral_entropy"], name_suffix="_high", freq_range=(2.0, 3.0)
    )

    assert trace.attributes["spectral_entropy_low"] == pytest.approx(0.0)
    assert trace.attributes["spectral_entropy_high"] == pytest.approx(1.0)


def test_void_index_bands_are_independent_of_freq_range(spectrum_trace) -> None:
    """The void index has its own two bands and ignores the general selection."""
    freqs = np.linspace(0.0, 1_000.0, 1_001)
    trace = spectrum_trace(np.where(freqs <= 200.0, 3.0, 1.5), freqs=freqs)

    calculate_signal_parameters([trace], attrs=["void_index"], name_suffix="_full")
    calculate_signal_parameters(
        [trace], attrs=["void_index"], name_suffix="_narrow", freq_range=(0.0, 50.0)
    )

    assert trace.attributes["void_index_narrow"] == trace.attributes["void_index_full"]


def test_peak_width_is_missing_when_the_band_has_no_crossings(spectrum_trace) -> None:
    freqs = np.linspace(0.0, 200.0, 4_001)
    trace = spectrum_trace(np.exp(-0.5 * ((freqs - 100.0) / 10.0) ** 2), freqs=freqs)

    with pytest.warns(RuntimeWarning, match="does not cross.*within freq_range"):
        calculate_signal_parameters(
            [trace], attrs=["spectral_peak_width"], freq_range=(98.0, 102.0)
        )

    assert np.isnan(trace.attributes["spectral_peak_width"])


def test_peak_width_also_stores_the_peak_frequency(spectrum_trace) -> None:
    """Documented side effect: the width needs the peak, so the peak is stored too."""
    freqs = np.linspace(0.0, 200.0, 4_001)
    trace = spectrum_trace(np.exp(-0.5 * ((freqs - 100.0) / 10.0) ** 2), freqs=freqs)

    calculate_signal_parameters(
        [trace], attrs=["spectral_peak_width"], freq_range=(50.0, 150.0), name_suffix="_w"
    )

    assert set(trace.attributes) == {"spectral_peak_width_w", "peak_frequency_w"}
    assert trace.attributes["peak_frequency_w"] == pytest.approx(100.0)

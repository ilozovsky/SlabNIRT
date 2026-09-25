"""The curves plot_traces draws: filtered and unfiltered signals, spectra, dB, markers, MSE.

All traces are synthetic with declared sampling parameters.
"""

from __future__ import annotations

from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from slabnirt import ButterworthFilterConfig, TraceData, preprocess_traces
from slabnirt.plot_config import TracePlotConfig
from slabnirt.plot_traces import plot_traces

plot_traces_module = import_module("slabnirt.plot_traces")

SAMPLING_RATE = 2_000.0
# plot_traces_module.plt is matplotlib.pyplot itself, so disabling its close()
# for a capture disables it everywhere; keep the real one for cleaning up.
_close_figures = plt.close


@pytest.fixture(autouse=True)
def _close_every_figure():
    yield
    _close_figures("all")


def _impulse(
    seed: int = 0, n: int = 512, frequency: float = 160.0, amplitude: float = 1.0
) -> np.ndarray:
    """Damped 160 Hz impulse with seeded noise, a NIRT-like test signal."""
    t = np.arange(n) / SAMPLING_RATE
    noise = np.random.default_rng(seed).standard_normal(n)
    return amplitude * (
        np.exp(-2 * np.pi * frequency * 0.03 * t) * np.sin(2 * np.pi * frequency * t) + 0.01 * noise
    )


def _filtered(synthetic_trace, name: str = "filtered") -> TraceData:
    trace = synthetic_trace(_impulse(), name=name)
    [trace] = preprocess_traces([trace], filter_options=ButterworthFilterConfig(20.0, 400.0))
    return trace


def _mse(scales, values, scalar: float = 1.0):
    return (
        scalar,
        {"Scale": np.asarray(scales, dtype=float), "Value": np.asarray(values, dtype=float)},
    )


def _figures(monkeypatch: pytest.MonkeyPatch, traces, config: TracePlotConfig) -> list:
    """Run plot_traces with figure closing disabled and return its figures in order."""
    _close_figures("all")
    monkeypatch.setattr(plot_traces_module.plt, "close", lambda *args, **kwargs: None)
    plot_traces(traces, config)
    return [plt.figure(number) for number in plt.get_fignums()]


def test_filtered_trace_draws_post_filter_solid_and_pre_filter_dashed(
    monkeypatch, synthetic_trace
) -> None:
    trace = _filtered(synthetic_trace)

    [figure] = _figures(
        monkeypatch, [trace], TracePlotConfig(show=False, spectrum_type="amplitude")
    )

    time_axis, spectrum_axis = figure.axes[:2]
    post_signal, pre_signal = time_axis.lines
    assert (post_signal.get_label(), post_signal.get_linestyle()) == ("Post-filtered signal", "-")
    assert (pre_signal.get_label(), pre_signal.get_linestyle()) == ("Pre-filtered signal", "--")
    np.testing.assert_array_equal(post_signal.get_ydata(), trace.signal_norm)
    np.testing.assert_array_equal(pre_signal.get_ydata(), trace.signal_norm_pre)
    post_spectrum, pre_spectrum = spectrum_axis.lines[:2]
    assert (
        post_spectrum.get_label(),
        post_spectrum.get_linestyle(),
        post_spectrum.get_marker(),
    ) == ("Post-filtered", "-", "None")
    assert (pre_spectrum.get_label(), pre_spectrum.get_linestyle()) == (
        "Pre-filtered Spectrum",
        "--",
    )
    np.testing.assert_array_equal(post_spectrum.get_xdata(), trace.freqs_filt)
    np.testing.assert_array_equal(post_spectrum.get_ydata(), trace.spec_filt)
    np.testing.assert_array_equal(pre_spectrum.get_xdata(), trace.freqs)
    np.testing.assert_array_equal(pre_spectrum.get_ydata(), trace.spectrum)


def test_filtered_trace_uses_the_configured_fill_and_unfiltered_colors(
    monkeypatch, synthetic_trace
) -> None:
    trace = _filtered(synthetic_trace)
    style = {"time_fill_color": "red", "pre_filtered_color": "blue"}

    [figure] = _figures(
        monkeypatch, [trace], TracePlotConfig(show=False, attrs_to_plot=None, style=style)
    )

    time_axis, spectrum_axis = figure.axes[:2]
    [fill] = time_axis.collections
    assert to_hex(fill.get_facecolor()[0]) == "#ff0000"
    assert time_axis.lines[1].get_color() == "blue"
    assert spectrum_axis.lines[1].get_color() == "blue"


def test_power_spectrum_is_the_squared_amplitude_spectrum(monkeypatch, synthetic_trace) -> None:
    trace = synthetic_trace(_impulse(), name="power")

    [figure] = _figures(monkeypatch, [trace], TracePlotConfig(show=False, spectrum_type="power"))

    spectrum_axis = figure.axes[1]
    assert spectrum_axis.get_ylabel() == "Power spectrum (a.u.)"
    [line] = spectrum_axis.lines
    assert line.get_label() == "Power Spectrum"
    np.testing.assert_array_equal(line.get_ydata(), trace.spectrum**2)


@pytest.mark.parametrize(
    ("spectrum_type", "expected_label"),
    [("amplitude", "Amplitude spectrum (dB re max)"), ("power", "Power spectrum (dB re max)")],
)
def test_db_spectrum_matches_the_hand_calculation(
    monkeypatch, spectrum_trace, spectrum_type, expected_label
) -> None:
    """Both modes give 0 dB at the peak and the same numbers: 20 log10(a/max) == 10 log10(a^2/max^2),
    with the display floor at -60 dB (amplitude ratio 1e-3, power ratio 1e-6)."""
    trace = spectrum_trace([1.0, 0.5, 0.1, 1e-3, 1e-4, 0.0])

    [figure] = _figures(
        monkeypatch,
        [trace],
        TracePlotConfig(show=False, spectrum_type=spectrum_type, use_db=True, attrs_to_plot=None),
    )

    spectrum_axis = figure.axes[1]
    assert spectrum_axis.get_ylabel() == expected_label
    expected = [0.0, 20 * np.log10(0.5), -20.0, -60.0, -60.0, -60.0]
    np.testing.assert_allclose(spectrum_axis.lines[0].get_ydata(), expected, rtol=0.0, atol=1e-12)
    assert spectrum_axis.get_ylim() == (-60.0, 3.0)


def test_marker_lines_follow_peak_and_centroid_attributes(monkeypatch, synthetic_trace) -> None:
    trace = synthetic_trace(_impulse(), name="markers")
    trace.attributes = {
        "peak_frequency": 150.0,
        "peak_frequency_10-200Hz": 120.0,
        "spectral_centroid": 140.0,
        "spectral_peak_width": 0.03,
    }

    [figure] = _figures(monkeypatch, [trace], TracePlotConfig(show=False))

    markers = [line for line in figure.axes[1].lines if len(line.get_xdata()) == 2]
    assert [(line.get_label(), line.get_xdata()[0]) for line in markers] == [
        ("Peak Frequency", 150.0),
        ("Peak Frequency 10-200Hz", 120.0),
        ("Spectral Centroid", 140.0),
    ]

    [figure] = _figures(
        monkeypatch,
        [trace],
        TracePlotConfig(show=False, style={"show_peak_lines": False, "show_centroid_lines": False}),
    )
    assert [line for line in figure.axes[1].lines if len(line.get_xdata()) == 2] == []


@pytest.mark.parametrize("reverse_order", [False, True])
def test_mse_panel_draws_every_variant_with_nan_gaps_and_combined_limits(
    monkeypatch, synthetic_trace, reverse_order
) -> None:
    items = [
        ("MSE_low", _mse([1.0, 2.0], [1.0, np.nan])),
        ("MSE_high", _mse([10.0, 20.0], [100.0, 200.0])),
    ]
    if reverse_order:
        items.reverse()
    trace = synthetic_trace(_impulse(), name="variants")
    trace.attributes = dict(items)

    [figure] = _figures(monkeypatch, [trace], TracePlotConfig(show=False, attrs_to_plot=None))

    mse_axis = figure.axes[2]
    assert (mse_axis.get_xlabel(), mse_axis.get_ylabel()) == (
        "Scale factor",
        "Sample entropy (nats)",
    )
    assert [line.get_label() for line in mse_axis.lines] == [key for key, _value in items]
    for line, (_key, (_scalar, details)) in zip(mse_axis.lines, items, strict=True):
        np.testing.assert_array_equal(line.get_xdata(), details["Scale"])
        np.testing.assert_array_equal(line.get_ydata(), details["Value"])
    assert mse_axis.get_xlim() == (1.0, 20.0)
    assert mse_axis.get_ylim() == (1.0, 210.0)

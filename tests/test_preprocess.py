"""Normalization, spectrum, filtering and amplification in preprocess_traces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

import slabnirt.preprocess as preprocess_module
from slabnirt import ButterworthFilterConfig, TraceData, preprocess_traces


def _trace(name: str, time: np.ndarray, signal: np.ndarray) -> TraceData:
    return TraceData(
        basename=name,
        channel=1,
        t=np.asarray(time, dtype=float),
        signal=np.asarray(signal, dtype=float),
        filepath=f"{name}.txt",
    )


def test_normalization_and_spectrum_match_numpy() -> None:
    """The spectrum tolerance allows for scipy.fft and NumPy 1.x's numpy.fft
    differing in the last bit."""
    sampling_interval = 0.0005
    time = np.arange(128, dtype=float) * sampling_interval
    signal = 2.5 * np.sin(2.0 * np.pi * 125.0 * time + 0.3)

    [processed] = preprocess_traces([_trace("sine", time, signal)])

    expected_normalized = signal / np.max(np.abs(signal))
    expected_spectrum = np.abs(np.fft.rfft(expected_normalized)) / signal.size
    np.testing.assert_array_equal(processed.signal, signal)
    np.testing.assert_array_equal(processed.signal_norm, expected_normalized)
    np.testing.assert_array_equal(
        processed.freqs, np.fft.rfftfreq(signal.size, d=sampling_interval)
    )
    np.testing.assert_allclose(
        processed.spectrum, expected_spectrum, rtol=0.0, atol=1e-12 * expected_spectrum.max()
    )


def test_zero_padding_makes_the_bins_denser_without_lowering_them() -> None:
    """Padding by two record lengths gives three bins per original bin; every
    third one is the original bin, with the same value."""
    sampling_rate = 2_000.0
    time = np.arange(200, dtype=float) / sampling_rate
    trace = _trace("tone", time, np.sin(2.0 * np.pi * 100.0 * time))

    [plain] = preprocess_traces([trace])
    [padded] = preprocess_traces([trace], pad_zeros_factor=2)

    assert padded.freqs.size == 3 * (plain.freqs.size - 1) + 1
    np.testing.assert_allclose(np.diff(padded.freqs), 10.0 / 3.0, rtol=1e-12)
    np.testing.assert_allclose(padded.freqs[::3], plain.freqs, rtol=1e-14)
    np.testing.assert_allclose(padded.spectrum[::3], plain.spectrum, rtol=0.0, atol=1e-12)
    assert plain.spectrum[10] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("signal_of_time", "integrate_times"),
    [(np.ones_like, 2), (lambda time: time, 1)],
    ids=["constant-twice", "ramp-once"],
)
def test_integration_is_the_cumulative_trapezoid(signal_of_time, integrate_times: int) -> None:
    """The trapezoid rule is exact for straight lines, so a constant integrated
    twice and a ramp integrated once both give t**2 / 2."""
    time = np.arange(200, dtype=float) / 1_000.0

    [processed] = preprocess_traces(
        [_trace("line", time, signal_of_time(time))],
        integrate_times=integrate_times,
        remove_offset=False,
    )

    np.testing.assert_allclose(processed.signal, time**2 / 2.0, rtol=1e-12, atol=0.0)


def test_preprocessed_copies_do_not_carry_the_input_attributes() -> None:
    """Attributes stored on the input describe its earlier processing."""
    time = np.arange(64, dtype=float) / 2_000.0
    trace = _trace("analyzed", time, np.sin(2.0 * np.pi * 100.0 * time))
    trace.attributes = {"peak_frequency": 100.0}
    trace.attributes_avg = {"peak_frequency": {"median": 100.0}}

    [processed] = preprocess_traces([trace])

    assert processed.attributes is None
    assert processed.attributes_avg is None
    assert trace.attributes == {"peak_frequency": 100.0}


@pytest.mark.parametrize(
    ("config", "wn", "btype"),
    [
        (ButterworthFilterConfig(max_frequency_hz=300.0, order=4), 0.3, "lowpass"),
        (ButterworthFilterConfig(min_frequency_hz=10.0, order=4), 0.01, "highpass"),
        (ButterworthFilterConfig(10.0, 300.0, order=4), [0.01, 0.3], "bandpass"),
    ],
)
def test_filter_is_the_zero_phase_scipy_butterworth(config, wn, btype) -> None:
    sampling_rate = 2_000.0
    time = np.arange(2_048, dtype=float) / sampling_rate
    signal = np.sin(2.0 * np.pi * 100.0 * time) + 0.25 * np.sin(2.0 * np.pi * 600.0 * time)
    expected_sos = butter(4, wn, btype=btype, output="sos")
    expected = sosfiltfilt(expected_sos, signal)

    [processed] = preprocess_traces([_trace("mix", time, signal)], filter_options=config)

    np.testing.assert_array_equal(
        preprocess_module.butter_design(config, sampling_rate), expected_sos
    )
    np.testing.assert_array_equal(processed.signal, expected)
    np.testing.assert_array_equal(processed.signal_norm, expected / np.max(np.abs(expected)))
    np.testing.assert_array_equal(processed.signal_norm_pre, signal / np.max(np.abs(signal)))
    assert processed.filter_freq_range == config.passband_hz


@pytest.mark.parametrize("shift_to_first_peak", [False, True])
def test_amplification_gain_starts_at_one_at_the_first_peak(
    monkeypatch: pytest.MonkeyPatch,
    shift_to_first_peak: bool,
) -> None:
    time = np.arange(6, dtype=float) * 0.1
    signal = np.arange(1.0, 7.0)
    first_peak = 2
    amplification_rate = 2.0

    monkeypatch.setattr(
        preprocess_module,
        "find_peaks",
        lambda *args, **kwargs: (np.array([first_peak]), {}),
    )

    [processed] = preprocess_traces(
        [_trace("ramp", time, signal)],
        shift_to_first_peak=shift_to_first_peak,
        amplification_factor=amplification_rate,
    )

    expected_time = time - first_peak * (time[1] - time[0]) if shift_to_first_peak else time
    expected_gain = np.concatenate(
        [
            np.ones(first_peak),
            np.exp((expected_time[first_peak:] - expected_time[first_peak]) * amplification_rate),
        ]
    )
    np.testing.assert_array_equal(processed.t, expected_time)
    np.testing.assert_array_equal(processed.signal, signal * expected_gain)

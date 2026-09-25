"""Preprocessing of raw traces: offset, detrend, integration, filtering and spectra."""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, find_peaks, sosfiltfilt

from ._console import safe_warn
from .core import TraceData


def compute_amplitude_spectrum(
    signal: Sequence[float], sampling_interval: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frequency bins in Hz and ``|rfft(signal, n_samples)| / n_samples``.

    This is the two-sided DFT magnitude at the non-negative frequencies: an
    interior bin holds half the amplitude of the sinusoid that produced it,
    the DC and Nyquist bins hold the full amplitude. ``sampling_interval``
    is in s.
    """
    y = np.asarray(signal)
    spectrum = np.abs(rfft(y, n=n_samples)) / n_samples
    freqs = rfftfreq(n_samples, d=sampling_interval)
    return freqs, spectrum


@dataclass(frozen=True)
class ButterworthFilterConfig:
    """Butterworth filter passband in Hz and filter order.

    Set only ``max_frequency_hz`` for a low-pass filter, only
    ``min_frequency_hz`` for a high-pass filter, or both for a band-pass
    filter. The filter is applied forward and backward (zero phase), which
    squares its response: a component at a band edge is reduced by 6 dB
    rather than 3 dB, and the roll-off is that of order ``2 * order``.

    Parameters
    ----------
    min_frequency_hz : float or None, default None
        Lower band edge in Hz, above 0.
    max_frequency_hz : float or None, default None
        Upper band edge in Hz, above ``min_frequency_hz``.
    order : int, default 6
        Order of the Butterworth design.
    """

    min_frequency_hz: float | None = None
    max_frequency_hz: float | None = None
    order: int = 6

    def __post_init__(self) -> None:
        if self.min_frequency_hz is None and self.max_frequency_hz is None:
            raise ValueError(
                "At least one of min_frequency_hz or max_frequency_hz "
                "must be given; use filter_options=None for no filtering."
            )

        for name in ("min_frequency_hz", "max_frequency_hz"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real frequency in Hz or None.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than 0 Hz; use None for an open passband edge."
                )
            object.__setattr__(self, name, value)

        if (
            self.min_frequency_hz is not None
            and self.max_frequency_hz is not None
            and self.min_frequency_hz >= self.max_frequency_hz
        ):
            raise ValueError("min_frequency_hz must be lower than max_frequency_hz.")

        if isinstance(self.order, bool) or not isinstance(self.order, Integral) or self.order <= 0:
            raise ValueError("order must be a positive integer.")
        object.__setattr__(self, "order", int(self.order))

    @property
    def passband_hz(self) -> tuple[float, float | None]:
        """Return the passband stored on filtered traces, ``(low or 0.0, high)``."""
        return (
            0.0 if self.min_frequency_hz is None else self.min_frequency_hz,
            self.max_frequency_hz,
        )


def _check_nyquist(filter_config: ButterworthFilterConfig, fs: float) -> None:
    nyquist = 0.5 * fs
    for name in ("min_frequency_hz", "max_frequency_hz"):
        value = getattr(filter_config, name)
        if value is not None and value >= nyquist:
            raise ValueError(
                f"{name} ({value:g} Hz) must be below the Nyquist frequency "
                f"({nyquist:g} Hz) for a {fs:g} Hz sampling rate."
            )


def butter_design(filter_config: ButterworthFilterConfig, fs: float) -> np.ndarray:
    """Return the second-order sections of the configured filter at ``fs`` Hz."""
    fs = float(fs)
    _check_nyquist(filter_config, fs)
    nyquist = 0.5 * fs
    minimum = filter_config.min_frequency_hz
    maximum = filter_config.max_frequency_hz

    if minimum is None:
        return butter(filter_config.order, maximum / nyquist, btype="lowpass", output="sos")
    if maximum is None:
        return butter(filter_config.order, minimum / nyquist, btype="highpass", output="sos")
    return butter(
        filter_config.order,
        [minimum / nyquist, maximum / nyquist],
        btype="bandpass",
        output="sos",
    )


def _warn_trace_omitted(filepath: str, reason: str) -> None:
    safe_warn(f"{filepath}: skipped because {reason}.", RuntimeWarning, stacklevel=3)


def _normalize_signal(signal: np.ndarray, *, filepath: str, step: str) -> np.ndarray | None:
    """Return ``signal / max(|signal|)``, or warn and return None when undefined."""
    if signal.size == 0:
        _warn_trace_omitted(filepath, f"the signal is empty after {step}")
        return None

    max_amplitude = np.max(np.abs(signal))
    if max_amplitude == 0:
        _warn_trace_omitted(filepath, f"the amplitude is zero after {step}")
        return None
    if not np.isfinite(max_amplitude):
        _warn_trace_omitted(filepath, f"the amplitude is non-finite after {step}")
        return None
    return signal / max_amplitude


def _spectrum(signal_norm: np.ndarray, dt: float, pad_zeros_factor: int):
    """Return the amplitude spectrum, zero-padded to ``1 + pad_zeros_factor`` lengths.

    The padded spectrum is rescaled to the unpadded record length, so that
    padding interpolates the bins without lowering their values.
    """
    n = len(signal_norm)
    if pad_zeros_factor <= 0:
        return compute_amplitude_spectrum(signal_norm, dt, n)
    padded = np.concatenate([signal_norm, np.zeros(pad_zeros_factor * n)])
    n_fft = len(padded)
    freqs, spectrum = compute_amplitude_spectrum(padded, dt, n_fft)
    return freqs, spectrum * (n_fft / n)


def _impact_onset(signal: np.ndarray) -> int | None:
    """Return the index of the first sample of the impact, or None without one.

    The impact starts at the first sample that departs from the mean of the
    first 40 samples by more than 5 % of the largest departure in the
    record. A constant record has no impact; a non-finite one is left for
    the normalization step to report.
    """
    level = np.mean(signal[:40])
    if not np.isfinite(level):
        return None
    departure = np.abs(signal - level)
    beyond = np.flatnonzero(departure > 0.05 * np.max(departure))
    return int(beyond[0]) if beyond.size else None


def _pre_impact_level(signal: np.ndarray, onset: int | None) -> float | None:
    """Return the mean of the samples before the impact, or None with fewer than 10.

    A record that starts during the impact has too few quiet samples, and
    their mean would include part of the pulse.
    """
    if onset is None or onset < 10:
        return None
    return float(np.mean(signal[:onset]))


def _remove_offset(signal: np.ndarray) -> np.ndarray:
    """Subtract the pre-impact level, when it can be measured."""
    level = _pre_impact_level(signal, _impact_onset(signal))
    if level is None:
        return signal
    return signal - level


def _highpass_detrend(signal: np.ndarray, fs: float, corner_hz: float) -> np.ndarray:
    """Subtract a low-pass baseline estimated with the impact left out.

    The baseline is the record passed through a zero-phase second-order
    Butterworth low-pass at ``corner_hz``, after the impact window (0.5 ms
    before the onset to 15 ms after it) has been replaced by a straight
    line so that the pulse does not leak into it. Before the onset the
    baseline is held at the pre-impact level, when measurable, and it joins
    the filtered curve over a 5 ms half-cosine. This keeps the start of the
    record at zero, which the non-causal response of a plain high-pass
    would not.
    """
    if corner_hz >= 0.5 * fs:
        raise ValueError(
            f"detrend_corner_hz ({corner_hz:g} Hz) must be below the Nyquist "
            f"frequency ({0.5 * fs:g} Hz) for a {fs:g} Hz sampling rate."
        )
    sos = butter_design(ButterworthFilterConfig(max_frequency_hz=corner_hz, order=2), fs)
    onset = _impact_onset(signal)
    bridged = np.array(signal, dtype=float)
    if onset is not None:
        start = max(onset - round(0.0005 * fs), 0)
        stop = min(onset + round(0.015 * fs), signal.size - 1)
        bridged[start : stop + 1] = np.linspace(signal[start], signal[stop], stop - start + 1)
    baseline = sosfiltfilt(sos, bridged)
    level = _pre_impact_level(signal, onset)
    if level is not None:
        baseline[:onset] = level
        n_blend = round(0.005 * fs)
        blend = slice(onset, min(onset + n_blend, signal.size))
        weight = 0.5 * (1.0 - np.cos(np.pi * np.arange(blend.stop - blend.start) / n_blend))
        baseline[blend] = level + weight * (baseline[blend] - level)
    return signal - baseline


def _polynomial_detrend(signal: np.ndarray) -> np.ndarray:
    """Subtract a second-order polynomial fitted after left-mirror padding.

    The record is mirrored to its left without repeating the first sample,
    a parabola is fitted over the padded record on a 0..100 grid, and the
    record's own span is cut back out.
    """
    left = signal[1:][::-1]
    padded = np.r_[left, signal]
    grid = np.linspace(0, 100, padded.size)
    detr_padded = padded - np.polyval(np.polyfit(grid, padded, 2), grid)
    return detr_padded[left.size : left.size + len(signal)]


def preprocess_traces(
    traces: list[TraceData],
    downsample: int = 1,
    polarity: int = 1,
    time_limit: float = 0.0,
    integrate_times: int = 0,
    shift_to_first_peak: bool = False,
    amplification_factor: float = 0.0,
    peak_params: tuple[int, tuple[float, float], int] = (2, (0.55, 2.0), 2),
    pad_zeros_factor: int = 0,
    filter_options: ButterworthFilterConfig | None = None,
    remove_offset: bool = True,
    detrend_method: str | None = None,
    detrend_corner_hz: float = 30.0,
) -> list[TraceData]:
    """Return preprocessed shallow copies of the traces.

    The steps run in this order: offset removal, downsampling, polarity,
    time trimming, detrending, integration, normalization, time shift,
    amplification, spectrum, filtering (with its own normalization and
    spectrum). The input traces are not changed.

    Parameters
    ----------
    traces : list of TraceData
        Loaded traces.
    downsample : int, default 1
        Keep every n-th sample. There is no anti-alias filter, and
        ``filter_options`` is applied after downsampling, so content above
        the new Nyquist frequency folds back into the record.
    polarity : {1, -1}, default 1
        Multiplier of the signal; -1 flips it.
    time_limit : float, default 0.0
        Keep only samples with t < time_limit, in s; 0 keeps everything.
    integrate_times : {0, 1, 2}, default 0
        Number of cumulative trapezoidal integrations, for example 1 for
        acceleration to velocity.
    shift_to_first_peak : bool, default False
        Shift the time axis so that the first detected peak is at t = 0.
    amplification_factor : float, default 0.0
        Rate of an exponential gain in 1/s, starting at 1 at the first
        detected peak; 0 disables it.
    peak_params : (int, (float, float), int), default (2, (0.55, 2.0), 2)
        ``distance``, ``prominence`` and ``width`` passed to
        ``scipy.signal.find_peaks`` for the two options above.
    pad_zeros_factor : int, default 0
        Zero-pad the record by this many record lengths before the FFT. The
        bins become denser; the resolution does not change.
    filter_options : ButterworthFilterConfig or None, default None
        Filter applied after the unfiltered spectrum is computed. The
        unfiltered normalized signal is kept in ``signal_norm_pre``.
    remove_offset : bool, default True
        Subtract the level recorded before the impact, so that the record
        starts at zero. The impact starts at the first sample that departs
        from the mean of the first 40 samples by more than 5 % of the
        largest departure in the record; the level is the mean of the
        samples before it and is subtracted only when there are at least 10
        of them. The step runs on the full record, before downsampling and
        trimming. It always runs when ``detrend_method`` is set.
    detrend_method : {None, 'highpass', 'polynomial'}, default None
        Detrend applied after trimming; None applies none.
        'highpass' subtracts a baseline obtained with a zero-phase
        second-order Butterworth low-pass at ``detrend_corner_hz``. The
        impact window (0.5 ms before the onset to 15 ms after it) is bridged
        by a straight line before the baseline is estimated, and the
        baseline is held at the pre-impact level before the onset and
        blended in over 5 ms, so the pulse is kept and the start stays at
        zero. It needs a record of at least 50 ms; a shorter record gets a
        RuntimeWarning and only the offset step. 'polynomial' subtracts a
        second-order polynomial fitted after left-mirror padding.
    detrend_corner_hz : float, default 30.0
        Corner frequency in Hz of the 'highpass' baseline, positive and
        below the Nyquist frequency of every trace. The detrend keeps
        ``(f/fc)**4 / (1 + (f/fc)**4)`` of a component at f: 88.5 % at
        50 Hz and 99.2 % at 100 Hz for 30 Hz. Choose fc at least 2.3 times
        a ringing frequency to remove (less than 5 % of it remains) and at
        most half the lowest structural frequency to keep.

    Returns
    -------
    list of TraceData
        The processed copies, with ``t``, ``signal``, ``signal_norm``,
        ``freqs``, ``spectrum`` and the filter fields set. A trace that
        cannot be processed is left out with a RuntimeWarning.

    Raises
    ------
    ValueError
        For an invalid parameter, a filter edge or detrend corner at or
        above the Nyquist frequency, or when no trace of a non-empty batch
        remains.
    TypeError
        For a ``filter_options`` that is not a ButterworthFilterConfig, or
        a ``detrend_corner_hz`` that is not a real number.
    """
    if downsample < 1:
        raise ValueError("downsample must be >= 1.")
    if polarity not in {1, -1}:
        raise ValueError("polarity must be 1 or -1.")
    if integrate_times not in {0, 1, 2}:
        raise ValueError("integrate_times must be 0, 1 or 2.")
    if detrend_method not in {None, "highpass", "polynomial"}:
        raise ValueError("detrend_method must be None, 'highpass' or 'polynomial'.")
    if detrend_method == "highpass":
        if isinstance(detrend_corner_hz, bool) or not isinstance(detrend_corner_hz, Real):
            raise TypeError("detrend_corner_hz must be a real frequency in Hz.")
        if not np.isfinite(detrend_corner_hz) or detrend_corner_hz <= 0:
            raise ValueError("detrend_corner_hz must be a finite frequency greater than 0 Hz.")
    # Both detrend methods expect a record that starts at zero.
    remove_offset = remove_offset or detrend_method is not None

    if filter_options is not None and not isinstance(filter_options, ButterworthFilterConfig):
        raise TypeError("filter_options must be a ButterworthFilterConfig or None.")

    dist, prom, width = peak_params
    preprocessed_traces: list[TraceData] = []

    for trace_data in traces:
        trace = copy.copy(trace_data)

        t, signal, filepath = (trace.t, trace.signal, trace.filepath)
        if signal.size == 0:
            _warn_trace_omitted(filepath, "the input signal is empty")
            continue
        if np.max(np.abs(signal)) == 0:
            _warn_trace_omitted(filepath, "the input signal has zero amplitude")
            continue

        # Done before downsampling, which could leave fewer than the 10 quiet
        # samples the rule needs.
        if remove_offset:
            signal = _remove_offset(signal)

        if downsample > 1:
            t = t[::downsample]
            signal = signal[::downsample]
            if t.size < 2 or signal.size < 2:
                _warn_trace_omitted(filepath, "fewer than two samples remain after downsampling")
                continue

        if polarity == -1:
            signal = -signal

        if time_limit > 0:
            mask = t < time_limit
            t, signal = t[mask], signal[mask]
            if t.size < 2 or signal.size < 2:
                _warn_trace_omitted(filepath, "fewer than two samples remain after time trimming")
                continue

        if t.size < 2 or signal.size < 2:
            _warn_trace_omitted(filepath, "it has fewer than two samples")
            continue

        dt = t[1] - t[0]
        # A non-positive or non-finite step would give negative or
        # non-finite frequency bins instead of an error.
        if not np.isfinite(dt) or dt <= 0:
            _warn_trace_omitted(
                filepath, f"the sampling interval is {dt:g} s; it must be finite and positive"
            )
            continue
        fs = 1.0 / dt
        if filter_options is not None:
            _check_nyquist(filter_options, fs)

        if detrend_method == "highpass":
            if signal.size < round(0.05 * fs):
                safe_warn(
                    f"{filepath}: the 'highpass' detrend needs a record of at least "
                    f"50 ms, this one spans {1e3 * signal.size * dt:.1f} ms; only the "
                    "offset step was applied.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                signal = _highpass_detrend(signal, fs, detrend_corner_hz)
        elif detrend_method == "polynomial":
            signal = _polynomial_detrend(signal)

        for _ in range(integrate_times):
            signal = cumulative_trapezoid(signal, dx=dt, initial=0)

        signal_norm = _normalize_signal(signal, filepath=filepath, step="detrending/integration")
        if signal_norm is None:
            continue

        if shift_to_first_peak:
            peaks, _ = find_peaks(signal_norm, distance=dist, prominence=prom, width=width)
            if peaks.size > 0:
                t = t - peaks[0] * dt

        if amplification_factor > 0:
            peaks, _ = find_peaks(signal_norm, distance=dist, prominence=prom, width=width)
            if peaks.size > 0:
                idx = peaks[0]
                envelope = np.concatenate(
                    [
                        np.ones(idx),
                        np.exp((t[idx:] - t[idx]) * amplification_factor),
                    ]
                )
                signal = signal * envelope
                signal_norm = _normalize_signal(signal, filepath=filepath, step="amplification")
                if signal_norm is None:
                    continue

        trace.freqs, trace.spectrum = _spectrum(signal_norm, dt, pad_zeros_factor)

        # Results of any earlier processing do not describe the new record.
        trace.signal_norm_pre = None
        trace.freqs_filt = trace.spec_filt = None
        trace.filter_freq_range = None
        trace.attributes = trace.attributes_avg = None

        if filter_options is not None:
            trace.signal_norm_pre = signal_norm.copy()
            trace.filter_freq_range = filter_options.passband_hz
            try:
                signal = sosfiltfilt(butter_design(filter_options, fs), signal)
            except ValueError as exc:
                _warn_trace_omitted(filepath, f"zero-phase filtering failed ({exc})")
                continue
            signal_norm = _normalize_signal(signal, filepath=filepath, step="filtering")
            if signal_norm is None:
                continue
            trace.freqs_filt, trace.spec_filt = _spectrum(signal_norm, dt, pad_zeros_factor)

        trace.t = t
        trace.signal = signal
        trace.signal_norm = signal_norm
        preprocessed_traces.append(trace)

    if traces and not preprocessed_traces:
        raise ValueError("preprocess_traces: every trace was skipped; see the warnings.")

    return preprocessed_traces

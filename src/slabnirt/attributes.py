"""Time-domain, spectral and entropy attributes of preprocessed traces."""

from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from ._attribute_selection import (
    AttributeSelection,
    normalize_attribute_selection,
    validate_known_attribute_names,
)
from ._console import safe_warn
from ._wu_rcmse import check_parameters, multiscale_entropy, sample_entropy
from .core import TraceData
from .preprocess import compute_amplitude_spectrum

# Entropy defaults: templates of two samples, r = 0.2 SD, and for MSE every
# scale that keeps at least 100 coarse-grained samples (see _wu_rcmse.rcmse).
DEFAULT_MSE_PARAMS = dict(dimension=2, tolerance="sd", scale=None)
DEFAULT_SAMPEN_PARAMS = dict(dimension=2, tolerance="sd")
DEFAULT_VOID_IDX_PARAMS = dict(
    max_freq_range=(0.0, 200.0), mav_freq_range=(200.0, 800.0), mav_method="median"
)

_SUPPORTED_ATTRS = [
    "norm_signal_energy",
    "normalized_spectrum_area",
    "spectral_centroid",
    "norm_spectrum_area_over_centroid",
    "peak_frequency",
    "spectral_peak_width",
    "spectral_flatness",
    "void_index",
    "sampen",
    "MSE",
    "spectral_entropy",
]

_SPECTRAL_ATTRS = {
    "normalized_spectrum_area",
    "spectral_centroid",
    "norm_spectrum_area_over_centroid",
    "peak_frequency",
    "spectral_peak_width",
    "spectral_flatness",
    "spectral_entropy",
    "void_index",
}


def _normalized_signal_energy(
    signal: np.ndarray,
    time: np.ndarray,
) -> tuple[float, str | None]:
    """Return ``dt * sum(signal**2)``, or NaN and the reason it is undefined."""
    values = np.asarray(signal, dtype=float)
    times = np.asarray(time, dtype=float)

    if values.ndim != 1 or times.ndim != 1:
        return np.nan, "the signal and time arrays must be one-dimensional"
    if values.size != times.size:
        return np.nan, "the signal and time arrays have different lengths"
    if values.size < 2:
        return np.nan, "at least two time samples are required"
    if not np.all(np.isfinite(values)):
        return np.nan, "the normalized signal contains non-finite values"
    if not np.all(np.isfinite(times)):
        return np.nan, "the time array contains non-finite values"

    intervals = np.diff(times)
    dt = float(intervals[0])
    if np.any(intervals <= 0.0):
        return np.nan, "the time array must be strictly increasing"
    if not np.allclose(intervals, dt, rtol=1e-7, atol=0.0):
        return np.nan, "the time array must be uniformly sampled"

    energy = dt * float(np.sum(values**2, dtype=float))
    if not np.isfinite(energy):
        return np.nan, "the time-integrated energy is non-finite"
    return energy, None


def _selected_spectrum_issue(spectrum: np.ndarray) -> str | None:
    """Return why a selected spectrum cannot give spectral attributes, or None.

    A non-finite, negative or all-zero selection would otherwise give a
    plausible number, often exactly 0.0, instead of a missing value.
    """
    values = np.asarray(spectrum, dtype=float)
    if not np.all(np.isfinite(values)):
        return "the selected spectrum contains non-finite values"
    if np.any(values < 0.0):
        return "the selected spectrum contains negative values"
    if float(np.max(values)) <= 0.0:
        return "the selected spectrum has zero total power"
    return None


def _normalized_spectral_entropy(spectrum: np.ndarray, *, spectrum_is_power: bool) -> float:
    """Return the Shannon entropy of the bin powers divided by ``log2(N)``, in [0, 1].

    The spectrum must pass ``_selected_spectrum_issue``.
    """
    values = np.asarray(spectrum, dtype=float)
    if values.size == 1:
        return 0.0

    # Dividing by the peak before squaring avoids overflow; the probabilities
    # do not change.
    scaled = values / float(np.max(values))
    power = scaled if spectrum_is_power else scaled**2
    probabilities = power / float(np.sum(power, dtype=float))
    positive = probabilities > 0.0
    entropy_bits = -float(
        np.sum(
            probabilities[positive] * np.log2(probabilities[positive]),
            dtype=float,
        )
    )
    # The clip removes round-off excursions beyond [0, 1].
    return float(np.clip(entropy_bits / np.log2(values.size), 0.0, 1.0))


def _spectral_flatness(spectrum: np.ndarray, *, spectrum_is_power: bool) -> float:
    """Return the geometric over the arithmetic mean of the bin powers, in [0, 1].

    The spectrum must pass ``_selected_spectrum_issue``.
    """
    values = np.asarray(spectrum, dtype=float)

    # The geometric mean is taken in log space after dividing by the peak,
    # which avoids overflow; a bin that is zero, or underflows to zero,
    # gives zero.
    scaled = values / float(np.max(values))
    if np.any(scaled == 0.0):
        return 0.0
    if spectrum_is_power:
        arithmetic_mean = float(np.mean(scaled))
        mean_log_power = float(np.mean(np.log(scaled)))
    else:
        arithmetic_mean = float(np.mean(scaled**2))
        mean_log_power = 2.0 * float(np.mean(np.log(scaled)))

    flatness = np.exp(mean_log_power) / arithmetic_mean
    return float(np.clip(flatness, 0.0, 1.0))


def _band_mask(freqs: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    """Return the bins with ``low <= f <= high``; a None edge is open."""
    mask = np.ones_like(freqs, dtype=bool)
    if low is not None:
        mask &= freqs >= low
    if high is not None:
        mask &= freqs <= high
    return mask


def _warn_nan(trace: TraceData, key: str, reason: str) -> None:
    safe_warn(f"{trace.basename}: {key} is NaN because {reason}.", RuntimeWarning, stacklevel=3)


def calculate_signal_parameters(
    traces: list[TraceData],
    attrs: AttributeSelection = "all",
    name_suffix: str = "",
    use_power_spectrum: bool = True,
    freq_range: tuple[float | None, float | None] = (0.0, None),
    centroid_threshold: float = 0.05,
    mse_params: dict[str, Any] | None = None,
    sampen_params: dict[str, Any] | None = None,
    void_index_params: dict[str, Any] | None = None,
) -> None:
    """Compute signal attributes and store them in ``trace.attributes``.

    Each result is stored under the attribute name plus ``name_suffix``;
    existing entries are kept. ``signal_norm`` is set to
    ``signal / max(|signal|)`` when absent, and ``freqs`` and ``spectrum``
    are computed when a spectral attribute needs them and none is stored.

    Parameters
    ----------
    traces : list of TraceData
        Traces to analyze, usually the output of ``preprocess_traces``.
    attrs : 'all', list of str or tuple of str, default 'all'
        Attributes to compute; see the list below.
    name_suffix : str, default ''
        Appended to every stored name, for example ``'_100-1000Hz'``, so that
        results of several configurations can be kept on one trace.
    use_power_spectrum : bool, default True
        Square the amplitude spectrum before every spectral attribute,
        ``void_index`` included. False uses the amplitude spectrum.
    freq_range : (float or None, float or None), default (0.0, None)
        Band in Hz of the spectral attributes other than ``void_index``.
        The default takes the stored filter passband of a filtered trace
        and the whole spectrum otherwise; any other value is used as given.
    centroid_threshold : float, default 0.05
        Fraction of the peak, 0 to 1, below which bins carry no weight in
        the centroid.
    mse_params : dict, optional
        ``'dimension'`` (template length m, default 2), ``'tolerance'``
        (match distance r: ``'sd'`` for 0.2 times the standard deviation of
        ``signal_norm``, or a number) and ``'scale'`` (a list of positive
        integers; the default None takes every scale at which the
        coarse-grained record keeps at least ``10**dimension`` samples).
    sampen_params : dict, optional
        ``'dimension'`` and ``'tolerance'`` as in ``mse_params``.
    void_index_params : dict, optional
        ``'max_freq_range'`` (band in Hz of the peak, default (0, 200)),
        ``'mav_freq_range'`` (band in Hz of the average level, default
        (200, 800)) and ``'mav_method'`` ('median', the default, or
        'mean'). On a filtered trace both bands read the filtered spectrum,
        so a band in the stopband gives a small level and a large ratio.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        For an invalid parameter, before any trace is changed, and when
        ``freq_range`` selects no frequency bin of a trace.
    TypeError
        For an ``attrs`` that is not 'all' or a list or tuple of names, and
        for an unknown key in ``mse_params``, ``sampen_params`` or
        ``void_index_params``.

    Notes
    -----
    Attributes:

    - norm_signal_energy: ``dt * sum(signal_norm**2)``, in s.
    - normalized_spectrum_area: trapezoidal area of the selected spectrum
      divided by its maximum, in Hz.
    - spectral_centroid: mean frequency in Hz, weighted by the
      peak-normalized spectrum over the bins at or above
      ``centroid_threshold``.
    - norm_spectrum_area_over_centroid: normalized_spectrum_area divided by
      spectral_centroid, dimensionless, always from the current band.
    - peak_frequency: frequency in Hz of the spectrum maximum.
    - spectral_peak_width: ``(f_high - f_low) / (2 * f_peak)`` from the
      interpolated half-power crossings within the band, dimensionless. An
      empirical width indicator, not a modal damping ratio. It also stores
      ``peak_frequency``.
    - spectral_flatness: geometric over arithmetic mean of the bin powers,
      in [0, 1]; 0 when a bin is exactly zero.
    - void_index: maximum of the spectrum in ``max_freq_range`` divided by
      its mean or median in ``mav_freq_range``, dimensionless.
    - sampen: sample entropy of ``signal_norm`` in nats (Richman and
      Moorman 2000); higher means a less regular waveform.
    - MSE: refined composite multiscale entropy of ``signal_norm`` after Wu
      et al. (2014), stored as ``(value, details)``. ``value`` is the mean
      of the finite per-scale entropies in nats; ``details`` holds the
      ``'Scale'`` and ``'Value'`` arrays, ``'Dimension'``, ``'Tolerance'``
      and the per-scale match ``'Counts'``. With the same dimension and
      tolerance, scale 1 equals ``sampen``; scale tau corresponds to the
      time scale ``tau * dt``. With the default
      scales the number of scales grows with the record length, so compare
      ``value`` only between records of equal length and sampling interval.
    - spectral_entropy: Shannon entropy of the bin powers over the band
      divided by ``log2(N)``: 0 for one occupied bin, 1 for uniform power.

    An attribute that the data cannot support is NaN, with a RuntimeWarning
    naming the trace, the attribute and the reason: a spectral band that is
    non-finite, negative or all zero; a spectral centroid of 0 Hz, for the
    area-over-centroid ratio; a void-index band that is empty or has a zero
    level; missing half-power crossings; no matching entropy templates. For
    a signal with non-finite samples ``sampen`` is NaN and ``MSE`` is not
    stored.
    """
    attribute_selection = normalize_attribute_selection(
        attrs,
        parameter_name="attrs",
    )

    if attribute_selection == "all":
        targets = set(_SUPPORTED_ATTRS)
    else:
        validate_known_attribute_names(
            attribute_selection,
            _SUPPORTED_ATTRS,
            parameter_name="attrs",
        )
        targets = set(attribute_selection)

    mse_params = {**DEFAULT_MSE_PARAMS, **(mse_params or {})}
    sampen_params = {**DEFAULT_SAMPEN_PARAMS, **(sampen_params or {})}
    min_freq, max_freq = freq_range
    if min_freq is not None and max_freq is not None and min_freq >= max_freq:
        raise ValueError("freq_range: the lower edge must be below the upper edge.")

    if "void_index" in targets:
        unknown_keys = [
            key for key in (void_index_params or {}) if key not in DEFAULT_VOID_IDX_PARAMS
        ]
        if unknown_keys:
            raise TypeError(f"void_index_params: unknown key(s) {unknown_keys}.")
        merged_void_params = {**DEFAULT_VOID_IDX_PARAMS, **(void_index_params or {})}
        max_freq_range = merged_void_params["max_freq_range"]
        mav_freq_range = merged_void_params["mav_freq_range"]
        mav_method = merged_void_params["mav_method"].lower()

        if len(max_freq_range) != 2 or len(mav_freq_range) != 2:
            raise ValueError("max_freq_range and mav_freq_range must be (low, high) pairs.")
        max_freq_min, max_freq_max = max_freq_range
        mav_freq_min, mav_freq_max = mav_freq_range
        for name, (low, high) in (
            ("max_freq_range", max_freq_range),
            ("mav_freq_range", mav_freq_range),
        ):
            if (low is not None and low < 0) or (high is not None and high < 0):
                raise ValueError(f"{name} must not contain negative frequencies.")
            if low is not None and high is not None and low > high:
                raise ValueError(f"{name}: the lower edge must not exceed the upper edge.")
        if mav_method not in ["mean", "median"]:
            raise ValueError("mav_method must be 'mean' or 'median'.")

    if targets & {"spectral_centroid", "norm_spectrum_area_over_centroid"}:
        if not (0 <= centroid_threshold <= 1):
            raise ValueError("centroid_threshold must be between 0 and 1.")

    if "sampen" in targets:
        check_parameters(**sampen_params)
    if "MSE" in targets:
        check_parameters(**mse_params)

    for trace in traces:
        if not isinstance(trace.attributes, dict):
            trace.attributes = {}

        if trace.signal_norm is None:
            trace.signal_norm = trace.signal / np.max(np.abs(trace.signal))
        sig_norm = trace.signal_norm

        if "norm_signal_energy" in targets:
            key = f"norm_signal_energy{name_suffix}"
            energy, reason = _normalized_signal_energy(sig_norm, trace.t)
            if reason is not None:
                _warn_nan(trace, key, reason)
            trace.attributes[key] = energy

        # A non-finite sample makes every entropy undefined. MSE is then not
        # stored, because its consumers expect a (value, details) pair.
        signal_is_finite = bool(np.all(np.isfinite(sig_norm)))

        if "sampen" in targets:
            key = f"sampen{name_suffix}"
            if not signal_is_finite:
                _warn_nan(trace, key, "the normalized signal contains non-finite values")
                trace.attributes[key] = np.nan
            else:
                trace.attributes[key] = sample_entropy(sig_norm, **sampen_params)
                if np.isnan(trace.attributes[key]):
                    _warn_nan(
                        trace, key, "no template pairs match at one of the two template lengths"
                    )

        if "MSE" in targets:
            key = f"MSE{name_suffix}"
            if not signal_is_finite:
                safe_warn(
                    f"{trace.basename}: {key} is not stored because the normalized "
                    "signal contains non-finite values.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                value, details = multiscale_entropy(sig_norm, **mse_params)
                if np.isnan(value):
                    _warn_nan(
                        trace,
                        key,
                        "no template pairs match at one of the two template lengths at any scale",
                    )
                trace.attributes[key] = (value, details)

        if not targets & _SPECTRAL_ATTRS:
            continue

        if trace.spectrum is None or trace.freqs is None:
            dt = trace.t[1] - trace.t[0]
            n = len(sig_norm)
            trace.freqs, trace.spectrum = compute_amplitude_spectrum(sig_norm, dt, n)

        using_filtered_spectrum = trace.spec_filt is not None
        spec = trace.spec_filt if using_filtered_spectrum else trace.spectrum
        freqs = trace.freqs_filt if using_filtered_spectrum else trace.freqs

        selected_min_freq, selected_max_freq = min_freq, max_freq
        if using_filtered_spectrum and min_freq == 0.0 and max_freq is None:
            filter_freq_range = trace.filter_freq_range
            if filter_freq_range is None:
                safe_warn(
                    f"{trace.basename}: the filtered spectrum has no stored passband; "
                    "the whole frequency axis is used. Give freq_range explicitly.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                selected_min_freq, selected_max_freq = filter_freq_range

        if use_power_spectrum:
            spec = spec**2

        mask = _band_mask(freqs, selected_min_freq, selected_max_freq)
        freqs_sel = freqs[mask]
        spec_sel = spec[mask]

        if len(spec_sel) == 0:
            raise ValueError(f"{trace.basename}: freq_range selects no frequency bin.")
        spectrum_issue = _selected_spectrum_issue(spec_sel)

        if "spectral_flatness" in targets:
            key = f"spectral_flatness{name_suffix}"
            if spectrum_issue is not None:
                _warn_nan(trace, key, spectrum_issue)
                trace.attributes[key] = np.nan
            else:
                trace.attributes[key] = _spectral_flatness(
                    spec_sel, spectrum_is_power=use_power_spectrum
                )

        normalized_targets = [
            name
            for name in (
                "normalized_spectrum_area",
                "spectral_centroid",
                "norm_spectrum_area_over_centroid",
            )
            if name in targets
        ]
        if normalized_targets and spectrum_issue is not None:
            for name in normalized_targets:
                key = f"{name}{name_suffix}"
                _warn_nan(trace, key, spectrum_issue)
                trace.attributes[key] = np.nan
        elif normalized_targets:
            spec_norm = spec_sel / np.max(spec_sel)
            area = trapezoid(spec_norm, freqs_sel)

            if "normalized_spectrum_area" in targets:
                trace.attributes[f"normalized_spectrum_area{name_suffix}"] = area

            if "spectral_centroid" in targets or "norm_spectrum_area_over_centroid" in targets:
                # The peak bin is exactly 1.0, so at least one bin is significant.
                significant = spec_norm >= centroid_threshold
                centroid = np.average(freqs_sel[significant], weights=spec_norm[significant])

                if "spectral_centroid" in targets:
                    trace.attributes[f"spectral_centroid{name_suffix}"] = centroid
                if "norm_spectrum_area_over_centroid" in targets:
                    key = f"norm_spectrum_area_over_centroid{name_suffix}"
                    if centroid == 0.0:
                        _warn_nan(trace, key, "the spectral centroid is 0 Hz")
                        trace.attributes[key] = np.nan
                    else:
                        trace.attributes[key] = area / centroid

        if "peak_frequency" in targets:
            key = f"peak_frequency{name_suffix}"
            if spectrum_issue is not None:
                _warn_nan(trace, key, spectrum_issue)
                trace.attributes[key] = np.nan
            else:
                trace.attributes[key] = freqs_sel[np.argmax(spec_sel)]

        if "spectral_peak_width" in targets:
            peak_key = f"peak_frequency{name_suffix}"
            width_key = f"spectral_peak_width{name_suffix}"
            peak_idx = int(np.argmax(spec_sel))
            peak_freq = freqs_sel[peak_idx]
            peak_value = spec_sel[peak_idx]
            trace.attributes[peak_key] = np.nan if spectrum_issue is not None else peak_freq

            if not np.all(np.isfinite(spec_sel)) or not np.isfinite(peak_freq):
                _warn_nan(trace, width_key, "the selected spectrum contains non-finite values")
                peak_width = np.nan
            elif peak_freq <= 0.0 or peak_value <= 0.0:
                _warn_nan(
                    trace,
                    width_key,
                    "the spectral peak must have positive frequency and magnitude",
                )
                peak_width = np.nan
            else:
                threshold = peak_value * (0.5 if use_power_spectrum else 1 / np.sqrt(2))

                left_below = np.flatnonzero(spec_sel[:peak_idx] < threshold)
                right_below = np.flatnonzero(spec_sel[peak_idx + 1 :] < threshold)

                if left_below.size == 0 or right_below.size == 0:
                    _warn_nan(
                        trace,
                        width_key,
                        "the spectrum does not cross the half-power level on both "
                        "sides of the peak within freq_range",
                    )
                    peak_width = np.nan
                else:
                    left_low_idx = int(left_below[-1])
                    left_high_idx = left_low_idx + 1
                    right_low_idx = peak_idx + 1 + int(right_below[0])
                    right_high_idx = right_low_idx - 1

                    f_low = np.interp(
                        threshold,
                        spec_sel[[left_low_idx, left_high_idx]],
                        freqs_sel[[left_low_idx, left_high_idx]],
                    )
                    f_high = np.interp(
                        threshold,
                        spec_sel[[right_low_idx, right_high_idx]],
                        freqs_sel[[right_low_idx, right_high_idx]],
                    )
                    peak_width = (f_high - f_low) / (2 * peak_freq)

            trace.attributes[width_key] = peak_width

        if "spectral_entropy" in targets:
            key = f"spectral_entropy{name_suffix}"
            if spectrum_issue is not None:
                _warn_nan(trace, key, spectrum_issue)
                trace.attributes[key] = np.nan
            else:
                trace.attributes[key] = _normalized_spectral_entropy(
                    spec_sel, spectrum_is_power=use_power_spectrum
                )

        if "void_index" in targets:
            max_band = spec[_band_mask(freqs, max_freq_min, max_freq_max)]
            mav_band = spec[_band_mask(freqs, mav_freq_min, mav_freq_max)]
            key = f"void_index{name_suffix}"

            if max_band.size == 0:
                reason = "no frequency bin falls in max_freq_range"
            elif mav_band.size == 0:
                reason = "no frequency bin falls in mav_freq_range"
            elif not (np.all(np.isfinite(max_band)) and np.all(np.isfinite(mav_band))):
                reason = "the spectrum in the void-index bands contains non-finite values"
            else:
                peak_value = np.max(max_band)
                if mav_method == "median":
                    mav_value = np.median(mav_band)
                else:
                    mav_value = np.mean(mav_band)
                reason = (
                    "the average spectrum level in mav_freq_range is zero"
                    if mav_value == 0
                    else None
                )

            if reason is not None:
                _warn_nan(trace, key, reason)
                trace.attributes[key] = np.nan
            else:
                trace.attributes[key] = peak_value / mav_value

"""Per-site statistics of the trace attributes."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._aggregation import aggregate_error_key
from ._attribute_selection import (
    AttributeSelection,
    normalize_attribute_selection,
    validate_known_attribute_names,
)
from ._console import safe_warn
from ._mse import unpack_mse_entry
from ._site import site_key
from .core import (
    AggregateStats,
    AveragedAttributeValue,
    MSEScaleStats,
    TraceData,
)

# Finite-sample bias A_n of the MAD about the sample median under normality,
# n = 2..100: Park, Kim and Wang (2022), Table A2,
# https://doi.org/10.1080/03610918.2019.1699114 (PKW).
_PKW_MAD_BIAS: dict[int, float] = {
    2: -0.1633880,
    3: -0.3275897,
    4: -0.2648275,
    5: -0.1781250,
    6: -0.1594213,
    7: -0.1210631,
    8: -0.1131928,
    9: -0.0920658,
    10: -0.0874503,
    11: -0.0741303,
    12: -0.0711412,
    13: -0.0620918,
    14: -0.0600210,
    15: -0.0534603,
    16: -0.0519047,
    17: -0.0467319,
    18: -0.0455579,
    19: -0.0417554,
    20: -0.0408248,
    21: -0.0376967,
    22: -0.0368350,
    23: -0.0342394,
    24: -0.0335390,
    25: -0.0313065,
    26: -0.0309765,
    27: -0.0290220,
    28: -0.0287074,
    29: -0.0269133,
    30: -0.0265451,
    31: -0.0250734,
    32: -0.0248177,
    33: -0.0236460,
    34: -0.0232808,
    35: -0.0222099,
    36: -0.0220756,
    37: -0.0210129,
    38: -0.0207309,
    39: -0.0199272,
    40: -0.0197140,
    41: -0.0188446,
    42: -0.0188203,
    43: -0.0180521,
    44: -0.0178185,
    45: -0.0171866,
    46: -0.0170796,
    47: -0.0165391,
    48: -0.0163509,
    49: -0.0157862,
    50: -0.0157372,
    51: -0.0152820,
    52: -0.0149951,
    53: -0.0146042,
    54: -0.0145007,
    55: -0.0140391,
    56: -0.0139674,
    57: -0.0136336,
    58: -0.0134819,
    59: -0.0130812,
    60: -0.0129708,
    61: -0.0126589,
    62: -0.0125598,
    63: -0.0122696,
    64: -0.0121523,
    65: -0.0118163,
    66: -0.0118244,
    67: -0.0115177,
    68: -0.0114479,
    69: -0.0111309,
    70: -0.0110816,
    71: -0.0108875,
    72: -0.0108319,
    73: -0.0106032,
    74: -0.0105424,
    75: -0.0102237,
    76: -0.0102132,
    77: -0.0099408,
    78: -0.0099776,
    79: -0.0097815,
    80: -0.0097399,
    81: -0.0094837,
    82: -0.0094713,
    83: -0.0092390,
    84: -0.0092875,
    85: -0.0091508,
    86: -0.0090145,
    87: -0.0088191,
    88: -0.0088205,
    89: -0.0086622,
    90: -0.0085714,
    91: -0.0084718,
    92: -0.0083861,
    93: -0.0082559,
    94: -0.0082650,
    95: -0.0080977,
    96: -0.0080708,
    97: -0.0078810,
    98: -0.0078492,
    99: -0.0077043,
    100: -0.0077614,
}

# Asymptotic consistency factor of the MAD under normality, 1 / Phi^-1(3/4).
_MAD_ASYM = 1.0 / 0.6744897501960817


def _pkw_bias_approx(n: int) -> float:
    """Return the PKW bias A_n for n > 100, A_n = a1/n + a2/n^2 (PKW Sec. 3, Hayes 2014)."""
    a1, a2 = -0.76213, -0.86413
    return a1 / n + a2 / (n * n)


def _mad_consistency_factor(n: int) -> float:
    """Return the finite-sample factor C_n of the MAD for n >= 2 (PKW 2022).

    C_n = 1.4826 / (1 + A_n), with A_n from PKW Table A2 up to n = 100 and
    from the Hayes approximation above.
    """
    An = _PKW_MAD_BIAS[n] if n in _PKW_MAD_BIAS else _pkw_bias_approx(n)
    return float(_MAD_ASYM / (1.0 + An))


def _aggregate_scalar(arr: np.ndarray, avg_algs: set, err_alg: str) -> AggregateStats:
    """Return min, max, the requested central values and their standard errors.

    Non-finite observations are left out; ``n_used`` counts the finite ones
    and ``n_dropped`` the others. With ``err_alg='mad'``, sigma = C_n * MAD
    with MAD = median(|x - median(x)|), SE(mean) = sigma / sqrt(n) and
    SE(median) = sqrt(pi/2) * sigma / sqrt(n). With ``'std'`` both errors
    are the sample standard deviation (ddof=1) over sqrt(n).
    """
    arr = np.asarray(arr, dtype=float)
    n_total = int(arr.size)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    entry = {
        "n_used": n,
        "n_dropped": n_total - n,
        "min": float(arr.min()) if n else np.nan,
        "max": float(arr.max()) if n else np.nan,
        "mean": None,
        aggregate_error_key("mean"): None,
        "median": None,
        aggregate_error_key("median"): None,
    }

    if n == 0:
        for statistic in avg_algs:
            entry[statistic] = np.nan
            entry[aggregate_error_key(statistic)] = np.nan
        return entry

    med = float(np.median(arr))

    if n < 2:
        if "median" in avg_algs:
            entry["median"] = med
            entry[aggregate_error_key("median")] = np.nan
        if "mean" in avg_algs:
            entry["mean"] = float(arr.mean())
            entry[aggregate_error_key("mean")] = np.nan
        return entry

    if err_alg == "mad":
        Cn = _mad_consistency_factor(n)
        mad = float(np.median(np.abs(arr - med)))
    else:
        sem = float(np.std(arr, ddof=1) / math.sqrt(n))

    if "median" in avg_algs:
        entry["median"] = med
        if err_alg == "mad":
            entry[aggregate_error_key("median")] = float(
                (math.sqrt(math.pi / 2.0) * Cn * mad) / math.sqrt(n)
            )
        else:
            entry[aggregate_error_key("median")] = sem

    if "mean" in avg_algs:
        entry["mean"] = float(arr.mean())
        if err_alg == "mad":
            entry[aggregate_error_key("mean")] = float((Cn * mad) / math.sqrt(n))
        else:
            entry[aggregate_error_key("mean")] = sem
    return entry


def _validate_mse_entries(
    entries: Sequence[Any],
    attribute_name: str,
    coordinate: tuple,
) -> tuple[np.ndarray, list[Any], list[np.ndarray]]:
    """Return the scalars, the scales and the value arrays of one site's MSE entries.

    Every scalar must be a single number and every entry must carry the
    scales of the first one.
    """
    scalars = []
    value_arrays = []
    first_scales: list[Any] = []
    first_scale_array = None

    for entry_index, entry in enumerate(entries, start=1):
        context = f"Site {coordinate}: MSE entry #{entry_index} of '{attribute_name}'"
        scalar, scale_array, value_array = unpack_mse_entry(entry, context)

        try:
            scalar_array = np.asarray(scalar, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context} scalar must be numeric.") from exc
        if scalar_array.ndim != 0:
            raise ValueError(f"{context} scalar must be a single numeric value.")
        scalars.append(float(scalar_array))

        if first_scale_array is None:
            first_scale_array = scale_array
            raw_scales = entry[1]["Scale"]
            if isinstance(raw_scales, np.ndarray):
                first_scales = raw_scales.tolist()
            else:
                first_scales = list(raw_scales)
        elif not np.array_equal(scale_array, first_scale_array):
            raise ValueError(f"{context} scales do not match those of entry #1.")
        value_arrays.append(value_array)

    return np.asarray(scalars, dtype=float), first_scales, value_arrays


def _aggregate_mse(
    entries: list[tuple[float, dict[str, Any]]],
    avg_algs: set,
    err_alg: str,
    *,
    attribute_name: str,
    coordinate: tuple,
) -> tuple[AggregateStats, MSEScaleStats]:
    """Return the statistics of the MSE scalars and ``{'Scale', 'Value'}`` per scale."""
    scalars, scales, value_arrays = _validate_mse_entries(
        entries,
        attribute_name,
        coordinate,
    )
    scalar_stats = _aggregate_scalar(scalars, avg_algs, err_alg)
    scale_stats: list[AggregateStats] = []
    for idx in range(len(scales)):
        arr = np.array([values[idx] for values in value_arrays], dtype=float)
        scale_stats.append(_aggregate_scalar(arr, avg_algs, err_alg))

    scale_summary: MSEScaleStats = {
        "Scale": scales,
        "Value": scale_stats,
    }
    return scalar_stats, scale_summary


def average_attributes(
    traces: list[TraceData],
    attrs: AttributeSelection = "all",
    average_algorithm: str | Sequence[str] = ("median", "mean"),
    error_algorithm: str = "mad",
) -> None:
    """Store per-site statistics of the attributes in ``attributes_avg``.

    Traces are grouped into sites by (x, y) rounded to six decimals; every
    trace at a site gives one observation, whatever its channel. An ``MSE``
    attribute is aggregated as its scalar and at every scale; all traces at
    a site must carry the same scales. A call replaces the whole
    ``attributes_avg`` of every trace, so statistics of an earlier call are
    not kept.

    Parameters
    ----------
    traces : list of TraceData
        Traces with ``attributes`` and finite ``x`` and ``y``.
    attrs : 'all', list of str or tuple of str, default 'all'
        Attributes to aggregate. 'all' takes every stored name in the order
        first seen across the traces.
    average_algorithm : str or sequence of str, default ('median', 'mean')
        Central values to compute: 'median', 'mean' or both.
    error_algorithm : {'mad', 'std'}, default 'mad'
        'mad': SE(mean) = C_n * MAD / sqrt(n) and
        SE(median) = sqrt(pi/2) * C_n * MAD / sqrt(n), with the
        finite-sample factor C_n of Park, Kim and Wang (2022).
        'std': sample standard deviation (ddof=1) / sqrt(n) for both, which
        for the median is only a rough estimate.

    Returns
    -------
    None
        Each statistics dictionary holds ``min``, ``max``, ``mean``,
        ``median``, their ``<statistic>_error``, ``n_used`` and
        ``n_dropped``; a central value that was not requested and its error
        are None.

    Raises
    ------
    ValueError
        For missing or non-finite coordinates, unknown names or algorithms,
        or inconsistent MSE entries. The traces are then left unchanged.
    TypeError
        For an ``attrs`` that is not 'all' or a list or tuple of names.

    Notes
    -----
    NaN and infinite values are left out and counted in ``n_dropped``, with
    a RuntimeWarning per site and attribute. A trace without the attribute
    is not counted; a site where no trace has it gets no entry for it and a
    UserWarning. One observation gives NaN errors; identical repeats give a
    MAD, and so a robust error, of zero. The errors can serve as
    inverse-variance weights, 1 / error**2.

    The traces of one site share one ``attributes_avg`` mapping; treat it
    as read-only.
    """
    attribute_selection = normalize_attribute_selection(
        attrs,
        parameter_name="attrs",
    )

    if not traces:
        safe_warn("average_attributes: no traces given.")
        return

    invalid_coordinates = []
    for tr in traces:
        try:
            coordinates_are_finite = np.isfinite(float(tr.x)) and np.isfinite(float(tr.y))
        except (TypeError, ValueError):
            coordinates_are_finite = False
        if not coordinates_are_finite:
            invalid_coordinates.append(str(tr.basename))
    if invalid_coordinates:
        names = ", ".join(invalid_coordinates)
        raise ValueError(f"average_attributes: missing or non-finite x or y for {names}.")

    all_supported: dict[str, None] = {}
    for tr in traces:
        if tr.attributes:
            all_supported.update(dict.fromkeys(tr.attributes))
    if not all_supported:
        raise ValueError("average_attributes: no trace has attributes.")

    if attribute_selection == "all":
        selected = list(all_supported)
    else:
        validate_known_attribute_names(
            attribute_selection,
            all_supported,
            parameter_name="attrs",
        )
        selected = list(attribute_selection)
        if not selected:
            raise ValueError("attrs selects no attribute.")

    if isinstance(average_algorithm, str):
        average_algorithm = [average_algorithm]
    avg_algs = {alg.lower() for alg in average_algorithm}
    if not avg_algs.issubset({"mean", "median"}):
        raise ValueError("average_algorithm may contain only 'mean' and 'median'.")
    err_alg = error_algorithm.lower()
    if err_alg not in {"std", "mad"}:
        raise ValueError("error_algorithm must be 'std' or 'mad'.")

    groups: dict[tuple, list[Any]] = {}
    for tr in traces:
        groups.setdefault(site_key(tr.x, tr.y), []).append(tr)

    # The traces are updated only after every site has succeeded.
    site_stats: dict[tuple, dict[str, AveragedAttributeValue]] = {}
    for coord, grp in groups.items():
        stats: dict[str, AveragedAttributeValue] = {}
        for attr in selected:
            raw = [tr.attributes.get(attr) for tr in grp if tr.attributes and attr in tr.attributes]
            if not raw:
                safe_warn(f"Site {coord}: no '{attr}' value; skipped.")
                continue

            if isinstance(attr, str) and attr.startswith("MSE"):
                scalar_stats, scale_dict = _aggregate_mse(
                    raw,
                    avg_algs,
                    err_alg,
                    attribute_name=attr,
                    coordinate=coord,
                )
                stats[attr] = (scalar_stats, scale_dict)
                n_dropped = scalar_stats["n_dropped"] + sum(
                    scale_stat["n_dropped"] for scale_stat in scale_dict["Value"]
                )
            else:
                arr = np.asarray(raw, dtype=float)
                stats[attr] = _aggregate_scalar(arr, avg_algs, err_alg)
                n_dropped = stats[attr]["n_dropped"]

            if n_dropped:
                safe_warn(
                    f"Site {coord}: {n_dropped} non-finite '{attr}' value(s) left out.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        site_stats[coord] = stats

    for coord, grp in groups.items():
        for tr in grp:
            tr.attributes_avg = site_stats[coord]

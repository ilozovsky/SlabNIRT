"""Sample entropy and refined composite multiscale entropy (RCMSE).

RCMSE follows Wu et al. (2014), Physics Letters A 378, 1369-1374:

1. Coarse-grain the record at scale ``tau`` into ``tau`` series of block
   means, one per start offset (their Eq. 4).
2. In each series count the pairs of ``m``-point templates whose Chebyshev
   distance is at most ``r``, and the pairs that still match with one more
   point; both template sets start at i = 1 .. N - m (their Eqs. 1-3).
3. RCMSE(tau) = -ln( sum of (m+1)-matches / sum of m-matches ) over the
   ``tau`` series (their Eq. 8).

Sample entropy is RCMSE at scale 1. A scale without m-point matches or
without (m+1)-point matches has no defined ratio and gives NaN.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree


def check_parameters(dimension: int = 2, tolerance: str | float = "sd", scale=None) -> None:
    """Raise ``ValueError`` for parameters ``rcmse`` cannot use (unknown ones raise ``TypeError``)."""
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError(f"dimension must be a positive integer, not {dimension!r}.")
    if isinstance(tolerance, str):
        if tolerance != "sd":
            raise ValueError(f"tolerance must be 'sd' or a number, not {tolerance!r}.")
    elif not float(tolerance) >= 0.0:
        raise ValueError(f"tolerance must be non-negative, not {tolerance!r}.")
    if scale is not None:
        scales = list(scale)
        if not scales or any(not isinstance(s, (int, np.integer)) or s < 1 for s in scales):
            raise ValueError(
                f"scale must be None or a sequence of positive integers, not {scale!r}."
            )


def coarse_grain(x: np.ndarray, tau: int) -> list[np.ndarray]:
    """Return the ``tau`` block-mean series of ``x`` at scale ``tau`` (Wu et al. Eq. 4)."""
    series = []
    for start in range(tau):
        blocks = (x.size - start) // tau
        if blocks:
            series.append(x[start : start + blocks * tau].reshape(blocks, tau).mean(axis=1))
    return series


def matched_pairs(y: np.ndarray, m: int, r: float) -> tuple[int, int]:
    """Return the ordered template pairs of ``y`` within ``r`` at lengths ``m`` and ``m + 1``."""
    if y.size - m < 2:
        return 0, 0
    templates = sliding_window_view(y, m)[:-1]  # same start points as the longer templates
    longer = sliding_window_view(y, m + 1)
    n_m = cKDTree(templates).count_neighbors(cKDTree(templates), r, p=np.inf) - len(templates)
    n_m1 = cKDTree(longer).count_neighbors(cKDTree(longer), r, p=np.inf) - len(longer)
    return int(n_m), int(n_m1)


def rcmse(
    signal,
    scale=None,
    dimension: int = 2,
    tolerance: str | float = "sd",
) -> tuple[np.ndarray, dict]:
    """Return RCMSE per scale and the details stored with the ``MSE`` attribute.

    Parameters
    ----------
    signal : array-like
        One-dimensional record with finite values.
    scale : sequence of positive int, optional
        Scale factors. None selects every scale at which the coarse-grained
        series keeps at least ``10**dimension`` samples, the lower end of
        the record length recommended for sample entropy (Wu et al. 2014,
        citing Liu et al. 2012); scale 1 is always included.
    dimension : int
        Template length m.
    tolerance : 'sd' or float
        Match distance r: 'sd' means 0.2 times the standard deviation
        (ddof=1) of ``signal``, a number is used as given.

    Returns
    -------
    values : ndarray
        RCMSE in nats per scale; NaN where the ratio is undefined.
    details : dict
        'Dimension', 'Tolerance' (the r used), 'Scale' (int array),
        'Value' (same as ``values``) and 'Counts', the (m, m+1) match
        totals per scale.
    """
    check_parameters(dimension, tolerance, scale)
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    r = 0.2 * np.std(x, ddof=1) if isinstance(tolerance, str) else float(tolerance)
    if scale is None:
        scales = list(range(1, max(x.size // 10**dimension, 1) + 1))
    else:
        scales = [int(s) for s in scale]

    values = np.full(len(scales), np.nan)
    counts = []
    for index, tau in enumerate(scales):
        n_m = n_m1 = 0
        for series in coarse_grain(x, tau):
            pairs_m, pairs_m1 = matched_pairs(series, dimension, r)
            n_m += pairs_m
            n_m1 += pairs_m1
        counts.append((n_m, n_m1))
        if n_m and n_m1:
            values[index] = 0.0 - np.log(n_m1 / n_m)  # 0.0 - ... keeps -ln(1) at +0.0

    details = {
        "Dimension": dimension,
        "Tolerance": r,
        "Scale": np.asarray(scales),
        "Value": values,
        "Counts": counts,
    }
    return values, details


def sample_entropy(signal, dimension: int = 2, tolerance: str | float = "sd") -> float:
    """Return the sample entropy of ``signal`` in nats: RCMSE at scale 1."""
    values, _ = rcmse(signal, scale=[1], dimension=dimension, tolerance=tolerance)
    return float(values[0])


def multiscale_entropy(
    signal,
    scale=None,
    dimension: int = 2,
    tolerance: str | float = "sd",
) -> tuple[float, dict]:
    """Return ``(value, details)`` for the ``MSE`` attribute.

    ``value`` is the mean of the finite RCMSE values over the scales (NaN if
    none is finite); ``details`` is the dictionary described in ``rcmse``.
    """
    values, details = rcmse(signal, scale=scale, dimension=dimension, tolerance=tolerance)
    finite = values[np.isfinite(values)]
    value = float(finite.mean()) if finite.size else np.nan
    return value, details

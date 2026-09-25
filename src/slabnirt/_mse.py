"""Internal helpers for exact handling of named MSE attribute variants."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def mse_attribute_keys(attribute_names: Iterable[str]) -> list[str]:
    """Return stored MSE keys in their existing insertion order."""
    return [key for key in attribute_names if isinstance(key, str) and key.startswith("MSE")]


def resolve_mse_feature(
    feature_name: str,
    available_mse_keys: Sequence[str],
) -> tuple[str, int | None] | None:
    """Resolve an exact MSE scalar or derived ``<key>_scaleN`` feature name."""
    if feature_name in available_mse_keys:
        return feature_name, None

    # Longest first prevents a base key from capturing a suffixed variant.
    for mse_key in sorted(available_mse_keys, key=len, reverse=True):
        prefix = f"{mse_key}_scale"
        if feature_name.startswith(prefix):
            scale_text = feature_name[len(prefix) :]
            if scale_text.isdigit():
                return mse_key, int(scale_text)
    return None


def selected_mse_keys(
    requested: str | Sequence[str],
    available_mse_keys: Sequence[str],
) -> list[str]:
    """Return the stored MSE variants selected by 'all' or a list of names."""
    if requested == "all":
        return list(available_mse_keys)
    return [key for key in available_mse_keys if key in requested]


def unpack_mse_entry(entry: Any, context: str) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return the scalar and the 'Scale' and 'Value' arrays of a stored MSE entry.

    ``entry`` must be a ``(scalar, details)`` pair whose details hold numeric,
    one-dimensional 'Scale' and 'Value' of equal, nonzero length, with finite
    scales. Otherwise a ValueError is raised whose message starts with
    ``context``.
    """
    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
        raise ValueError(f"{context} must be a (scalar, details) pair.")
    scalar, details = entry
    if not isinstance(details, Mapping) or "Scale" not in details or "Value" not in details:
        raise ValueError(f"{context} details must be a mapping with 'Scale' and 'Value'.")
    try:
        scales = np.asarray(details["Scale"], dtype=float)
        values = np.asarray(details["Value"], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} 'Scale' and 'Value' must be numeric.") from exc
    if scales.ndim != 1 or values.ndim != 1 or scales.size == 0 or scales.size != values.size:
        raise ValueError(
            f"{context} 'Scale' and 'Value' must be one-dimensional, nonempty and of equal length."
        )
    if not np.all(np.isfinite(scales)):
        raise ValueError(f"{context} scales must be finite.")
    return scalar, scales, values

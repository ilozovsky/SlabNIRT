"""Export of signals, spectra and attributes to delimited text or Excel files."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._aggregation import AGGREGATE_EXPORT_KEYS
from ._attribute_selection import (
    AttributeSelection,
    normalize_attribute_selection,
    validate_known_attribute_names,
)
from ._logging import log_info
from ._site import site_key
from .core import TraceData

EXPORT_MODES = ("signals", "spectra", "attributes", "attributes_avg")


def _value_at(values: np.ndarray | None, index: int) -> float:
    """Return ``values[index]``, or NaN when the array is absent or too short."""
    return values[index] if values is not None and len(values) > index else np.nan


def export_tracedata(
    traces: list[TraceData],
    mode: str | list[str] = "attributes",
    attributes: AttributeSelection = "all",
    MSE_scales: list[int] | None = None,
    output_prefix: str | Path | None = None,
    output_dir: str | Path | None = None,
    to_excel: bool = False,
    float_decimals: int = 6,
    include_filepath: bool = False,
    sep: str = ",",
) -> None:
    """Write the traces to one delimited text or Excel file per mode.

    Each file is named ``<prefix>_<mode>.dat`` or ``.xlsx``; an existing
    file is overwritten. The traces are not changed.

    Parameters
    ----------
    traces : list of TraceData
        Traces to export.
    mode : str or list of str, default 'attributes'
        'signals' (one row per sample), 'spectra' (one row per frequency
        bin), 'attributes' (one row per trace) and 'attributes_avg' (one row
        per site, identified by its first trace: basename, channel and
        unrounded coordinates). Traces without the data are skipped.
    attributes : 'all', list of str or tuple of str, default 'all'
        Attributes of the two attribute modes. Explicit names set the column
        order; 'all' keeps the stored order.
    MSE_scales : list of int, optional
        Scales to export for every selected MSE variant, as ``<name>_scaleN``
        columns (``<name>_scaleN_<statistic>`` in 'attributes_avg') after
        the variant's own columns.
    output_prefix : str or Path, optional
        Prefix of the file names, 'export' when None. Without
        ``output_dir`` the files go to the prefix's folder, which must
        exist; with ``output_dir`` only the last part of the prefix is used.
    output_dir : str or Path, optional
        Directory for all files, created if needed.
    to_excel : bool, default False
        Write .xlsx files at full precision instead of .dat text files.
    float_decimals : int, default 6
        Digits after the decimal point in the scientific notation of text
        files, so 6 keeps seven significant digits; 16 round-trips exactly.
        Coordinates are always written in full. No effect on Excel files.
    include_filepath : bool, default False
        Add a 'filepath' column.
    sep : str, default ','
        One-character delimiter of .dat files.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        For an unknown mode or attribute name, before any file is written.
    TypeError
        For an ``attributes`` that is not 'all' or a list or tuple of names.

    Notes
    -----
    The columns are 'basename', 'channel', 'x', 'y' (and 'filepath'),
    followed by the data. An averaged attribute gives ``<name>_mean``,
    ``_mean_error``, ``_median``, ``_median_error``, ``_n_used`` and
    ``_n_dropped``. A column that is missing in every row is left out: the
    statistics that were not requested, but also the error columns when
    every site has one observation, an attribute that is NaN on every
    trace, and an MSE scale that no trace has.
    """
    attribute_selection = normalize_attribute_selection(
        attributes,
        parameter_name="attributes",
    )
    modes = [mode] if isinstance(mode, str) else mode
    unknown_modes = [m for m in modes if m not in EXPORT_MODES]
    if unknown_modes:
        raise ValueError(
            f"mode contains unknown export mode(s): {unknown_modes}. "
            f"Valid modes: {list(EXPORT_MODES)}."
        )

    if attribute_selection != "all" and any(m in ("attributes", "attributes_avg") for m in modes):
        available_attributes = {
            name
            for td in traces
            for selected_mode in modes
            if selected_mode in ("attributes", "attributes_avg")
            for name in (
                td.attributes or {} if selected_mode == "attributes" else td.attributes_avg or {}
            )
        }
        validate_known_attribute_names(
            attribute_selection,
            available_attributes,
            parameter_name="attributes",
        )

    dir_path = Path(output_dir) if output_dir is not None else None
    if dir_path:
        dir_path.mkdir(parents=True, exist_ok=True)

    prefix = Path(output_prefix) if output_prefix else None
    ext = ".xlsx" if to_excel else ".dat"
    base = prefix.name if prefix else "export"
    common_cols = ["basename", "channel", "x", "y"] + (["filepath"] if include_filepath else [])

    def add_common_fields(row: dict[str, Any], td: TraceData) -> dict[str, Any]:
        row["basename"] = td.basename
        row["channel"] = td.channel
        row["x"] = str(td.x) if td.x is not None else ""
        row["y"] = str(td.y) if td.y is not None else ""
        if include_filepath:
            row["filepath"] = td.filepath
        return row

    for m in modes:
        fname = f"{base}_{m}{ext}"
        if dir_path:
            outpath = dir_path / fname
        elif prefix:
            outpath = prefix.with_name(fname)
        else:
            outpath = Path(fname)

        if m == "attributes_avg":
            seen = set()
            trace_list = []
            for td in traces:
                coord = site_key(td.x, td.y)
                if coord not in seen:
                    seen.add(coord)
                    trace_list.append(td)
        else:
            trace_list = traces

        rows: list[dict[str, Any]] = []
        for td in trace_list:
            if m == "signals":
                for i, t_i in enumerate(td.t):
                    row = {
                        "t": t_i,
                        "signal": td.signal[i] if td.signal is not None else np.nan,
                        "signal_norm": td.signal_norm[i] if td.signal_norm is not None else np.nan,
                        "signal_norm_pre": td.signal_norm_pre[i]
                        if td.signal_norm_pre is not None
                        else np.nan,
                    }
                    rows.append(add_common_fields(row, td))

            elif m == "spectra":
                if td.freqs is None or td.freqs.size == 0:
                    continue
                for i, f_i in enumerate(td.freqs):
                    row = {
                        "freqs": f_i,
                        "spectrum": _value_at(td.spectrum, i),
                        "freqs_filt": _value_at(td.freqs_filt, i),
                        "spec_filt": _value_at(td.spec_filt, i),
                    }
                    rows.append(add_common_fields(row, td))

            else:
                adict = td.attributes if m == "attributes" else td.attributes_avg or {}
                if not adict:
                    continue

                if attribute_selection == "all":
                    selected_attrs = list(adict)
                else:
                    selected_attrs = []
                    for attr in attribute_selection:
                        if attr in adict and attr not in selected_attrs:
                            selected_attrs.append(attr)
                if not selected_attrs:
                    continue

                row: dict[str, Any] = {}
                for attr in selected_attrs:
                    raw = adict.get(attr)
                    if raw is None:
                        continue

                    if m == "attributes":
                        if attr.startswith("MSE"):
                            scalar_val, detail = raw
                            row[attr] = scalar_val
                            if MSE_scales:
                                scales = list(detail.get("Scale", []))
                                values = detail.get("Value", [])
                                for s in MSE_scales:
                                    row[f"{attr}_scale{s}"] = (
                                        values[scales.index(s)] if s in scales else np.nan
                                    )
                        else:
                            row[attr] = raw

                    elif attr.startswith("MSE"):
                        scalar_stats, scale_stats = raw
                        for st in AGGREGATE_EXPORT_KEYS:
                            row[f"{attr}_{st}"] = scalar_stats.get(st, np.nan)
                        if MSE_scales and isinstance(scale_stats, dict):
                            scales = list(scale_stats.get("Scale", []))
                            values = scale_stats.get("Value", [])
                            for s in MSE_scales:
                                stats = values[scales.index(s)] if s in scales else {}
                                for st in AGGREGATE_EXPORT_KEYS:
                                    row[f"{attr}_scale{s}_{st}"] = stats.get(st, np.nan)
                    else:
                        for st in AGGREGATE_EXPORT_KEYS:
                            row[f"{attr}_{st}"] = raw.get(st, np.nan)
                rows.append(add_common_fields(row, td))

        if not rows:
            log_info(f"{m!r}: no data to export; skipped.")
            continue

        df = pd.DataFrame(rows)
        other_cols = [c for c in df.columns if c not in common_cols]
        df = df[common_cols + other_cols]
        df = df.dropna(axis=1, how="all")

        if to_excel:
            df.to_excel(outpath, index=False)
        else:
            df.to_csv(outpath, index=False, float_format=f"%.{float_decimals}e", sep=sep)

        log_info(f"Exported {m!r} to {outpath}.")

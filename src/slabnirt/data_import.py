"""Readers for text and SEG-Y traces and for coordinate catalogs."""

import glob
import os
import re

import numpy as np
import pandas as pd
import segyio

from ._console import safe_warn
from ._logging import log_info
from .core import TraceData

# pandas' parser and decoder and the time-axis checks below all raise
# subclasses of ValueError.
_TXT_READER_ERRORS = (OSError, ValueError)
_SEGY_READER_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    IndexError,
    KeyError,
)


def _read_txt_table(filepath: str) -> tuple[pd.DataFrame, float]:
    """Return the table of a text file and its sampling interval.

    Raises ValueError for a time column that is too short, non-finite, not
    strictly increasing or not uniformly sampled.
    """
    frame = pd.read_csv(filepath, sep=r"\s+", header=None)
    time = frame[0].to_numpy(dtype=float)
    if time.size < 2:
        raise ValueError("time column must contain at least two samples")
    if not np.all(np.isfinite(time)):
        raise ValueError("time column values must be finite")

    time_steps = np.diff(time)
    if not np.all(time_steps > 0.0):
        raise ValueError("time column must be strictly increasing")

    dt = time_steps[0]
    if not np.allclose(time_steps, dt, rtol=1e-5, atol=0.0):
        raise ValueError("time column must be uniformly sampled")
    return frame, dt


def _read_segy_channels(
    filepath: str,
    channels: list[int],
    time_scaling: float,
) -> tuple[
    list[tuple[int, np.ndarray, np.ndarray]],
    list[str],
    Exception | None,
]:
    """Return the (channel, t, signal) triples read, the unavailable channels and any error.

    Channels read before an error are kept.
    """
    raw_traces = []
    unavailable = []
    try:
        with segyio.open(filepath, ignore_geometry=True) as segy_file:
            trace_count = len(segy_file.trace)
            for ch in channels:
                if ch < 1 or ch > trace_count:
                    unavailable.append(
                        f"{filepath} (channel {ch}: unavailable; file "
                        f"has {trace_count} trace channel(s))"
                    )
                    continue

                signal = segy_file.trace[ch - 1]
                dt_us = segy_file.header[ch - 1][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
                dt = (dt_us / 1e6) * time_scaling
                t = np.linspace(0, signal.size * dt, signal.size, endpoint=False)
                raw_traces.append((ch, t, signal))
    except _SEGY_READER_ERRORS as exc:
        return raw_traces, unavailable, exc
    return raw_traces, unavailable, None


def read_coordinates_from_catalog(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a coordinate catalog of trace files.

    Parameters
    ----------
    path : str
        A .csv file or an .xlsx file (first sheet) with the columns
        ``filename``, ``x`` and ``y``, in any letter case. A filename may
        carry its extension or not.

    Returns
    -------
    filenames : ndarray of str
    x, y : ndarray of float
        Coordinates in survey units.

    Raises
    ------
    ValueError
        For another file type or a missing column.
    """

    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"{path}: the catalog must be a .csv or .xlsx file.")

    df.columns = [col.lower() for col in df.columns]

    required_columns = {"filename", "x", "y"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{path}: the catalog must contain the columns filename, x and y.")

    filenames = df["filename"].astype(str).to_numpy()
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    return filenames, x, y


def load_traces(
    file_type: str = "txt",
    directory: str = ".",
    channel: int | list[int] = 1,
    time_scaling: float = 1.0,
    filenames_with_coordinates: np.ndarray | list[str] | pd.Series | None = None,
    x_position: np.ndarray | list[float] | pd.Series | None = None,
    y_position: np.ndarray | list[float] | pd.Series | None = None,
) -> list[TraceData]:
    """Load every .txt or .sgy file of a directory as TraceData objects.

    Files are read in natural name order (``file2`` before ``file10``).

    A .txt file holds whitespace-separated columns without a header: column 0
    is time, columns 1, 2, ... are signals. The time column must be finite,
    strictly increasing and uniform within a relative tolerance of 1e-5.
    For a .sgy file each trace is a channel; the sampling interval is read
    from the trace headers in microseconds and converted to seconds. The
    SEG-Y reader was written for files from IDS (Logicheskie Sistemy)
    equipment; other SEG-Y layouts may need changes.

    Parameters
    ----------
    file_type : {'txt', 'sgy'}, default 'txt'
        Extension of the files to load.
    directory : str, default '.'
        Directory to search.
    channel : int or list of int, default 1
        Channels to load: the signal column of a .txt file (1 is the first
        signal column) or the 1-based trace number of a .sgy file.
    time_scaling : float, default 1.0
        Factor applied to the sampling interval. For .txt files it is the
        length of one time-column unit in s: 0.001 for a time column in
        ms, 1e-6 for microseconds. For .sgy files the interval is already
        in s, so keep 1.0. Frequencies are in Hz only when the time axis
        ends up in s.
    filenames_with_coordinates : array-like of str, optional
        Catalog file names, with or without extension; each must occur once.
    x_position, y_position : array-like of float, optional
        Catalog coordinates, one per file name. The three catalog arguments
        are given together or not at all.

    Returns
    -------
    list of TraceData
        One trace per file and channel, with ``basename``, ``channel``,
        ``t`` in s, ``signal``, ``filepath`` and, for catalog files, ``x``
        and ``y``. ``t`` starts at 0 and advances by the first time step of
        the file; the file's other time values are not kept. A directory
        without matching files gives an empty list.

    Warns
    -----
    RuntimeWarning
        When some files or channels fail but at least one trace loads.
    UserWarning
        For a loaded file missing from the catalog, and for catalog entries
        without a loaded file.

    Raises
    ------
    ValueError
        For invalid arguments, or when files match but no trace loads.
    FileNotFoundError
        For a directory that does not exist.
    """

    def natural_key(string: str):
        """Sort key that puts 'file2' before 'file10'."""
        return [
            int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", string)
        ]

    if file_type not in {"txt", "sgy"}:
        raise ValueError("file_type must be either 'txt' or 'sgy'.")

    if isinstance(channel, (int, np.integer)):
        channel = [int(channel)]
    else:
        channel = list(channel)

    if file_type == "txt":
        invalid_channels = [
            ch for ch in channel if not isinstance(ch, (int, np.integer)) or int(ch) < 1
        ]
        if invalid_channels:
            raise ValueError(
                "channel must hold integers >= 1 for .txt files (column 0 is time); "
                f"invalid: {invalid_channels}."
            )
        channel = [int(ch) for ch in channel]

    # Validate coordinate metadata before file discovery so an invalid catalog
    # cannot be hidden by an empty or incorrect input directory.
    catalog = (filenames_with_coordinates, x_position, y_position)
    if any(value is None for value in catalog) and any(value is not None for value in catalog):
        raise ValueError(
            "filenames_with_coordinates, x_position and y_position must be given together."
        )
    if filenames_with_coordinates is not None:
        filenames_with_coordinates = np.asarray(
            [os.path.splitext(fname)[0] for fname in filenames_with_coordinates], dtype=str
        )
        x_position = np.asarray(x_position, dtype=float)
        y_position = np.asarray(y_position, dtype=float)
        if not len(filenames_with_coordinates) == len(x_position) == len(y_position):
            raise ValueError(
                "filenames_with_coordinates, x_position and y_position must have the same length."
            )

        seen_names = set()
        duplicate_names = []
        for fname in filenames_with_coordinates:
            name = str(fname)
            if name in seen_names and name not in duplicate_names:
                duplicate_names.append(name)
            seen_names.add(name)
        if duplicate_names:
            names = ", ".join(duplicate_names)
            raise ValueError(
                f"filenames_with_coordinates: duplicate names after the extensions are "
                f"removed: {names}."
            )

    coord_lookup = {}
    if filenames_with_coordinates is not None:
        coord_lookup = {
            str(fname): (float(x), float(y))
            for fname, x, y in zip(filenames_with_coordinates, x_position, y_position, strict=True)
        }

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"{directory}: no such directory.")
    pattern = os.path.join(directory, f"*.{file_type}")
    file_list = sorted(glob.glob(pattern), key=natural_key)

    if not file_list:
        log_info(f"No .{file_type} files found in '{directory}'.")
        return []

    traces = []
    loaded_basenames = set()
    load_failures = []

    for filepath in file_list:
        basename = os.path.splitext(os.path.basename(filepath))[0]

        if file_type == "txt":
            try:
                df, dt = _read_txt_table(filepath)
            except _TXT_READER_ERRORS as exc:
                load_failures.append(f"{filepath} ({type(exc).__name__}: {exc})")
                continue

            for ch in channel:
                if ch >= len(df.columns):
                    load_failures.append(
                        f"{filepath} (channel {ch}: unavailable; file has "
                        f"{max(len(df.columns) - 1, 0)} signal channel(s))"
                    )
                    continue

                t = np.arange(len(df)) * dt * time_scaling
                signal = df[ch].to_numpy()
                x, y = coord_lookup.get(basename, (None, None))
                if filenames_with_coordinates is not None and x is None:
                    safe_warn(f"{basename}: not in the coordinate catalog.")
                new_trace = TraceData(basename, ch, t, signal, filepath, x=x, y=y)
                traces.append(new_trace)
                loaded_basenames.add(basename)

        else:
            raw_traces, unavailable, reader_failure = _read_segy_channels(
                filepath,
                channel,
                time_scaling,
            )
            load_failures.extend(unavailable)
            for ch, t, signal in raw_traces:
                x, y = coord_lookup.get(basename, (None, None))
                if filenames_with_coordinates is not None and x is None:
                    safe_warn(f"{basename}: not in the coordinate catalog.")
                new_trace = TraceData(basename, ch, t, signal, filepath, x=x, y=y)
                traces.append(new_trace)
                loaded_basenames.add(basename)

            if reader_failure is not None:
                load_failures.append(
                    f"{filepath} ({type(reader_failure).__name__}: {reader_failure})"
                )

    failure_details = "; ".join(load_failures)
    if not traces:
        message = (
            f"{directory}: no trace could be loaded from {len(file_list)} .{file_type} file(s)."
        )
        if failure_details:
            message = f"{message} Failures: {failure_details}."
        else:
            message = (
                f"{message} No file/channel request produced a trace; check the channel selection."
            )
        raise ValueError(message)

    if load_failures:
        safe_warn(
            f"{directory}: {len(load_failures)} file/channel request(s) failed, "
            f"{len(traces)} trace(s) loaded. Failures: {failure_details}.",
            RuntimeWarning,
            stacklevel=2,
        )

    if filenames_with_coordinates is not None:
        missing = set(filenames_with_coordinates) - loaded_basenames
        if missing:
            missing_names = ", ".join(sorted(missing))
            safe_warn(f"filenames_with_coordinates: no trace loaded for {missing_names}.")

    log_info(f"Loaded {len(traces)} traces.")
    return traces

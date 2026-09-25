"""Columns, rows and precision of the files written by export_tracedata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slabnirt import average_attributes, export_tracedata

COMMON_COLUMNS = ["basename", "channel", "x", "y"]


def _mse(value: float, scales: list[int]) -> tuple[float, dict[str, np.ndarray]]:
    return value, {
        "Scale": np.asarray(scales, dtype=int),
        "Value": np.asarray([value + scale / 10.0 for scale in scales]),
    }


def test_columns_with_no_finite_value_are_omitted(site_trace, tmp_path: Path) -> None:
    """Unrequested statistics, single-observation error bars, the statistics of
    an attribute that is NaN everywhere, and a requested MSE scale nobody has
    all leave no column; the finite counts of the all-NaN attribute survive."""
    traces = [
        site_trace("p", {"f": 1.0, "g": np.nan, "MSE": _mse(0.5, [1, 2])}, x=0.0),
        site_trace("q", {"f": 2.0, "g": np.nan, "MSE": _mse(0.7, [1, 2])}, x=1.0),
    ]
    with pytest.warns(RuntimeWarning, match="1 non-finite 'g' value"):
        average_attributes(traces, attrs="all", average_algorithm=["median"])
    export_tracedata(
        traces,
        mode=["attributes", "attributes_avg"],
        MSE_scales=[2, 9],
        output_prefix="rule",
        output_dir=tmp_path,
    )

    raw = pd.read_csv(tmp_path / "rule_attributes.dat")
    averaged = pd.read_csv(tmp_path / "rule_attributes_avg.dat")

    assert list(raw.columns) == COMMON_COLUMNS + ["f", "MSE", "MSE_scale2"]
    assert list(averaged.columns) == COMMON_COLUMNS + [
        "f_median",
        "f_n_used",
        "f_n_dropped",
        "g_n_used",
        "g_n_dropped",
        "MSE_median",
        "MSE_n_used",
        "MSE_n_dropped",
        "MSE_scale2_median",
        "MSE_scale2_n_used",
        "MSE_scale2_n_dropped",
    ]


def test_averaged_export_writes_one_row_per_site_keyed_by_its_first_trace(
    site_trace, tmp_path: Path
) -> None:
    """The row carries the first trace's name, channel and unrounded coordinates."""
    traces = [
        site_trace("late", {"f": 1.0}, x=2.0000004, channel=2),
        site_trace("early", {"f": 3.0}, x=1.9999996, channel=1),
        site_trace("other", {"f": 9.0}, x=5.0),
        site_trace("apart", {"f": 4.0}, x=5.0000011),
    ]
    average_attributes(traces, attrs=["f"])

    export_tracedata(traces, mode="attributes_avg", output_prefix="sites", output_dir=tmp_path)

    exported = pd.read_csv(tmp_path / "sites_attributes_avg.dat")
    assert exported["basename"].tolist() == ["late", "other", "apart"]
    assert exported["channel"].tolist() == [2, 1, 1]
    assert exported["x"].tolist() == [2.0000004, 5.0, 5.0000011]
    assert exported["f_n_used"].tolist() == [2, 1, 1]
    assert exported["f_mean"].tolist() == [2.0, 9.0, 4.0]


def test_text_export_rounds_to_float_decimals_and_keeps_coordinates_exact(
    site_trace, tmp_path: Path
) -> None:
    """Six decimals in scientific notation keep seven significant digits."""
    value = 1.23456789012345
    trace = site_trace("p", {"f": value}, x=123456.123456789, y=-0.75)

    export_tracedata([trace], mode="attributes", output_prefix="six", output_dir=tmp_path)
    export_tracedata(
        [trace], mode="attributes", output_prefix="full", output_dir=tmp_path, float_decimals=16
    )

    six = pd.read_csv(tmp_path / "six_attributes.dat")
    full = pd.read_csv(tmp_path / "full_attributes.dat")
    assert six.loc[0, "f"] == float(f"{value:.6e}")
    assert six.loc[0, "f"] != value
    assert full.loc[0, "f"] == value
    assert (six.loc[0, "x"], six.loc[0, "y"]) == (123456.123456789, -0.75)
    first_row = (tmp_path / "six_attributes.dat").read_text().splitlines()[1]
    assert first_row == "p,1,123456.123456789,-0.75,1.234568e+00"


def test_excel_export_keeps_full_precision(site_trace, tmp_path: Path) -> None:
    value = 1.23456789012345
    trace = site_trace("p", {"f": value, "MSE": _mse(0.5, [1, 2])}, x=0.25)

    export_tracedata(
        [trace],
        mode="attributes",
        MSE_scales=[2],
        output_prefix="xl",
        output_dir=tmp_path,
        to_excel=True,
        float_decimals=2,
    )

    exported = pd.read_excel(tmp_path / "xl_attributes.xlsx")
    assert list(exported.columns) == COMMON_COLUMNS + ["f", "MSE", "MSE_scale2"]
    assert exported.loc[0, "f"] == value
    assert exported.loc[0, "MSE_scale2"] == 0.7


def test_signal_and_spectrum_exports_write_one_row_per_sample(
    synthetic_trace, tmp_path: Path
) -> None:
    """An unfiltered trace gives no filtered-spectrum columns."""
    trace = synthetic_trace([0.0, 2.0, -1.0], name="s")
    trace.x, trace.y = 1.0, 2.0
    trace.signal_norm = trace.signal / 2.0
    trace.freqs = np.array([0.0, 500.0])
    trace.spectrum = np.array([0.5, 0.25])

    export_tracedata(
        [trace],
        mode=["signals", "spectra"],
        output_prefix="rows",
        output_dir=tmp_path,
        include_filepath=True,
    )

    signals = pd.read_csv(tmp_path / "rows_signals.dat")
    spectra = pd.read_csv(tmp_path / "rows_spectra.dat")
    assert list(signals.columns) == COMMON_COLUMNS + ["filepath", "t", "signal", "signal_norm"]
    assert signals["t"].tolist() == [0.0, 0.0005, 0.001]
    assert signals["signal_norm"].tolist() == [0.0, 1.0, -0.5]
    assert list(spectra.columns) == COMMON_COLUMNS + ["filepath", "freqs", "spectrum"]
    assert spectra["spectrum"].tolist() == [0.5, 0.25]


def test_export_selects_the_exact_mse_variant_and_scales(site_trace, tmp_path: Path) -> None:
    trace = site_trace("p", {"MSE_alt": _mse(101.0, [1, 2, 3]), "MSE": _mse(1.0, [1, 2])})

    export_tracedata(
        [trace],
        mode="attributes",
        attributes=["MSE_alt"],
        MSE_scales=[2, 3],
        output_prefix="exact",
        output_dir=tmp_path,
    )

    exported = pd.read_csv(tmp_path / "exact_attributes.dat")
    assert list(exported.columns) == COMMON_COLUMNS + [
        "MSE_alt",
        "MSE_alt_scale2",
        "MSE_alt_scale3",
    ]
    assert exported.loc[0, ["MSE_alt", "MSE_alt_scale2", "MSE_alt_scale3"]].tolist() == [
        101.0,
        101.2,
        101.3,
    ]

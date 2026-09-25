"""The shipped slab-model example, run end to end.

``examples/slab_model_workflow.py`` is run the way a user runs it, in a
fresh interpreter, on the 1788-trace dataset next to it. One run is shared
by the tests: a second fresh run with another hash seed must produce the
same tables and figures of the same size, and the exported site table must
show the model's known property - the air void under the slab center
lowers the peak frequency and the normalized spectrum area and raises the
void index and the sample entropy at the sites above it, compared with the
sites on the sound slab away from the free edges. The children import the
same package as this test session (this checkout's ``src``, or
site-packages when the installed package is tested), never another
installed slabnirt.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import slabnirt
from slabnirt import load_traces, preprocess_traces, read_coordinates_from_catalog

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "slab_model_workflow.py"
PACKAGE_ROOT = Path(slabnirt.__file__).resolve().parents[1]
CHILD_ENV = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT), "MPLBACKEND": "Agg"}


def run_example(output_dir: Path, hash_seed: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), str(output_dir)],
        capture_output=True,
        text=True,
        env={**CHILD_ENV, "PYTHONHASHSEED": hash_seed},
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "run.log").read_text(encoding="utf-8").count("Loaded 1788 traces.") == 1


def inventory(output_dir: Path) -> dict[str, bytes | tuple[int, int]]:
    """Every output file: exported bytes, or the pixel size of a figure.

    The run report carries timestamps and absolute paths, so it is left out.
    An .xlsx is a zip whose ``docProps/core.xml`` records the minute openpyxl
    wrote it, so the sheet itself stands for the workbook.
    """
    result: dict[str, bytes | tuple[int, int]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.name == "run.log" or not path.is_file():
            continue
        name = path.relative_to(output_dir).as_posix()
        if path.suffix == ".xlsx":
            with zipfile.ZipFile(path) as workbook:
                result[name] = workbook.read("xl/worksheets/sheet1.xml")
            continue
        data = path.read_bytes()
        result[name] = struct.unpack(">II", data[16:24]) if path.suffix == ".png" else data
    return result


@pytest.fixture(scope="module")
def example_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    imported = subprocess.run(
        [sys.executable, "-c", "import slabnirt; print(slabnirt.__file__)"],
        capture_output=True,
        text=True,
        env=CHILD_ENV,
        check=False,
    )
    assert Path(imported.stdout.strip()).resolve() == Path(slabnirt.__file__).resolve()
    output_dir = tmp_path_factory.mktemp("slab_model") / "first"
    run_example(output_dir, "1")
    return output_dir


def test_example_is_repeatable_across_fresh_processes(example_output: Path, tmp_path: Path) -> None:
    second = tmp_path / "second"
    run_example(second, "99")

    first_files = inventory(example_output)
    sample_figures = [
        "mirror_Vertical_velocity_114_Source_x_0.075_Source_y_0.075_Ch1.png",
        "mirror_Vertical_velocity_117_Source_x_-0.975_Source_y_-0.075_Ch1.png",
    ]
    assert sorted(first_files) == [
        *(f"filtered/{name}" for name in sample_figures),
        *(
            f"maps/slab_model_map_{attribute}.png"
            for attribute in (
                "MSE",
                "MSE_scale1",
                "MSE_scale2",
                "MSE_scale3",
                "norm_signal_energy",
                "norm_spectrum_area_over_centroid",
                "normalized_spectrum_area",
                "peak_frequency",
                "sampen",
                "spectral_centroid",
                "spectral_entropy",
                "spectral_flatness",
                "spectral_peak_width",
                "void_index",
            )
        ),
        "maps_variants/slab_model_absolute_map_void_index.png",
        "maps_variants/slab_model_scatter_scatter_void_index.png",
        "maps_variants/slab_model_spread_map_void_index.png",
        "sample_sites.txt",
        "slab_model_attributes.dat",
        "slab_model_attributes_avg.dat",
        "slab_model_attributes_avg.xlsx",
        "slab_model_samples_signals.dat",
        "slab_model_samples_spectra.dat",
        *(f"traces/{name}" for name in sample_figures),
    ]
    assert inventory(second) == first_files


def test_site_table_shows_the_void_under_the_slab_center(example_output: Path) -> None:
    """Observed medians, void / sound: 0.58, 0.59, 1.8 and 1.9; the bounds allow
    a different spectral discretization, not a lost signature. The three
    mirrored quadrants are copies of the computed one."""
    table = pd.read_csv(example_output / "slab_model_attributes_avg.dat")
    assert len(table) == 462
    assert set(table["peak_frequency_n_used"]) == {3, 4}
    assert not table.filter(like="_median").isna().any().any()

    footprint = (table["x"].abs() <= 0.33) & (table["y"].abs() <= 0.29)
    edge_band = (table["x"].abs() >= 1.2) | (table["y"].abs() >= 0.75)
    sound = ~footprint & ~edge_band
    assert footprint.sum() == 32 and sound.sum() == 262

    for attribute, bound, lower_is_void in (
        ("peak_frequency", 0.75, True),
        ("normalized_spectrum_area", 0.75, True),
        ("void_index", 1.3, False),
        ("sampen", 1.5, False),
    ):
        values = table[f"{attribute}_median"]
        ratio = values[footprint].median() / values[sound].median()
        assert (ratio <= bound) if lower_is_void else (ratio >= bound), (attribute, ratio)
    peak = table["peak_frequency_median"]
    assert peak[footprint].max() < peak[sound].median()

    mirrored = table.assign(x=-table["x"] + 0.0, y=-table["y"] + 0.0)
    paired = table.merge(mirrored, on=["x", "y"], suffixes=("", "_mirror"))
    assert len(paired) == 462
    for column in table.filter(like="_median").columns:
        assert paired[column].equals(paired[f"{column}_mirror"]), column


def test_default_offset_step_leaves_the_slab_model_traces_unchanged() -> None:
    """The simulated records begin during the impact, so no pre-impact level exists.

    With fewer than 10 samples before the 5 % crossing the offset step
    subtracts nothing.
    """
    data_dir = EXAMPLE.with_name("slab_model_data")
    names, x, y = read_coordinates_from_catalog(str(data_dir / "catalog.csv"))
    traces = load_traces(
        file_type="txt",
        directory=str(data_dir),
        channel=1,
        filenames_with_coordinates=names,
        x_position=x,
        y_position=y,
    )

    with_step = preprocess_traces(traces, polarity=-1, pad_zeros_factor=2)
    without_step = preprocess_traces(traces, polarity=-1, pad_zeros_factor=2, remove_offset=False)

    assert len(with_step) == len(without_step) == 1788
    for first, second in zip(with_step, without_step, strict=True):
        np.testing.assert_array_equal(first.signal, second.signal)
        np.testing.assert_array_equal(first.spectrum, second.spectrum)

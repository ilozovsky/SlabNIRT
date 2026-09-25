"""The text and SEG-Y readers of load_traces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import segyio

from slabnirt.data_import import load_traces


def _write_valid_txt(path: Path, *, columns: int = 2) -> None:
    time = np.arange(8, dtype=float) / 2_000.0
    data = [time]
    for channel in range(1, columns):
        data.append(np.sin(2.0 * np.pi * (100.0 + channel) * time))
    np.savetxt(path, np.column_stack(data))


def _write_txt_with_time(path: Path, time: np.ndarray) -> None:
    signal = np.arange(time.size, dtype=float)
    np.savetxt(path, np.column_stack((time, signal)))


def _write_valid_segy(path: Path) -> None:
    spec = segyio.spec()
    spec.tracecount = 2
    spec.format = 5
    spec.samples = np.arange(5, dtype=float) * 0.5
    with segyio.create(str(path), spec) as segy_file:
        segy_file.trace[0] = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        segy_file.trace[1] = np.array([-1, -2, -3, -4, -5], dtype=np.float32)
        segy_file.header[0] = {
            segyio.TraceField.TRACE_SAMPLE_INTERVAL: 500,
        }
        segy_file.header[1] = {
            segyio.TraceField.TRACE_SAMPLE_INTERVAL: 500,
        }


@pytest.mark.parametrize(
    ("time", "message"),
    [
        (np.array([0.0, 1.0, 2.1, 3.1]), "uniformly sampled"),
        (np.array([0.0, 1.0, 1.0, 2.0]), "strictly increasing"),
        (np.array([0.0, 1.0, np.nan, 3.0]), "finite"),
    ],
    ids=["nonuniform", "nonincreasing", "nonfinite"],
)
def test_load_traces_rejects_invalid_txt_time_axis(
    tmp_path: Path,
    time: np.ndarray,
    message: str,
) -> None:
    _write_txt_with_time(tmp_path / "invalid-time.txt", time)

    with pytest.raises(ValueError, match=message):
        load_traces(directory=str(tmp_path), file_type="txt", channel=1)


def test_load_traces_accepts_small_txt_time_rounding_within_tolerance(
    tmp_path: Path,
) -> None:
    _write_txt_with_time(
        tmp_path / "rounded-time.txt",
        np.array([0.0, 1.0, 2.000005, 3.000005]),
    )

    traces = load_traces(directory=str(tmp_path), file_type="txt", channel=1)

    np.testing.assert_allclose(traces[0].t, [0.0, 1.0, 2.0, 3.0])


def test_load_traces_warns_once_and_returns_partial_file_success(
    tmp_path: Path,
) -> None:
    _write_valid_txt(tmp_path / "valid.txt")
    (tmp_path / "corrupt.txt").write_text("not enough rows\n", encoding="utf-8")

    with pytest.warns(RuntimeWarning) as warning_records:
        traces = load_traces(
            directory=str(tmp_path),
            file_type="txt",
            channel=1,
            filenames_with_coordinates=["valid"],
            x_position=[1.0],
            y_position=[2.0],
        )

    assert len(warning_records) == 1
    message = str(warning_records[0].message)
    assert "1 file/channel request(s) failed" in message
    assert "corrupt.txt" in message
    assert "valid.txt" not in message
    assert [(trace.basename, trace.channel) for trace in traces] == [("valid", 1)]
    np.testing.assert_allclose(traces[0].x, 1.0)
    np.testing.assert_allclose(traces[0].y, 2.0)


def test_load_traces_raises_when_every_matched_file_fails(tmp_path: Path) -> None:
    (tmp_path / "corrupt1.txt").write_text("one row\n", encoding="utf-8")
    (tmp_path / "corrupt2.txt").write_text("also one row\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"no trace could be loaded from 2 \.txt file\(s\)"
    ) as error:
        load_traces(directory=str(tmp_path), file_type="txt", channel=1)

    message = str(error.value)
    assert "corrupt1.txt" in message
    assert "corrupt2.txt" in message


def test_load_traces_preserves_exact_valid_segy_arrays(tmp_path: Path) -> None:
    _write_valid_segy(tmp_path / "valid.sgy")

    traces = load_traces(
        directory=str(tmp_path),
        file_type="sgy",
        channel=[1, 2],
        time_scaling=1.0,
    )

    assert [(trace.basename, trace.channel) for trace in traces] == [
        ("valid", 1),
        ("valid", 2),
    ]
    for trace in traces:
        np.testing.assert_allclose(
            trace.t,
            np.arange(5, dtype=float) * 0.0005,
            rtol=0.0,
            atol=0.0,
        )
    np.testing.assert_array_equal(
        traces[0].signal,
        np.array([1, 2, 3, 4, 5], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        traces[1].signal,
        np.array([-1, -2, -3, -4, -5], dtype=np.float32),
    )


def test_load_traces_keeps_corrupt_segy_as_a_partial_load_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / "corrupt.sgy").write_bytes(b"")
    _write_valid_segy(tmp_path / "valid.sgy")

    with pytest.warns(RuntimeWarning) as warning_records:
        traces = load_traces(
            directory=str(tmp_path),
            file_type="sgy",
            channel=1,
        )

    assert [(trace.basename, trace.channel) for trace in traces] == [
        ("valid", 1),
    ]
    assert len(warning_records) == 1
    message = str(warning_records[0].message)
    assert "1 file/channel request(s) failed" in message
    assert "corrupt.sgy" in message
    assert "OSError" in message
    assert "valid.sgy" not in message


def test_txt_channels_are_the_signal_columns_after_time(tmp_path: Path) -> None:
    rows = (f"{i / 2_000.0} {i} {-2 * i}" for i in range(6))
    (tmp_path / "two-channels.txt").write_text("\n".join(rows), encoding="utf-8")

    traces = load_traces(directory=str(tmp_path), channel=[2, 1])

    assert [trace.channel for trace in traces] == [2, 1]
    np.testing.assert_array_equal(traces[0].signal, -2.0 * np.arange(6))
    np.testing.assert_array_equal(traces[1].signal, np.arange(6))


def test_an_incomplete_coordinate_catalog_is_rejected(tmp_path: Path) -> None:
    _write_valid_txt(tmp_path / "a.txt")

    with pytest.raises(ValueError, match="must be given together"):
        load_traces(directory=str(tmp_path), filenames_with_coordinates=["a"], x_position=[1.0])


def test_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no_such_folder"):
        load_traces(directory=str(tmp_path / "no_such_folder"))


def test_time_scaling_converts_a_millisecond_time_column_to_seconds(tmp_path: Path) -> None:
    rows = (f"{i * 0.5} {float(i)}" for i in range(5))
    (tmp_path / "probe.txt").write_text("\n".join(rows), encoding="utf-8")

    [trace] = load_traces(directory=str(tmp_path), time_scaling=0.001)

    np.testing.assert_allclose(trace.t, np.arange(5) / 2_000.0)

"""Offset removal and the two detrend methods of preprocess_traces.

The records are built from declared parameters and carry no survey data.
The field-like record imitates what a geophone delivers after a hammer
impact: a level offset, a damped 800 Hz pulse, and the instrument's own
slow ringing, a damped 13 Hz cosine that starts at its negative extreme at
the impact. The offset (1/16) is exactly representable, so the offset step
can be checked bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from slabnirt.core import TraceData
from slabnirt.preprocess import preprocess_traces

SAMPLING_RATE = 20_000.0
ONSET = 50  # quiet samples before the impact (2.5 ms)
OFFSET = 0.0625
RINGING_AMPLITUDE = 0.08


def _trace(name: str, time: np.ndarray, signal: np.ndarray) -> TraceData:
    return TraceData(
        basename=name,
        channel=1,
        t=np.asarray(time, dtype=float),
        signal=np.asarray(signal, dtype=float),
        filepath=f"{name}.txt",
    )


def field_like_record(
    n: int = 4000,
    onset: int = ONSET,
    offset: float = OFFSET,
    ringing_amplitude: float = RINGING_AMPLITUDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return t, signal and its pulse and ringing parts (200 ms by default)."""
    t = np.arange(n, dtype=float) / SAMPLING_RATE
    after = np.clip(t - onset / SAMPLING_RATE, 0.0, None)
    active = (t >= onset / SAMPLING_RATE).astype(float)
    pulse = active * np.exp(-after / 0.0015) * np.sin(2.0 * np.pi * 800.0 * after)
    ringing = (
        active
        * ringing_amplitude
        * np.exp(-after / 0.25)
        * np.cos(2.0 * np.pi * 13.0 * after + np.pi)
    )
    return t, offset + pulse + ringing, pulse, ringing


def mirror_parabola(signal: np.ndarray) -> np.ndarray:
    """The polynomial detrend, written out by hand."""
    left = signal[1:][::-1]
    padded = np.concatenate([left, signal])
    grid = np.linspace(0.0, 100.0, padded.size)
    quadratic = np.polyval(np.polyfit(grid, padded, 2), grid)
    return (padded - quadratic)[left.size : left.size + signal.size]


def test_offset_step_removes_the_pre_impact_level_exactly() -> None:
    t, signal, _, _ = field_like_record()

    [processed] = preprocess_traces([_trace("quiet-start", t, signal)])

    np.testing.assert_array_equal(processed.signal, signal - OFFSET)
    assert np.all(processed.signal[:ONSET] == 0.0)


def test_offset_step_leaves_a_record_that_starts_during_the_impact() -> None:
    t, signal, _, _ = field_like_record(onset=0, ringing_amplitude=0.0)

    [processed] = preprocess_traces([_trace("during-impact", t, signal)])

    np.testing.assert_array_equal(processed.signal, signal)


@pytest.mark.parametrize(
    ("options", "keep"),
    [({"downsample": 8}, slice(None, None, 8)), ({"time_limit": 0.030}, slice(0, 600))],
    ids=["downsample", "time_limit"],
)
def test_offset_step_is_measured_before_downsampling_and_time_trimming(
    options: dict, keep: slice
) -> None:
    """Eight-fold decimation leaves 7 quiet samples, fewer than the 10 the rule
    needs; the level is measured before decimation, so it is still removed."""
    t, signal, _, _ = field_like_record()

    [processed] = preprocess_traces([_trace("quiet-start", t, signal)], **options)

    np.testing.assert_array_equal(processed.signal, (signal - OFFSET)[keep])


def test_highpass_removes_the_ringing_and_keeps_the_pulse_and_the_start() -> None:
    """The start stays at zero, because the baseline is held at the pre-impact
    level. The pulse is kept: its peak and first trough move by less than
    0.5 % relative to the offset-corrected record (the ringing hidden under
    the impact window is left in place by design). The ringing is gone from
    40 ms after the impact to 20 ms before the end: below 5 % of its
    amplitude at the impact, where without the detrend more than half of it
    is still there. Nearer the impact the part of the ringing bridged with
    the impact window remains, and the last 20 ms carry the edge of the
    zero-phase filter."""
    t, signal, pulse, _ = field_like_record()
    [offset_only] = preprocess_traces([_trace("ringing", t, signal)])

    [processed] = preprocess_traces([_trace("ringing", t, signal)], detrend_method="highpass")

    assert np.all(processed.signal[:ONSET] == 0.0)
    peak = int(np.argmax(offset_only.signal))
    trough = int(np.argmin(offset_only.signal))
    for index in (peak, trough):
        assert abs(processed.signal[index] / offset_only.signal[index] - 1.0) < 0.005
    settled = (t >= t[ONSET] + 0.040) & (t <= t[-1] - 0.020)
    assert np.max(np.abs(offset_only.signal[settled])) > 0.5 * RINGING_AMPLITUDE
    assert np.max(np.abs(processed.signal[settled] - pulse[settled])) < 0.05 * RINGING_AMPLITUDE


def test_highpass_after_a_50_ms_cut_agrees_with_the_full_record() -> None:
    """In the pulse region (20 ms after the impact) the two agree within 0.1 %
    of the peak; nearer the cut the shorter record carries its own filter
    edge."""
    t, signal, _, _ = field_like_record()
    full_trace = _trace("full", t, signal)
    cut_trace = _trace("cut", t, signal)

    [full] = preprocess_traces([full_trace], detrend_method="highpass")
    [cut] = preprocess_traces([cut_trace], time_limit=0.050, detrend_method="highpass")

    peak = np.max(np.abs(full.signal))
    window = cut.t < t[ONSET] + 0.020
    assert cut.t.size == 1000
    assert np.all(cut.signal[:ONSET] == 0.0)
    assert np.max(np.abs(cut.signal[window] - full.signal[: cut.t.size][window])) < 0.001 * peak


def test_highpass_warns_and_keeps_only_the_offset_step_on_a_short_record() -> None:
    t, signal, _, _ = field_like_record(n=520)
    [offset_only] = preprocess_traces([_trace("short", t, signal)])

    with pytest.warns(RuntimeWarning, match=r"short\.txt.*at least 50 ms.*26\.0 ms"):
        [processed] = preprocess_traces([_trace("short", t, signal)], detrend_method="highpass")

    np.testing.assert_array_equal(processed.signal, offset_only.signal)


def test_polynomial_detrend_matches_a_hand_written_mirror_parabola() -> None:
    """A record that starts during the impact has no offset to subtract, so the
    parabola sees the raw record. With a quiet start it sees the
    offset-corrected record: a detrend method applies the offset step even
    when remove_offset is False."""
    t, signal, _, _ = field_like_record(onset=0, ringing_amplitude=0.0)
    [processed] = preprocess_traces(
        [_trace("during-impact", t, signal)], detrend_method="polynomial"
    )
    np.testing.assert_array_equal(processed.signal, mirror_parabola(signal))

    t, signal, _, _ = field_like_record()
    [processed] = preprocess_traces(
        [_trace("quiet-start", t, signal)], remove_offset=False, detrend_method="polynomial"
    )
    np.testing.assert_array_equal(processed.signal, mirror_parabola(signal - OFFSET))

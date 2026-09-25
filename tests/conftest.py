"""Shared synthetic-trace fixtures for the attribute and plotting tests.

Every trace here is built from declared sampling parameters and carries no
survey data. The spectrum-carrying variant lets a test hand the attribute
calculation an exact spectrum, so the oracle is the hand-written spectrum
itself rather than an FFT of a signal.

Matplotlib is switched to the non-interactive Agg backend before the package
is imported, so the plotting tests never try to open a window.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from slabnirt import TraceData


@pytest.fixture
def synthetic_trace() -> Callable[..., TraceData]:
    """Return a factory for a TraceData built from a signal and a sampling rate."""

    def make(
        signal,
        *,
        sampling_rate: float = 2_000.0,
        name: str = "trace",
    ) -> TraceData:
        values = np.asarray(signal, dtype=float)
        return TraceData(
            basename=name,
            channel=1,
            filepath="synthetic",
            t=np.arange(values.size, dtype=float) / sampling_rate,
            signal=values,
            x=0.0,
            y=0.0,
        )

    return make


@pytest.fixture
def spectrum_trace(synthetic_trace) -> Callable[..., TraceData]:
    """Return a factory for a trace that already carries a given spectrum.

    The signal is a short placeholder; ``signal_norm`` is set so the attribute
    calculation does not renormalize it, and ``freqs`` defaults to one bin per
    unit frequency so hand calculations stay simple.
    """

    def make(spectrum, *, freqs=None, name: str = "spectrum") -> TraceData:
        values = np.asarray(spectrum, dtype=float)
        trace = synthetic_trace([0.0, 1.0, -0.5, 0.25], name=name)
        trace.signal_norm = trace.signal.copy()
        trace.freqs = (
            np.arange(values.size, dtype=float) if freqs is None else np.asarray(freqs, dtype=float)
        )
        trace.spectrum = values
        return trace

    return make


@pytest.fixture
def site_trace(synthetic_trace) -> Callable[..., TraceData]:
    """Return a factory for a trace with attributes at a given site and channel."""

    def make(
        name: str,
        attributes: dict[str, Any] | None = None,
        *,
        x: float = 0.0,
        y: float = 0.0,
        channel: int = 1,
    ) -> TraceData:
        trace = synthetic_trace([0.0, 1.0, -0.5, 0.25], name=name)
        trace.x = x
        trace.y = y
        trace.channel = channel
        trace.attributes = attributes
        return trace

    return make

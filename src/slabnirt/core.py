"""Core data structures shared by the SlabNIRT processing pipeline."""

from dataclasses import dataclass
from typing import Any, TypeAlias, TypedDict

import numpy as np

# TypeAlias rather than a `type` statement: typing.get_type_hints expands these.
AggregateStatValue: TypeAlias = int | float | None  # noqa: UP040
AggregateStats: TypeAlias = dict[str, AggregateStatValue]  # noqa: UP040


class MSEScaleStats(TypedDict):
    """Per-scale aggregate statistics for one averaged MSE variant."""

    Scale: list[int | float]
    Value: list[AggregateStats]


AveragedAttributeValue: TypeAlias = (  # noqa: UP040
    AggregateStats | tuple[AggregateStats, MSEScaleStats]
)
AttributesAverage: TypeAlias = dict[str, AveragedAttributeValue]  # noqa: UP040


@dataclass(eq=False)
class TraceData:
    """One channel of one source file and the results computed from it.

    Each pipeline step sets its fields; preprocess_traces sets them on the
    copies it returns:

    load_traces
        basename (file name without extension), channel, t (time in s),
        signal (raw samples in acquisition units), filepath, and the
        optional survey coordinates x and y.
    preprocess_traces
        signal_norm (signal scaled to a maximum absolute value of 1), freqs
        and spectrum (frequency bins in Hz and amplitude spectrum), and for
        a filtered trace freqs_filt, spec_filt, filter_freq_range (passband
        in Hz, upper edge None for a high-pass) and signal_norm_pre (the
        normalized signal before filtering).
    calculate_signal_parameters
        attributes, a dictionary of attribute values.
    average_attributes
        attributes_avg, the per-site statistics: one dictionary per scalar
        attribute, a (scalar statistics, per-scale statistics) pair per MSE
        variant.

    Construction validates nothing; each function checks what it needs.
    Arrays and dictionaries are stored by reference, so copy.copy and
    dataclasses.replace give copies that share them; use copy.deepcopy for
    independent data. The traces of one site share one attributes_avg
    mapping; treat it as read-only. Equality and hashing are by identity.
    """

    basename: str
    channel: int
    t: np.ndarray
    signal: np.ndarray
    filepath: str
    x: float | None = None
    y: float | None = None

    signal_norm: np.ndarray | None = None
    freqs: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    freqs_filt: np.ndarray | None = None
    spec_filt: np.ndarray | None = None
    filter_freq_range: tuple[float, float | None] | None = None
    signal_norm_pre: np.ndarray | None = None

    attributes: dict[str, Any] | None = None
    attributes_avg: AttributesAverage | None = None

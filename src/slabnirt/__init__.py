"""Normalized Impulse Response Testing analysis tools."""

__version__ = "0.7.0"

from ._logging import configure_logging
from .analysis import average_attributes
from .attributes import calculate_signal_parameters
from .core import TraceData
from .data_export import export_tracedata
from .data_import import load_traces, read_coordinates_from_catalog
from .plot_config import MapPlotConfig, TracePlotConfig
from .plot_maps import plot_maps
from .plot_traces import plot_traces
from .preprocess import ButterworthFilterConfig, preprocess_traces

__all__ = [
    "TraceData",
    "load_traces",
    "read_coordinates_from_catalog",
    "preprocess_traces",
    "ButterworthFilterConfig",
    "calculate_signal_parameters",
    "average_attributes",
    "export_tracedata",
    "TracePlotConfig",
    "plot_traces",
    "MapPlotConfig",
    "plot_maps",
    "configure_logging",
]

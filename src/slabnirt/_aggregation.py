"""Shared keys for aggregated attribute statistics."""

from typing import Literal

AggregateStatistic = Literal["mean", "median"]

AGGREGATE_EXPORT_KEYS = (
    "mean",
    "mean_error",
    "median",
    "median_error",
    "n_used",
    "n_dropped",
)


def aggregate_error_key(statistic: AggregateStatistic) -> str:
    """Return the canonical uncertainty key for an aggregate statistic."""
    return f"{statistic}_error"

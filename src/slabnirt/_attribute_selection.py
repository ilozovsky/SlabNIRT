"""Shared validation for public attribute-name selectors."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal, TypeAlias

# TypeAlias rather than a `type` statement: typing.get_type_hints expands these.
AttributeNames: TypeAlias = list[str] | tuple[str, ...]  # noqa: UP040
AttributeSelection: TypeAlias = Literal["all"] | AttributeNames  # noqa: UP040
OptionalAttributeSelection: TypeAlias = AttributeSelection | None  # noqa: UP040
NormalizedAttributeSelection: TypeAlias = Literal["all"] | list[str]  # noqa: UP040


def normalize_attribute_selection(
    selection: object,
    *,
    parameter_name: str,
    allow_none: bool = False,
) -> NormalizedAttributeSelection | None:
    """Validate selector shape and return a stable list for explicit names."""
    if selection is None:
        if allow_none:
            return None
        raise TypeError(f"{parameter_name} may not be None.")

    if isinstance(selection, str):
        if selection == "all":
            return "all"
        raise ValueError(
            f"{parameter_name} must be exactly 'all' or a list or tuple of "
            f"attribute names. Use {parameter_name}=[{selection!r}] for one "
            "attribute."
        )

    if not isinstance(selection, (list, tuple)):
        raise TypeError(
            f"{parameter_name} must be exactly 'all' or a list or tuple of attribute names."
        )

    invalid_positions = [index for index, name in enumerate(selection) if not isinstance(name, str)]
    if invalid_positions:
        raise TypeError(
            f"{parameter_name} entries must be strings; invalid position(s): {invalid_positions}."
        )

    return list(selection)


def validate_known_attribute_names(
    names: list[str],
    available_names: Iterable[str],
    *,
    parameter_name: str,
    is_additional_known: Callable[[str], bool] | None = None,
) -> None:
    """Reject unknown explicit names while preserving caller-provided order."""
    available = set(available_names)
    unknown = []
    for name in names:
        is_known = name in available or (
            is_additional_known is not None and is_additional_known(name)
        )
        if not is_known and name not in unknown:
            unknown.append(name)

    if unknown:
        available_display = sorted(available, key=str)
        raise ValueError(
            f"{parameter_name} contains unknown attribute name(s): {unknown}. "
            f"Available attributes: {available_display}."
        )

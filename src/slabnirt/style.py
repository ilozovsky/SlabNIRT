"""Default plot styles, one complete dictionary per plot kind."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_MAP_STYLE_DEFAULTS: dict[str, Any] = {
    "axis_label_font_size": 12,
    "tick_font_size": 12,
    "grid": True,
    "grid_color": "#dddddd",
    "grid_style": "--",
    "grid_alpha": 0.8,
    "save_format": "png",
    "suptitle_font_size": 14,
    "label_font_size": 9,
    "marker_edge_width": 0.8,
    "marker_shape": "s",
    "polygon_style": {
        "ls": "--",
        "lw": 0.8,
        "alpha": 1.0,
        "color": "#C81D25",
    },
    "custom_cmap_colors": [
        (105 / 255, 34 / 255, 141 / 255),
        (13 / 255, 20 / 255, 101 / 255),
        (43 / 255, 72 / 255, 145 / 255),
        (105 / 255, 175 / 255, 207 / 255),
        (153 / 255, 255 / 255, 255 / 255),
        (188 / 255, 221 / 255, 122 / 255),
        (254 / 255, 255 / 255, 110 / 255),
        (255 / 255, 183 / 255, 48 / 255),
        (255 / 255, 125 / 255, 86 / 255),
        (171 / 255, 18 / 255, 18 / 255),
        (85 / 255, 23 / 255, 6 / 255),
    ],
    "grid_lw": 0.5,
    "marker_size": 200,
    "marker_edge_color": "black",
    "error_marker_shape": "s",
    "avg_marker_shape": "o",
    "overlay_marker_size_factor": 0.5,
    "overlay_edge_width": 0.8,
    "plot_site_only_color": "k",
    "marker_zorder": 5,
    "attribute_labels": {
        "norm_signal_energy": "Normalized Signal Energy",
        "normalized_spectrum_area": "Normalized Spectrum Area",
        "spectral_centroid": "Spectral Centroid",
        "norm_spectrum_area_over_centroid": "Norm. Spectrum Area/Centroid",
        "peak_frequency": "Peak Frequency",
        "spectral_peak_width": "Spectral Peak Width",
        "spectral_flatness": "Spectral Flatness",
        "void_index": "Void Index",
        "sampen": "Sample Entropy",
        "MSE": "\N{GREEK CAPITAL LETTER SIGMA}MSE",
        "spectral_entropy": "Spectral Entropy",
    },
    "contour_kwargs": {
        "colors": "k",
        "linewidths": 0.3,
        "alpha": 0.3,
        "linestyles": "-",
        "zorder": 1,
    },
}

_TRACE_STYLE_DEFAULTS: dict[str, Any] = {
    "axis_label_font_size": 12,
    "tick_font_size": 12,
    "grid": True,
    "grid_color": "#dddddd",
    "grid_style": "--",
    "grid_alpha": 0.8,
    "save_format": "png",
    "legend_font_size": 12,
    "line_width": 1,
    "time_color": "k",
    "time_fill_color": "#27292B",
    "time_fill_alpha": 0.9,
    "spectrum_color": "#084698",
    "spectrum_fill_alpha": 0.5,
    "pre_filtered_color": "#8B0000",
    "vertical_line_style": {
        "ls": "--",
        "lw": 0.8,
        "alpha": 1.0,
        "color": "#32CD32",
    },
    "show_panel_labels": False,
    "panel_label_style": {
        "x": 0.03,
        "y": 0.25,
        "boxstyle": "circle,pad=0.25",
        "facecolor": "#FFF9E2",
        "edgecolor": "black",
        "linewidth": 1,
        "fontsize": 13,
        "text_color": "black",
        "ha": "center",
        "va": "top",
    },
    "panel_hspace": 0.4,
    "subplots_adjust": {
        "top": 0.94,
        "bottom": 0.08,
        "left": 0.1,
        "right": 0.95,
    },
    "show_filename": True,
    "suptitle_font_size": 14,
    "table_font_size": 10,
    "axis_bg_color": "white",
    "spec_marker_size": 1,
    "show_peak_lines": True,
    "peak_line_style": {
        "ls": "--",
        "lw": 2,
        "alpha": 1.0,
        "color": "#FFA500",
    },
    "show_centroid_lines": True,
    "centroid_line_style": {
        "ls": "--",
        "lw": 2,
        "alpha": 1.0,
        "color": "#FF1493",
    },
    "mse_color": "maroon",
    "mse_marker_size": 3,
    "mse_ylim": None,
    "attr_numfmt": ".3g",
}

_STYLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "maps": _MAP_STYLE_DEFAULTS,
    "traces": _TRACE_STYLE_DEFAULTS,
}


def style_defaults(
    plot_kind: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deep copy of one plot kind's defaults with the caller's overrides applied.

    Every key of ``overrides`` must be a known key of that plot kind. A nested
    dictionary such as ``panel_label_style`` replaces the default one whole,
    so give all of its keys. Neither the defaults nor ``overrides`` share any
    object with the result.
    """
    try:
        resolved = deepcopy(_STYLE_DEFAULTS[plot_kind])
    except KeyError as exc:
        choices = ", ".join(sorted(_STYLE_DEFAULTS))
        raise ValueError(f"plot_kind must be one of: {choices}; got {plot_kind!r}.") from exc

    if overrides:
        unknown = [key for key in overrides if key not in resolved]
        if unknown:
            raise ValueError(
                f"style: unknown key(s) for {plot_kind!r}: {unknown}. "
                f"Valid keys: {sorted(resolved)}."
            )
        resolved.update(deepcopy(dict(overrides)))
    return resolved

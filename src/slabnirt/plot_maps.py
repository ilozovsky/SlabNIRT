"""Maps of site attributes: colored site markers and interpolated surfaces."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import verde as vd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._aggregation import aggregate_error_key
from ._attribute_selection import (
    normalize_attribute_selection,
    validate_known_attribute_names,
)
from ._console import safe_warn
from ._logging import log_warning
from ._mse import mse_attribute_keys, resolve_mse_feature, selected_mse_keys
from ._site import site_key
from .core import TraceData
from .plot_config import MapPlotConfig
from .style import style_defaults

_CONCRETE_GRAY = "#9e9e9e"
_PLOT_TYPES = ("map", "scatter")
_ERROR_MODES = (None, "error", "minmax")
_COLOR_NORMS = ("absolute", "mad", "mad-scaled", "mean-std", "median-std")
# Inches; a shorter vertical color bar cannot hold eight two-line labels.
_MIN_COLORBAR_HEIGHT_IN = 2.5
# Attributes mapped with the reversed colormap, so that the values typical of
# a void take the same colors on every map; a name containing one of these (a
# suffixed variant, a scale name) is reversed too.
_REVERSED_COLORMAP_ATTRIBUTES = (
    "normalized_spectrum_area",
    "spectral_centroid",
    "norm_spectrum_area_over_centroid",
    "peak_frequency",
    "spectral_flatness",
    "spectral_entropy",
)


def plot_maps(
    traces: list[TraceData],
    config: MapPlotConfig | None = None,
) -> None:
    """Draw one figure per attribute and plot type from the site coordinates of ``traces``.

    Sites are the six-decimal coordinate buckets of
    :func:`slabnirt.average_attributes`; the first trace at each site
    represents it. With ``averaged_attribute_values=True`` (default) the site
    value is the median or mean stored in ``attributes_avg`` and its
    uncertainty the matching ``median_error``/``mean_error``; otherwise the
    raw value in ``attributes``, which must then be unique per site.
    ``attrs_to_plot='all'`` takes every attribute stored at any site, in
    first-seen order. MSE variants are addressed by their stored key, their
    scales as ``<key>_scaleN`` through ``MSE_scales``.

    A 'scatter' figure shows the sites as markers colored by value, with a
    color bar; a 'map' figure shows them on a biharmonic-spline surface
    (:class:`verde.Spline`) with contour lines at the color-bar ticks. The
    spline has no trend term, so outside the site hull the surface can leave
    the data range quickly (see ``buffer`` and ``mask_distance``). The color
    bar is at the right of the map, or below a map shorter than 2.5 inches.

    ``color_norm='absolute'`` spans the site minimum to maximum with up to
    eight ticks. The robust modes center the scale on the median (or mean),
    put ticks at whole multiples of the MAD (or std) inside the data range,
    labeled ``med``/``mean`` and ``+kMAD (value)``, and saturate beyond
    ``n_scale_levels`` units from the center. A robust scale below 1e-8
    gives minimum-to-maximum colors without ticks or contours, with a
    warning. The attributes normalized_spectrum_area, spectral_centroid,
    norm_spectrum_area_over_centroid, peak_frequency, spectral_flatness and
    spectral_entropy (and names containing them) use the reversed colormap.

    Figures are saved as ``<output_prefix>_<plot_type>_<attribute>.<save_format>``,
    overwriting existing files; every figure is closed, also when drawing or
    saving fails.

    Parameters
    ----------
    traces : list of TraceData
        Traces with finite ``x`` and ``y`` and a non-empty attribute source.
    config : MapPlotConfig, optional
        Plot options; None uses the defaults.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        For a config that is not a MapPlotConfig.
    ValueError
        Before the first figure: for an invalid option or attribute name, no
        traces, missing or non-finite coordinates, an empty attribute source,
        or several traces at one site with raw values. Before the figures of
        an attribute: for a value that is non-finite or missing at a site, or
        on a weighted map an uncertainty that is not finite and positive; the
        files of earlier attributes stay.

    Notes
    -----
    Overlays (``external_polygons``) are dicts with one of:
    ``{'x': [...], 'y': [...]}`` a polyline, filled when 'fill' is given;
    ``{'circle': {'center': (cx, cy), 'radius': r}}`` an outline, filled
    with 'fill'; ``{'ring': {'center': (cx, cy), 'r_inner': ri,
    'r_outer': ro}}`` a filled annulus with its two circles drawn unless
    'edge' is False; ``{'shp': 'path.shp'}`` the Polygon, MultiPolygon
    (every part) and LineString geometries of a shapefile read with
    geopandas (optional dependency, extra ``shapefiles``); a missing
    dependency, an unreadable file or a malformed geometry is written to the
    log as a warning and the overlay is skipped. Common keys: 'style' (line
    kwargs over the ``polygon_style`` defaults), 'fill' ({'color', 'alpha',
    'zorder'}, default a concrete gray) and 'label' (text at the first vertex
    of a polyline, Polygon or LineString, or above a circle or ring).

    Style keys (``config.style``, see ``slabnirt.style``):
    axis_label_font_size, tick_font_size, grid, grid_color, grid_style,
    grid_alpha, grid_lw, save_format, suptitle_font_size (the axes title),
    label_font_size (site and overlay labels), marker_size, marker_shape,
    marker_edge_width, marker_edge_color, marker_zorder,
    error_marker_shape, avg_marker_shape, overlay_marker_size_factor,
    overlay_edge_width, plot_site_only_color, polygon_style,
    custom_cmap_colors (the 'slabnirt' palette), attribute_labels (title
    per attribute key) and contour_kwargs.
    """
    if config is None:
        config = MapPlotConfig()
    elif not isinstance(config, MapPlotConfig):
        raise TypeError("config must be a MapPlotConfig instance or None.")

    attrs_to_plot = normalize_attribute_selection(
        config.attrs_to_plot,
        parameter_name="attrs_to_plot",
    )
    plot_error = config.plot_error
    output_name_prefix = "" if config.output_prefix is None else f"{config.output_prefix}_"

    if not (isinstance(config.panel_size, (list, tuple)) and len(config.panel_size) == 2):
        raise ValueError("panel_size must be (width, height).")

    st = style_defaults("maps", config.style)

    if config.color_norm not in _COLOR_NORMS:
        raise ValueError(
            f"color_norm must be one of {list(_COLOR_NORMS)}; got {config.color_norm!r}."
        )

    plot_types = [config.plot_type] if isinstance(config.plot_type, str) else list(config.plot_type)
    unknown_types = [name for name in plot_types if name not in _PLOT_TYPES]
    if unknown_types:
        raise ValueError(
            f"plot_type entries must be 'map', 'scatter' or both; got {unknown_types}."
        )
    if plot_error not in _ERROR_MODES:
        raise ValueError(f"plot_error must be None, 'error' or 'minmax'; got {plot_error!r}.")

    if plot_error is not None and config.plot_site_only:
        safe_warn(
            "plot_error is ignored when plot_site_only=True.",
            UserWarning,
            stacklevel=2,
        )
        plot_error = None
    elif plot_error is not None and not config.averaged_attribute_values:
        safe_warn(
            "plot_error is ignored when averaged_attribute_values=False because "
            "raw attributes have no aggregate uncertainties.",
            UserWarning,
            stacklevel=2,
        )
        plot_error = None

    if not traces:
        raise ValueError("plot_maps: no traces given.")

    invalid_coordinates = []
    for td in traces:
        try:
            coordinates_are_finite = np.isfinite(float(td.x)) and np.isfinite(float(td.y))
        except (TypeError, ValueError):
            coordinates_are_finite = False
        if not coordinates_are_finite:
            invalid_coordinates.append(str(td.basename))
    if invalid_coordinates:
        names = ", ".join(invalid_coordinates)
        raise ValueError(f"plot_maps: missing or non-finite x or y for {names}.")

    source_key = "attributes_avg" if config.averaged_attribute_values else "attributes"
    missing_sources = [str(td.basename) for td in traces if not getattr(td, source_key)]
    if missing_sources:
        names = ", ".join(missing_sources)
        raise ValueError(f"plot_maps: '{source_key}' is missing or empty for {names}.")

    traces, duplicate_coordinates = deduplicate_site_traces(traces)
    if duplicate_coordinates and not config.averaged_attribute_values:
        raise ValueError(
            "plot_maps: several traces share a site; "
            "set averaged_attribute_values=True to map them."
        )

    all_attrs = list(dict.fromkeys(key for td in traces for key in getattr(td, source_key)))
    base_attrs = all_attrs if attrs_to_plot == "all" else list(attrs_to_plot)

    available_mse = mse_attribute_keys(all_attrs)
    if attrs_to_plot != "all":
        validate_known_attribute_names(
            base_attrs,
            all_attrs,
            parameter_name="attrs_to_plot",
            is_additional_known=lambda name: resolve_mse_feature(name, available_mse) is not None,
        )
    selected_mse = selected_mse_keys(attrs_to_plot, available_mse)

    plot_list = list(base_attrs)

    if config.MSE_scales:
        if selected_mse:
            for mse_key in selected_mse:
                for scale in config.MSE_scales:
                    scale_name = f"{mse_key}_scale{scale}"
                    if scale_name not in plot_list:
                        plot_list.append(scale_name)
        else:
            safe_warn("MSE_scales is ignored: no MSE variant is selected.")

    xs = np.array([td.x for td in traces], float)
    ys = np.array([td.y for td in traces], float)
    site_names = np.array([str(td.basename) for td in traces], dtype=object)
    aggregate_stat = "median" if config.plot_median_averaged_attributes else "mean"

    base_cmap = (
        LinearSegmentedColormap.from_list("base", st["custom_cmap_colors"], 256)
        if config.colorscale == "slabnirt"
        else plt.get_cmap(config.colorscale)
    )
    base = st["marker_size"]
    overlay = base * st["overlay_marker_size_factor"]

    for attr in plot_list:
        vals, errs, mins, maxs = _site_values(
            traces,
            source_key,
            attr,
            resolve_mse_feature(attr, available_mse),
            config.averaged_attribute_values,
            aggregate_stat,
        )

        invalid_values = ~np.isfinite(vals)
        if np.any(invalid_values):
            names = ", ".join(site_names[invalid_values])
            raise ValueError(f"{attr}: non-finite or missing value at sites {names}.")

        if "map" in plot_types and config.use_weights:
            if errs is None:
                invalid_uncertainties = np.ones(vals.shape, dtype=bool)
            else:
                invalid_uncertainties = ~np.isfinite(errs) | (errs <= 0)
            if np.any(invalid_uncertainties):
                names = ", ".join(site_names[invalid_uncertainties])
                raise ValueError(
                    f"{attr}: the weighted map needs a finite, positive uncertainty at "
                    f"every site; missing or invalid at {names}. Set use_weights=False "
                    "to map without weights."
                )

        attr_cmap = base_cmap
        if any(name in attr for name in _REVERSED_COLORMAP_ATTRIBUTES):
            attr_cmap = attr_cmap.reversed()

        cmap, norm, ticks, labels = build_cmap(
            attr_cmap, vals, config.color_norm, config.n_scale_levels
        )

        for pt in plot_types:
            fig, ax = plt.subplots(figsize=config.panel_size)
            colorbar_source = None
            try:
                if pt == "map":
                    region = (
                        xs.min() - config.buffer,
                        xs.max() + config.buffer,
                        ys.min() - config.buffer,
                        ys.max() + config.buffer,
                    )

                    if config.use_weights:
                        weights = 1 / (errs**2)
                        spline = vd.Spline(damping=config.damping).fit(
                            (xs, ys), vals, weights=weights
                        )
                    else:
                        spline = vd.Spline(damping=config.damping).fit((xs, ys), vals)

                    grid = spline.grid(region=region, spacing=config.spacing)

                    if config.mask_distance is not None and config.mask_distance > 0:
                        grid = vd.distance_mask(
                            data_coordinates=(xs, ys),
                            maxdist=config.mask_distance,
                            grid=grid,
                        )

                    xi = grid.coords["easting"].values
                    yi = grid.coords["northing"].values
                    z = grid["scalars"].values

                    mesh = ax.pcolormesh(xi, yi, z, cmap=cmap, norm=norm)
                    if config.plot_site_only:
                        colorbar_source = mesh

                    if ticks and config.show_contours:
                        ax.contour(xi, yi, z, levels=ticks, **st["contour_kwargs"])

                if plot_error in ("error", "minmax") and errs is not None:
                    for x0, y0, v, e, mi, ma in zip(xs, ys, vals, errs, mins, maxs, strict=True):
                        lo, hi = (v - e, v + e) if plot_error == "error" else (mi, ma)
                        for half, val in [("left", lo), ("right", hi)]:
                            ax.plot(
                                x0,
                                y0,
                                marker=st["error_marker_shape"],
                                linestyle="None",
                                markersize=np.sqrt(base),
                                markerfacecolor=cmap(norm(val)),
                                markeredgecolor=st["marker_edge_color"],
                                markeredgewidth=st["marker_edge_width"],
                                fillstyle=half,
                                zorder=st["marker_zorder"],
                            )
                        ax.plot(
                            x0,
                            y0,
                            marker=st["avg_marker_shape"],
                            linestyle="None",
                            markersize=np.sqrt(overlay),
                            markerfacecolor=cmap(norm(v)),
                            markeredgecolor=st["marker_edge_color"],
                            markeredgewidth=st["overlay_edge_width"],
                            zorder=st["marker_zorder"],
                        )

                    if not config.plot_site_only:
                        sm = ScalarMappable(norm=norm, cmap=cmap)
                        sm.set_array(vals)
                        colorbar_source = sm

                else:
                    sc = ax.scatter(
                        xs,
                        ys,
                        c=st["plot_site_only_color"] if config.plot_site_only else vals,
                        s=base,
                        cmap=None if config.plot_site_only else cmap,
                        norm=None if config.plot_site_only else norm,
                        marker=st["marker_shape"],
                        edgecolor=st["marker_edge_color"],
                        linewidths=st["marker_edge_width"],
                        zorder=st["marker_zorder"],
                    )

                    if not config.plot_site_only:
                        colorbar_source = sc

                ax.set_title(
                    st["attribute_labels"].get(attr, attr), fontsize=st["suptitle_font_size"]
                )

                ax.set_xlabel(
                    f"Distance ({config.coordinate_units})", fontsize=st["axis_label_font_size"]
                )
                ax.set_ylabel(
                    f"Distance ({config.coordinate_units})", fontsize=st["axis_label_font_size"]
                )

                if config.xlim:
                    ax.set_xlim(*config.xlim)
                if config.ylim:
                    ax.set_ylim(*config.ylim)

                ax.set_aspect("equal")

                ax.tick_params(labelsize=st["tick_font_size"])

                if st["grid"]:
                    # Grid lines stay below the filled overlays and the markers.
                    ax.set_axisbelow(True)
                    ax.grid(
                        True,
                        color=st["grid_color"],
                        linestyle=st["grid_style"],
                        lw=st["grid_lw"],
                        alpha=st["grid_alpha"],
                    )

                if config.add_labels:
                    for td, x0, y0 in zip(traces, xs, ys, strict=True):
                        ax.text(
                            x0,
                            y0,
                            td.basename,
                            fontsize=st["label_font_size"],
                            ha="center",
                            va="top",
                        )

                if config.external_polygons:
                    plot_external_polygons(ax, config.external_polygons, st)

                # After the overlays, which can widen the axis limits.
                if colorbar_source is not None:
                    _add_colorbar(fig, ax, colorbar_source, ticks, labels)

                if config.output_dir:
                    outdir = Path(config.output_dir)
                    outdir.mkdir(parents=True, exist_ok=True)
                    # The tight crop measures axis labels only across their
                    # axis and would cut the y label of a flat map.
                    fig.savefig(
                        outdir / f"{output_name_prefix}{pt}_{attr}.{st['save_format']}",
                        dpi=config.dpi,
                        bbox_inches="tight",
                        bbox_extra_artists=[
                            *fig.get_default_bbox_extra_artists(),
                            ax.xaxis.label,
                            ax.yaxis.label,
                        ],
                    )
                if config.show:
                    plt.show()
            finally:
                plt.close(fig)


def _add_colorbar(fig, ax, mappable, ticks: list[float], labels: list[str]) -> None:
    """Add the color bar of ``mappable`` at the right of the map, or below a
    map shorter than _MIN_COLORBAR_HEIGHT_IN, with the given ticks and labels."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    height_per_width = abs(y_max - y_min) / abs(x_max - x_min)
    box = ax.get_position(original=True)
    box_width = box.width * fig.get_figwidth()
    box_height = box.height * fig.get_figheight()
    # A bar at the right takes about 1 inch of the width.
    map_height = min(box_height, (box_width - 1.0) * height_per_width)

    divider = make_axes_locatable(ax)
    if map_height >= _MIN_COLORBAR_HEIGHT_IN:
        cax = divider.append_axes("right", size=0.2, pad=0.15)
        orientation = "vertical"
    else:
        cax = divider.append_axes("bottom", size=0.2, pad=0.75)
        orientation = "horizontal"
    colorbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    if ticks:
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(labels)


def _site_values(
    traces: Sequence[TraceData],
    source_key: str,
    attr: str,
    mse_feature: tuple[str, int | None] | None,
    averaged: bool,
    aggregate_stat: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return one value per site for ``attr`` plus its uncertainty, minimum and maximum.

    Averaged sources give the ``aggregate_stat`` entry and its error key of the
    stored statistics; raw sources give the value alone and ``None`` for the
    other three arrays. A key, MSE scale or statistic that is missing at a
    site becomes NaN, which the caller reports with the site name.
    """
    vals, errs, mins, maxs = [], [], [], []
    error_key = aggregate_error_key(aggregate_stat)
    for td in traces:
        source = getattr(td, source_key)
        if mse_feature is not None:
            mse_key, desired_scale = mse_feature
            entry = source.get(mse_key)
            if entry is None:
                stat = None
            else:
                scalar_stats, scale_stats = entry
                if desired_scale is None:
                    stat = scalar_stats
                else:
                    scales = list(scale_stats.get("Scale", []))
                    stat = (
                        scale_stats["Value"][scales.index(desired_scale)]
                        if desired_scale in scales
                        else None
                    )
            if averaged:
                stat = {} if stat is None else stat
                val = stat.get(aggregate_stat, np.nan)
                err = stat.get(error_key, np.nan)
                mn = stat.get("min", np.nan)
                mx = stat.get("max", np.nan)
            else:
                val = np.nan if stat is None else stat
                err = mn = mx = None
        else:
            d = source.get(attr)
            if d is None:
                val = err = mn = mx = np.nan
            elif averaged:
                val = d[aggregate_stat]
                err = d.get(error_key, np.nan)
                mn = d.get("min", np.nan)
                mx = d.get("max", np.nan)
            else:
                val = d
                err = mn = mx = None

        vals.append(val)
        errs.append(err)
        mins.append(mn)
        maxs.append(mx)

    return (
        np.array(vals, float),
        np.array(errs, float) if any(e is not None for e in errs) else None,
        np.array(mins, float) if any(m is not None for m in mins) else None,
        np.array(maxs, float) if any(m is not None for m in maxs) else None,
    )


def build_cmap(base_cmap, data, color_norm="mad", n_scale_levels=5, n_colors=256):
    """Return the colormap, norm, color-bar ticks and tick labels of one attribute.

    Parameters
    ----------
    base_cmap : Colormap
        Base colormap to sample from.
    data : array-like
        1D array of finite values that sets the color scale.
    color_norm : {'absolute', 'mad', 'mad-scaled', 'mean-std', 'median-std'}, default 'mad'
        'absolute': the data range [min, max]; 'mad': median -/+
        n_scale_levels * MAD; 'mad-scaled': median -/+ n_scale_levels *
        1.4826 * MAD; 'mean-std': mean -/+ n_scale_levels * std;
        'median-std': median -/+ n_scale_levels * std.
    n_scale_levels : int, default 5
        Multiplier of the MAD or std in the robust modes.
    n_colors : int, default 256
        Number of discrete colors.

    Returns
    -------
    cmap : Colormap
    norm : Normalize
    ticks : list of float
    labels : list of str
        Empty lists when a robust scale is below 1e-8 and the absolute
        scale is used instead.
    """
    data = np.asarray(data)
    data_min, data_max = data.min(), data.max()

    if color_norm == "absolute":
        vmin, vmax = data_min, data_max
        cmap = base_cmap
        norm = Normalize(vmin=vmin, vmax=vmax)
        n_ticks = 7

        data_range = vmax - vmin

        if data_range > 50:
            fmt = "{:.0f}"
        elif data_range > 10:
            fmt = "{:.1f}"
        elif data_range > 1:
            fmt = "{:.2f}"
        elif data_range > 0.1:
            fmt = "{:.3f}"
        elif data_range > 1e-3:
            fmt = "{:.4f}"
        elif data_range > 1e-5:
            fmt = "{:.5f}"
        else:
            fmt = "{:.1e}"

        def format_tick(tick, fmt):
            s = fmt.format(tick)
            if "e" not in s and "." in s:
                s = s.rstrip("0").rstrip(".")
            return s

        ticks_raw = [
            tick for tick in MaxNLocator(n_ticks).tick_values(vmin, vmax) if vmin <= tick <= vmax
        ]
        labels = [format_tick(tick, fmt) for tick in ticks_raw]

        return cmap, norm, ticks_raw, labels

    if color_norm == "mad":
        center = np.median(data)
        scale = np.median(np.abs(data - center))
        center_label = "med"
        unit = "MAD"

    elif color_norm == "mad-scaled":
        center = np.median(data)
        scale = np.median(np.abs(data - center)) * 1.4826
        center_label = "med"
        unit = "σ̂"

    elif color_norm == "mean-std":
        center = np.mean(data)
        scale = np.std(data)
        center_label = "mean"
        unit = "σ"

    elif color_norm == "median-std":
        center = np.median(data)
        scale = np.std(data)
        center_label = "med"
        unit = "σ"

    else:
        raise ValueError(f"color_norm must be one of {list(_COLOR_NORMS)}; got {color_norm!r}.")

    if scale < 1e-8:
        safe_warn(
            f"color_norm {color_norm!r}: the scale is below 1e-8; the colors span the data "
            "range without ticks."
        )
        vmin, vmax = data_min, data_max
        cmap = base_cmap
        norm = Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm, [], []

    vmin = center - n_scale_levels * scale
    vmax = center + n_scale_levels * scale

    # The data range is mapped onto the part of the base colormap that
    # [vmin, vmax] covers, so values beyond n_scale_levels scale units from
    # the center saturate.
    values = np.linspace(data_min, data_max, n_colors)
    t = (values - vmin) / (vmax - vmin if vmax != vmin else 1e-6)
    t = np.clip(t, 0.0, 1.0)
    colors = base_cmap(t)
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_colors)
    norm = Normalize(vmin=data_min, vmax=data_max)

    # Integer offsets k so that the ticks center + k * scale cover the data range.
    k_min = int(np.floor((data_min - center) / scale))
    k_max = int(np.ceil((data_max - center) / scale))

    ticks_raw = []
    k_values = []
    for k in range(k_min, k_max + 1):
        tick = center + k * scale
        if data_min <= tick <= data_max:
            ticks_raw.append(tick)
            k_values.append(k)

    # The value in parentheses keeps two decimals down to a scale step of 0.1 and
    # gains one per decade below it, so that ticks one step apart print
    # differently. A label longer than eight characters uses scientific
    # notation.
    decimals = max(2, 1 - int(np.floor(np.log10(scale))))

    def value_label(tick: float) -> str:
        text = f"{tick:.{decimals}f}"
        return text if len(text) <= 8 else f"{tick:.2e}"

    # Every tick is labeled below 7 ticks, the odd offsets below 12, the
    # offsets 3, 7, 11, ... up to 28, and above that the multiples of the
    # smallest 1-2-5 step that leaves at most seven labels besides the
    # center, which is always labeled.
    total = len(ticks_raw)
    steps = [5] + [m * 10**e for e in range(1, 9) for m in (1, 2, 5)]
    label_step = next(step for step in steps if total <= 7 * step)
    labels = []
    for tick, k in zip(ticks_raw, k_values, strict=True):
        if k == 0:
            labels.append(center_label)
            continue
        if total < 7:
            show_label = True
        elif total < 12:
            show_label = abs(k) % 2 == 1
        elif total <= 28:
            show_label = abs(k) % 4 == 3
        else:
            show_label = abs(k) % label_step == 0
        sign = "+" if k > 0 else "−"
        labels.append(f"{sign}{abs(k)}{unit}\n({value_label(tick)})" if show_label else "")

    return cmap, norm, ticks_raw, labels


def deduplicate_site_traces(
    traces: Sequence[TraceData],
) -> tuple[list[TraceData], bool]:
    """Return the first trace of every six-decimal site and whether any site repeated."""
    seen = set()
    unique = []
    for trace in traces:
        coordinate = site_key(trace.x, trace.y)
        if coordinate not in seen:
            seen.add(coordinate)
            unique.append(trace)
    return unique, len(unique) < len(traces)


def _fill_arguments(fill: Mapping[str, Any]) -> dict[str, Any]:
    """Return the patch arguments of an overlay fill, with the package defaults."""
    return {
        "facecolor": fill.get("color", _CONCRETE_GRAY),
        "edgecolor": "none",
        "alpha": fill.get("alpha", 0.55),
        "zorder": fill.get("zorder", 1),
    }


def plot_external_polygons(
    ax: plt.Axes,
    external_polygons: Sequence[Mapping[str, Any]],
    style: Mapping[str, Any],
) -> None:
    """Draw the ``external_polygons`` overlays of :func:`plot_maps` on ``ax``.

    ``style`` is the resolved map style: its ``polygon_style`` gives the line
    defaults and ``label_font_size`` the label size. A shapefile that cannot
    be read is reported through the package logger and skipped; Matplotlib
    failures while drawing propagate.
    """
    default_line_style = style["polygon_style"]
    label_font_size = style["label_font_size"]
    label_kwargs: dict[str, Any] = {"ha": "center", "va": "center", "fontsize": label_font_size}

    for polygon in external_polygons:
        line_style = dict(default_line_style)
        line_style.update(polygon.get("style", {}))

        if "ring" in polygon:
            specification = polygon["ring"]
            center_x, center_y = specification.get("center", (0.0, 0.0))
            inner_radius = specification["r_inner"]
            outer_radius = specification["r_outer"]
            fill = polygon.get("fill", {})
            ax.add_patch(
                Wedge(
                    (center_x, center_y),
                    outer_radius,
                    0,
                    360,
                    width=outer_radius - inner_radius,
                    **_fill_arguments(fill),
                )
            )
            if polygon.get("edge", True):
                for radius in (inner_radius, outer_radius):
                    ax.add_patch(
                        Circle(
                            (center_x, center_y),
                            radius,
                            fill=False,
                            **line_style,
                        )
                    )
            if "label" in polygon:
                ax.text(
                    center_x,
                    center_y + outer_radius,
                    polygon["label"],
                    fontsize=label_font_size,
                    ha="center",
                    va="bottom",
                )
            continue

        if "circle" in polygon:
            specification = polygon["circle"]
            center_x, center_y = specification.get("center", (0.0, 0.0))
            radius = specification["radius"]
            fill = polygon.get("fill")
            if fill is not None:
                ax.add_patch(Circle((center_x, center_y), radius, **_fill_arguments(fill)))
            ax.add_patch(
                Circle(
                    (center_x, center_y),
                    radius,
                    fill=False,
                    **line_style,
                )
            )
            if "label" in polygon:
                ax.text(
                    center_x,
                    center_y + radius,
                    polygon["label"],
                    fontsize=label_font_size,
                    ha="center",
                    va="bottom",
                )
            continue

        if "shp" in polygon:
            shapefile_path = polygon["shp"]
            try:
                import geopandas as gpd
                from pyogrio.errors import DataLayerError, DataSourceError
                from shapely.errors import GEOSException
                from shapely.geometry.base import BaseGeometry
            except (ImportError, OSError) as exc:
                log_warning(f"{shapefile_path}: shapefile not drawn ({exc}).")
                continue

            read_errors: tuple[type[BaseException], ...] = (
                DataLayerError,
                DataSourceError,
                OSError,
                ValueError,
            )
            if gpd.options.io_engine == "fiona":
                try:
                    from fiona.errors import FionaError
                except (ImportError, OSError) as exc:
                    log_warning(f"{shapefile_path}: shapefile not drawn ({exc}).")
                    continue
                read_errors += (FionaError,)

            try:
                frame = gpd.read_file(shapefile_path)
            except read_errors as exc:
                log_warning(f"{shapefile_path}: shapefile not drawn ({exc}).")
                continue

            try:
                if not isinstance(frame, gpd.GeoDataFrame):
                    raise ValueError("reader did not return a GeoDataFrame")
                geometries = list(frame.geometry)
                line_coordinates = []
                for geometry in geometries:
                    if geometry is None:
                        raise ValueError("shapefile contains a missing geometry")
                    if not isinstance(geometry, BaseGeometry):
                        raise ValueError("shapefile contains a non-Shapely geometry")
                    if geometry.geom_type == "Polygon":
                        x_values, y_values = geometry.exterior.xy
                        line_coordinates.append((x_values, y_values))
                    elif geometry.geom_type == "MultiPolygon":
                        for part in geometry.geoms:
                            x_values, y_values = part.exterior.xy
                            line_coordinates.append((x_values, y_values))
                    elif geometry.geom_type == "LineString":
                        x_values, y_values = geometry.xy
                        line_coordinates.append((x_values, y_values))

                label_position = None
                if "label" in polygon:
                    if not geometries:
                        raise ValueError("shapefile contains no geometry to label")
                    first_geometry = geometries[0]
                    if first_geometry.geom_type == "Polygon":
                        first_x_values, first_y_values = first_geometry.exterior.xy
                        if len(first_x_values) == 0 or len(first_y_values) == 0:
                            raise ValueError("first shapefile geometry has no label position")
                        label_position = (first_x_values[0], first_y_values[0])
                    elif first_geometry.geom_type == "LineString":
                        first_x_values, first_y_values = first_geometry.xy
                        if len(first_x_values) == 0 or len(first_y_values) == 0:
                            raise ValueError("first shapefile geometry has no label position")
                        label_position = (first_x_values[0], first_y_values[0])
            except (GEOSException, ValueError) as exc:
                log_warning(f"{shapefile_path}: shapefile not drawn ({exc}).")
                continue

            # Only reading is guarded; drawing errors propagate.
            for x_values, y_values in line_coordinates:
                ax.plot(x_values, y_values, **line_style)
            if label_position is not None:
                ax.text(
                    label_position[0],
                    label_position[1],
                    polygon["label"],
                    **label_kwargs,
                )
            continue

        if "x" in polygon and "y" in polygon:
            x_values, y_values = polygon["x"], polygon["y"]
            fill = polygon.get("fill")
            if fill is not None:
                ax.fill(x_values, y_values, **_fill_arguments(fill))
            ax.plot(x_values, y_values, **line_style)
            if "label" in polygon:
                ax.text(
                    x_values[0],
                    y_values[0],
                    polygon["label"],
                    **label_kwargs,
                )

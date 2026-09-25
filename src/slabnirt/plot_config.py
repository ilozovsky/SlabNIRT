"""Configuration objects of plot_maps and plot_traces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._attribute_selection import AttributeSelection, OptionalAttributeSelection


@dataclass(frozen=True)
class MapPlotConfig:
    """Configuration of :func:`slabnirt.plot_maps`.

    Parameters
    ----------
    attrs_to_plot : 'all', list of str or tuple of str, default 'all'
        'all' maps every attribute stored at any site, in first-seen order.
        A name unknown to every site raises ValueError. MSE variants go by
        their stored key.
    MSE_scales : list of int, optional
        Scales to map for every selected MSE variant, as '<key>_scaleN'.
        None maps only the scalar MSE values.
    averaged_attribute_values : bool, default True
        Map the per-site statistics in ``attributes_avg`` rather than the
        raw values in ``attributes``.
    plot_type : {'map', 'scatter'} or list of both, default 'map'
        'map' is the interpolated surface with the site markers, 'scatter'
        the site markers only.
    spacing : float, default 0.2
        Grid cell of the surface in coordinate units; verde adjusts it so
        that the region divides evenly.
    damping : float or None, default 1e-4
        Regularization of the spline fit (verde.Spline). Larger values give
        a smoother surface that does not pass exactly through the site values;
        None fits the sites exactly and ignores the weights.
    use_weights : bool, default False
        Weight the fit by 1 / uncertainty**2. Every site then needs a
        finite, positive uncertainty; identical repeat values, such as
        peak_frequency on the FFT bin grid, give zero. At the default
        damping the weights change the surface only slightly; they matter
        from a damping of about 1e-2 upwards.
    buffer : float, default 0.1
        Margin around the site bounds covered by the surface, in coordinate
        units. The spline diverges away from the sites, so keep it small or
        set ``mask_distance``.
    mask_distance : float or None, default None
        Hide surface cells farther than this from every site; None or a
        value <= 0 shows the whole region.
    show_contours : bool, default True
        Draw contour lines at the color-bar ticks on the surface.
    colorscale : str, default 'slabnirt'
        'slabnirt' (the package palette) or a Matplotlib colormap name.
    color_norm : {'absolute', 'mad', 'mad-scaled', 'mean-std', 'median-std'}, default 'mad'
        'absolute' (site minimum to maximum), 'mad' (median and MAD),
        'mad-scaled' (MAD * 1.4826), 'mean-std' or 'median-std'.
    n_scale_levels : int, default 5
        Scale units (MAD or std) on each side of the center that the colors
        span in the robust modes.
    plot_site_only : bool, default False
        Draw every site marker in one color. A scatter figure then has no
        color bar; a map keeps the color bar of its surface.
    plot_median_averaged_attributes : bool, default True
        Map the median of the averaged statistics, with its uncertainty;
        False maps the mean.
    plot_error : {None, 'error', 'minmax'}, default None
        Draw two half-filled markers per site, at value -/+ uncertainty or
        at the site minimum and maximum, and a marker at the value. Ignored
        with a warning for site-only and raw-value maps.
    add_labels : bool, default False
        Write the basename of the representing trace at every site.
    coordinate_units : str, default 'm'
        Unit in the axis labels 'Distance (<unit>)'.
    xlim, ylim : (float, float), optional
        Axis limits; None fits the data.
    external_polygons : list of dict, optional
        Overlays drawn on every figure; see :func:`slabnirt.plot_maps`.
    panel_size : (float, float), default (8, 4)
        Figure width and height in inches. Saved files are cropped to the
        drawing, so their size follows the shape of the survey.
    output_prefix : str or Path, optional
        Prefix of the saved file names; None means no prefix.
    output_dir : str or Path, optional
        Directory to save into, created if missing; None saves nothing.
    dpi : int, optional
        Resolution of saved figures; None uses Matplotlib's savefig.dpi.
    show : bool, default True
        Call plt.show() after each figure.
    style : dict, optional
        Overrides of the style keys listed in :func:`slabnirt.plot_maps`.
    """

    attrs_to_plot: AttributeSelection = "all"
    MSE_scales: list[int] | None = None
    averaged_attribute_values: bool = True
    plot_type: str | list[str] = "map"
    spacing: float = 0.2
    damping: float | None = 1e-4
    use_weights: bool = False
    buffer: float = 0.1
    mask_distance: float | None = None
    show_contours: bool = True
    colorscale: str = "slabnirt"
    color_norm: str = "mad"
    n_scale_levels: int = 5
    plot_site_only: bool = False
    plot_median_averaged_attributes: bool = True
    plot_error: str | None = None
    add_labels: bool = False
    coordinate_units: str = "m"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    external_polygons: list[dict[str, Any]] | None = None
    panel_size: tuple[float, float] = (8, 4)
    output_prefix: str | Path | None = None
    output_dir: str | Path | None = None
    dpi: int | None = None
    show: bool = True
    style: dict[str, Any] | None = None


@dataclass(frozen=True)
class TracePlotConfig:
    """Configuration of :func:`slabnirt.plot_traces`.

    Parameters
    ----------
    attrs_to_plot : 'all', list of str, tuple of str or None, default 'all'
        Attributes listed in the table. 'all' lists every non-MSE
        attribute; a name absent on a trace is skipped with a warning, a
        name unknown to every trace raises ValueError; None omits the table.
    time_in_ms : bool, default False
        Show the time axis in milliseconds.
    time_limits : (float, float), optional
        Limits of the time axis, in ms with ``time_in_ms`` and in s
        otherwise; None fits the data.
    spectrum_type : {'power', 'amplitude'}, default 'power'
        'power' draws the square of the stored spectrum, as the spectral
        attributes use by default; 'amplitude' the stored spectrum.
    use_db : bool, default False
        Draw the spectrum in decibels relative to each curve's maximum.
    spectrum_freq_lim : (float, float), optional
        Limits of the frequency axis in Hz; None shows the whole spectrum.
    log_freq : bool, default False
        Logarithmic frequency axis, starting at 0.1 Hz when the lower limit
        is not positive.
    plot_pre_filtered : bool, default True
        Also draw the unfiltered signal and spectrum, dashed, for filtered
        traces.
    freq_lines : list of float, optional
        Frequencies in Hz to mark with vertical lines.
    panel_size : (float, float), default (8, 2)
        Width and height in inches of one panel; the figure height adds up
        the panels and table rows.
    output_dir : str or Path, optional
        Directory to save into, created if missing; None saves nothing.
    dpi : int, optional
        Resolution of saved figures; None uses Matplotlib's savefig.dpi.
    show : bool, default True
        Call plt.show() after each figure.
    unify_ylims : bool, default False
        Use the same y-limits, taken over all traces, in every figure.
    mse_log_y : bool, default False
        Logarithmic y-axis of the MSE panel.
    style : dict, optional
        Overrides of the style keys listed in :func:`slabnirt.plot_traces`.
    """

    attrs_to_plot: OptionalAttributeSelection = "all"
    time_in_ms: bool = False
    time_limits: tuple[float, float] | None = None
    spectrum_type: str = "power"
    use_db: bool = False
    spectrum_freq_lim: tuple[float, float] | None = None
    log_freq: bool = False
    plot_pre_filtered: bool = True
    freq_lines: list[float] | None = None
    panel_size: tuple[float, float] = (8, 2)
    output_dir: str | Path | None = None
    dpi: int | None = None
    show: bool = True
    unify_ylims: bool = False
    mse_log_y: bool = False
    style: dict[str, Any] | None = None

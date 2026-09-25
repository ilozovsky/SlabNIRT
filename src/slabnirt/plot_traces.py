"""Per-trace figures: signal, spectrum, multiscale entropy and attribute table."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._attribute_selection import (
    normalize_attribute_selection,
    validate_known_attribute_names,
)
from ._console import safe_warn
from ._mse import unpack_mse_entry
from .core import TraceData
from .plot_config import TracePlotConfig
from .preprocess import compute_amplitude_spectrum
from .style import style_defaults


def _maybe_grid(ax: plt.Axes, style: Mapping[str, Any]) -> None:
    """Apply the grid style when grid display is enabled."""
    if style["grid"]:
        ax.grid(
            color=style["grid_color"],
            linestyle=style["grid_style"],
            alpha=style["grid_alpha"],
        )


def _ensure_spectrum(trace: TraceData) -> tuple[np.ndarray, np.ndarray]:
    """Return the trace's amplitude spectrum, computing and storing it when absent.

    A trace that carries no ``freqs`` or no ``spectrum`` gets both replaced by
    the spectrum of ``signal_norm`` or, when the trace was not preprocessed,
    of the raw ``signal``; that spectrum is not normalized and scales with the
    acquisition units. The sampling interval is taken from the first two time
    samples. ``signal_norm`` itself is not stored.
    """
    if trace.freqs is None or trace.spectrum is None:
        dt = trace.t[1] - trace.t[0]
        signal = trace.signal_norm if trace.signal_norm is not None else trace.signal
        trace.freqs, trace.spectrum = compute_amplitude_spectrum(signal, dt, len(signal))
    return trace.freqs, trace.spectrum


def _convert_to_db(values: Any, spectrum_type: str) -> np.ndarray:
    """Convert a spectrum to decibels relative to its own maximum, floored at -60 dB.

    ``20 * log10`` for an amplitude spectrum and ``10 * log10`` for a power
    spectrum, so a squared amplitude spectrum gives the same curve as the
    amplitude spectrum. Values below the floor are clipped to it. A spectrum
    with no finite positive value has no dynamic range and returns zeros;
    otherwise NaN positions stay NaN.
    """
    values_array = np.asarray(values, dtype=float)
    scale = 20 if spectrum_type == "amplitude" else 10
    finite_values = values_array[np.isfinite(values_array)]
    if finite_values.size == 0:
        return np.zeros_like(values_array)
    reference = np.max(finite_values)
    if reference <= 0:
        return np.zeros_like(values_array)
    minimum = reference * 10 ** (-60 / scale)
    return scale * np.log10(np.clip(values_array, minimum, None) / reference)


def _validated_mse_variants(trace: TraceData) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return (key, scales, values) of every MSE variant stored on one trace.

    The values may contain NaN, which leaves a gap, but not infinity, and at
    least one of them must be finite.
    """
    variants = []
    for key, stored in (trace.attributes or {}).items():
        if not key.startswith("MSE"):
            continue

        context = f"{trace.basename}: MSE variant '{key}'"
        _, scales, values = unpack_mse_entry(stored, context)
        if np.any(np.isinf(values)):
            raise ValueError(f"{context} values may contain NaN but not infinity.")
        if not np.any(np.isfinite(values)):
            raise ValueError(f"{context} must contain at least one finite value.")

        variants.append((key, scales, values))
    return variants


def _spectrum_curves(trace, spectrum_type, use_db, plot_pre_filtered):
    """Return the spectrum curves of one trace as drawn, for the panel and the shared limits.

    Gives ``(main_freqs, main, raw_freqs, unfiltered)``: the filtered spectrum
    when the trace was filtered, otherwise its stored spectrum (computed and
    stored first when absent), and the unfiltered spectrum to draw dashed, or
    None. Power mode squares the values; dB mode converts each curve against
    its own maximum.
    """
    raw_freqs, raw_spectrum = _ensure_spectrum(trace)
    if trace.spec_filt is not None:
        main_freqs, main = trace.freqs_filt, trace.spec_filt
        unfiltered = raw_spectrum if plot_pre_filtered else None
    else:
        main_freqs, main = raw_freqs, raw_spectrum
        unfiltered = None

    if spectrum_type == "power":
        main = main**2
        unfiltered = None if unfiltered is None else unfiltered**2
    if use_db:
        main = _convert_to_db(main, spectrum_type)
        unfiltered = None if unfiltered is None else _convert_to_db(unfiltered, spectrum_type)
    return main_freqs, main, raw_freqs, unfiltered


def plot_traces(
    traces: list[TraceData],
    config: TracePlotConfig | None = None,
) -> None:
    """Draw one figure per trace and save or show it.

    Panels, top to bottom: the normalized signal, the spectrum, the multiscale
    entropy curve when the trace holds any ``MSE*`` attribute, and a table of
    the other attributes. A filtered trace shows its filtered signal and
    spectrum; ``plot_pre_filtered`` adds the unfiltered ones as dashed curves.
    Peak-frequency and spectral-centroid attributes are drawn as vertical
    marker lines on the spectrum; a non-finite value draws no line.

    Spectrum values are the two-sided DFT magnitudes stored by preprocessing
    (``spectrum_type='power'`` squares them). With ``use_db`` each curve is
    converted separately to decibels relative to its own maximum, 20*log10
    for amplitude and 10*log10 for power, floored at -60 dB; every curve
    therefore peaks at 0 dB and the level difference between the filtered
    and unfiltered curves is not shown.

    Attributes may be NaN (shown as ``nan``); non-numeric values are shown
    as text. A trace without any ``MSE*`` key gets no entropy panel. Every
    ``MSE*`` entry must be a ``(scalar, {'Scale', 'Value'})`` pair with
    finite scales and at least one finite value; NaN values leave gaps.

    Mutation: a trace without ``freqs``/``spectrum`` gets both computed and
    stored on it, from ``signal_norm`` or, for a trace that was not
    preprocessed, from the raw signal (then not normalized, so its scale
    follows the acquisition units). Nothing else on the traces changes.

    Output: ``output_dir`` is created if needed and each figure is saved as
    ``<basename>_Ch<channel>.<save_format>`` (PNG by default), overwriting an
    existing file; traces sharing basename and channel overwrite each other.
    ``dpi=None`` uses Matplotlib's ``savefig.dpi`` setting, the figure dpi by
    default. ``show=True`` calls ``plt.show()`` after each figure. Every
    figure is closed before the next trace is drawn, also when drawing fails.

    Parameters
    ----------
    traces : list of TraceData
        Traces to draw.
    config : TracePlotConfig, optional
        Plot options; None uses the defaults.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        For a config that is not a TracePlotConfig.
    ValueError
        For an unknown attribute name, ``spectrum_type`` or style key, or a
        malformed ``MSE*`` entry, before the first figure is created. An
        error while drawing propagates after the earlier figures were saved.

    Notes
    -----
    Style keys accepted in ``config.style`` (an unknown key raises
    ValueError; a nested dictionary replaces the default one whole):
      panel_hspace         (float): Vertical spacing between panels.
      subplots_adjust      (dict):  top/bottom/left/right figure margins.
      show_filename        (bool):  Suptitle ``<basename>, Ch<channel>``.
      suptitle_font_size, axis_label_font_size, tick_font_size,
      legend_font_size, table_font_size (int): Font sizes.
      line_width           (float): Width of plotted lines.
      axis_bg_color        (str):   Background color of each panel.
      grid                 (bool):  Draw grid lines.
      grid_color, grid_style, grid_alpha: Color, linestyle, alpha of the grid.
      time_color           (str):   Color of the signal curve.
      time_fill_color      (str):   Fill color under the signal curve.
      time_fill_alpha      (float): Alpha of that fill.
      spectrum_color       (str):   Color of the spectrum curve and its fill.
      spectrum_fill_alpha  (float): Alpha of the spectrum fill.
      spec_marker_size     (int):   Marker size on an unfiltered spectrum.
      pre_filtered_color   (str):   Color of the dashed unfiltered curves.
      vertical_line_style  (dict):  Line style of the ``freq_lines`` markers.
      show_peak_lines      (bool):  Draw peak-frequency markers.
      peak_line_style      (dict):  Their line style.
      show_centroid_lines  (bool):  Draw spectral-centroid markers.
      centroid_line_style  (dict):  Their line style.
      mse_color            (str):   Color of a single MSE curve.
      mse_marker_size      (int):   Marker size on MSE curves.
      mse_ylim             (tuple|None): Fixed y-limits of the MSE panel.
      attr_numfmt          (str):   Format spec for table numbers.
      show_panel_labels    (bool):  Draw a, b, c badges on the panels.
      panel_label_style    (dict):  Badge position, box and font.
      save_format          (str):   File extension when saving.
    """
    if config is None:
        config = TracePlotConfig()
    elif not isinstance(config, TracePlotConfig):
        raise TypeError("config must be a TracePlotConfig instance or None.")

    attrs_to_plot = normalize_attribute_selection(
        config.attrs_to_plot,
        parameter_name="attrs_to_plot",
        allow_none=True,
    )

    if config.spectrum_type not in ("amplitude", "power"):
        raise ValueError(
            f"spectrum_type must be 'amplitude' or 'power'; got {config.spectrum_type!r}."
        )
    st = style_defaults("traces", config.style)
    numfmt = st["attr_numfmt"]

    # Validate every MSE structure before any figure or output file is created.
    mse_variants_by_trace = {id(trace): _validated_mse_variants(trace) for trace in traces}
    if isinstance(attrs_to_plot, list):
        available_attributes = {name for trace in traces for name in (trace.attributes or {})}
        validate_known_attribute_names(
            attrs_to_plot,
            available_attributes,
            parameter_name="attrs_to_plot",
        )

    global_ylims = (
        _compute_global_ylims(
            traces,
            config.spectrum_type,
            config.use_db,
            config.plot_pre_filtered,
            config.spectrum_freq_lim,
            config.mse_log_y,
            mse_variants_by_trace,
        )
        if config.unify_ylims
        else None
    )

    for trace in traces:
        trace_attributes = trace.attributes or {}
        mse_variants = mse_variants_by_trace[id(trace)]

        has_mse = bool(mse_variants)

        if isinstance(attrs_to_plot, list):
            missing = [k for k in attrs_to_plot if k not in trace_attributes]
            if missing:
                safe_warn(f"{trace.basename}: attributes {missing} not found; skipped.")

        if attrs_to_plot == "all":
            keys = [k for k in trace_attributes if not k.startswith("MSE")]
        elif isinstance(attrs_to_plot, list):
            keys = [k for k in attrs_to_plot if k in trace_attributes and not k.startswith("MSE")]
        else:
            keys = []

        want_table = bool(keys)
        n_rows_table = len(keys)
        ratios = [1.0, 1.0] + ([1.0] if has_mse else [])
        if want_table:
            ratios.append(0.2 + 0.12 * n_rows_table)

        panel_w, panel_h = config.panel_size
        fig_h = panel_h * (2 + int(has_mse)) + panel_h * 0.3 * n_rows_table
        fig = plt.figure(figsize=(panel_w, fig_h), constrained_layout=False)
        try:
            gs = fig.add_gridspec(
                nrows=len(ratios), ncols=1, height_ratios=ratios, hspace=st["panel_hspace"]
            )

            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2]) if has_mse else None
            ax_tab = fig.add_subplot(gs[-1]) if want_table else None

            if st["show_filename"]:
                fig.suptitle(
                    f"{trace.basename}, Ch{trace.channel}",
                    fontsize=st["suptitle_font_size"],
                    y=0.98,
                )

            plot_axes = [ax0, ax1] + ([ax2] if has_mse else [])
            for ax in plot_axes:
                ax.set_facecolor(st["axis_bg_color"])
                ax.tick_params(labelsize=st["tick_font_size"])

            if st["show_panel_labels"]:
                panel_style = st["panel_label_style"]
                for ax, lbl in zip(plot_axes, ["a", "b", "c"], strict=False):
                    ax.text(
                        panel_style["x"],
                        panel_style["y"],
                        lbl,
                        transform=ax.transAxes,
                        fontsize=panel_style["fontsize"],
                        color=panel_style["text_color"],
                        ha=panel_style["ha"],
                        va=panel_style["va"],
                        bbox=dict(
                            boxstyle=panel_style["boxstyle"],
                            facecolor=panel_style["facecolor"],
                            edgecolor=panel_style["edgecolor"],
                            linewidth=panel_style["linewidth"],
                        ),
                    )

            # Time-domain panel
            t = trace.t * (1000 if config.time_in_ms else 1)

            sig = (
                trace.signal_norm
                if trace.signal_norm is not None
                else trace.signal / np.max(np.abs(trace.signal))
            )
            sig_pre = trace.signal_norm_pre

            if config.plot_pre_filtered and sig_pre is not None:
                ax0.plot(
                    t,
                    sig,
                    color=st["time_color"],
                    lw=st["line_width"],
                    label="Post-filtered signal",
                )
                ax0.fill_between(
                    t,
                    0,
                    sig,
                    where=sig >= 0,
                    color=st["time_fill_color"],
                    alpha=st["time_fill_alpha"],
                )
                ax0.plot(
                    t,
                    sig_pre,
                    color=st["pre_filtered_color"],
                    lw=st["line_width"],
                    ls="--",
                    label="Pre-filtered signal",
                )
            else:
                ax0.plot(
                    t, sig, color=st["time_color"], lw=st["line_width"], label="Normalized signal"
                )
                ax0.fill_between(
                    t,
                    0,
                    sig,
                    where=sig >= 0,
                    color=st["time_fill_color"],
                    alpha=st["time_fill_alpha"],
                )

            if config.time_limits:
                ax0.set_xlim(*config.time_limits)
            else:
                ax0.set_xlim(t.min(), t.max())
            if global_ylims is not None:
                ax0.set_ylim(*global_ylims["time"])
            else:
                ax0.set_ylim(-1.1, 1.1)

            time_unit = "ms" if config.time_in_ms else "s"
            ax0.set_xlabel(f"Time ({time_unit})", fontsize=st["axis_label_font_size"])
            ax0.set_ylabel("Amplitude (norm., a.u.)", fontsize=st["axis_label_font_size"])
            _maybe_grid(ax0, st)

            # Frequency-spectrum panel
            main_freqs, spec_main, raw_freqs, spec_other = _spectrum_curves(
                trace, config.spectrum_type, config.use_db, config.plot_pre_filtered
            )
            filtered = trace.spec_filt is not None

            if filtered:
                label_main = "Post-filtered"
            elif config.spectrum_type == "amplitude":
                label_main = "Amplitude Spectrum"
            else:
                label_main = "Power Spectrum"
            ax1.plot(
                main_freqs,
                spec_main,
                color=st["spectrum_color"],
                lw=st["line_width"],
                marker=None if filtered else "o",
                markersize=None if filtered else st["spec_marker_size"],
                label=label_main,
            )

            ax1.fill_between(
                main_freqs,
                spec_main.min() if config.use_db else 0,
                spec_main,
                color=st["spectrum_color"],
                alpha=st["spectrum_fill_alpha"],
            )

            if spec_other is not None:
                ax1.plot(
                    raw_freqs,
                    spec_other,
                    color=st["pre_filtered_color"],
                    lw=st["line_width"],
                    ls="--",
                    label="Pre-filtered Spectrum",
                )

            if config.freq_lines:
                for i, f in enumerate(config.freq_lines):
                    label = "Frequency marker" if i == 0 else None
                    ax1.axvline(f, **st["vertical_line_style"], label=label)

            _configure_spectrum_axes(
                ax1, main_freqs, spec_main, config.use_db, config.log_freq, config.spectrum_freq_lim
            )
            if global_ylims is not None:
                ax1.set_ylim(*global_ylims["spectrum"])

            quantity = (
                "Amplitude spectrum" if config.spectrum_type == "amplitude" else "Power spectrum"
            )
            ax1.set_xlabel("Frequency (Hz)", fontsize=st["axis_label_font_size"])
            ax1.set_ylabel(
                f"{quantity} ({'dB re max' if config.use_db else 'a.u.'})",
                fontsize=st["axis_label_font_size"],
            )
            _maybe_grid(ax1, st)

            if st["show_peak_lines"]:
                peak_keys = [k for k in trace_attributes if k.startswith("peak_frequency")]
                if peak_keys:
                    _plot_frequency_lines(
                        ax1,
                        trace,
                        "Peak Frequency",
                        peak_keys,
                        st["peak_line_style"],
                        colormap_name="Set2",
                    )

            if st["show_centroid_lines"]:
                centroid_keys = [k for k in trace_attributes if k.startswith("spectral_centroid")]
                if centroid_keys:
                    _plot_frequency_lines(
                        ax1,
                        trace,
                        "Spectral Centroid",
                        centroid_keys,
                        st["centroid_line_style"],
                        colormap_name="Set1",
                    )

            ax1.legend(fontsize=st["legend_font_size"])
            ax0.legend(fontsize=st["legend_font_size"])

            # Multiscale-entropy panel
            if has_mse:
                colors = _distinct_colors(
                    [key for key, _, _ in mse_variants],
                    st["mse_color"],
                    "Set1",
                )

                for i, (key, scales, values) in enumerate(mse_variants):
                    ax2.plot(
                        scales,
                        values,
                        color=colors[i],
                        lw=st["line_width"],
                        marker="o",
                        markersize=st["mse_marker_size"],
                        label=key,
                    )

                all_scales = np.concatenate([scales for _, scales, _ in mse_variants])
                all_finite_values = np.concatenate(
                    [values[np.isfinite(values)] for _, _, values in mse_variants]
                )

                ax2.set_xlabel("Scale factor", fontsize=st["axis_label_font_size"])
                ax2.set_ylabel("Sample entropy (nats)", fontsize=st["axis_label_font_size"])

                # The scale is set before the limits, so a log axis keeps them.
                if config.mse_log_y:
                    ax2.set_yscale("log")

                if st["mse_ylim"] is not None:
                    lo, hi = st["mse_ylim"]
                elif global_ylims is not None and global_ylims["mse"] is not None:
                    lo, hi = global_ylims["mse"]
                else:
                    lo = np.min(all_finite_values)
                    hi = np.max(all_finite_values) * 1.05
                    # A log axis needs a positive lower limit.
                    if config.mse_log_y and lo <= 0:
                        pos = all_finite_values[all_finite_values > 0]
                        lo = pos.min() if pos.size else hi / 1000.0

                ax2.set_ylim(lo, hi)
                ax2.set_xlim(np.min(all_scales), np.max(all_scales))

                _maybe_grid(ax2, st)
                ax2.legend(fontsize=st["legend_font_size"])

            # Attribute-table panel
            if want_table:
                rows = []
                for k in keys:
                    v = trace.attributes[k]
                    try:
                        num = float(v)
                        txt = format(num, numfmt)
                    except (TypeError, ValueError, OverflowError):
                        txt = str(v)
                    rows.append((k, txt))

                ax_tab.axis("off")
                table = ax_tab.table(
                    cellText=rows,
                    colLabels=["Attribute", "Value"],
                    cellLoc="left",
                    colLoc="left",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(st["table_font_size"])

                for (r, _column), cell in table.get_celld().items():
                    if r == 0:
                        cell.set_text_props(weight="bold", color="white")
                        cell.set_facecolor("#40466E")
                    else:
                        cell.set_facecolor("#F9F9F9" if r % 2 else "white")
                    cell.set_edgecolor("none")

            fig.subplots_adjust(**st["subplots_adjust"])

            if config.output_dir:
                out = Path(config.output_dir).expanduser().resolve()
                out.mkdir(parents=True, exist_ok=True)
                fname = out / f"{trace.basename}_Ch{trace.channel}.{st['save_format']}"
                fig.savefig(
                    fname, dpi=config.dpi or plt.rcParams["savefig.dpi"], bbox_inches="tight"
                )

            if config.show:
                plt.show()

        finally:
            plt.close(fig)


def _compute_global_ylims(
    traces,
    spectrum_type,
    use_db,
    plot_pre_filtered,
    spectrum_freq_lim,
    mse_log_y,
    mse_variants_by_trace,
):
    """Return the y-limits shared by every figure: ``{'time', 'spectrum', 'mse'}``.

    Each value is (ymin, ymax), taken over the curves as drawn; 'mse' is None
    when no trace has MSE values.
    """
    t_min, t_max = np.inf, -np.inf
    s_min, s_max = np.inf, -np.inf
    m_min, m_max = np.inf, -np.inf
    m_min_pos = np.inf  # smallest positive MSE value, for a log axis
    has_mse_any = False

    for trace in traces:
        sig = (
            trace.signal_norm
            if trace.signal_norm is not None
            else trace.signal / np.max(np.abs(trace.signal))
        )
        sig_pre = trace.signal_norm_pre
        stack = [sig]
        if plot_pre_filtered and sig_pre is not None:
            stack.append(sig_pre)
        for arr in stack:
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                t_min = min(t_min, arr.min())
                t_max = max(t_max, arr.max())

        main_freqs, spec_main, raw_freqs, spec_other = _spectrum_curves(
            trace, spectrum_type, use_db, plot_pre_filtered
        )
        for fr, sp in ((main_freqs, spec_main), (raw_freqs, spec_other)):
            if sp is None:
                continue
            fr = np.asarray(fr)
            sp = np.asarray(sp)
            if spectrum_freq_lim:
                x_min, x_max = spectrum_freq_lim
                mask = (fr >= x_min) & (fr <= x_max)
                vis = sp[mask] if np.any(mask) else sp
            else:
                vis = sp
            vis = vis[np.isfinite(vis)]
            if vis.size:
                s_min = min(s_min, vis.min())
                s_max = max(s_max, vis.max())

        for _, _, values in mse_variants_by_trace[id(trace)]:
            vals = values[np.isfinite(values)]
            if vals.size:
                has_mse_any = True
                m_min = min(m_min, vals.min())
                m_max = max(m_max, vals.max())
                pos = vals[vals > 0]
                if pos.size:
                    m_min_pos = min(m_min_pos, pos.min())

    if np.isfinite(t_min) and np.isfinite(t_max):
        pad = 0.05 * (t_max - t_min) if t_max > t_min else 0.1
        time_lim = (t_min - pad, t_max + pad)
    else:
        time_lim = (-1.1, 1.1)

    if np.isfinite(s_min) and np.isfinite(s_max):
        spectrum_lim = (s_min, 3) if use_db else (0, s_max * 1.05)
    else:
        spectrum_lim = (0, 1)

    if not has_mse_any:
        mse_lim = None
    elif mse_log_y:
        lo = m_min if m_min > 0 else (m_min_pos if np.isfinite(m_min_pos) else m_max / 1000.0)
        mse_lim = (lo, m_max * 1.05)
    else:
        mse_lim = (m_min, m_max * 1.05)

    return {"time": time_lim, "spectrum": spectrum_lim, "mse": mse_lim}


def _distinct_colors(keys, base_color, colormap_name, max_colors=3):
    """Return ``base_color`` for one key, else colors of ``colormap_name`` (tab20 above 3)."""
    if len(keys) == 1:
        return [base_color]
    elif len(keys) <= max_colors:
        colormap = plt.get_cmap(colormap_name, max_colors)
        return [colormap(i) for i in range(len(keys))]
    else:
        colormap = plt.get_cmap("tab20", len(keys))
        return [colormap(i) for i in range(len(keys))]


def _plot_frequency_lines(ax, trace, label_prefix, keys, line_style, colormap_name):
    """Draw a vertical line at each finite frequency attribute in ``keys``.

    The label is the prefix plus the key's suffix, for example
    'Peak Frequency 10-200Hz' for 'peak_frequency_10-200Hz'.
    """
    base_color = line_style.get("color", "#000000")
    colors = _distinct_colors(keys, base_color, colormap_name)

    for i, key in enumerate(keys):
        freq = trace.attributes[key]
        if not np.isfinite(freq):
            continue

        label_detail = key
        for prefix in ("peak_frequency", "spectral_centroid"):
            if key.startswith(prefix):
                label_detail = key.removeprefix(prefix)
                break
        label_detail = label_detail.strip("_")
        label = f"{label_prefix} {label_detail}" if label_detail else f"{label_prefix}"

        style_with_color = {**line_style, "color": colors[i]}
        ax.axvline(freq, **style_with_color, label=label)


def _configure_spectrum_axes(ax, freqs, y_values, use_db, log_freq, spectrum_freq_lim=None):
    """Set the frequency axis and the y-limits of the visible part of the spectrum."""
    if spectrum_freq_lim:
        x_min, x_max = spectrum_freq_lim

        if log_freq and x_min <= 0:
            safe_warn("spectrum_freq_lim starts at or below 0 Hz; the log axis starts at 0.1 Hz.")
            x_min = 0.1

        ax.set_xlim(x_min, x_max)
        mask = (freqs >= x_min) & (freqs <= x_max)
        y_visible = y_values[mask] if np.any(mask) else y_values
    else:
        x_min = 0.1 if log_freq else 0
        x_max = freqs.max()
        ax.set_xlim(x_min, x_max)
        y_visible = y_values

    if use_db:
        y_min = y_visible.min()
        ax.set_ylim(y_min, 3)
    else:
        y_max = y_visible.max()
        ax.set_ylim(0, y_max * 1.05)

    ax.set_xscale("log" if log_freq else "linear")

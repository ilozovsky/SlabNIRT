"""Complete SlabNIRT workflow on a simulated impulse-response survey of a slab.

Usage::

    python slab_model_workflow.py [output_dir] [--show] [--fast]

Reads the traces and coordinate catalog in ``slab_model_data/`` next to this
script and runs every public function of the package, in order::

    read_coordinates_from_catalog -> load_traces -> preprocess_traces
    -> calculate_signal_parameters -> average_attributes -> export_tracedata
    -> plot_traces -> plot_maps

Everything is written into ``output_dir`` (default ``slab_model_output`` in
the current directory). Expect under a minute; ``--fast`` drops the
multiscale entropy and the extra figures and runs about three times faster.
``--show`` opens every figure on screen as well as saving it.

The stages are separate functions, each one a self-contained example of the
call it demonstrates, with the options that matter for real surveys spelled
out and commented. Turn individual stages off with the flags below the
imports (they only control the optional extras; the main pipeline always
runs, since every later stage needs the results of the earlier ones).

What each stage shows
---------------------
1.  ``configure_logging``    - run report and log file.
2.  ``read_coordinates_from_catalog`` + ``load_traces`` - the catalog layout,
    text records, channel selection and time scaling.
3.  ``preprocess_traces``    - polarity, zero-padding, offset removal,
    detrending, trimming, decimation, integration, gain, and a separate
    band-pass demonstration with ``ButterworthFilterConfig``.
4.  ``calculate_signal_parameters`` - all eleven attributes, the void-index
    bands, and the entropy parameters (``sampen_params``, ``mse_params``).
5.  ``average_attributes``   - median and mean per site with MAD errors.
6.  ``TraceData`` - reading signals, spectra, attributes and per-site
    statistics straight off the objects, without any plotting.
7.  ``export_tracedata``     - all four export modes, text and Excel.
8.  ``plot_traces``          - ``TracePlotConfig``: spectra, decibels,
    frequency markers, the MSE panel, and filtered vs unfiltered overlays.
9.  ``plot_maps``            - ``MapPlotConfig``: interpolated attribute
    maps, a scatter variant, robust and absolute color scaling, per-site
    uncertainty markers and geometry overlays.

The dataset
-----------
A numerical model of a concrete slab, 2800 x 1900 x 170 mm, resting on a
150 mm sand layer inside a surrounding soil block, with one built-in
defect: an air void of 660 x 580 x 220 mm under the center of the slab,
its top face at the underside of the slab. Material properties (Vp, Vs,
density): concrete 3800 m/s, 2200 m/s, 2400 kg/m3; sand 650, 300, 1700;
surrounding soil 1400, 700, 2000; air 330 m/s, -, 1.23 kg/m3; no material
attenuation. The wavefield was computed with the finite-element method
(COMSOL Multiphysics, Structural Mechanics module) in three dimensions.

Sources (vertical point forces with a short-pulse source time function of
1.5 kHz center frequency) and receivers (vertical particle velocity, m/s)
alternate on a 75 mm grid starting 50 mm from the slab edges. Each trace
is the response at one receiver to one of its nearest sources, 75 mm away,
so a receiver is a "site" with three or four "strikes" - the
impulse-response test geometry. Coordinates are meters from the slab
center. One quadrant (x <= 0, y >= 0) was computed; the model is symmetric
about both center lines, so the files of the other three quadrants are
copies of the quadrant files listed in the catalog with mirrored
coordinates. The 1 us simulation output was decimated by 40 to 40 us
(25 kHz), 176 samples per trace (7 ms); the records have decayed to below
2 % of their peak by then.

The model is by Ilya Lozovsky, Ruslan Zhostkov and Aleksei Churkin; a
publication describing it is in preparation. The 1788 trace files are
shipped with their values unchanged; ``catalog.csv`` (filename, x, y of the
receiver, x, y of the source) was derived from the model's coordinate
spreadsheet with the coordinates rounded to millimeters. The dataset is
released under the CC BY 4.0 license (``slab_model_data/LICENSE``).

What to look for: the void lowers the peak frequency, narrows the spectrum
and raises the void index and the sample entropy at the sites above it and
around it; the free slab edges raise the peak frequency. The dashed
rectangle on the maps is the void footprint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slabnirt import (
    ButterworthFilterConfig,
    MapPlotConfig,
    TraceData,
    TracePlotConfig,
    average_attributes,
    calculate_signal_parameters,
    configure_logging,
    export_tracedata,
    load_traces,
    plot_maps,
    plot_traces,
    preprocess_traces,
    read_coordinates_from_catalog,
)

DATA_DIR = Path(__file__).with_name("slab_model_data")
PROJECT = "slab_model"

# Optional stages. The pipeline itself (load, preprocess, attributes,
# averaging) always runs; these only add outputs.
EXPORT_TABLES = True  # stage 7: attribute tables for every trace and site
EXPORT_SIGNALS = True  # stage 7: signals and spectra of the sample traces
PLOT_TRACES = True  # stage 8: single-trace figures
PLOT_FILTER_DEMO = True  # stage 8: unfiltered vs band-passed overlay
PLOT_MAPS = True  # stage 9: one map per attribute
PLOT_MAP_VARIANTS = True  # stage 9: scatter, absolute scale, error markers

# The attributes calculate_signal_parameters can produce. 'MSE' (multiscale
# entropy) is the one slow attribute; main() adds it unless --fast is given.
ATTRIBUTES = [
    "norm_signal_energy",
    "normalized_spectrum_area",
    "spectral_centroid",
    "norm_spectrum_area_over_centroid",
    "peak_frequency",
    "spectral_peak_width",
    "spectral_flatness",
    "void_index",
    "sampen",
    "spectral_entropy",
]

# Scales to compute, export and plot for the multiscale entropy. Scale tau
# averages tau neighboring samples before the entropy is measured, so it
# looks at structure tau times slower; scale 1 is the sample entropy itself.
# A scale needs at least 10**dimension = 100 samples to be reliable, and
# these records have 176 (zero-padding lengthens the spectrum, not the
# signal), so only scale 1 meets the rule and the default scale=None returns
# it alone. Scales 2 and 3 are computed here to show the per-scale output;
# they rest on 88 and 58 samples.
MSE_SCALES = [1, 2, 3]

# Void index: spectral maximum in the band where the slab above the void
# resonates, divided by the median level of the band above it. Both bands
# are survey-specific - they follow from the plate thickness and the
# expected defect depth, so set them from the structure, not from defaults.
VOID_INDEX_BANDS = {
    "max_freq_range": (40.0, 200.0),
    "mav_freq_range": (200.0, 800.0),
    "mav_method": "median",  # 'median' (robust) or 'mean'
}

# Entropy settings: templates of two samples (dimension) counted as matching
# when they stay within 0.2 standard deviations (tolerance). These are the
# defaults, written out because they decide what the entropy measures; a
# fixed number instead of 'sd' makes the tolerance independent of the
# record's amplitude distribution.
ENTROPY_PARAMS = {"dimension": 2, "tolerance": "sd"}

# Footprint of the air void, drawn on every map. Any number of overlays can
# be given: polylines like this one, circles, rings or shapefiles (see
# help(plot_maps)), each with its own Matplotlib style and an optional
# 'label'.
VOID_OUTLINE = {
    "x": [-0.33, -0.33, 0.33, 0.33, -0.33],
    "y": [-0.29, 0.29, 0.29, -0.29, -0.29],
    "style": {"ls": "--", "lw": 1.8, "color": "k"},
}

# Two sites to look at in detail: above the void center, and on sound slab
# between the void and the left edge.
SAMPLE_SITES = {"above_void": (0.0, 0.0), "sound_slab": (-0.9, 0.0)}


def trace_nearest(traces: list[TraceData], x: float, y: float) -> TraceData:
    """Return the first trace whose site is closest to (x, y)."""
    return min(traces, key=lambda trace: (trace.x - x) ** 2 + (trace.y - y) ** 2)


def site_traces(traces: list[TraceData], x: float, y: float) -> list[TraceData]:
    """Return every strike recorded at the site closest to (x, y)."""
    site = trace_nearest(traces, x, y)
    return [t for t in traces if (t.x, t.y) == (site.x, site.y)]


# 2. Load
def load() -> list[TraceData]:
    """Read the coordinate catalog and the trace files it lists."""
    # The catalog is a CSV or Excel table with columns filename, x, y: one
    # row per record, repeated strikes at a site sharing its coordinates.
    # Extensions in the filename column are optional.
    names, x, y = read_coordinates_from_catalog(str(DATA_DIR / "catalog.csv"))

    traces = load_traces(
        file_type="txt",  # 'txt' (ZBL and plain tables) or 'sgy' (IDS SEG-Y)
        directory=str(DATA_DIR),
        channel=1,  # 1 = first signal column; a list reads several channels
        time_scaling=1.0,  # 0.001 for exports whose time column is in ms
        filenames_with_coordinates=names,
        x_position=x,
        y_position=y,
    )
    # Without a catalog, load_traces reads every file in the directory and
    # leaves the coordinates unset - enough for stages 3, 4, 7 and 8, but
    # not for the per-site averaging or the maps.
    print(f"{len(traces)} traces loaded from {DATA_DIR.name}/")
    return traces


# 3. Preprocess
def preprocess(traces: list[TraceData]) -> list[TraceData]:
    """Condition the records and prepare their spectra."""
    return preprocess_traces(
        traces,
        # The simulated traces start with a negative swing; the flip only
        # turns the plotted first arrival upward, every attribute is
        # sign-invariant.
        polarity=-1,
        # The 7 ms records give 142 Hz FFT bins; padding with two extra
        # record lengths interpolates the spectrum to 47 Hz bins. It adds no
        # resolution, but the spectral attributes are computed on the denser
        # bins, so compare surveys processed with the same padding.
        pad_zeros_factor=2,
        # Subtract the level recorded before the impact (on by default; the
        # modeled records are already centered, field records rarely are).
        remove_offset=True,
        # Options this dataset does not need, with the values a field survey
        # would typically use:
        #
        # downsample=4,           # keep every 4th sample (no anti-alias
        #                         # filter: the filter below runs afterwards)
        # time_limit=0.05,        # keep the first 50 ms of each record
        # detrend_method="highpass",   # remove drift with the impact-aware
        #                              # baseline, after the offset step
        # detrend_corner_hz=30.0,      # (needs records of 50 ms or more,
        #                              # so not these 7 ms traces)
        # integrate_times=1,      # accelerometer records -> velocity
        # shift_to_first_peak=True,    # put the first peak at t = 0
        # amplification_factor=50.0,   # exponential gain from the peak on
        # peak_params=(2, (0.55, 2.0), 2),  # find_peaks distance,
        #                                   # prominence and width, used by
        #                                   # the two options above
        filter_options=None,  # see filter_demo() for the filtered variant
    )


def filter_demo(traces: list[TraceData]) -> list[TraceData]:
    """Preprocess the same traces through a band-pass, for comparison.

    ``preprocess_traces`` keeps the unfiltered signal and spectrum next to
    the filtered ones (``signal_norm_pre``), so ``plot_pre_filtered=True``
    draws both and the filter's effect stays visible.
    """
    bandpass = ButterworthFilterConfig(
        min_frequency_hz=200.0,  # omit for a low-pass
        max_frequency_hz=4000.0,  # omit for a high-pass
        order=6,
    )
    return preprocess_traces(traces, polarity=-1, pad_zeros_factor=2, filter_options=bandpass)


# 4. Attributes
def attributes(traces: list[TraceData], attrs: list[str]) -> None:
    """Calculate every attribute on every trace, in place."""
    calculate_signal_parameters(
        traces,
        attrs=attrs,  # or 'all' for every supported attribute
        use_power_spectrum=True,  # spectral attributes from |S|^2, not |S|
        freq_range=(0.0, 5000.0),  # band the spectral attributes look at
        # Bins below 5 % of the peak do not count for the centroid.
        centroid_threshold=0.05,
        void_index_params=VOID_INDEX_BANDS,
        sampen_params=ENTROPY_PARAMS,
        mse_params={**ENTROPY_PARAMS, "scale": MSE_SCALES},
        # name_suffix="_raw",   # store as 'peak_frequency_raw' etc., so two
        #                       # processing variants can live on one trace
    )


# 5. Averaging
def average(traces: list[TraceData], attrs: list[str]) -> None:
    """Combine the strikes recorded at each site."""
    average_attributes(
        traces,
        attrs=attrs,
        # Both statistics are stored; the maps below choose between them
        # with plot_median_averaged_attributes.
        average_algorithm=["median", "mean"],
        # Uncertainty of the central value: 'mad' (median absolute
        # deviation, robust and the default) or 'std'.
        error_algorithm="mad",
    )


# 6. Reading results off the objects
def report_sites(traces: list[TraceData], attrs: list[str], output_dir: Path) -> None:
    """Write a small text summary straight from the TraceData objects.

    Everything the plots and exports use is available as plain attributes,
    so results can be taken into any other analysis without going through a
    file. ``attributes`` holds one value per trace, ``attributes_avg`` the
    per-site statistics (shared by the traces of that site).
    """
    lines: list[str] = []
    for label, (x, y) in SAMPLE_SITES.items():
        strikes = site_traces(traces, x, y)
        first = strikes[0]
        lines.append(f"{label}: site ({first.x:.3f}, {first.y:.3f}) m, {len(strikes)} strikes")
        lines.append(
            f"  signal: {first.signal.size} samples, "
            f"dt = {(first.t[1] - first.t[0]) * 1e6:.0f} us, "
            f"spectrum: {first.freqs.size} bins up to {first.freqs[-1]:.0f} Hz"
        )
        for name in attrs:
            if name == "MSE":
                # MSE stores (mean over the scales, per-scale details); the
                # details also carry the dimension, tolerance and the
                # matched-pair counts behind each value.
                value, details = first.attributes["MSE"]
                per_scale = ", ".join(
                    f"tau={int(s)}: {v:.3f}"
                    for s, v in zip(details["Scale"], details["Value"], strict=True)
                )
                lines.append(f"  {'MSE':<32} mean {value:>10.4g} nats ({per_scale})")
                continue
            stats = first.attributes_avg[name]
            lines.append(
                f"  {name:<32} trace {first.attributes[name]:>10.4g} | "
                f"site median {stats['median']:>10.4g} "
                f"+- {stats.get('median_error', float('nan')):.2g} "
                f"(n = {stats['n_used']})"
            )
        lines.append("")

    text = "\n".join(lines)
    (output_dir / "sample_sites.txt").write_text(text, encoding="utf-8")
    print(text)


# 7. Export
def export_tables(traces: list[TraceData], attrs: list[str], output_dir: Path) -> None:
    """Write the attribute tables, as text and as Excel."""
    export_tracedata(
        traces,
        mode=["attributes", "attributes_avg"],  # one row per trace / per site
        attributes=attrs,  # explicit names also set the column order
        # Adds MSE_scale1... columns; None when MSE was not calculated.
        MSE_scales=MSE_SCALES if "MSE" in attrs else None,
        output_prefix=PROJECT,
        output_dir=output_dir,
        float_decimals=6,  # 16 for a lossless round trip
        include_filepath=False,
        sep=",",  # delimiter of the .dat files
    )
    export_tracedata(
        traces,
        mode="attributes_avg",
        attributes=attrs,
        MSE_scales=MSE_SCALES if "MSE" in attrs else None,
        output_prefix=PROJECT,
        output_dir=output_dir,
        to_excel=True,  # .xlsx at full precision
    )


def export_signals(traces: list[TraceData], output_dir: Path) -> None:
    """Write the waveforms and spectra of the sample sites.

    The 'signals' and 'spectra' modes write one row per sample and per
    frequency bin, so they are usually restricted to the records actually
    being examined rather than run over a whole survey.
    """
    selected = [trace_nearest(traces, x, y) for x, y in SAMPLE_SITES.values()]
    export_tracedata(
        selected,
        mode=["signals", "spectra"],
        output_prefix=f"{PROJECT}_samples",
        output_dir=output_dir,
    )


# 8. Trace figures
def figures_traces(traces: list[TraceData], attrs: list[str], output_dir: Path, show: bool) -> None:
    """One figure per trace: signal, spectrum, attribute table, MSE curve."""
    selected = [trace_nearest(traces, x, y) for x, y in SAMPLE_SITES.values()]
    plot_traces(
        selected,
        TracePlotConfig(
            attrs_to_plot=attrs,  # 'all', a list, or None for no table
            time_in_ms=True,
            time_limits=(0.0, 7.0),  # in the units of the time axis above
            spectrum_type="power",  # 'amplitude' or 'power'
            use_db=False,  # dB relative to each curve's own maximum
            spectrum_freq_lim=(0.0, 5000.0),
            log_freq=False,
            freq_lines=[200.0, 800.0],  # mark the void-index bands
            mse_log_y="MSE" in attrs,  # log y-axis on the MSE panel
            # One set of y-limits over all the figures, so that the two sites
            # can be compared directly.
            unify_ylims=True,
            panel_size=(8, 2),
            output_dir=output_dir / "traces",
            dpi=150,
            show=show,
            style={
                "panel_hspace": 0.25,
                "save_format": "png",  # 'pdf' and 'svg' also work
                "subplots_adjust": {"top": 0.94, "bottom": 0.12, "left": 0.1, "right": 0.95},
            },
        ),
    )


def figures_filter(traces: list[TraceData], output_dir: Path, show: bool) -> None:
    """The same two traces band-passed, drawn over the unfiltered ones."""
    selected = [trace_nearest(traces, x, y) for x, y in SAMPLE_SITES.values()]
    plot_traces(
        filter_demo(selected),
        TracePlotConfig(
            attrs_to_plot=None,  # signal and spectrum only
            time_in_ms=True,
            spectrum_type="power",
            use_db=True,  # decibels show the rejected bands clearly
            spectrum_freq_lim=(50.0, 8000.0),  # a log axis cannot start at 0
            log_freq=True,
            plot_pre_filtered=True,  # dashed: the unfiltered version
            freq_lines=[200.0, 4000.0],  # the passband edges
            output_dir=output_dir / "filtered",
            dpi=150,
            show=show,
        ),
    )


# 9. Maps
def figures_maps(traces: list[TraceData], attrs: list[str], output_dir: Path, show: bool) -> None:
    """One interpolated map per attribute, over the whole slab."""
    plot_maps(
        traces,
        MapPlotConfig(
            attrs_to_plot=attrs,
            # One map per scale as well; None when MSE was not calculated.
            MSE_scales=MSE_SCALES if "MSE" in attrs else None,
            # Per-site statistics rather than individual strikes.
            averaged_attribute_values=True,
            plot_median_averaged_attributes=True,  # median, not mean
            plot_type="map",  # interpolated surface plus site markers
            spacing=0.01,  # grid cell, in coordinate units
            # Spline regularization: raise it if the surface overshoots
            # between the sites.
            damping=1e-4,
            # buffer extends the 2.7 x 1.8 m sensor grid to the real slab
            # outline; mask_distance would instead blank the cells too far
            # from any site.
            buffer=0.05,
            mask_distance=None,
            # The site uncertainties are not used as weights: the strikes at
            # a site are different source positions, not repeats.
            use_weights=False,
            show_contours=True,
            colorscale="slabnirt",  # or any Matplotlib colormap name
            # 'mad-scaled' centers the color bar on the survey median and
            # steps it in units of 1.4826 x MAD, so the maps show anomalies
            # rather than values and stay comparable between attributes.
            color_norm="mad-scaled",
            n_scale_levels=5,
            coordinate_units="m",
            external_polygons=[VOID_OUTLINE],
            panel_size=(8, 4),
            output_prefix=PROJECT,
            output_dir=output_dir / "maps",
            dpi=150,
            show=show,
            style={"marker_size": 20},
        ),
    )


def figures_map_variants(traces: list[TraceData], output_dir: Path, show: bool) -> None:
    """Three more views of the void index: sites, values, spread."""
    common = dict(
        attrs_to_plot=["void_index"],
        spacing=0.01,
        buffer=0.05,
        external_polygons=[VOID_OUTLINE],
        output_dir=output_dir / "maps_variants",
        dpi=150,
        show=show,
        style={"marker_size": 20},
    )
    # Sites only, no interpolation: shows where data exist.
    plot_maps(
        traces,
        MapPlotConfig(plot_type="scatter", output_prefix=f"{PROJECT}_scatter", **common),
    )
    # Absolute color scale, from the smallest to the largest site value,
    # with every site labeled by its representing record.
    plot_maps(
        traces,
        MapPlotConfig(
            # The alternatives are 'mad', 'mad-scaled', 'mean-std' and
            # 'median-std'.
            color_norm="absolute",
            add_labels=True,
            xlim=(-1.4, 1.4),
            ylim=(-0.95, 0.95),
            output_prefix=f"{PROJECT}_absolute",
            **common,
        ),
    )
    # Two half-filled markers per site: the spread over its strikes.
    plot_maps(
        traces,
        MapPlotConfig(
            plot_error="minmax",  # or 'error' for +- the MAD error
            output_prefix=f"{PROJECT}_spread",
            **common,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="slab_model_output",
        type=Path,
        help="directory for the tables, figures and log (default: slab_model_output)",
    )
    parser.add_argument("--show", action="store_true", help="display the figures as well")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the multiscale entropy and the extra figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Logging: progress messages and the package's warnings go to the
    # console and to this file.
    configure_logging(report_file=output_dir / "run.log")

    attrs = ATTRIBUTES if args.fast else [*ATTRIBUTES, "MSE"]
    extras = not args.fast

    traces = load()
    processed = preprocess(traces)
    print(f"{len(processed)} traces preprocessed.")

    attributes(processed, attrs)
    average(processed, attrs)
    report_sites(processed, attrs, output_dir)

    if EXPORT_TABLES:
        export_tables(processed, attrs, output_dir)
    if EXPORT_SIGNALS and extras:
        export_signals(processed, output_dir)
    if PLOT_TRACES:
        figures_traces(processed, attrs, output_dir, args.show)
    if PLOT_FILTER_DEMO and extras:
        figures_filter(traces, output_dir, args.show)
    if PLOT_MAPS:
        figures_maps(processed, attrs, output_dir, args.show)
    if PLOT_MAP_VARIANTS and extras:
        figures_map_variants(processed, output_dir, args.show)

    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

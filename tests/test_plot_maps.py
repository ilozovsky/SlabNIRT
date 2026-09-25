"""What plot_maps draws: the values handed to the spline, the surface, the color scale, overlays.

The interpolation oracles use an analytic plane and brute-force distances.
Shapefile overlays use synthetic GeoDataFrames in place of geopandas' reader,
so no file is read. All traces are synthetic with declared parameters.
"""

from __future__ import annotations

import builtins
import warnings
from importlib import import_module
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
import verde as vd
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Wedge
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon

from slabnirt import MapPlotConfig, TraceData, average_attributes, plot_maps
from slabnirt.plot_maps import build_cmap, plot_external_polygons
from slabnirt.style import style_defaults

plot_maps_module = import_module("slabnirt.plot_maps")
MAP_STYLE = style_defaults("maps")
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="Could not detect GDAL data files.*", category=RuntimeWarning
    )
    gpd = import_module("geopandas")
    import_module("pyogrio.errors")  # loads GDAL, which may warn about its data files
# plot_maps_module.plt is matplotlib.pyplot itself, so disabling its close()
# for a capture disables it everywhere; keep the real one for cleaning up.
_close_figures = plt.close


@pytest.fixture(autouse=True)
def _close_every_figure():
    yield
    _close_figures("all")


def _stats(value: float, error: float = 0.1) -> dict[str, float | int]:
    return {
        "n_used": 3,
        "n_dropped": 0,
        "min": value - error,
        "max": value + error,
        "mean": value,
        "mean_error": error,
        "median": value,
        "median_error": error,
    }


@pytest.fixture
def site_trace(synthetic_trace):
    """Return a factory for a trace at (x, y) with raw or averaged attributes."""

    def make(
        name: str,
        x: float | None,
        y: float | None,
        *,
        attributes: dict[str, Any] | None = None,
        averaged: dict[str, Any] | None = None,
    ) -> TraceData:
        trace = synthetic_trace([0.0, 1.0, -0.5, 0.25], name=name)
        trace.x = x
        trace.y = y
        trace.attributes = attributes
        trace.attributes_avg = averaged
        return trace

    return make


def _square(site_trace, values=(1.0, 2.0, 3.0, 4.0), errors=None, name="site") -> list[TraceData]:
    """Four averaged sites on the unit square, one 'feature' each."""
    errors = [0.1] * len(values) if errors is None else errors
    return [
        site_trace(
            f"{name}{index}",
            float(index % 2),
            float(index // 2),
            averaged={"feature": _stats(value, error)},
        )
        for index, (value, error) in enumerate(zip(values, errors, strict=True))
    ]


def _config(**overrides) -> MapPlotConfig:
    """A scatter of 'feature' with weights off and no window; tests override."""
    fields = {
        "attrs_to_plot": ["feature"],
        "plot_type": "scatter",
        "use_weights": False,
        "show": False,
    }
    fields.update(overrides)
    return MapPlotConfig(**fields)


def _figures(monkeypatch, traces, config: MapPlotConfig) -> list:
    """Run plot_maps with figure closing disabled and return its figures in order."""
    _close_figures("all")
    monkeypatch.setattr(plot_maps_module.plt, "close", lambda *args, **kwargs: None)
    plot_maps(traces, config)
    return [plt.figure(number) for number in plt.get_fignums()]


def _capture_fit(monkeypatch) -> dict[str, Any]:
    """Record the coordinates, data and weights of every verde spline fit."""
    observed: dict[str, Any] = {"calls": []}
    original_fit = vd.Spline.fit

    def capture(self, coordinates, data, weights=None):
        observed["calls"].append(
            {
                "x": np.asarray(coordinates[0], dtype=float),
                "y": np.asarray(coordinates[1], dtype=float),
                "data": np.asarray(data, dtype=float),
                "weights": None if weights is None else np.asarray(weights, dtype=float),
            }
        )
        return original_fit(self, coordinates, data, weights=weights)

    monkeypatch.setattr(vd.Spline, "fit", capture)
    return observed


def _capture_surface(monkeypatch) -> dict[str, np.ndarray]:
    """Record the first pcolormesh call of a run: the map surface (the color bar draws one too)."""
    surface: dict[str, np.ndarray] = {}
    original = Axes.pcolormesh

    def capture(self, xi, yi, z, *args, **kwargs):
        if "z" not in surface:
            surface.update(xi=np.asarray(xi), yi=np.asarray(yi), z=np.asarray(z))
        return original(self, xi, yi, z, *args, **kwargs)

    monkeypatch.setattr(Axes, "pcolormesh", capture)
    return surface


def test_site_identity_is_shared_with_average_attributes(monkeypatch, site_trace) -> None:
    traces = []
    for name, x, value in (
        ("site0_repeat0", 0.0, 1.0),
        ("site0_repeat1", 0.0000004, 3.0),
        ("site1", 1.0, 5.0),
        ("site2", 2.0, 7.0),
    ):
        traces.append(site_trace(name, x, 0.0, attributes={"feature": value}))
    average_attributes(traces, attrs=["feature"])
    assert traces[0].attributes_avg is traces[1].attributes_avg

    [figure] = _figures(monkeypatch, traces, _config())
    offsets = figure.axes[0].collections[0].get_offsets().data
    np.testing.assert_array_equal(offsets[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(figure.axes[0].collections[0].get_array(), [2.0, 5.0, 7.0])


def test_weights_are_the_inverse_variance_of_the_selected_statistic(
    monkeypatch, site_trace
) -> None:
    traces = _square(site_trace, errors=(0.1, 0.2, 0.25, 0.5))
    for trace in traces:
        trace.attributes_avg["feature"]["mean_error"] = 0.5
    observed = _capture_fit(monkeypatch)

    plot_maps(traces, _config(plot_type="map", use_weights=True, spacing=0.5))
    plot_maps(
        traces,
        _config(
            plot_type="map", use_weights=True, spacing=0.5, plot_median_averaged_attributes=False
        ),
    )

    median_fit, mean_fit = observed["calls"]
    np.testing.assert_allclose(median_fit["weights"], [100.0, 25.0, 16.0, 4.0])
    np.testing.assert_allclose(mean_fit["weights"], [4.0, 4.0, 4.0, 4.0])


def _plane_sites(site_trace, n: int = 4) -> list[TraceData]:
    """n x n sites at unit spacing carrying the plane 1 + 2x + 3y."""
    return [
        site_trace(
            f"p{index}",
            float(index % n),
            float(index // n),
            averaged={"plane": _stats(1.0 + 2.0 * (index % n) + 3.0 * (index // n))},
        )
        for index in range(n * n)
    ]


@pytest.mark.parametrize(("damping", "tolerance"), [(1e-8, 1e-6), (1e-4, 2e-3)])
def test_surface_reproduces_the_site_values_within_the_damping(
    monkeypatch, site_trace, damping: float, tolerance: float
) -> None:
    """Analytic plane: the spline passes through the sites as closely as the damping allows."""
    sites = _plane_sites(site_trace)
    surface = _capture_surface(monkeypatch)

    plot_maps(
        sites,
        _config(attrs_to_plot=["plane"], plot_type="map", spacing=0.5, buffer=1.0, damping=damping),
    )

    xi, yi, z = surface["xi"], surface["yi"], surface["z"]
    assert z.shape == (yi.size, xi.size)
    at_sites = [
        z[np.argmin(np.abs(yi - trace.y)), np.argmin(np.abs(xi - trace.x))]
        - trace.attributes_avg["plane"]["median"]
        for trace in sites
    ]
    assert np.max(np.abs(at_sites)) < tolerance


def test_grid_covers_the_site_bounds_plus_buffer_at_the_adjusted_spacing(
    monkeypatch, site_trace
) -> None:
    """verde keeps the region and adjusts the spacing: 5 / 0.3 rounds to 17 cells."""
    surface = _capture_surface(monkeypatch)

    plot_maps(
        _plane_sites(site_trace),
        _config(attrs_to_plot=["plane"], plot_type="map", spacing=0.3, buffer=1.0),
    )

    for axis in (surface["xi"], surface["yi"]):
        assert (axis[0], axis[-1]) == (-1.0, 4.0)
        assert axis.size == 18
        np.testing.assert_allclose(np.diff(axis), 5.0 / 17.0)
    assert surface["z"].shape == (18, 18)
    assert np.isfinite(surface["z"]).all()


@pytest.mark.parametrize("mask_distance", [0.55, 1.1])
def test_mask_hides_cells_farther_than_the_distance_from_every_site(
    monkeypatch, site_trace, mask_distance: float
) -> None:
    sites = _square(site_trace)
    surface = _capture_surface(monkeypatch)

    plot_maps(
        sites,
        _config(plot_type="map", spacing=0.25, buffer=1.0, mask_distance=mask_distance),
    )

    xs = np.array([trace.x for trace in sites])
    ys = np.array([trace.y for trace in sites])
    grid_x, grid_y = np.meshgrid(surface["xi"], surface["yi"])
    nearest = np.min(
        np.hypot(grid_x[..., None] - xs, grid_y[..., None] - ys),
        axis=-1,
    )
    np.testing.assert_array_equal(np.isnan(surface["z"]), nearest > mask_distance)
    assert 0 < np.isnan(surface["z"]).sum() < surface["z"].size


def test_robust_scale_ticks_at_whole_scale_units_and_the_three_label_rules() -> None:
    """Median 0 and MAD 1 throughout. With fewer than 7 ticks every tick is
    labeled, with 7 to 11 the odd multiples, with 12 or more the multiples
    3, 7, 11, ..."""
    cmap = plt.get_cmap("viridis")
    minus = "\N{MINUS SIGN}"

    _cmap, norm, ticks, labels = build_cmap(cmap, [-2.0, -1.0, 0.0, 1.0, 2.0], color_norm="mad")
    assert (norm.vmin, norm.vmax) == (-2.0, 2.0)
    np.testing.assert_array_equal(ticks, [-2.0, -1.0, 0.0, 1.0, 2.0])
    assert labels == [
        f"{minus}2MAD\n(-2.00)",
        f"{minus}1MAD\n(-1.00)",
        "med",
        "+1MAD\n(1.00)",
        "+2MAD\n(2.00)",
    ]

    _cmap, _norm, ticks, labels = build_cmap(cmap, [-4.0, -1.0, 0.0, 1.0, 4.0], color_norm="mad")
    np.testing.assert_array_equal(ticks, np.arange(-4.0, 5.0))
    assert [label for label in labels if label] == [
        f"{minus}3MAD\n(-3.00)",
        f"{minus}1MAD\n(-1.00)",
        "med",
        "+1MAD\n(1.00)",
        "+3MAD\n(3.00)",
    ]

    _cmap, _norm, ticks, labels = build_cmap(
        cmap, [-10.0, -1.0, -1.0, 0.0, 1.0, 1.0, 10.0], color_norm="mad"
    )
    np.testing.assert_array_equal(ticks, np.arange(-10.0, 11.0))
    assert [(tick, label) for tick, label in zip(ticks, labels, strict=True) if label] == [
        (-7.0, f"{minus}7MAD\n(-7.00)"),
        (-3.0, f"{minus}3MAD\n(-3.00)"),
        (0.0, "med"),
        (3.0, "+3MAD\n(3.00)"),
        (7.0, "+7MAD\n(7.00)"),
    ]


def test_color_bar_matches_the_map_and_the_saved_file_keeps_every_label(
    monkeypatch, site_trace, tmp_path
) -> None:
    """Square survey: bar at the right, as tall as the map. 20 x 1 strip:
    bar below, as wide as the map. Neither saved file is cut at its edges."""
    strip = [
        site_trace(f"strip{i}", float(x), float(y), averaged={"feature": _stats(x + y)})
        for i, (x, y) in enumerate([(0, 0), (20, 0), (0, 1), (20, 1)])
    ]
    for traces, prefix in [(_square(site_trace), "square"), (strip, "strip")]:
        (fig,) = _figures(monkeypatch, traces, _config(output_dir=tmp_path, output_prefix=prefix))
        fig.canvas.draw()
        map_box, bar_box = (ax.get_window_extent() for ax in fig.axes)
        if prefix == "square":
            assert bar_box.x0 > map_box.x1
            assert bar_box.height == pytest.approx(map_box.height, abs=1)
        else:
            assert bar_box.y1 < map_box.y0
            assert bar_box.width == pytest.approx(map_box.width, abs=1)

        image = plt.imread(tmp_path / f"{prefix}_scatter_feature.png")[..., :3]
        border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
        assert (border > 0.97).all()


def test_robust_scale_labels_stay_few_when_an_outlier_spans_many_units() -> None:
    """Median 0, MAD 1 and one site at 67: 69 ticks, labeled every 10."""
    cmap = plt.get_cmap("viridis")
    data = [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 67.0]

    _cmap, _norm, ticks, labels = build_cmap(cmap, data, color_norm="mad")

    np.testing.assert_array_equal(ticks, np.arange(-1.0, 68.0))
    labeled = [tick for tick, label in zip(ticks, labels, strict=True) if label]
    assert labeled == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


@pytest.fixture
def messages(monkeypatch) -> list[str]:
    """Collect the diagnostics plot_maps sends to the package logger."""
    collected: list[str] = []
    monkeypatch.setattr(plot_maps_module, "log_warning", collected.append)
    return collected


def _frame(*geometries):
    return gpd.GeoDataFrame(geometry=list(geometries))


def test_polyline_circle_and_ring_overlays_with_fills_and_labels() -> None:
    """The four circles are the filled disc, its outline and the ring's two edges."""
    fig, ax = plt.subplots()
    overlays = [
        {
            "x": [0.0, 1.0, 0.0],
            "y": [0.0, 0.0, 1.0],
            "fill": {"color": "red", "alpha": 0.4, "zorder": 2},
            "label": "polygon",
        },
        {
            "circle": {"center": (2.0, 2.0), "radius": 0.5},
            "fill": {"color": "blue", "alpha": 0.3},
            "label": "circle",
        },
        {
            "ring": {"center": (4.0, 4.0), "r_inner": 0.25, "r_outer": 0.75},
            "fill": {"color": "green", "alpha": 0.2},
            "label": "ring",
        },
    ]

    plot_external_polygons(ax, overlays, MAP_STYLE)

    [outline] = ax.lines
    np.testing.assert_array_equal(outline.get_xdata(), [0.0, 1.0, 0.0])
    assert outline.get_linestyle() == "--" and outline.get_color() == "#C81D25"
    fills = [patch for patch in ax.patches if isinstance(patch, MplPolygon)]
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    wedges = [patch for patch in ax.patches if isinstance(patch, Wedge)]
    assert len(fills) == 1 and fills[0].get_alpha() == 0.4 and fills[0].get_zorder() == 2
    assert len(circles) == 4
    assert sorted(circle.get_radius() for circle in circles) == [0.25, 0.5, 0.5, 0.75]
    assert len(wedges) == 1
    assert wedges[0].width == pytest.approx(0.5) and wedges[0].get_alpha() == 0.2
    assert [text.get_text() for text in ax.texts] == ["polygon", "circle", "ring"]
    assert [text.get_position() for text in ax.texts] == [(0.0, 0.0), (2.0, 2.5), (4.0, 4.75)]
    assert all(text.get_fontsize() == MAP_STYLE["label_font_size"] for text in ax.texts)


@pytest.mark.parametrize(
    ("geometry", "expected_lines"),
    [
        (Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), 1),
        (
            MultiPolygon(
                [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                    Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
                ]
            ),
            2,
        ),
    ],
    ids=["polygon", "multipolygon"],
)
def test_shapefile_overlay_draws_every_polygon_part(
    monkeypatch, geometry, expected_lines: int
) -> None:
    """Every part of a MultiPolygon is drawn."""
    monkeypatch.setattr(gpd, "read_file", lambda _: _frame(geometry))
    fig, ax = plt.subplots()

    plot_external_polygons(ax, [{"shp": "synthetic.shp"}], MAP_STYLE)

    assert len(ax.lines) == expected_lines


def test_missing_geopandas_is_reported_and_the_overlay_skipped(monkeypatch, messages) -> None:
    original_import = builtins.__import__

    def import_without_geopandas(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "geopandas":
            raise ModuleNotFoundError("geopandas is unavailable")
        return original_import(name, *args, **kwargs)

    fig, ax = plt.subplots()
    monkeypatch.setattr(builtins, "__import__", import_without_geopandas)

    plot_external_polygons(
        ax, [{"shp": "missing-dependency.shp"}, {"x": [0.0, 1.0], "y": [0.0, 1.0]}], MAP_STYLE
    )

    assert messages == ["missing-dependency.shp: shapefile not drawn (geopandas is unavailable)."]
    assert len(ax.lines) == 1

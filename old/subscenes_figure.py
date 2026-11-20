"""The retired Fig. 3: scene context plus two 20 m subscenes -> figures/fig3_subscenes.png.

Its successor, the 20 m insets figure, is retired too. The window search and lidar helpers
this draws on live beside it in old/fig3_20m.py.

    conda run -n nival python old/subscenes_figure.py
"""
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nival import paper_figures, paths, retrieval   # noqa: E402
from nival.paper_figures import _save_print   # noqa: E402
import unwrap_20m   # noqa: E402  (old/unwrap_20m.py)
from fig3_20m import (BURNED, SUB_WIN_X, SUB_WIN_Y, _fraction_on_grid,   # noqa: E402
                      _lidar_on, subscene_windows)

SUB_LIM = 55.0                 # mm SWE, diverging limit on the NISAR inset panels
SUB_LIM_HS = 30.0              # cm depth, diverging limit on the lidar inset panels


def subscenes_figure():
    """Fig. 3: scene context masked to the lidar footprint, plus two 20 m subscenes."""
    ds = xr.open_dataset(unwrap_20m.product())
    dswe, coh = ds["dswe"].values, ds["coherence"].values
    xs, ys = ds.x.values, ds.y.values
    depth = _lidar_on(xs, ys)                       # lidar dHS in metres
    hs_cm = depth * 100                            # shown as measured, no density assumed
    lidar = depth * retrieval.SNOW_DENSITY / retrieval.WATER_DENSITY * 1000
    # the product's own 20 m incidence angle, not the 80 m field upsampled: theta enters
    # Eq. 1 as theta^2.5, and the interpolation also NaNs the edge of the AOI
    lia = ds["lia"].values
    valid = np.isfinite(dswe) & np.isfinite(lidar)

    reference = xr.DataArray(np.zeros_like(dswe), coords={"y": ds.y, "x": ds.x},
                             dims=("y", "x")).rio.write_crs(paths.EPSG)
    burn = _fraction_on_grid(paths.MTBS, lambda a: a.isin(BURNED), reference)
    forest = _fraction_on_grid(paths.WORLDCOVER, lambda a: a == 10, reference)
    regions = subscene_windows(dswe, lidar, lia, burn, xs, ys)

    # the inset panels are 1.5 x 3 km on equal axes, so their columns are sized near that
    # 1:2 aspect; a wider column just centres the panel and leaves dead space either side
    fig = plt.figure(figsize=(9.9, 9.6), constrained_layout=True)
    if hasattr(fig, "get_layout_engine"):
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.0, hspace=0.02)
    gs = fig.add_gridspec(6, 3, width_ratios=[1.27, 1.0, 1.05], wspace=0.02)
    scene = dswe - np.nanmean(dswe[valid])
    rr, cc = np.where(valid.any(axis=1))[0], np.where(valid.any(axis=0))[0]
    crop = (slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1))
    extent = [xs[cc[0]], xs[cc[-1]], ys[rr[-1]], ys[rr[0]]]
    corners = [c for _, c in regions]

    def context(ax, field, cmap, vmin, vmax, label):
        im = ax.imshow(field, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        fig.colorbar(im, ax=ax, shrink=0.75, label=label)
        for k, (y0, x0) in enumerate(corners):
            ax.add_patch(plt.Rectangle((xs[x0] - 10.0, ys[y0 + SUB_WIN_Y - 1] - 10.0),
                                       SUB_WIN_X * 20.0, SUB_WIN_Y * 20.0,
                                       fill=False, ec="k", lw=1.4))
            ax.text(xs[x0] + 40, ys[y0] - 40, str(k + 1), fontsize=12, fontweight="bold",
                    ha="left", va="top", color="k", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    seen = np.isfinite(lidar)
    context(fig.add_subplot(gs[0:2, 0]), np.where(valid, scene, np.nan)[crop], "RdBu_r",
            -SUB_LIM, SUB_LIM, "mm SWE from scene mean")
    fig.axes[-2].set_title("dSWE departure from scene mean", fontsize=10)
    context(fig.add_subplot(gs[2:4, 0]), np.where(seen, lia, np.nan)[crop], "cividis",
            np.nanpercentile(lia[seen], 2), np.nanpercentile(lia[seen], 98),
            "incidence angle (deg)")
    fig.axes[-2].set_title("local incidence angle", fontsize=10)
    context(fig.add_subplot(gs[4:6, 0]), np.where(seen, forest, np.nan)[crop], "Greens", 0, 1,
            "canopy fraction")
    ax_fcf = fig.axes[-2]
    rings, polygon = paper_figures.pioneer_perimeter()
    for part in (polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]):
        if not part.is_empty:
            ax_fcf.add_patch(MplPolygon(np.array(part.exterior.coords), closed=True,
                                        facecolor=paper_figures.FIRE_C, edgecolor="none",
                                        alpha=0.16, zorder=1.5))
    for ring in rings:
        ax_fcf.plot(ring[:, 0], ring[:, 1], color=paper_figures.FIRE_C, lw=1.6, ls="--",
                    zorder=3)
    ax_fcf.set_xlim(extent[0], extent[1]); ax_fcf.set_ylim(extent[2], extent[3])
    ax_fcf.set_title("canopy fraction and burn scar", fontsize=10)

    nisar_axes, lidar_axes = [], []
    for k, (name, (y0, x0)) in enumerate(regions):
        sl = (slice(y0, y0 + SUB_WIN_Y), slice(x0, x0 + SUB_WIN_X))
        S, D = scene[sl], (lidar - np.nanmean(lidar[valid]))[sl]
        H = hs_cm[sl]
        f = np.isfinite(S) & np.isfinite(D)
        r = float(np.corrcoef(D[f], S[f])[0, 1])
        slope = np.polyfit(D[f], S[f], 1)[0]
        ir, ic = np.where(f.any(axis=1))[0], np.where(f.any(axis=0))[0]
        tight = (slice(ir[0], ir[-1] + 1), slice(ic[0], ic[-1] + 1))
        Sa = np.where(f, S - np.nanmedian(S[f]), np.nan)[tight]
        Da = np.where(f, H - np.nanmedian(H[f]), np.nan)[tight]
        ext = [xs[x0 + ic[0]], xs[x0 + ic[-1]], ys[y0 + ir[-1]], ys[y0 + ir[0]]]

        a0 = fig.add_subplot(gs[3 * k : 3 * k + 3, 1])
        a1 = fig.add_subplot(gs[3 * k : 3 * k + 3, 2])
        im_n = a0.imshow(Sa, extent=ext, cmap="RdBu_r", vmin=-SUB_LIM, vmax=SUB_LIM,
                         origin="upper")
        im_h = a1.imshow(Da, extent=ext, cmap="RdBu_r", vmin=-SUB_LIM_HS, vmax=SUB_LIM_HS,
                         origin="upper")
        nisar_axes.append(a0)
        lidar_axes.append(a1)
        if k == 0:
            a0.set_title("NISAR dSWE departure", fontsize=10)
            a1.set_title("Airborne dHS departure", fontsize=10)
        a0.set_ylabel(f"{k + 1}  {name}\n$r$ = {r:+.2f}", fontsize=10, fontweight="bold")
        for a_ in (a0, a1):
            a_.set_xticks([]); a_.set_yticks([]); a_.set_aspect("equal")
        print(f"  {k + 1} {name}: r={r:+.2f}, slope={slope:.2f}, n={int(f.sum())}, "
              f"dHS span {np.nanmin(Da):+.0f} to {np.nanmax(Da):+.0f} cm, "
              f"coh {np.nanmedian(coh[sl]):.2f}, burn {100 * np.nanmean(burn[sl]):.0f}%, "
              f"canopy {100 * np.nanmean(forest[sl]):.0f}%, LIA {np.nanmedian(lia[sl]):.0f} deg")
    fig.colorbar(im_n, ax=nisar_axes, shrink=0.45, pad=0.008,
                 label="NISAR dSWE from window median (mm)")
    fig.colorbar(im_h, ax=lidar_axes, shrink=0.45, pad=0.02,
                 label="Airborne dHS from window median (cm)")

    out = paths.FIGURES_DIR / "fig3_subscenes.png"
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_print(fig, out, 18)
    print("saved ->", out)


if __name__ == "__main__":
    subscenes_figure()

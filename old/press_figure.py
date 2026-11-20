"""Press figure: four 20 m subscenes, NISAR beside lidar, over the Mores Creek scene.

Layout follows mores_closure_testing/render_insets_20m.py: a context column on the left
and one row per subscene. Here the scene panels are the NISAR and lidar fields at the same
size, the scatter column and colour bars are dropped, and the fields come from this repo's
own product (the delivered 20 m wrapped phase unwrapped in nival.unwrap_20m).

The lidar depth is scaled to SWE by the slope fitted against NISAR in each window, as the
original did, rather than by the SNOTEL density: the true density is not known per window,
and matching the amplitudes lets the eye compare the patterns.

NOT a paper figure. The four windows scored best on correlation in an earlier sweep, and
the amplitude match is fitted rather than measured; neither is how the manuscript's Fig. 3
is built.

    conda run -n nival python src/nival/press_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cmocean
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nival import paths, retrieval, unwrap_20m   # noqa: E402
from nival.paper_figures import _lidar_on   # noqa: E402

OUT = paths.FIGURES_DIR / "NISAR_PRESS"
OUTNAME = "press_insets.png"   # overridable, so colour-map variants can sit side by side
DX = 20.0
LIDAR_DX = 3.0        # the lidar is shown at its own finer posting, NISAR stays at 20 m
CMAP = cmocean.cm.ice          # deep to pale: more snow reads brighter
#: condensed face for the titles: tall and narrow, so long labels stay inside the panel
TITLE_FONT = {"fontfamily": "Avenir Next Condensed"}
DENS = retrieval.SNOW_DENSITY / retrieval.WATER_DENSITY
#: (centre easting, centre northing, window size in m), numbered north to south
PICKS = sorted([(606327, 4868683, 3000),
                (608307, 4866523, 1000),
                (606887, 4866403, 1500),
                (608307, 4865083, 1000)], key=lambda p: -p[1])


def window(xs, ys, cx, cy, size_m):
    """Row and column slices for a square window centred on a UTM coordinate."""
    n = max(8, round(size_m / DX))
    j = int(np.clip(np.argmin(np.abs(xs - cx)) - n // 2, 0, len(xs) - n))
    i = int(np.clip(np.argmin(np.abs(ys - cy)) - n // 2, 0, len(ys) - n))
    return slice(i, i + n), slice(j, j + n)


def lidar_at(x0, x1, y0, y1, res=LIDAR_DX):
    """NIVAL depth change on a fresh grid at `res` metres over a bounding box."""
    gx = np.arange(x0, x1 + res, res)
    gy = np.arange(y0, y1 - res, -res)
    return gx, gy, _lidar_on(gx, gy) * DENS * 1000


def departure(field, mask):
    """Field minus its own median over `mask`, NaN elsewhere."""
    return np.where(mask, field - np.nanmedian(field[mask]), np.nan)


def fitted_slope(lidar, nisar, mask):
    """mm of NISAR SWE per mm of lidar-implied SWE, least squares over `mask`."""
    return float(np.polyfit(lidar[mask], nisar[mask], 1)[0])


def main():
    ds = xr.open_dataset(unwrap_20m.product(filtered=True))
    xs, ys = ds.x.values, ds.y.values
    nisar = ds["dswe"].values
    lidar = _lidar_on(xs, ys) * DENS * 1000
    seen = np.isfinite(nisar) & np.isfinite(lidar)

    fig = plt.figure(figsize=(9.5, 3.0 * len(PICKS)), constrained_layout=True)
    rows = 6 * len(PICKS)      # finer grid so the context gap can be a half row
    gs = fig.add_gridspec(rows, 3, width_ratios=[1.05, 1.0, 1.0])

    # --- context column: the two scene fields, same size, stacked ---
    rr, cc = np.where(seen.any(axis=1))[0], np.where(seen.any(axis=0))[0]
    crop = (slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1))
    extent = [xs[cc[0]], xs[cc[-1]], ys[rr[-1]], ys[rr[0]]]
    # the scene panels show the real accumulation, not a departure. NISAR is anchored so
    # its scene median matches the lidar-implied SWE, the same anchor the paper uses.
    _, _, fine_scene = lidar_at(extent[0], extent[1], extent[3], extent[2])
    anchored = nisar - np.nanmedian(nisar[seen]) + np.nanmedian(lidar[seen])
    scene_nisar = np.where(seen, anchored, np.nan)[crop]   # same footprint as the lidar
    scene_lidar = fine_scene / (DENS * 1000) * 100        # back to depth change in cm
    for k, (field, title, unit) in enumerate(
            [(scene_nisar, "February 7-19 NISAR", "NISAR $\\Delta$SWE (mm)"),
             (scene_lidar, "February 7-22 Lidar", "lidar $\\Delta H_s$ (cm)")]):
        # a third of the column each, with a small gap between and even margins
        span, gap = rows // 3, rows // 12
        margin = (rows - 2 * span - gap) // 2
        top = margin + k * (span + gap)
        ax = fig.add_subplot(gs[top:top + span, 0])
        a = np.where(np.isfinite(field), field, np.nan)
        v = float(np.ceil(np.nanpercentile(a[np.isfinite(a)], 98) / 10) * 10)
        im = ax.imshow(a, extent=extent, cmap=CMAP, vmin=0, vmax=v, origin="upper")
        for n, (cx, cy, size_m) in enumerate(PICKS, start=1):
            rs, cs = window(xs, ys, cx, cy, size_m)
            ax.add_patch(plt.Rectangle((xs[cs.start], ys[rs.stop - 1]),   # rows -> y, cols -> x
                                       (cs.stop - cs.start) * DX, (rs.stop - rs.start) * DX,
                                       fill=False, ec="k", lw=1.4))
            # inside the box, so a box at the top edge does not collide with the title
            ax.text(xs[cs.start] + 3 * DX, ys[rs.start] - 3 * DX, str(n), fontsize=14,
                    fontweight="bold", ha="left", va="top", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=19, **TITLE_FONT)
        # the scene panels carry the scale for the whole column
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", location="bottom",
                            shrink=0.85, pad=0.02, aspect=28)
        cbar.set_label(unit, fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        if k == 0:
            x0 = extent[0] + 0.06 * (extent[1] - extent[0])
            y0 = extent[2] + 0.07 * (extent[3] - extent[2])
            ax.plot([x0, x0 + 1000], [y0, y0], color="k", lw=3, solid_capstyle="butt")
            ax.text(x0 + 500, y0 + 0.02 * (extent[3] - extent[2]), "1 km",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    # --- one row per subscene: NISAR then lidar, on that row's shared scale ---
    for k, (cx, cy, size_m) in enumerate(PICKS):
        rs, cs = window(xs, ys, cx, cy, size_m)
        sl = (rs, cs)
        good = seen[sl]
        slope = fitted_slope(lidar[sl], nisar[sl], good)
        a = departure(nisar[sl], good)
        v = np.nanpercentile(np.abs(a[good]), 96)
        r = float(np.corrcoef(lidar[sl][good], nisar[sl][good])[0, 1])
        ext = [xs[cs.start], xs[cs.stop - 1], ys[rs.stop - 1], ys[rs.start]]
        # r and the slope come from the matched 20 m grids; only the display is finer
        _, _, fine = lidar_at(ext[0], ext[1], ext[3], ext[2])
        b = departure(fine, np.isfinite(fine)) * slope
        for j, (field, title) in enumerate([(a, "NISAR Snow Volume Change"),
                                            (b, "Lidar Snow Depth Change")]):
            ax = fig.add_subplot(gs[6 * k:6 * k + 6, j + 1])
            ax.imshow(field, extent=ext, cmap=CMAP, vmin=-v, vmax=v, origin="upper")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if k == 0:
                ax.set_title(title, fontsize=18, **TITLE_FONT)
            if j == 0:
                ax.set_ylabel(f"{k + 1}", fontsize=16, fontweight="bold", rotation=0,
                              labelpad=14, va="center")
        print(f"  {k + 1}: {size_m} m at ({cx},{cy}), r = {r:+.2f}, slope = {slope:.2f} "
              f"({slope * retrieval.SNOW_DENSITY:.0f} kg/m3), n = {int(good.sum())}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / OUTNAME
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

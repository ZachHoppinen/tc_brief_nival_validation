"""RETIRED: the 20 m Fig. 3 and everything that hung off it. The paper's Fig. 3 is now the
SWE timeline (nival.paper_figures.swe_timeline_figure).

  insets_figure()    two scene locators + three 20 m NISAR / lidar insets -> figures/fig3_insets.png
  subscene_windows() the two 20 m subscene windows (old/subscenes_figure.py draws them)
  posting_stats()    the 80 m vs 20 m agreement block that used to be in paper_stats

All of it reads the 20 m product from old/unwrap_20m.py, which must be run first, and
insets_figure() needs the hand-drawn data/isnets/insets.shp.

    conda run -n nival python old/fig3_20m.py
"""
import sys
from pathlib import Path

from matplotlib.colors import LightSource, ListedColormap, LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nival import paths, retrieval   # noqa: E402
from nival.paper_figures import (_save_print, pioneer_perimeter)   # noqa: E402
import unwrap_20m   # noqa: E402  (old/unwrap_20m.py)


# =============================== figure 3: analysis ===============================
WINDOW = 11          # canopy-fraction (FCF) smoothing window in pixels (~440 m at the 40 m Fig-3 posting)
LOCAL_R_WINDOW = 11  # local-correlation window in pixels (sigma~110 m): ~420 m physical, held from 21 px @ 20 m
OPEN_NAMES = ["forest", "open", "burn-low", "burn-mod", "burn-high"]



# --- the two 20 m subscene windows paper_stats reports on --------------------
SUB_WIN_X, SUB_WIN_Y = 75, 150 # 1.5 km across by 3 km down, at the 20 m posting
BURNED = (2, 3, 4)             # MTBS low, moderate and high severity
FREEMAN_LL = (-115.7018528, 43.94091389)   # MET_STATIONS
STEEP_NUDGE = (-50, -50)       # from Freeman: 1 km up-slope and left
BURN_NUDGE = (12, 0)           # rows down from the searched burn window: 240 m


def _fraction_on_grid(raster_path, selector, reference):
    """Fraction of each cell meeting `selector` in a finer categorical raster."""
    src = rioxarray.open_rasterio(raster_path, masked=True).squeeze()
    hit = selector(src).astype("float32")
    hit.rio.write_crs(src.rio.crs, inplace=True)
    hit.rio.write_nodata(np.nan, inplace=True)
    return np.asarray(hit.rio.reproject(paths.EPSG)
                      .rio.reproject_match(reference, resampling=Resampling.average).values)


def _best_window(score, valid):
    """Top-left corner of the window maximising mean score, over windows at least 90% covered."""
    size = (SUB_WIN_Y, SUB_WIN_X)
    s = uniform_filter(np.where(valid, np.nan_to_num(score), 0.0), size, mode="constant")
    v = uniform_filter(valid.astype(float), size, mode="constant")
    total = np.where(v > 0.9, s, -1.0)
    for sl in (np.s_[: SUB_WIN_Y // 2, :], np.s_[-SUB_WIN_Y // 2 :, :],
               np.s_[:, : SUB_WIN_X // 2], np.s_[:, -SUB_WIN_X // 2 :]):
        total[sl] = -1.0
    cy, cx = np.unravel_index(np.argmax(total), total.shape)
    return cy - SUB_WIN_Y // 2, cx - SUB_WIN_X // 2


def _lidar_on(xs, ys):
    """NIVAL depth change on an arbitrary grid, same outlier gate as the 80 m product."""
    later = rioxarray.open_rasterio(paths.LIDAR_A22, masked=True).squeeze()
    earlier = rioxarray.open_rasterio(paths.LIDAR_A07, masked=True).squeeze()
    change = later - earlier
    change = change.where((change > -1.5) & (change < 1.5))
    # without an explicit nodata the average resampling treats every NaN as opaque and
    # discards two thirds of the lidar coverage
    change.rio.write_nodata(np.nan, inplace=True)
    reference = xr.DataArray(np.zeros((len(ys), len(xs))), coords={"y": ys, "x": xs},
                             dims=("y", "x")).rio.write_crs(paths.EPSG)
    return np.asarray(change.rio.reproject_match(reference, resampling=Resampling.average).values)


def subscene_windows(dswe, lidar, lia, burn, xs, ys):
    """The two Fig. 3 windows.

    Window 1 starts from a search for the greatest burn fraction among pixels below the
    scene median incidence angle, window 2 from the Freeman station, and both are then
    nudged by hand to sit on the terrain we wanted to show. They are placed from the burn,
    canopy and incidence-angle fields only. Neither is placed by its correlation with the
    lidar, which is the property that matters for the comparison.
    """
    valid = np.isfinite(dswe) & np.isfinite(lidar)
    low = np.isfinite(lia) & (lia < np.nanmedian(lia))
    fx, fy = Transformer.from_crs("EPSG:4326", f"EPSG:{paths.EPSG}",
                                  always_xy=True).transform(*FREEMAN_LL)
    y = int(np.clip(int(np.argmin(np.abs(ys - fy))) - SUB_WIN_Y // 2 + STEEP_NUDGE[0],
                    0, len(ys) - SUB_WIN_Y))
    x = int(np.clip(int(np.argmin(np.abs(xs - fx))) - SUB_WIN_X // 2 + STEEP_NUDGE[1],
                    0, len(xs) - SUB_WIN_X))
    by, bx = _best_window(np.where(low, burn, 0.0), valid)
    by = int(np.clip(by + BURN_NUDGE[0], 0, len(ys) - SUB_WIN_Y))
    bx = int(np.clip(bx + BURN_NUDGE[1], 0, len(xs) - SUB_WIN_X))
    return [("burned, low incidence", (by, bx)),
            ("unburned, steep incidence", (y, x))]


# =============================== figure 3: insets ================================
INSET_SHP = paths.DATA / "isnets" / "insets.shp"
#: per inset: index of the drawn box in INSET_SHP, then width, height and a nudge on
#: that box's centre, all in metres. One footprint for all three, so their local r are
#: comparable; the nudges are where the boxes were moved after inspecting the scene.
INSET_WINDOWS = [(0, 2016.0, 1400.0, +250.0, -75.0),
                 (1, 2016.0, 1400.0, 0.0, +75.0),
                 (3, 2016.0, 1400.0, +850.0, -850.0)]
INSET_CMAP = matplotlib.colormaps["RdBu_r"].copy()
INSET_CMAP.set_bad("0.82")            # nodata must not read as a pixel at the median
DENSE_CANOPY = 0.75                   # the canopy locator is a mask, not a fraction
DENSE_CMAP = ListedColormap(["0.90", "#1b5e20"])
ZLIM = 2.5                            # inset colour range, in standard deviations
LIDAR_DX = 3.0                        # lidar drawn at its own posting, NISAR stays at 20 m
N_LOCATORS = 2                        # canopy and incidence
INSET_COL_IN = 3.68                   # width of one inset column (in), sets the figure size


def _inset_halo(lw=2.5, colour="w"):
    return [pe.withStroke(linewidth=lw, foreground=colour)]


def _inset_scene():
    """The 20 m unwrapped scene, lidar on its grid, and the ancillary layers."""
    fine = xr.open_dataset(unwrap_20m.product())
    xs, ys = fine.x.values, fine.y.values
    dswe = fine["dswe"].values / 1000.0
    # the 20 m incidence, which is what unwrap_20m used in kappa for this dSWE and what
    # resolves the terrain facets the locator panel is there to show. The 46 deg quartile
    # and 50 deg threshold elsewhere belong to the local-r work on the 80 m product
    lia = fine["lia"].values
    dhs = _lidar_on(xs, ys)
    both = np.isfinite(dswe) & np.isfinite(dhs)
    density = retrieval.SNOW_DENSITY / retrieval.WATER_DENSITY
    dswe = dswe + np.nanmedian(dhs[both] * density - dswe[both])
    ref = xr.DataArray(np.zeros_like(dswe), coords={"y": fine.y, "x": fine.x},
                       dims=("y", "x")).rio.write_crs(paths.EPSG)
    burn = _fraction_on_grid(paths.MTBS, lambda a: a.isin(BURNED), ref)
    fcf = _fraction_on_grid(paths.WORLDCOVER, lambda a: a == 10, ref)
    return xs, ys, dswe, dhs, lia, both, fcf, burn, fine["coherence"].values


def _inset_centres():
    """Centre, width and height of each inset, from its drawn box plus its nudge."""
    import geopandas as gpd
    boxes = gpd.read_file(INSET_SHP).to_crs(paths.EPSG)
    out = []
    for idx, w, h, dx, dy in INSET_WINDOWS:
        b = boxes.geometry.iloc[idx].bounds
        out.append(((b[0] + b[2]) / 2 + dx, (b[1] + b[3]) / 2 + dy, w, h))
    return out


def _inset_slices(xs, ys, cx, cy, w, h):
    dx = abs(xs[1] - xs[0])
    nx, ny = int(round(w / dx)), int(round(h / dx))
    j = int(np.clip(np.argmin(np.abs(xs - cx)) - nx // 2, 0, len(xs) - nx))
    i = int(np.clip(np.argmin(np.abs(ys - cy)) - ny // 2, 0, len(ys) - ny))
    return slice(i, i + ny), slice(j, j + nx)


def insets_figure():
    """Fig. 3: canopy and incidence locators beside three 20 m NISAR/lidar inset pairs.

    Each inset is drawn as a departure from its own window median in standard
    deviations, so one colour bar describes all six panels. NISAR is standardised on
    its own 20 m field and the lidar on its 20 m field, with the result applied to a
    3 m drawing, so the finer posting adds detail rather than amplitude. NISAR is
    blanked wherever the lidar is missing, so each pair covers one set of pixels.
    """
    xs, ys, dswe, dhs, lia, both, fcf, burn, coh = _inset_scene()
    density = retrieval.SNOW_DENSITY / retrieval.WATER_DENSITY
    picks = _inset_centres()
    seen = np.isfinite(dhs)
    rr, cc = np.where(both.any(axis=1))[0], np.where(both.any(axis=0))[0]
    crop = (slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1))
    extent = [xs[cc[0]], xs[cc[-1]], ys[rr[-1]], ys[rr[0]]]
    scene_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])

    plt.rcParams.update({"font.size": 10})
    # the locator column stacks N_LOCATORS panels into the height of the inset rows, and
    # the figure width follows from a fixed inset column, so changing the number of
    # locators narrows the left column instead of shrinking the insets
    heights = [h / w for _, _, w, h in picks]
    left_ratio = sum(heights) / (N_LOCATORS * scene_aspect)
    fig = plt.figure(figsize=(INSET_COL_IN * (left_ratio + 2.0),
                              sum(heights) * INSET_COL_IN + 0.9), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[left_ratio, 2.0])
    gs_left = outer[0, 0].subgridspec(N_LOCATORS, 1)
    gs_right = outer[0, 1].subgridspec(len(picks), 2, height_ratios=heights)

    def clamp(ax):
        """Hold the panel on the swath; the fire perimeter runs well outside it."""
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    def locator(ax, field, cmap, vmin, vmax, title):
        ax.imshow(field, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        for k, (cx, cy, w, h) in enumerate(picks, start=1):
            ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                       fill=False, ec="k", lw=1.4, zorder=4))
            ax.text(cx, cy + h / 2 + 70, str(k), fontsize=11, fontweight="bold",
                    ha="center", va="bottom", color="k", zorder=5,
                    path_effects=_inset_halo())
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        clamp(ax)
        return ax

    dense = np.where(seen, (fcf > DENSE_CANOPY).astype(float), np.nan)
    ax_f = locator(fig.add_subplot(gs_left[0]), dense[crop], DENSE_CMAP, 0, 1,
                   f"canopy > {DENSE_CANOPY:.2f} and burn scar")
    _, polygon = pioneer_perimeter()
    for part in (polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]):
        if not part.is_empty:
            ax_f.fill(*part.exterior.xy, color="darkorange", alpha=0.18, lw=0, zorder=2)
            ax_f.plot(*part.exterior.xy, color="orangered", lw=1.4, ls="--", zorder=3)
    clamp(ax_f)                       # the perimeter autoscaled the panel out
    lia_lo, lia_hi = np.nanpercentile(lia[seen], [2, 98])
    # cividis runs dark blue at vmin to yellow at vmax: shallow blue, steep yellow
    locator(fig.add_subplot(gs_left[1]), np.where(seen, lia, np.nan)[crop], "cividis",
            lia_lo, lia_hi,
            f"local incidence angle\n{lia_lo:.0f}$^\\circ$ (blue) to {lia_hi:.0f}$^\\circ$ (yellow)")

    rows, inset_axes = [], []
    for k, (cx, cy, w, h) in enumerate(picks):
        rs, cs = _inset_slices(xs, ys, cx, cy, w, h)
        sub_n, sub_d, sub_l = dswe[rs, cs], dhs[rs, cs], lia[rs, cs]
        # blank NISAR where the lidar is missing, so both panels of a pair carry the
        # same nodata and every statistic below is over one set of pixels
        sub_n = np.where(np.isfinite(sub_d), sub_n, np.nan)
        g = np.isfinite(sub_n) & np.isfinite(sub_d)
        r = float(np.corrcoef(sub_n[g], sub_d[g])[0, 1])
        rows.append({"inset": k + 1, "box": INSET_WINDOWS[k][0] + 1, "w": w, "h": h,
                     "lia": float(np.nanmean(sub_l[g])),
                     "coh": float(np.nanmedian(coh[rs, cs][g])),
                     "burn": float(np.nanmean(burn[rs, cs][g])),
                     "fcf": float(np.nanmean(fcf[rs, cs][g])),
                     "r": r, "r2": r ** 2,
                     # slope of dSWE on dHS inside the window: the density the inset
                     # implies. Plain fit, the same estimator paper_stats uses scene-wide
                     "density": float(np.polyfit(sub_d[g], sub_n[g], 1)[0]) * 1000.0,
                     "dhs_cm": float(np.nanmean(sub_d[g])) * 100.0,
                     "n": int(g.sum()), "cover": float(g.mean())})

        def zscore(field, ref=None):
            """Departure from the window median, in standard deviations.

            `ref` supplies the median and spread from another field: the lidar panel is
            drawn at 3 m but standardised on its own 20 m statistics, so the finer
            posting cannot inflate its spread relative to NISAR."""
            source = field if ref is None else ref
            return (field - np.nanmedian(source)) / max(float(np.nanstd(source)), 1e-9)

        e = [xs[cs.start], xs[cs.stop - 1], ys[rs.stop - 1], ys[rs.start]]
        gx = np.arange(e[0], e[1] + LIDAR_DX, LIDAR_DX)
        gy = np.arange(e[3], e[2] - LIDAR_DX, -LIDAR_DX)
        fine_d = _lidar_on(gx, gy)
        e_fine = [gx[0], gx[-1], gy[-1], gy[0]]
        for i, (f, ref, ex, lab) in enumerate(
                ((sub_n, None, e, r"NISAR $\Delta$SWE"),
                 (fine_d, sub_d, e_fine, r"Airborne $\Delta\mathbf{H_s}$"))):
            ax = fig.add_subplot(gs_right[k, i])
            im = ax.imshow(zscore(f, ref), extent=ex, cmap=INSET_CMAP, vmin=-ZLIM,
                           vmax=ZLIM, origin="upper")
            inset_axes.append(ax)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.text(0.03, 0.95, f"{k + 1}{'ab'[i]}", transform=ax.transAxes, ha="left",
                    va="top", fontsize=12, fontweight="bold", color="0.15",
                    path_effects=_inset_halo(2.6))
            if k == 0:
                ax.set_title(lab, fontsize=11, fontweight="bold")
            if i == 0:
                x0 = e[0] + 0.05 * (e[1] - e[0]); y0 = e[2] + 0.07 * (e[3] - e[2])
                ax.plot([x0, x0 + 500], [y0, y0], color="0.15", lw=2.6,
                        solid_capstyle="butt", path_effects=_inset_halo(4.0))
                ax.text(x0 + 250, y0 + 0.02 * (e[3] - e[2]), "500 m", color="0.15",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        path_effects=_inset_halo(2.2))

    cb = fig.colorbar(im, ax=inset_axes, location="right", fraction=0.03, pad=0.02,
                      ticks=[-ZLIM, 0, ZLIM])
    cb.set_label(r"departure from window median ($\sigma$)")

    lines = ["| inset | drawn box | window (m) | mean LIA | med coh | mean fcf | mean burn | "
             "burn area | r | r2 | mean dHS | density | n | coverage |",
             "|" + "---|" * 14]
    for d in rows:
        lines.append(
            f"| {d['inset']} | {d['box']} | {d['w']:.0f} x {d['h']:.0f} | {d['lia']:.1f} deg | "
            f"{d['coh']:.2f} | "
            f"{d['fcf']:.2f} | {d['burn']:.2f} | {d['burn'] * d['w'] * d['h'] / 1e6:.2f} km2 | "
            f"{d['r']:+.2f} | {d['r2']:.2f} | {d['dhs_cm']:.1f} cm | "
            f"{d['density']:.0f} kg/m3 | {d['n']:,} | {d['cover']:.0%} |")
    table = "\n".join(lines)
    print(table)
    stats_path = paths.OUTPUTS / "fig3_inset_stats.md"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(table + "\n")

    out = paths.FIGURES_DIR / "fig3_insets.png"
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_print(fig, out, 18)
    print("saved ->", out)


def posting_stats():
    """Scene-wide agreement at the delivered 80 m posting and at the 20 m posting we unwrap.

    The operational GUNW ships the wrapped interferogram at 20 m but only unwraps at 80 m.
    Fig. 3 is drawn from the 20 m layer unwrapped here (nival.unwrap_20m). The third row
    averages that field onto the delivered product's own cell centres, so the three rows are
    the same scene at three effective resolutions.
    """

    def block_average(field, xs, ys, ox, oy):
        """Average a fine regular grid onto a coarser one by nearest-cell assignment.

        coarsen().mean() blocks do not land on the coarse product's own cell centres, so
        the two grids are matched here explicitly rather than by a nearest-neighbour interp.
        """
        iy = np.rint((oy[0] - ys) / (oy[0] - oy[1])).astype(int)
        ix = np.rint((xs - ox[0]) / (ox[1] - ox[0])).astype(int)
        keep_y, keep_x = (iy >= 0) & (iy < len(oy)), (ix >= 0) & (ix < len(ox))
        total = np.zeros((len(oy), len(ox)))
        count = np.zeros((len(oy), len(ox)))
        sub = field[np.ix_(keep_y, keep_x)]
        good = np.isfinite(sub)
        np.add.at(total, (iy[keep_y][:, None].repeat(keep_x.sum(), 1)[good],
                          ix[keep_x][None, :].repeat(keep_y.sum(), 0)[good]), sub[good])
        np.add.at(count, (iy[keep_y][:, None].repeat(keep_x.sum(), 1)[good],
                          ix[keep_x][None, :].repeat(keep_y.sum(), 0)[good]), 1.0)
        return np.where(count > 0, total / np.where(count > 0, count, 1), np.nan)

    dens = retrieval.SNOW_DENSITY / retrieval.WATER_DENSITY

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return (np.corrcoef(a[m], b[m])[0, 1], int(m.sum())) if m.sum() else (np.nan, 0)

    o = xr.open_dataset(paths.product())
    lid80 = o["dhs"].values * dens * 1000
    fine = xr.open_dataset(unwrap_20m.product())
    xs, ys = fine.x.values, fine.y.values
    lid20 = _lidar_on(xs, ys) * dens * 1000
    up = block_average(fine["dswe"].values, xs, ys, o.x.values, o.y.values)

    print("\n=== POSTING (delivered 80 m unwrap vs the 20 m wrapped layer unwrapped here) ===")
    print(f"{'field':>26}{'r':>7}{'n':>9}")
    for label, a, b in [("80 m, delivered unwrap", o["dswe"].values * 1000, lid80),
                        ("20 m, SNAPHU here", fine["dswe"].values, lid20),
                        ("20 m averaged to 80 m", up, lid80)]:
        r, n = corr(a, b)
        print(f"{label:>26}{r:>7.2f}{n:>9d}")

    # the two Fig. 3 subscenes, placed by external masks rather than by correlation
    dswe20 = fine["dswe"].values
    coh20 = fine["coherence"].values
    lia20 = fine["lia"].values          # the product's own 20 m incidence angle
    reference = xr.DataArray(np.zeros_like(dswe20), coords={"y": fine.y, "x": fine.x},
                             dims=("y", "x")).rio.write_crs(paths.EPSG)
    burn = _fraction_on_grid(paths.MTBS, lambda a: a.isin(BURNED), reference)
    fcf = _fraction_on_grid(paths.WORLDCOVER, lambda a: a == 10, reference)
    print("=== FIG-3 SUBSCENES (20 m, 1.5 x 3 km) ===")
    print(f"{'subscene':>28}{'r':>7}{'slope':>8}{'coh':>7}{'burn%':>7}{'fcf%':>7}{'LIA':>6}{'n':>8}")
    for name, (y0, x0) in subscene_windows(dswe20, lid20, lia20, burn, xs, ys):
        sl = (slice(y0, y0 + SUB_WIN_Y), slice(x0, x0 + SUB_WIN_X))
        S, D = dswe20[sl], lid20[sl]
        f = np.isfinite(S) & np.isfinite(D)
        r = np.corrcoef(D[f], S[f])[0, 1]
        slope = np.polyfit(D[f], S[f], 1)[0]
        print(f"{name:>28}{r:>7.2f}{slope:>8.2f}{np.nanmedian(coh20[sl]):>7.2f}"
              f"{100 * np.nanmean(burn[sl]):>7.0f}{100 * np.nanmean(fcf[sl]):>7.0f}"
              f"{np.nanmedian(lia20[sl]):>6.0f}{int(f.sum()):>8d}")


if __name__ == "__main__":
    posting_stats()
    insets_figure()

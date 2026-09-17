"""The three paper figures for the TC brief communication.

  context_figure()  -> fig1_context.png : study-area map + SWE/depth + snowpack temp
  results_figure()  -> fig2_results.png : scene dSWE/dHS maps + density hexbin
  subscenes_figure()-> fig3_subscenes.png: scene context + two 20 m subscenes
  insets_figure()   -> fig3_insets.png   : two scene locators + three 20 m insets

Composites that reuse the draw helpers (rotate, contours, km ticks) from
validation_plotting and the met-series readers / panel functions from
ancillary_plotting. This is the single figure driver -- run after run_workflow.py:

    python src/nival/paper_figures.py        # in the project env (see environment.yml)
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the src/ dir
from nival import ancillary_plotting, paths, retrieval, unwrap_20m   # noqa: E402
from nival.ancillary_plotting import (LIDAR, NISAR, NISAR_COLOR,   # noqa: E402
                                      load_met_series, mark_acquisitions,
                                      panel_snowpack_temperature, panel_swe, study_area)
from nival.validation_plotting import (DHS_VMAX, DHS_VMIN, DSWE_VMAX, DSWE_VMIN,   # noqa: E402
                                       LS, add_contours, add_north_arrow, km_ticks, rotate)


# =============================== figure 1: context ===============================

def _save_print(fig, out, width_cm):
    """Save at 300 dpi for the printed width (Copernicus 300 dpi, 5 MB per figure)."""
    from PIL import Image
    tmp = out.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=300, bbox_inches="tight"); plt.close(fig)
    im = Image.open(tmp)
    w = round(width_cm / 2.54 * 300)
    im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(out, dpi=(300, 300))
    tmp.unlink()

def context_figure():
    """Study-area map (left) + SWE/depth and snowpack-temperature panels (right)."""
    data = load_met_series()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold"})

    fig = plt.figure(figsize=(14.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1, 1],
                          hspace=0.30, wspace=0.10)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_swe = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[1, 1], sharex=ax_swe)

    study_area(ax=ax_map)                                      # the existing map, drawn into our axis

    for ax in (ax_swe, ax_temp):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    panel_swe(ax_swe, data, mark_acq=False); mark_acquisitions(ax_swe, top=False)   # NISAR/Lidar legend only on bottom panel
    panel_snowpack_temperature(ax_temp, data, mark_acq=True); mark_acquisitions(ax_temp, top=False)

    # InSAR Pair span: arrow + label inside the SWE panel, centered (in x) between the two
    # NISAR acquisitions (Feb 7 / Feb 19 -> Feb 13) at SWE = 280 mm (data coords)
    ax_swe.annotate("", xy=(dt.date(2026, 2, 19), 292), xytext=(dt.date(2026, 2, 7), 292),
                    arrowprops=dict(arrowstyle="<->", color=NISAR_COLOR, lw=1.6))
    ax_swe.annotate("InSAR Pair", (dt.date(2026, 2, 13), 294), ha="center", va="bottom",
                    fontsize=11, color=NISAR_COLOR, fontweight="bold")

    # shared date x-axis: tick only on our NISAR overpass + lidar flight dates
    tick_dates = sorted({dt.date.fromisoformat(d) for d, _ in NISAR}
                        | {dt.date.fromisoformat(d) for d, _ in LIDAR})
    ax_temp.set_xticks([mdates.date2num(d) for d in tick_dates])
    ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_temp.set_xlim(dt.date.fromisoformat(paths.D0), dt.date.fromisoformat(paths.D1))
    ax_temp.set_xlabel("2026")
    plt.setp(ax_swe.get_xticklabels(), visible=False)

    # the last two dates (Feb 19, Feb 22) are only 3 days apart -- nudge their labels
    # apart horizontally (ticks/markers stay on the true dates)
    from matplotlib.transforms import ScaledTranslation
    nudge = ScaledTranslation(11 / 72, 0, fig.dpi_scale_trans)
    labels = ax_temp.get_xticklabels()
    if len(labels) >= 3:
        labels[-2].set_transform(labels[-2].get_transform() - nudge)   # Feb 19 left
        labels[-1].set_transform(labels[-1].get_transform() + nudge)   # Feb 22 right

    out = paths.FIGURES_DIR / "fig1_context.png"
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_print(fig, out, 12)
    print("saved ->", out)


# =============================== figure 2: results ===============================
def _crop_to_valid(*arrays, mask):
    """Crop every array to the bounding box of True cells in `mask`."""
    rows, cols = np.where(mask)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    return [a[box] for a in arrays], mask[box]


def _draw_map(ax, field, valid, elev, caps, posting, contour_index, title, north=False):
    """Hillshade + field + contours into one map axis; return the field image."""
    cmap = cmocean.cm.ice.copy(); cmap.set_bad("white")        # nodata blends into the page
    extent = [0, field.shape[1] * posting, 0, field.shape[0] * posting]
    ax.set_facecolor("white")
    finite_elev = np.isfinite(elev)
    if finite_elev.sum() >= 9 and min(elev.shape) >= 3:        # guard: hillshade needs a real grid
        hillshade = LS.hillshade(np.where(finite_elev, elev, np.nanmedian(elev[finite_elev])),
                                 vert_exag=1.5, dx=posting, dy=posting)
        ax.imshow(np.where(valid, hillshade, np.nan), extent=extent, cmap="gray", vmin=0, vmax=1, origin="upper")
        lo, hi = np.nanpercentile(elev[finite_elev], [2, 98])
        levels = np.arange(np.ceil(lo / contour_index) * contour_index, hi, contour_index)
        add_contours(ax, np.where(valid, elev, np.nan), extent, levels, contour_index * 5)
    image = ax.imshow(np.where(valid, field, np.nan), extent=extent, cmap=cmap,
                      vmin=caps[0], vmax=caps[1], alpha=0.85, origin="upper")
    ax.set_title(title, fontsize=13); ax.set_aspect("equal")
    ax.set_box_aspect(field.shape[0] / field.shape[1])        # box hugs the swath: no letterbox
    if north:
        add_north_arrow(ax)
    return image


def results_figure():
    """Maps + density hexbin, all three panels on the delivered 80 m operational product."""
    product = retrieval.load_product()
    dswe, dhs, coh, elev = product["dswe"], product["dhs"], product["coherence"], product["elevation"]
    posting = float(abs(product["xs"][1] - product["xs"][0]))   # grid posting (m), from the product

    common = np.isfinite(dswe) & np.isfinite(dhs)
    slope, intercept = np.polyfit(dhs[common], dswe[common], 1)
    r_all = np.corrcoef(dswe[common], dhs[common])[0, 1]
    high = common & (coh > 0.5)
    r_high = np.corrcoef(dswe[high], dhs[high])[0, 1]
    xlim = np.nanpercentile(dhs[common], [0.5, 99.5])
    ylim = np.nanpercentile(dswe[common], [0.5, 99.5])
    print(f"all r={r_all:.2f} n={int(common.sum())}; high-coh r={r_high:.2f} n={int(high.sum())}")

    # every panel on one grid: the maps read the same 80 m arrays the scatter does
    m_dswe, m_dhs, m_elev, m_common, m_posting = dswe, dhs, elev, common, posting
    median_elev = float(np.nanmedian(m_elev[np.isfinite(m_elev)]))
    # scene: rotate + crop to the valid swath
    sn = rotate(m_dswe, np.nan); sd = rotate(np.where(m_common, m_dhs, np.nan), np.nan)
    se = rotate(m_elev, median_elev)
    svalid = np.isfinite(sn) & np.isfinite(sd)
    (sn, sd, se), svalid = _crop_to_valid(sn, sd, se, mask=svalid)

    # headline layout: two prominent maps + a compact density panel, packed tight by
    # constrained_layout so colorbars and labels never collide
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax_sdswe, ax_sdhs, ax_hall = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    img_dswe = _draw_map(ax_sdswe, sn, svalid, se, (DSWE_VMIN, DSWE_VMAX), m_posting, 100,
                         r"NISAR $\Delta$SWE (m)", north=True)
    img_dhs = _draw_map(ax_sdhs, sd, svalid, se, (DHS_VMIN, DHS_VMAX), m_posting, 100,
                        r"Airborne $\Delta\mathbf{H_s}$ (m)")
    km_ticks(ax_sdswe, m_posting, 2000, left=True); km_ticks(ax_sdhs, m_posting, 2000, left=False)

    # density hexbin of NISAR dSWE vs NIVAL dHS (log-scaled pixel count)
    dhs_c, dswe_c = dhs[common], dswe[common]
    # bin size scales with sqrt(n) so the counts per bin stay comparable whatever the posting
    # (the 80 m operational product has ~4x fewer pixels than the old 40 m one)
    gridsize = max(12, int(round(np.sqrt(dhs_c.size) / 3.1)))
    mincnt = max(3, int(round(15 * dhs_c.size / 19505)))
    hb = ax_hall.hexbin(dhs_c, dswe_c, gridsize=gridsize, cmap="magma", mincnt=mincnt,
                        norm=LogNorm(), extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    hb.set_norm(LogNorm(vmin=mincnt, vmax=hb.get_array().max()))   # vmax = peak bin count
    ax_hall.plot(xlim, slope * np.array(xlim) + intercept, color="#4575b4", lw=1.8, ls="--", label="linear fit")
    ax_hall.set_xlim(xlim); ax_hall.set_ylim(ylim)
    ax_hall.set_box_aspect(sn.shape[0] / sn.shape[1])         # same box shape as the maps -> equal heights
    ax_hall.set_xlabel(r"Airborne $\Delta H_s$ (m)"); ax_hall.set_ylabel(r"NISAR $\Delta$SWE (m)")
    ax_hall.set_title(f"r = {r_all:.2f},  n = {int(common.sum())}")
    ax_hall.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # colorbars attached to each panel; constrained_layout keeps them snug and overlap-free
    fig.colorbar(img_dswe, ax=ax_sdswe, fraction=0.05, pad=0.02).set_label(r"$\Delta$SWE (m)")
    fig.colorbar(img_dhs, ax=ax_sdhs, fraction=0.05, pad=0.02).set_label(r"$\Delta H_s$ (m)")
    cb_count = fig.colorbar(hb, ax=ax_hall, fraction=0.05, pad=0.02)
    cb_count.set_label("count")
    # show real counts (20, 30, 40, ...) instead of LogNorm's default 2x10^1 sci notation
    int_fmt = FuncFormatter(lambda value, pos: f"{value:.0f}")
    cb_count.ax.yaxis.set_major_formatter(int_fmt)
    cb_count.ax.yaxis.set_minor_formatter(int_fmt)

    out = paths.FIGURES_DIR / "fig2_results.png"
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_print(fig, out, 18)
    print("saved ->", out)


# =============================== figure 3: analysis ===============================
WINDOW = 11          # canopy-fraction (FCF) smoothing window in pixels (~440 m at the 40 m Fig-3 posting)
LOCAL_R_WINDOW = 11  # local-correlation window in pixels (sigma~110 m): ~420 m physical, held from 21 px @ 20 m
OPEN_NAMES = ["forest", "open", "burn-low", "burn-mod", "burn-high"]
OPEN_COLORS = ["#1b5e20", "#bcaa66", "#ffd54f", "#fb8c00", "#b71c1c"]



# --- Fig. 3: two 20 m subscenes from the delivered wrapped phase ------------
SUB_LIM = 55.0                 # mm SWE, diverging limit on the NISAR inset panels
SUB_LIM_HS = 30.0              # cm depth, diverging limit on the lidar inset panels
SUB_WIN_X, SUB_WIN_Y = 75, 150 # 1.5 km across by 3 km down, at the 20 m posting
BURNED = (2, 3, 4)             # MTBS low, moderate and high severity
FREEMAN_LL = (-115.7018528, 43.94091389)   # ancillary_plotting.MET_STATIONS
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
    rings, polygon = ancillary_plotting.pioneer_perimeter()
    for part in (polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]):
        if not part.is_empty:
            ax_fcf.add_patch(MplPolygon(np.array(part.exterior.coords), closed=True,
                                        facecolor=ancillary_plotting.FIRE_C, edgecolor="none",
                                        alpha=0.16, zorder=1.5))
    for ring in rings:
        ax_fcf.plot(ring[:, 0], ring[:, 1], color=ancillary_plotting.FIRE_C, lw=1.6, ls="--",
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
    _, polygon = ancillary_plotting.pioneer_perimeter()
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


if __name__ == "__main__":
    context_figure()
    results_figure()
    subscenes_figure()
    insets_figure()

    # mirror the three paper figures into the paper repo (sibling checkout), if it is present
    import shutil
    if paths.PAPER_FIGURES_DIR.parent.exists():
        paths.PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        names = ("fig1_context.png", "fig2_results.png", "fig3_subscenes.png",
                 "fig3_insets.png")
        for name in names:
            shutil.copy2(paths.FIGURES_DIR / name, paths.PAPER_FIGURES_DIR / name)
        print(f"copied {len(names)} figures -> {paths.PAPER_FIGURES_DIR}")

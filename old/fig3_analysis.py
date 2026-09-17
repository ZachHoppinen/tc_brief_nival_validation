"""RETIRED: the original Fig. 3, a five-panel local-correlation map over the upper-right
quarter of the scene. Replaced in the manuscript by the 20 m subscene figure built by
nival.paper_figures.subscenes_figure(). Kept for reference; imports assume src/ on the path.

    conda run -n nival python old/fig3_analysis.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nival import paths, retrieval   # noqa: E402
from nival.ancillary_plotting import _raster_on_grid   # noqa: E402
from nival.paper_figures import (WINDOW, LOCAL_R_WINDOW, OPEN_COLORS, OPEN_NAMES,  # noqa: E402
                                 _crop_to_valid)
from nival.validation_plotting import (DHS_VMAX, DHS_VMIN, DSWE_VMAX, DSWE_VMIN,  # noqa: E402
                                       LS, add_contours, add_north_arrow, km_ticks,
                                       rotate)


def _openness_class(xs, ys):
    """Vegetation/burn class on the grid: 0 forest, 1 open (unburned, split by canopy
    fraction); 2 burn-low, 3 burn-mod, 4 burn-high. Burn comes from MTBS severity, not
    WorldCover canopy: tree-cover classes can't tell live canopy from fire-killed snags.
    MTBS codes 5 (greenness) and 6 (non-mapping) are unburned ground -> classed by FCF."""
    fcf = uniform_filter((_raster_on_grid(paths.WORLDCOVER, xs, ys, Resampling.nearest) == 10).astype(float),
                         WINDOW, mode="constant")             # local forest canopy fraction 0-1
    sev = _raster_on_grid(paths.MTBS, xs, ys, Resampling.nearest)        # MTBS class 0-6
    unburned = np.isin(sev, [0, 5, 6])
    opn = np.full(sev.shape, np.nan)
    opn[unburned & (fcf >= 0.5)] = 0
    opn[unburned & (fcf < 0.5)] = 1
    opn[np.isin(sev, [1, 2])] = 2
    opn[sev == 3] = 3
    opn[sev == 4] = 4
    return opn


def analysis_figure():
    """Fig 3: why this pair was retrievable, in five panels over the deep-accumulation
    upper-right quarter of the scene -- dSWE, dHS, local r(dSWE,dHS), vegetation/burn class,
    and local incidence angle.

    Story, left to right: NISAR recovered deep snow here (dSWE); the lidar shows the same deep
    snow (dHS); the two agree on the open/burned slopes but not in the forested deep drifts (local r);
    that split tracks vegetation -- forest vs burned/open (veg); and the underlying geometric
    control is the look angle -- grazing/unfavorable LIA is where retrieval degrades (LIA).
    LIA is the strongest single predictor of local r and survives controlling for
    coherence, which it partly drives. Runs on the operational L2 GUNW at its 80 m posting,
    with the local-r window held at 440 m physical."""
    product = retrieval.load_product()
    dswe, dhs, lia = product["dswe"], product["dhs"], product["lia"]
    xs, ys = product["xs"], product["ys"]
    common = np.isfinite(dswe) & np.isfinite(dhs)

    local_r = retrieval.local_correlation(dswe, dhs, LOCAL_R_WINDOW)
    opn = _openness_class(xs, ys)

    # rotate every field to the display frame, mask to the common-snow swath, crop to it,
    # then keep the upper-right quarter (the deep-accumulation corner that carries the story)
    sn = rotate(np.where(common, dswe, np.nan), np.nan)
    sd = rotate(np.where(common, dhs, np.nan), np.nan)
    slr = rotate(np.where(common, local_r, np.nan), np.nan)   # clip to the data footprint:
    #     the Gaussian windows smear valid pixels into the nodata margin, so without this
    #     mask local r bleeds ~10-15 px past the real edge (asymmetric -> looks like a shift)
    sop = rotate(np.where(common, opn, np.nan), np.nan)
    slia = rotate(np.where(common, lia, np.nan), np.nan)
    valid = np.isfinite(sn) & np.isfinite(sd)
    (sn, sd, slr, sop, slia), valid = _crop_to_valid(sn, sd, slr, sop, slia, mask=valid)
    H, W = sn.shape
    # upper-right quarter, nudged down (wind-drift cornice stays at the top, but we pick up
    # more lower terrain) and left (trims the white nodata wedge off the east edge, keeping a bit)
    qr = slice(int(0.10 * H), int(0.62 * H))
    qc = slice(int(0.42 * W), int(0.94 * W))
    sd_full = sd                                              # keep the whole dHS swath for the locator inset
    sn, sd, slr, sop, slia = sn[qr, qc], sd[qr, qc], slr[qr, qc], sop[qr, qc], slia[qr, qc]

    print(f"quarter {sn.shape}: local r median={np.nanmedian(slr):.2f}; "
          f"LIA median={np.nanmedian(slia):.1f} deg")

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.6), constrained_layout=True)   # a little height for the (d) legend below the panel

    def finish(ax, title, letter):
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.text(0.96, 0.96, letter, transform=ax.transAxes, va="top", ha="right", fontsize=12,
                fontweight="bold", color="k", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))

    # (a) NISAR dSWE, (b) lidar dHS -- shared viridis stretch with the scene maps
    im = axes[0].imshow(sn, cmap="viridis", vmin=DSWE_VMIN, vmax=DSWE_VMAX, origin="upper")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02).set_label("dSWE (m)")
    add_north_arrow(axes[0]); finish(axes[0], "NISAR dSWE", "(a)")
    im = axes[1].imshow(sd, cmap="viridis", vmin=DHS_VMIN, vmax=DHS_VMAX, origin="upper")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02).set_label("dHS (m)")
    finish(axes[1], "NIVAL dHS", "(b)")

    # (c) local correlation -- sequential bad->good on a LINEAR stretch (-0.25..0.75): green
    # saturates by r=0.75 so good agreement reads solidly green, while r~0 (no agreement)
    # sits at red/orange -- as bad as anti-correlation; yellow lands at r~0.25, not at zero
    im = axes[2].imshow(slr, cmap="RdYlGn", vmin=-0.25, vmax=0.75, origin="upper")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02, ticks=[-0.25, 0, 0.25, 0.5, 0.75]).set_label("local r")
    finish(axes[2], "local r(dSWE, dHS)", "(c)")

    # (d) vegetation / burn class -- the control on where it worked (burn comes from MTBS
    # severity, not canopy, so the radar-open burn scar isn't mislabelled as forest)
    cmap = ListedColormap(OPEN_COLORS); cmap.set_bad("0.9")
    axes[3].imshow(sop, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 5.5), cmap.N), origin="upper")
    axes[3].legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(OPEN_COLORS, OPEN_NAMES)],
                   fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                   frameon=False)                                 # BELOW the panel: frees the low-r blob region
    finish(axes[3], "vegetation / burn", "(d)")

    # (e) local incidence angle -- the geometric control. Purples (a hue unused by the other
    # panels, and not blue=snow): pale=favorable (low LIA, facing the radar), deep purple=
    # grazing/unfavorable (high LIA), which is where retrieval degrades and coherence collapses
    im = axes[4].imshow(slia, cmap="Purples", vmin=15, vmax=60, origin="upper")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.02).set_label("LIA (deg)")
    finish(axes[4], "local incidence angle", "(e)")
    # locator inset in (e)'s LOWER-RIGHT (the low-r blob is lower-LEFT, good-r green there, so
    # nothing to read is hidden): the whole swath in dHS (viridis) with this figure's quarter
    # boxed in red -- the viridis mini-map stands out against the Purples panel
    loc = axes[4].inset_axes([0.63, 0.035, 0.34, 0.34]); loc.set_facecolor("white")
    cmap_loc = plt.cm.viridis.copy(); cmap_loc.set_bad("white")
    loc.imshow(sd_full, cmap=cmap_loc, vmin=DHS_VMIN, vmax=DHS_VMAX, origin="upper")
    loc.add_patch(plt.Rectangle((qc.start, qr.start), qc.stop - qc.start, qr.stop - qr.start,
                                fill=False, edgecolor="red", lw=1.5))
    loc.set_xticks([]); loc.set_yticks([]); loc.set_aspect("equal")
    for spine in loc.spines.values():
        spine.set_edgecolor("k")

    out = paths.FIGURES_DIR / "fig3_analysis.png"
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved ->", out)



if __name__ == "__main__":
    analysis_figure()

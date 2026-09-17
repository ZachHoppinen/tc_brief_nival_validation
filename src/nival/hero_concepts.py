"""Ten variations on the two-panel hero image, in figures/NISAR_PRESS/hero_concepts/.

All build on the same base: NISAR beside lidar, flipped ice colours, a gamma 1.5 stretch
fitted on the coherent pixels, no numbers on the colour bar. They vary the framing, the
palette, the background and how much context is shown.

    conda run -n nival python src/nival/hero_concepts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cmocean
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cmcrameri import cm as cmc
from matplotlib.colors import LightSource, PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nival import paths, unwrap_20m   # noqa: E402
from nival.press_concepts import DX, equalize, limits, load   # noqa: E402
from nival.press_figure import PICKS, lidar_at, window   # noqa: E402

OUT = paths.FIGURES_DIR / "NISAR_PRESS" / "hero_concepts"
FONT = {"fontfamily": "Helvetica Neue"}
SEQ = cmocean.cm.ice
GAMMA, COH_MIN, PCT = 1.5, 0.30, (2, 98)
HEADLINE = "Mapping Snow Accumulation: NISAR vs Lidar"
T_NISAR = "NISAR Feb 7-19\nSnow Volume Change"
T_LIDAR = "Lidar Feb 7-22\nSnow Depth Change"


def _swath_angle(valid):
    """Rotation in degrees that stands the swath's dominant straight edge upright."""
    from scipy.ndimage import sobel
    a = valid.astype(float)
    gy, gx = sobel(a, axis=0), sobel(a, axis=1)
    mag = np.hypot(gx, gy)
    edge = mag > 0.5 * mag.max()
    ang = np.degrees(np.arctan2(gy[edge], gx[edge])) % 180.0
    hist, edges = np.histogram(ang, bins=180, range=(0, 180), weights=mag[edge])
    return float((0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])) % 90.0)


def _rotate_nan(field, deg):
    """Rotate a field holding NaN without letting the NaN bleed into the data."""
    from scipy.ndimage import rotate
    good = np.isfinite(field).astype(float)
    filled = rotate(np.where(good > 0, field, 0.0), deg, reshape=True, order=1, cval=0.0)
    weight = rotate(good, deg, reshape=True, order=1, cval=0.0)
    return np.where(weight > 0.99, filled / np.where(weight > 0, weight, 1), np.nan)


def fields(d):
    """NISAR at 20 m, lidar at 3 m, plus the coherent-pixel stretch limits for each."""
    xs, ys, nisar, lidar, seen, crop, ext = d
    from nival.press_concepts import COHERENCE as coh
    ref_n = np.where(coh >= COH_MIN, nisar, np.nan)[crop]
    _, _, fine = lidar_at(ext[0], ext[1], ext[3], ext[2])
    la = limits(ref_n, lo=PCT[0], hi=PCT[1])
    lb = limits(lidar[crop], lo=PCT[0], hi=PCT[1])
    a, ea = crop_corner(nisar[crop], ext, **CROP)
    b, _ = crop_corner(fine, ext, **CROP)
    return a, b, la, lb, ea


def norm(lim, gamma=GAMMA):
    return PowerNorm(gamma=gamma, vmin=lim[0], vmax=lim[1], clip=True)


def inset_title(ax, text, colour="k", fontsize=19, halo="w"):
    """Upper left, inside the panel; a thin halo keeps it legible over bright snow."""
    effects = [pe.withStroke(linewidth=3, foreground=halo)] if halo else None
    ax.text(0.06, 0.96, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, color=colour, **FONT, path_effects=effects)


def bare(ax, face=None, anchor=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    if anchor:                      # push the letterboxed image to the inner edge
        ax.set_anchor(anchor)
    for s in ax.spines.values():
        s.set_visible(False)
    if face:
        ax.set_facecolor(face)
    return ax


def qbar(fig, im, rect, colour="k", orientation="horizontal"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    cb.set_ticks([]); cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor(colour)
    if orientation == "horizontal":
        for x, s, ha in ((0.0, "Less Snow", "left"), (1.0, "More Snow", "right")):
            cb.ax.text(x, -0.35, s, ha=ha, va="top", transform=cb.ax.transAxes,
                       fontsize=12, color=colour, **FONT)
    else:
        for y, s, va in ((0.0, "Less Snow", "bottom"), (1.0, "More Snow", "top")):
            cb.ax.text(1.3, y, s, ha="left", va=va, rotation=90, transform=cb.ax.transAxes,
                       fontsize=12, color=colour, **FONT)
    return cb


def save(fig, name, face="white", dpi=220, out=None):
    target = (out or OUT)
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / name, dpi=dpi, bbox_inches="tight", facecolor=face)
    plt.close(fig)
    print(f"  {target / name}")


# ------------------------------------------------------------------- 1 baseline
def c01_baseline(d):
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(13, 7))
    axes = [fig.add_axes([0.02, 0.20, 0.47, 0.74]), fig.add_axes([0.51, 0.20, 0.47, 0.74])]
    for ax, f, lim, t, anc in ((axes[0], a, la, T_NISAR, "E"), (axes[1], b, lb, T_LIDAR, "W")):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax, anchor=anc); inset_title(ax, t)
    qbar(fig, im, [0.30, 0.10, 0.40, 0.030])
    save(fig, "01_baseline.png")


# --------------------------------------------------------------- 2 vertical pair
def c02_stacked(d):
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(7.5, 13))
    axes = [fig.add_axes([0.06, 0.50, 0.74, 0.46]), fig.add_axes([0.06, 0.10, 0.74, 0.46])]
    for ax, f, lim, t in ((axes[0], a, la, T_NISAR), (axes[1], b, lb, T_LIDAR)):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax); inset_title(ax, t, fontsize=18)
    qbar(fig, im, [0.84, 0.30, 0.022, 0.40], orientation="vertical")
    save(fig, "02_stacked.png")


# ------------------------------------------------------------------ 3 dark card
def c03_dark(d):
    a, b, la, lb, ext = fields(d)
    bg = "#0d1117"
    fig = plt.figure(figsize=(13, 7), facecolor=bg)
    axes = [fig.add_axes([0.02, 0.20, 0.47, 0.74]), fig.add_axes([0.51, 0.20, 0.47, 0.74])]
    for ax, f, lim, t in ((axes[0], a, la, T_NISAR), (axes[1], b, lb, T_LIDAR)):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax, face=bg); inset_title(ax, t, colour="w", halo="0.1")
    qbar(fig, im, [0.30, 0.10, 0.40, 0.030], colour="w")
    save(fig, "03_dark.png", face=bg)


# ------------------------------------------------------------- 4 rotated upright
def c04_upright(d):
    xs, ys, nisar, lidar, seen, crop, ext = d
    deg = _swath_angle(seen)
    rn, rl = _rotate_nan(nisar, deg), _rotate_nan(lidar, deg)
    g = np.isfinite(rn) & np.isfinite(rl)
    rr, cc = np.where(g.any(axis=1))[0], np.where(g.any(axis=0))[0]
    sl = (slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1))
    rn, rl = rn[sl], rl[sl]
    la, lb = limits(rn, lo=PCT[0], hi=PCT[1]), limits(rl, lo=PCT[0], hi=PCT[1])
    fig = plt.figure(figsize=(13, 7))
    axes = [fig.add_axes([0.02, 0.20, 0.46, 0.74]), fig.add_axes([0.52, 0.20, 0.46, 0.74])]
    for ax, f, lim, t in ((axes[0], rn, la, T_NISAR), (axes[1], rl, lb, T_LIDAR)):
        im = ax.imshow(f, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax); inset_title(ax, t)
    qbar(fig, im, [0.30, 0.10, 0.40, 0.030])
    save(fig, "04_upright.png")


# ------------------------------------------------------------------ 5 hillshade
def c05_hillshade(d):
    xs, ys, nisar, lidar, seen, crop, ext = d
    a, b, la, lb, _ = fields(d)
    coarse = xr.open_dataset(paths.product())
    elev = (coarse["elevation"].to_dataset(name="e")
            .interp(y=ys, x=xs, method="linear")["e"].values)[crop]
    ls = LightSource(azdeg=315, altdeg=42)
    shade = ls.hillshade(np.where(np.isfinite(elev), elev, np.nanmean(elev)),
                         vert_exag=3, dx=DX, dy=DX)
    fig = plt.figure(figsize=(13, 7))
    axes = [fig.add_axes([0.02, 0.20, 0.47, 0.74]), fig.add_axes([0.51, 0.20, 0.47, 0.74])]
    for ax, f, lim, t in ((axes[0], a, la, T_NISAR), (axes[1], b, lb, T_LIDAR)):
        n = norm(lim)(f)
        sh = shade if f.shape == shade.shape else np.ones(f.shape)
        rgb = SEQ(np.nan_to_num(n))[..., :3] * (0.5 + 0.5 * sh)[..., None]
        ax.imshow(np.dstack([rgb, np.isfinite(f).astype(float)]), extent=ext, origin="upper")
        bare(ax); inset_title(ax, t)
    im = axes[0].imshow(np.full(a.shape, np.nan), extent=ext, cmap=SEQ, norm=norm(la))
    qbar(fig, im, [0.30, 0.10, 0.40, 0.030])
    save(fig, "05_hillshade.png")


# --------------------------------------------------------------- 6 warm palette
def c06_warm(d):
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(13, 7))
    axes = [fig.add_axes([0.02, 0.20, 0.47, 0.74]), fig.add_axes([0.51, 0.20, 0.47, 0.74])]
    for ax, f, lim, t in ((axes[0], a, la, T_NISAR), (axes[1], b, lb, T_LIDAR)):
        im = ax.imshow(f, extent=ext, cmap=cmc.lapaz_r, norm=norm(lim), origin="upper")
        bare(ax); inset_title(ax, t)
    qbar(fig, im, [0.30, 0.10, 0.40, 0.030])
    save(fig, "06_warm.png")


# ------------------------------------------------------------ 7 rank, no titles
def c07_minimal(d):
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(13, 6.4))
    axes = [fig.add_axes([0.02, 0.06, 0.47, 0.88]), fig.add_axes([0.51, 0.06, 0.47, 0.88])]
    for ax, f, ref, t in ((axes[0], a, a, "NISAR"), (axes[1], b, b, "lidar")):
        ax.imshow(equalize(f, reference=ref), extent=ext, cmap=SEQ, vmin=0, vmax=1,
                  origin="upper")
        bare(ax)
        ax.text(0.03, 0.95, t, transform=ax.transAxes, fontsize=22, va="top", color="w",
                **FONT, path_effects=[pe.withStroke(linewidth=3, foreground="0.15")])
    save(fig, "07_minimal.png")


# ------------------------------------------------------------- 8 with a close up
def c08_with_zoom(d):
    xs, ys, nisar, lidar, seen, crop, ext = d
    a, b, la, lb, _ = fields(d)
    pick = PICKS[0]
    sl = window(xs, ys, *pick)
    e = [xs[sl[1].start], xs[sl[1].stop - 1], ys[sl[0].stop - 1], ys[sl[0].start]]
    fig = plt.figure(figsize=(13, 9.5))
    top = [fig.add_axes([0.02, 0.42, 0.47, 0.52]), fig.add_axes([0.51, 0.42, 0.47, 0.52])]
    bot = [fig.add_axes([0.10, 0.06, 0.36, 0.32]), fig.add_axes([0.54, 0.06, 0.36, 0.32])]
    for ax, f, lim, t in ((top[0], a, la, T_NISAR), (top[1], b, lb, T_LIDAR)):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax); inset_title(ax, t, fontsize=18)
        ax.add_patch(plt.Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2],
                                   fill=False, ec="w", lw=2))
    _, _, fine_zoom = lidar_at(e[0], e[1], e[3], e[2])
    for ax, f, lim in ((bot[0], nisar[sl], la), (bot[1], fine_zoom, lb)):
        ax.imshow(f, extent=e, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax)
    bot[0].set_title("close up, 3 km across", fontsize=15, loc="left", **FONT)
    save(fig, "08_with_zoom.png")


# ----------------------------------------------------------- 9 interleaved bands
def c09_bands(d, bands=10):
    a, b, la, lb, ext = fields(d)
    bl = np.array([[np.nan]])
    coarse = np.asarray(b)
    # put the lidar on the NISAR grid so the bands line up pixel for pixel
    step = max(1, coarse.shape[0] // a.shape[0])
    bres = coarse[::step, ::step][:a.shape[0], :a.shape[1]]
    if bres.shape != a.shape:
        bres = np.pad(bres, ((0, a.shape[0] - bres.shape[0]), (0, a.shape[1] - bres.shape[1])),
                      constant_values=np.nan)
    rows = np.arange(a.shape[0])
    take = (rows // max(1, a.shape[0] // bands)) % 2 == 0
    mix = np.where(take[:, None], norm(la)(a), norm(lb)(bres))
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.82])
    im = ax.imshow(mix, extent=ext, cmap=SEQ, vmin=0, vmax=1, origin="upper")
    height = (ext[3] - ext[2]) / bands
    for k in range(bands):
        y = ext[3] - (k + 0.5) * height
        ax.text(ext[0] + 250, y, "NISAR" if k % 2 == 0 else "lidar", ha="left", va="center",
                fontsize=12, color="w", **FONT,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="0.2")])
    bare(ax); ax.set_title("alternating bands", fontsize=19, **FONT)
    qbar(fig, im, [0.30, 0.06, 0.40, 0.026])
    save(fig, "09_bands.png")


# -------------------------------------------------------------- 10 poster crop
def c10_poster(d):
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(15, 6.2), facecolor="#f4f1ea")
    axes = [fig.add_axes([0.02, 0.12, 0.44, 0.80]), fig.add_axes([0.40, 0.12, 0.44, 0.80])]
    for ax, f, lim in ((axes[0], a, la), (axes[1], b, lb)):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax, face="#f4f1ea")
    fig.text(0.86, 0.72, "the same snow,\ntwo ways of seeing it", fontsize=24, va="top",
             **FONT)
    fig.text(0.86, 0.52, "NISAR radar, from orbit\nAirborne lidar, from a plane\n"
                         "Mores Creek Summit, Idaho\nFebruary 2026", fontsize=13, va="top",
             color="0.3", **FONT)
    qbar(fig, im, [0.86, 0.20, 0.11, 0.028])
    save(fig, "10_poster.png", face="#f4f1ea")


#: the standard crop: a quarter off the left and bottom, a fifth off the right
CROP = dict(drop_bottom=0.20, drop_left=0.25, drop_right=1 / 6)


def crop_corner(field, ext, drop_bottom=0.20, drop_left=0.25, drop_right=1 / 6):
    """Trim the empty corners, keeping the white space inside what remains."""
    n, m = field.shape
    r1 = n - int(round(drop_bottom * n))
    c0 = int(round(drop_left * m))
    c1 = m - int(round(drop_right * m))
    out = field[:r1, c0:c1]
    x0 = ext[0] + (ext[1] - ext[0]) * c0 / m
    x1 = ext[0] + (ext[1] - ext[0]) * c1 / m
    y0 = ext[3] - (ext[3] - ext[2]) * r1 / n
    return out, [x0, x1, y0, ext[3]]


def c12_suptitle(d):
    """One shared headline over the pair, with just the instrument named on each panel."""
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(13, 7.6))
    axes = [fig.add_axes([0.02, 0.14, 0.47, 0.72]), fig.add_axes([0.51, 0.14, 0.47, 0.72])]
    for ax, f, lim, t, anc in ((axes[0], a, la, "NISAR\nFeb 7-19", "E"),
                               (axes[1], b, lb, "Airborne Lidar\nFeb 7-22", "W")):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax, anchor=anc); inset_title(ax, t)
    fig.text(0.5, 0.95, HEADLINE, ha="center", va="top", fontsize=30, **FONT)

    # 1 km scale bar in the lower right of the lidar panel
    x0 = ext[0] + 0.06 * (ext[1] - ext[0])
    y0 = ext[2] + 0.06 * (ext[3] - ext[2])
    axes[1].plot([x0, x0 + 1000], [y0, y0], color="w", lw=3.5, solid_capstyle="butt",
                 path_effects=[pe.withStroke(linewidth=5.5, foreground="0.15")])
    axes[1].text(x0 + 500, y0 + 0.02 * (ext[3] - ext[2]), "1 km", ha="center", va="bottom",
                 fontsize=13, color="w", **FONT,
                 path_effects=[pe.withStroke(linewidth=3, foreground="0.15")])

    qbar(fig, im, [0.30, 0.075, 0.40, 0.028])
    save(fig, "12_suptitle.png")


def c13_subtle(d):
    """The shared-headline layout with terrain relief and a quiet Idaho locator."""
    xs, ys, nisar, lidar, seen, crop, raw_ext = d
    a, b, la, lb, ext = fields(d)          # ext is the CROPPED extent, matching a and b
    coarse = xr.open_dataset(paths.product())
    elev = (coarse["elevation"].to_dataset(name="e")
            .interp(y=ys, x=xs, method="linear")["e"].values)[crop]
    elev, _ = crop_corner(elev, raw_ext, **CROP)
    ls = LightSource(azdeg=315, altdeg=42)
    base = ls.hillshade(np.where(np.isfinite(elev), elev, np.nanmean(elev)),
                        vert_exag=2, dx=DX, dy=DX)

    fig = plt.figure(figsize=(13, 7.6))
    axes = [fig.add_axes([0.02, 0.14, 0.47, 0.72]), fig.add_axes([0.51, 0.14, 0.47, 0.72])]
    for ax, f, lim, t, anc in ((axes[0], a, la, "NISAR\nFeb 7-19", "E"),
                               (axes[1], b, lb, "Airborne Lidar\nFeb 7-22", "W")):
        n = norm(lim)(f)
        shade = base if f.shape == base.shape else np.ones(f.shape)
        # a light touch: the relief textures the colour rather than darkening it
        rgb = SEQ(np.nan_to_num(n))[..., :3] * (0.80 + 0.20 * shade)[..., None]
        ax.imshow(np.dstack([rgb, np.isfinite(f).astype(float)]), extent=ext, origin="upper")
        bare(ax, anchor=anc); inset_title(ax, t)
    im = axes[0].imshow(np.full(a.shape, np.nan), extent=ext, cmap=SEQ, norm=norm(la))

    x0 = ext[0] + 0.06 * (ext[1] - ext[0])
    y0 = ext[2] + 0.06 * (ext[3] - ext[2])
    axes[1].plot([x0, x0 + 1000], [y0, y0], color="w", lw=3.5, solid_capstyle="butt",
                 path_effects=[pe.withStroke(linewidth=5.5, foreground="0.15")])
    axes[1].text(x0 + 500, y0 + 0.02 * (ext[3] - ext[2]), "1 km", ha="center", va="bottom",
                 fontsize=13, color="w", **FONT,
                 path_effects=[pe.withStroke(linewidth=3, foreground="0.15")])

    # locator in the figure's own palette: pale ice fill, mid-blue edge, deep navy marker
    from nival.ancillary_plotting import idaho_locator
    loc = axes[1].inset_axes([0.78, 0.66, 0.20, 0.34])
    idaho_locator(loc)
    for txt in loc.texts:
        txt.set_visible(False)
    for patch in loc.patches:
        patch.set_facecolor(SEQ(0.88)); patch.set_edgecolor(SEQ(0.35))
        patch.set_linewidth(1.4); patch.set_alpha(1.0)
    for line in loc.lines:
        line.set_color(SEQ(0.06)); line.set_markerfacecolor(SEQ(0.06))
        line.set_markeredgecolor(SEQ(0.06)); line.set_markersize(7)

    fig.text(0.5, 0.95, HEADLINE, ha="center", va="top", fontsize=30, **FONT)
    qbar(fig, im, [0.30, 0.075, 0.40, 0.028])
    save(fig, "13_subtle.png")


def c14_hero_with_insets(d, n_boxes=3):
    """The hero pair above a row of close ups, NISAR over lidar for each box."""
    xs, ys, nisar, lidar, seen, crop, raw_ext = d
    a, b, la, lb, ext = fields(d)
    picks = PICKS[:n_boxes]

    fig = plt.figure(figsize=(13, 11.5))
    top = [fig.add_axes([0.02, 0.44, 0.47, 0.46]), fig.add_axes([0.51, 0.44, 0.47, 0.46])]
    for ax, f, lim, t, anc in ((top[0], a, la, "NISAR\nFeb 7-19", "E"),
                               (top[1], b, lb, "Airborne Lidar\nFeb 7-22", "W")):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax, anchor=anc); inset_title(ax, t)
        for k, pick in enumerate(picks, start=1):
            rs, cs = window(xs, ys, *pick)
            e = [xs[cs.start], xs[cs.stop - 1], ys[rs.stop - 1], ys[rs.start]]
            ax.add_patch(plt.Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2],
                                       fill=False, ec="w", lw=1.8))
            ax.text(e[0] + 120, e[3] - 120, str(k), fontsize=12, fontweight="bold",
                    color="w", ha="left", va="top", **FONT,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="0.15")])

    # close ups: NISAR row above lidar row, one column per box
    w = 0.27
    left0 = (1.0 - (n_boxes * w + (n_boxes - 1) * 0.02)) / 2
    for k, pick in enumerate(picks):
        rs, cs = window(xs, ys, *pick)
        e = [xs[cs.start], xs[cs.stop - 1], ys[rs.stop - 1], ys[rs.start]]
        _, _, fine = lidar_at(e[0], e[1], e[3], e[2])
        x = left0 + k * (w + 0.02)
        for i, (f, lim) in enumerate(((nisar[rs, cs], la), (fine, lb))):
            ax = fig.add_axes([x, 0.22 - i * 0.155, w, 0.15])
            ax.imshow(f, extent=e, cmap=SEQ, norm=norm(lim), origin="upper")
            bare(ax)
            if i == 0:
                ax.set_title(f"{k + 1}", fontsize=13, **FONT)
        fig.text(x - 0.012, 0.295 - 0.155 / 2, "", fontsize=1)
    fig.text(left0 - 0.02, 0.295, "NISAR", ha="right", va="center", fontsize=12, **FONT)
    fig.text(left0 - 0.02, 0.14, "Airborne", ha="right", va="center", fontsize=12, **FONT)

    fig.text(0.5, 0.965, HEADLINE, ha="center", va="top", fontsize=30, **FONT)
    qbar(fig, im, [0.34, 0.045, 0.32, 0.022])
    save(fig, "14_hero_with_insets.png")


BOXES = [0, 2]           # areas 1 and 3 of the inset figure, zero-indexed
ROW_LABELS = ["Ex 1. Wind Drifting", "Ex 2. Aspect and Vegetation"]
CLOSEUP_W, CLOSEUP_H = 1950, 1500   # metres; wider than tall, same scale on both rows


#: per-box nudges in metres, (east, north)
CLOSEUP_NUDGE = {0: (-100, +100)}


def closeup_slices(xs, ys, box):
    """Row and column slices for a rectangular close up centred on the box."""
    cx, cy, _ = PICKS[box]
    dx, dy = CLOSEUP_NUDGE.get(box, (0, 0))
    nx, ny = round(CLOSEUP_W / DX), round(CLOSEUP_H / DX)
    j = int(np.clip(np.argmin(np.abs(xs - (cx + dx))) - nx // 2, 0, len(xs) - nx))
    i = int(np.clip(np.argmin(np.abs(ys - (cy + dy))) - ny // 2, 0, len(ys) - ny))
    return slice(i, i + ny), slice(j, j + nx)


def _closeups(fig, d, la, lb, rects, label_col=True, row_labels=False):
    """Fill `rects` (one per box) with a NISAR and an airborne panel, side by side."""
    xs, ys, nisar, lidar, seen, crop, raw_ext = d
    for (k, box), (x, y, w, h) in zip(enumerate(BOXES), rects):
        rs, cs = closeup_slices(xs, ys, box)
        e = [xs[cs.start], xs[cs.stop - 1], ys[rs.stop - 1], ys[rs.start]]
        _, _, fine = lidar_at(e[0], e[1], e[3], e[2])
        half = (w - 0.008) / 2
        for i, (f, lim) in enumerate(((nisar[rs, cs], la), (fine, lb))):
            ax = fig.add_axes([x + i * (half + 0.008), y, half, h])
            ax.imshow(f, extent=e, cmap=SEQ, norm=norm(lim), origin="upper")
            bare(ax, anchor="E" if i == 0 else "W")
            # a for the radar panel, b for the lidar panel of the same box
            ax.text(0.04, 0.94, f"{k + 1}{'ab'[i]}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=14, fontweight="bold", color="w", **FONT,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="0.15")])
            if k == 0 and label_col:
                ax.set_title(["NISAR", "Airborne"][i], fontsize=13, **FONT)
            if i == 0 and row_labels:
                ax.set_ylabel(ROW_LABELS[k], fontsize=14, labelpad=10, **FONT)
                ax.yaxis.set_visible(True); ax.set_yticks([])


def scale_bar(ax, ext, km=1.0, colour="w"):
    """A 1 km bar in the panel's lower left."""
    x0 = ext[0] + 0.06 * (ext[1] - ext[0])
    y0 = ext[2] + 0.06 * (ext[3] - ext[2])
    ax.plot([x0, x0 + 1000 * km], [y0, y0], color=colour, lw=3.2, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=5, foreground="0.15")])
    ax.text(x0 + 500 * km, y0 + 0.02 * (ext[3] - ext[2]), f"{km:.0f} km", ha="center",
            va="bottom", fontsize=12, color=colour, **FONT,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="0.15")])


def locator(ax, rect=(0.78, 0.66, 0.20, 0.34)):
    """Idaho in the figure's palette, marker only, no label."""
    from nival.ancillary_plotting import idaho_locator
    loc = ax.inset_axes(list(rect))
    idaho_locator(loc)
    for txt in loc.texts:
        txt.set_visible(False)
    for patch in loc.patches:
        patch.set_facecolor(SEQ(0.88)); patch.set_edgecolor(SEQ(0.35))
        patch.set_linewidth(1.4); patch.set_alpha(1.0)
    for line in loc.lines:
        line.set_color(SEQ(0.06)); line.set_markerfacecolor(SEQ(0.06))
        line.set_markeredgecolor(SEQ(0.06)); line.set_markersize(7)


def _mark(ax, d, suffix="", ec="w"):
    xs, ys = d[0], d[1]
    for box in BOXES:
        rs, cs = closeup_slices(xs, ys, box)
        e = [xs[cs.start], xs[cs.stop - 1], ys[rs.stop - 1], ys[rs.start]]
        ax.add_patch(plt.Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2],
                                   fill=False, ec=ec, lw=1.8))
        ax.text(e[0] + 120, e[3] - 120, f"{BOXES.index(box) + 1}{suffix}", fontsize=12,
                fontweight="bold", color=ec, ha="left", va="top", **FONT,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="0.15")])


def c15_column(d):
    """Single column: the hero pair, then a 2 x 2 of close ups beneath."""
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(9.5, 13))
    top = [fig.add_axes([0.05, 0.56, 0.455, 0.34]), fig.add_axes([0.525, 0.56, 0.455, 0.34])]
    for j, (ax, f, lim, t, anc) in enumerate(((top[0], a, la, "NISAR\nFeb 7-19", "E"),
                                              (top[1], b, lb, "Airborne Lidar\nFeb 7-22", "W"))):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax, anchor=anc); inset_title(ax, t, fontsize=16); _mark(ax, d, suffix="ab"[j])
        if j == 0:
            scale_bar(ax, ext)
        else:
            locator(ax)
    # the close ups are square, so size their boxes to the image and centre the pair;
    # otherwise the axes letterbox and leave a wide empty margin either side
    fw, fh = fig.get_size_inches()
    ch = 0.21
    aspect = CLOSEUP_W / CLOSEUP_H          # boxes follow the window's own shape
    pair_w = 2 * ch * fh / fw * aspect + 0.008
    x0 = (1.0 - pair_w) / 2 + 0.03
    _closeups(fig, d, la, lb, [(x0, 0.29, pair_w, ch), (x0, 0.055, pair_w, ch)],
              label_col=False, row_labels=True)
    fig.text(x0 - 0.055, 0.29 + ch / 2 - 0.117, "Ability to Capture Finer Scale Features",
             rotation=90, ha="center", va="center", fontsize=15, **FONT)
    fig.text(0.5, 0.965, HEADLINE, ha="center", va="top", fontsize=21, **FONT)
    qbar(fig, im, [0.33, 0.535, 0.34, 0.016])
    save(fig, "feb_nisar_lidar_ex.png", dpi=600, out=OUT.parent)


def c16_wide(d):
    """Wide: the hero pair on the left, the close ups stacked on the right."""
    a, b, la, lb, ext = fields(d)
    fig = plt.figure(figsize=(16, 8))
    top = [fig.add_axes([0.02, 0.10, 0.29, 0.76]), fig.add_axes([0.31, 0.10, 0.29, 0.76])]
    for j, (ax, f, lim, t, anc) in enumerate(((top[0], a, la, "NISAR\nFeb 7-19", "E"),
                                              (top[1], b, lb, "Airborne Lidar\nFeb 7-22", "W"))):
        im = ax.imshow(f, extent=ext, cmap=SEQ, norm=norm(lim), origin="upper")
        bare(ax, anchor=anc); inset_title(ax, t, fontsize=16); _mark(ax, d, suffix="ab"[j])
        if j == 0:
            scale_bar(ax, ext)
        else:
            locator(ax)
    _closeups(fig, d, la, lb, [(0.66, 0.50, 0.32, 0.36), (0.66, 0.11, 0.32, 0.36)])
    fig.text(0.31, 0.965, HEADLINE, ha="center", va="top", fontsize=21, **FONT)
    qbar(fig, im, [0.15, 0.045, 0.32, 0.020])
    save(fig, "16_wide.png")


if __name__ == "__main__":
    d = load()
    OUT.mkdir(parents=True, exist_ok=True)
    print("hero concepts:")
    for fn in (c01_baseline, c02_stacked, c03_dark, c04_upright, c05_hillshade,
               c06_warm, c07_minimal, c08_with_zoom, c09_bands, c10_poster,
               c12_suptitle, c13_subtle, c14_hero_with_insets, c15_column, c16_wide):
        try:
            fn(d)
        except Exception as exc:
            print(f"  {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    print(f"-> {OUT}")

"""Draw helpers + shared colour caps for the NISAR dSWE vs NIVAL dHS maps.

Used by paper_figures.py (the figure driver); nothing here saves a figure.
Everything is in metres. Keep this module tight and correct -- it feeds the
load-bearing result maps.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LightSource
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import rotate as ndrotate

from nival.retrieval import SNOW_DENSITY, WATER_DENSITY

LS = LightSource(azdeg=315, altdeg=45)        # hillshade light (NW, 45 deg up)
ANG = -36.0                                    # map rotation so the swath fills the panel

# shared colour caps for the maps [metres] so all resolutions are directly comparable.
# The dSWE caps are the depth caps converted at the new-snow density, so the same colour
# means the same water in both maps; derived here rather than typed twice.
DHS_VMIN, DHS_VMAX = 0.0, 0.75
_DENSITY = SNOW_DENSITY / WATER_DENSITY
DSWE_VMIN, DSWE_VMAX = DHS_VMIN * _DENSITY, DHS_VMAX * _DENSITY


# ------------------------------- small helpers -------------------------------
def rotate(field, cval):
    """Nearest-neighbour rotation by ANG degrees, filling outside with `cval`."""
    return ndrotate(field, ANG, reshape=True, order=0, cval=cval, mode="constant")


def add_contours(ax, elevation, extent, levels, index_metres):
    """Subtle elevation contours, every `index_metres` drawn slightly bolder."""
    levels = np.asarray(levels, float)
    is_index = (np.isclose(np.mod(levels, index_metres), 0, atol=1)
                | np.isclose(np.mod(levels, index_metres), index_metres, atol=1))
    for subset, width, alpha in [(levels[~is_index], 0.35, 0.3), (levels[is_index], 0.9, 0.4)]:
        if subset.size:
            ax.contour(elevation, levels=subset, extent=extent, origin="upper",
                       colors="k", linewidths=width, alpha=alpha)


def add_north_arrow(ax):
    """North arrow in the upper-left for the rotated map (true north is up-and-left)."""
    theta = np.radians(-45)
    base_x, base_y, length = 0.12, 0.82, 0.08
    tip_x, tip_y = base_x + length * np.sin(theta), base_y + length * np.cos(theta)
    ax.annotate("", xy=(tip_x, tip_y), xytext=(base_x, base_y), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
    ax.text(tip_x - 0.012, tip_y + 0.015, "N", transform=ax.transAxes, ha="center",
            va="center", fontsize=14, fontweight="bold", color="k")


def km_ticks(ax, posting_metres, step_metres, left=True):
    """Label map axes in km (every `step_metres`) -- a scale for the rotated swath."""
    to_km = FuncFormatter(lambda value, pos: f"{value / 1000:.0f}")
    ax.xaxis.set_major_locator(MultipleLocator(step_metres))
    ax.xaxis.set_major_formatter(to_km)
    ax.set_xlabel("km", fontsize=13)
    if left:
        ax.yaxis.set_major_locator(MultipleLocator(step_metres))
        ax.yaxis.set_major_formatter(to_km)
        ax.set_ylabel("km", fontsize=13)
    else:
        ax.set_yticks([])


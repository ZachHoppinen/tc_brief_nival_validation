"""RETIRED: Fig. S2, snowpack temperature at both stations through the winter
-> old/figures/supplement/figS2_temperature_profiles.png. Fig. 1 now carries the Freeman
temperature record across the season. Also holds the commented-out three-pair figure that
was dropped before it.

    conda run -n nival python old/figS2_temperature.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nival.pairs import ANALYSED, on_track   # noqa: E402
from nival.supplement import wetness   # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "supplement"
T0, T1 = "2025-11-01", "2026-04-05"


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# The three 12-day pairs side by side. Dropped from the supplement: the wet-at
# column in Table S1 already says which of them are wet, so the figure repeated
# the point. Kept commented in case it is wanted again.
# def fig_twelve_day(frame):
#     twelve = frame[frame.days == 12]
#     fig, axes = plt.subplots(len(twelve), 3, figsize=(11, 3.5 * len(twelve)), dpi=300,
#                              squeeze=False)
#     for i, (_, x) in enumerate(twelve.iterrows()):
#         ds = xr.open_dataset(PRODUCTS / f"{x['key']}.nc")
#         ext = [0, (ds.x.values.max() - ds.x.values.min()) / 1000,
#                0, (ds.y.values.max() - ds.y.values.min()) / 1000]
#         lidar = ds["dhs"].values * DENS * 1000
#         nisar = np.where(np.isfinite(lidar), ds["dswe"].values * 1000, np.nan)
#         panels = [(ds["coherence"].values, "magma", 0, 0.8, "coherence"),
#                   (nisar, "viridis", -50, 150, "NISAR dSWE (mm)"),
#                   (lidar, "viridis", -50, 150, "lidar SWE (mm)")]
#         for j, (field, cmap, lo, hi, lab) in enumerate(panels):
#             ax = axes[i, j]
#             im = ax.imshow(field, extent=ext, origin="upper", cmap=cmap, vmin=lo, vmax=hi)
#             ax.set_xticks([]); ax.set_yticks([])
#             fig.colorbar(im, ax=ax, shrink=0.82, label=lab)
#             if j == 0:
#                 ax.set_ylabel(f"{x['lab']}\n{'wet at an acquisition' if x['wet'] else 'dry at both'}",
#                               fontsize=10)
#             if i == 0:
#                 ax.set_title(["coherence", "NISAR", "lidar"][j], fontsize=11)
#     fig.tight_layout()
#     _save(fig, "figS3_twelve_day.png")
#
#
def _blank_above_surface(frame, surface):
    """Sensors above the snow read air, not pack."""
    out = frame.copy()
    surf = surface.reindex(out.index, method="nearest")
    for when in out.index:
        h = surf.get(when, np.nan)
        if np.isfinite(h):
            out.loc[when, [c for c in out.columns if c > h - wetness.SURFACE_SKIP_M]] = np.nan
    return out


def fig_temperature():
    from matplotlib.colors import Normalize
    from nival.paper_figures import SNOW_TEMP_VMIN, snow_temp_colormap
    # same scale as figure 1: linear white-to-blue over the cold range, red at the melting
    # point. A gamma-4 power norm used to spend most of the colorbar on the last degree.
    norm = Normalize(vmin=SNOW_TEMP_VMIN, vmax=0)
    cmap = snow_temp_colormap()
    ticks = [-8, -6, -4, -2, 0]
    acqs = sorted({d for p in on_track("077") for d in (p[1], p[2])})
    stamps = [wetness.overpass("077", wetness.date_of(d)) for d in acqs]
    analysed = [wetness.overpass("077", wetness.date_of(d)) for d in ANALYSED[1:]]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=300, sharex=True)
    for ax, (frame, depth, title) in zip(axes, [
            (*wetness.snotel_profile(), f"Mores Creek Summit SNOTEL, {wetness.SNOTEL_ELEV_M:.0f} m"),
            (*wetness.freeman_profile(), f"Freeman, {wetness.FREEMAN_ELEV_M:.0f} m")]):
        # both records are hourly; daily means smear the diurnal melt signal we care about
        hourly = frame.loc[T0:T1].resample("1h").mean()
        dep = depth.resample("1h").mean().interpolate(limit=24).loc[T0:T1]
        blank = _blank_above_surface(hourly, dep)
        mesh = ax.pcolormesh(blank.index, np.asarray(blank.columns, float), blank.values.T,
                             cmap=cmap, norm=norm, shading="nearest")
        ax.plot(dep.index, dep.values, color="k", lw=1.0)
        ax.set_ylim(0, 2.2)
        ax.set_ylabel("height above ground (m)")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label="snow temperature (°C)", pad=0.01, ticks=ticks, extend="both")

    for ax in axes:
        for s in stamps:
            ax.axvline(s, color="0.35", lw=0.7)
        for s in analysed:
            ax.axvline(s, color="k", lw=2.0)
        ax.set_xlim(pd.Timestamp(T0), pd.Timestamp(T1))
    axes[1].set_xlabel("date")
    axes[0].plot([], [], color="0.35", lw=0.7, label="NISAR acquisition")
    axes[0].plot([], [], color="k", lw=2.0, label="analyzed pair")
    axes[0].plot([], [], color="k", lw=1.2, label="snow surface")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "figS2_temperature_profiles.png")


if __name__ == "__main__":
    fig_temperature()

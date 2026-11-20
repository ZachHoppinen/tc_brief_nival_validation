"""Supplement assets for the track-077 pair census.

  table_s1.tex   the seven candidate pairs on track 077
  figS1          coherence for all seven
  figS2          snowpack temperature at both stations through the winter
  figS3          the three 12-day pairs, coherence / NISAR / lidar

Everything the paper depends on is built from tracked code: the pair list comes from
nival.pairs and the wetness from nival.wetness, not from scratch.

    conda run -n nival python src/nival/supplement_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nival import paths, retrieval, wetness   # noqa: E402
from nival.pairs import ANALYSED, label, on_track   # noqa: E402

OUT = paths.FIGURES_DIR / "supplement"
PAPER = Path.home() / "Documents/nival_paper"
PRODUCTS = paths.OUTPUTS / "retrieval_multipair"
DENS = retrieval.SNOW_DENSITY / retrieval.WATER_DENSITY
MONTH = {"11": "Nov", "12": "Dec", "01": "Jan", "02": "Feb", "03": "Mar"}
T0, T1 = "2025-11-01", "2026-04-05"


def nice(stamp):
    """'20260207' -> 'Feb 7' (month first, matching the manuscript's date style)."""
    return f"{MONTH[stamp[4:6]]} {int(stamp[6:])}"


def _snotel_swe():
    """SNOTEL 637 daily SWE in metres, converted from inches at ingest."""
    from nival.stage_ancillary import fetch_snotel_season
    if not paths.SNOTEL_SEASON.exists():
        fetch_snotel_season()
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"])
    metres = season["wteq_in"].values * 0.0254
    if np.nanmax(metres) > 3.0:
        raise ValueError("SNOTEL SWE is not plausible in metres; check units")
    swe = pd.Series(metres, index=season["date"]).to_dict()
    # the 06:00 overpass snapshot is what the main text reports; it only spans February onward
    snap = pd.read_csv(paths.SNOTEL_SNAP, parse_dates=["date"])
    swe.update(dict(zip(snap["date"], snap["swe_in_06"].values * 0.0254)))
    return swe


def census(track="077"):
    """One row per candidate pair: interval, coherence, wetness, accumulation, r."""
    rep = pd.read_csv(paths.OUTPUTS / "multipair_stats.csv").set_index("pair") \
        if (paths.OUTPUTS / "multipair_stats.csv").exists() else None
    wet_ends = wetness.acquisition_table().set_index(["pair", "end"])
    swe = _snotel_swe()
    rows = []
    for pair in on_track(track):
        t, ref, sec = pair[0], pair[1], pair[2]
        name = f"{t} {ref[4:]}-{sec[4:]}"
        path = PRODUCTS / f"{label(pair)}.nc"
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        dswe, dhs = ds["dswe"].values, ds["dhs"].values
        good = np.isfinite(dswe) & np.isfinite(dhs)
        days = (pd.Timestamp(sec) - pd.Timestamp(ref)).days
        rows.append(dict(
            name=name, key=label(pair), lab=f"{nice(ref)}--{nice(sec)}", days=days,
            coh=float(np.nanmedian(ds["coherence"].values)),
            r=float(np.corrcoef(dhs[good], dswe[good])[0, 1]),
            dhs_cm=float(np.nanmean(dhs[good]) * 100),
            dswe_mm=(swe.get(pd.Timestamp(sec), np.nan) - swe.get(pd.Timestamp(ref), np.nan)) * 1000,
            snotel_ref=wet_ends.loc[(name, "ref"), "snotel_top_c"],
            snotel_sec=wet_ends.loc[(name, "sec"), "snotel_top_c"],
            freeman_ref=wet_ends.loc[(name, "ref"), "freeman_top_c"],
            freeman_sec=wet_ends.loc[(name, "sec"), "freeman_top_c"],
            wet_ref=bool(wet_ends.loc[(name, "ref"), "wet"]),
            wet_sec=bool(wet_ends.loc[(name, "sec"), "wet"]),
            wet=bool(wet_ends.loc[(name, "ref"), "wet"] or wet_ends.loc[(name, "sec"), "wet"]),
            analysed=(pair[:3] == ANALYSED)))
    return pd.DataFrame(rows)      # on_track() already returns them in date order


def table_s1(frame):
    """One column naming which acquisition was wet, rather than four temperature columns."""
    def wet_at(x):
        return {(True, True): "both", (True, False): "first",
                (False, True): "second", (False, False): "--"}[(x["wet_ref"], x["wet_sec"])]

    lines = [r"\begin{tabular}{lrrlrr}", r"\hline",
             r"Pair & $\Delta t$ (d) & $\bar{\gamma}$ & wet at & $\Delta$SWE (mm) & $r$ \\",
             r"\hline"]
    for _, x in frame.iterrows():
        bold = r"\textbf{" if x["analysed"] else ""
        end = "}" if bold else ""
        lines.append(f"{bold}{x['lab']}{end} & {x['days']} & {x['coh']:.2f} & "
                     f"{wet_at(x)} & ${x['dswe_mm']:+.0f}$ & ${x['r']:+.2f}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (PAPER / "table_s1.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {PAPER / 'table_s1.tex'}")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (PAPER / "figures" / name).write_bytes((OUT / name).read_bytes())
    print(f"wrote {name}")


def fig_coherence(frame):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), dpi=300)
    axes = axes.ravel()
    for ax in axes[len(frame):]:
        ax.axis("off")
    for ax, (_, x) in zip(axes, frame.iterrows()):
        ds = xr.open_dataset(PRODUCTS / f"{x['key']}.nc")
        coh = ds["coherence"].values
        ext = [0, (ds.x.values.max() - ds.x.values.min()) / 1000,
               0, (ds.y.values.max() - ds.y.values.min()) / 1000]
        im = ax.imshow(coh, extent=ext, origin="upper", cmap="magma", vmin=0, vmax=0.8)
        mark = "  (analyzed)" if x["analysed"] else ""
        ax.set_title(f"{x['lab']}, {x['days']} d{mark}\nmedian {np.nanmedian(coh):.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="coherence")
    fig.tight_layout()
    _save(fig, "figS1_coherence_077.png")


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
    from nival.ancillary_plotting import SNOW_TEMP_VMIN, snow_temp_colormap
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
    frame = census()
    print(f"{len(frame)} track-077 pairs; {int((~frame.wet).sum())} dry at both stations")
    print(frame[["lab", "days", "coh", "r", "wet", "analysed"]].to_string(index=False))
    table_s1(frame)
    fig_coherence(frame)
    # fig_twelve_day(frame)
    fig_temperature()

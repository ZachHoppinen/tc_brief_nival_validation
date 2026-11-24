"""Every figure-side piece of the TC brief communication, in one module.

  context_figure()  -> fig1_context.png : study-area map, then season-long snow depth + snowpack temp
  results_figure()  -> fig2_results.png : scene dSWE/dHS maps + density hexbin
  swe_timeline_figure() -> fig3a_pair_timeline.png, fig3b_season_totals.png : SWE change per pair with SNOTEL and lidar; season totals

Laid out top to bottom as: the map draw helpers and shared colour caps, the met-series
readers and panel builders, the study-area map, then the three figures. paper_stats and
download_paper_data read the met readers, the IONEX reader and the NAIP fetch from here too.
Run after run_paper_workflow.py's process step:

    python src/nival/paper_figures.py        # in the project env (see environment.yml)
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import cartopy.io.shapereader as shpreader
import cmocean
import h5py
import matplotlib
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from matplotlib.colors import LightSource, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the src/ dir
from nival import paths, retrieval   # noqa: E402
from nival.build_multipair import PRODUCTS   # noqa: E402
from nival.download_paper_data import TRACK, lidar_file, local_gunws   # noqa: E402
from nival.pairs import ANALYSED, PAIRS, label   # noqa: E402
from nival.supplement import wetness   # noqa: E402   (the melting-point tolerance Fig. 1 marks)
from nival.retrieval import (SNOW_DENSITY, WATER_DENSITY,   # noqa: E402
                             _elevation_on_grid, _raster_on_grid)


# ============================ names used on every figure ============================
# What the two measurements are called on colour bars, axes and panel titles, so Figs. 2 and 3
# say the same thing the same way. NISAR is always SWE in mm, the lidar always snow depth in m;
# "change" is there when the field is a difference between two dates.
SWE_CHANGE = "NISAR SWE change"
DEPTH_CHANGE = "Lidar snow depth change"
DEPTH = "Lidar snow depth"
SWE_UNIT, DEPTH_UNIT = "mm", "m"
SWE_CHANGE_LABEL = f"{SWE_CHANGE} ({SWE_UNIT})"
DEPTH_CHANGE_LABEL = f"{DEPTH_CHANGE} ({DEPTH_UNIT})"
DEPTH_LABEL = f"{DEPTH} ({DEPTH_UNIT})"


# ===================== map draw helpers + shared colour caps =====================
LS = LightSource(azdeg=315, altdeg=45)        # hillshade light (NW, 45 deg up)

# shared colour caps for the maps [metres] so all resolutions are directly comparable.
# The dSWE caps are the depth caps converted at the new-snow density, so the same colour
# means the same water in both maps; derived here rather than typed twice.
DHS_VMIN, DHS_VMAX = 0.0, 0.75
_DENSITY = SNOW_DENSITY / WATER_DENSITY
DSWE_VMIN, DSWE_VMAX = DHS_VMIN * _DENSITY, DHS_VMAX * _DENSITY


# ------------------------------- small helpers -------------------------------
def density_scatter(ax, x, y, extent):
    """Point density of `y` against `x` on a log count scale, with r and n in the title: the
    one scatter style Figs. 2 and 3 share. `extent` is (xmin, xmax, ymin, ymax). Returns the
    hexbin for its colour bar."""
    both = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[both], y[both])[0, 1]
    hexes = ax.hexbin(x[both], y[both], gridsize=38, cmap="magma", mincnt=1, norm=LogNorm(), extent=extent)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"r = {r:.2f},  n = {int(both.sum())}")
    ax.spines[["top", "right"]].set_visible(False)
    return hexes


def density_fit(ax, depth_m, swe_mm, loc, fontsize=9):
    """Draw the straight-line fit of SWE [mm] on snow depth [m] across a density scatter and
    label it with the density its slope implies: mm of SWE per m of depth is kg/m3. An ordinary
    fit with a free intercept, not forced through zero. Returns (slope, intercept)."""
    both = np.isfinite(depth_m) & np.isfinite(swe_mm)
    slope, intercept = np.polyfit(depth_m[both], swe_mm[both], 1)
    ends = np.array(ax.get_xlim())
    ax.plot(ends, slope * ends + intercept, color="#4575b4", lw=1.8, ls="--",
            label=f"Implied {slope:.0f} kg m$^{{-3}}$ density")
    ax.legend(loc=loc, fontsize=fontsize, framealpha=0.9)
    return slope, intercept


def panel_letter(ax, letter, fontsize=12):
    """Write a bold panel letter, like (a), just above the upper right corner of the axis, clear of the data."""
    ax.text(1.0, 1.015, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=fontsize,
            fontweight="bold")


def count_colorbar(colorbar):
    """Label a density scatter's colour bar and show plain counts on its log scale instead of
    powers of ten. Only 1, 2 and 5 in each decade are labelled, so the labels do not collide."""
    colorbar.set_label("pixel count")
    sparse = FuncFormatter(lambda value, pos: f"{value:.0f}" if f"{value:.0f}"[0] in "125" else "")
    colorbar.ax.yaxis.set_major_formatter(sparse)
    colorbar.ax.yaxis.set_minor_formatter(sparse)


def add_contours(ax, elevation, extent, levels, index_metres):
    """Subtle elevation contours, every `index_metres` drawn slightly bolder."""
    levels = np.asarray(levels, float)
    is_index = (np.isclose(np.mod(levels, index_metres), 0, atol=1)
                | np.isclose(np.mod(levels, index_metres), index_metres, atol=1))
    for subset, width, alpha in [(levels[~is_index], 0.35, 0.3), (levels[is_index], 0.9, 0.4)]:
        if subset.size:
            ax.contour(elevation, levels=subset, extent=extent, origin="upper",
                       colors="k", linewidths=width, alpha=alpha)


EPSG = paths.EPSG
NISAR_COLOR, LIDAR_COLOR = "#1db954", "#8f3fe0"


# ======================= met time series: read the data ========================
OVERPASS_UTC_H = 12 + 46 / 60 + 36 / 3600        # ASC-077 dawn pass (05:46 MST)
IN_TO_MM = 25.4

# acquisitions marked through every panel
NISAR = [("2026-02-07", "NISAR"), ("2026-02-19", "NISAR")]
LIDAR = [("2026-02-07", "Lidar"), ("2026-02-22", "Lidar")]
PIT_DATE, PIT_SWE_MM = "2026-02-07", 338.0       # Freeman snow pit at NISAR A (in-situ truth)

# Freeman DTC snow-temperature string: 37 channels, 4-inch beads, ground at sensor 35, 20 dead
DTC_COLS = ["DTC1_Avg"] + [f"DTC_Avg({i})" for i in range(2, 38)]
DTC_SPACING = 4 * 25.4 / 1000                    # 0.1016 m

# Snowpack temperature colormap: linear white-to-blue over the cold range, with the top
# wetness.MELT_TOL_C of the scale in red so pack at the melting point (the criterion the
# dry-snow argument rests on) is the one thing that stands out. Shared with figure S2.
SNOW_TEMP_VMIN = -8.0


def snow_temp_colormap(vmin=SNOW_TEMP_VMIN, tol=None, n=4096):
    """Blues_r with its warmest `tol` degrees replaced by red, so 0 C reads at a glance.

    A bin turns red only when its whole width lies within `tol` of melting, so the red
    never runs colder than the criterion it stands for (256 bins put the edge at -0.094 C
    rather than -0.1; 4096 holds it to a thousandth of a degree)."""
    tol = wetness.MELT_TOL_C if tol is None else tol
    edges = np.linspace(0, 1, n, endpoint=False)          # left (cold) edge of each color bin
    colors = plt.get_cmap("Blues_r")(edges + 0.5 / n)
    colors[edges >= 1 - tol / abs(vmin)] = [0.84, 0.19, 0.15, 1.0]
    return ListedColormap(colors)

DTC_GROUND, DTC_DEAD = 35, 20
SOIL_DEPTHS = [5, 10, 20]


def read_snotel():
    """SNOTEL 637 06:00-local series from the frozen CSV: daily axis + SWE (mm),
    snow depth [cm], air temp [C]."""
    df = pd.read_csv(paths.SNOTEL_SNAP, parse_dates=["date"])
    t = np.array([d.date() for d in df["date"]])
    swe = np.where(df["swe_in_06"].notna(), df["swe_in_06"], df["swe_in_daily"]).astype(float) * IN_TO_MM
    depth = df["snwd_in_daily"].to_numpy(float) * 2.54
    temp = (df["tobs_F_06"].to_numpy(float) - 32.0) * 5.0 / 9.0
    return t, swe, depth, temp


def read_ionex(path):
    """(epoch_hours[N], lats, lons, tec[N,nlat,nlon]) in TECU from an IGS GIM file."""
    lines = Path(path).read_text().splitlines()
    exponent = -1
    for ln in lines:
        if "EXPONENT" in ln:
            exponent = int(ln.split()[0])
        elif "LAT1 / LAT2 / DLAT" in ln:
            lat1, lat2, dlat = map(float, ln.split()[:3])
        elif "LON1 / LON2 / DLON" in ln:
            lon1, lon2, dlon = map(float, ln.split()[:3])
        elif "END OF HEADER" in ln:
            break
    lats = np.arange(lat1, lat2 + np.sign(dlat) * 1e-6, dlat)
    lons = np.arange(lon1, lon2 + np.sign(dlon) * 1e-6, dlon)
    scale = 10.0 ** exponent
    epochs, maps, i = [], [], 0
    while i < len(lines):
        if "START OF TEC MAP" in lines[i]:
            i += 1
            ep = list(map(int, lines[i].split()[:6]))
            epochs.append(ep[3] + ep[4] / 60 + ep[5] / 3600)
            grid, i = np.full((len(lats), len(lons)), np.nan), i + 1
            while "END OF TEC MAP" not in lines[i]:
                if "LAT/LON1/LON2/DLON/H" in lines[i]:
                    j = int(round((float(lines[i][:8]) - lat1) / dlat))
                    vals, i = [], i + 1
                    while "LAT/LON1/LON2/DLON/H" not in lines[i] and "END OF TEC MAP" not in lines[i]:
                        vals += [int(lines[i][k:k + 5]) for k in range(0, len(lines[i].rstrip()), 5)]
                        i += 1
                    grid[j, :len(vals)] = np.array(vals) * scale
                else:
                    i += 1
            maps.append(grid)
        i += 1
    ep = np.array(epochs, float)
    for k in range(1, len(ep)):                  # unwrap the 24:00-as-0 rollover
        while ep[k] <= ep[k - 1]:
            ep[k] += 24.0
    return ep, lats, lons, np.array(maps)


def _bilinear(lats, lons, grid, lat, lon):
    fi = np.interp(lat, lats[::-1], np.arange(len(lats))[::-1])
    fj = np.interp(lon, lons, np.arange(len(lons)))
    i0, j0 = int(np.floor(fi)), int(np.floor(fj))
    i1, j1 = min(i0 + 1, len(lats) - 1), min(j0 + 1, len(lons) - 1)
    di, dj = fi - i0, fj - j0
    return ((1 - di) * (1 - dj) * grid[i0, j0] + (1 - di) * dj * grid[i0, j1]
            + di * (1 - dj) * grid[i1, j0] + di * dj * grid[i1, j1])


def vtec_overpass(doy):
    """Site VTEC [TECU] interpolated to the overpass time from one IONEX day file."""
    ep, lats, lons, tec = read_ionex(paths.IONEX_DIR / f"IGS0OPSFIN_{paths.YEAR}{doy:03d}0000_01D_02H_GIM.INX")
    series = np.array([_bilinear(lats, lons, tec[k], paths.SITE_LAT, paths.SITE_LON) for k in range(len(ep))])
    return float(np.interp(OVERPASS_UTC_H, ep, series))


def freeman_06():
    """Freeman on-site 06:00 air temp [C] + SnoDAR snow depth [cm], date-keyed."""
    s = pd.read_csv(paths.MET_SOIL, usecols=["TIMESTAMP", "AirTC_Avg", "SnoDAR_snow_depth_Avg"])
    s["TIMESTAMP"] = pd.to_datetime(s["TIMESTAMP"])
    s["temp_C"] = pd.to_numeric(s["AirTC_Avg"], errors="coerce")
    s["hs_cm"] = pd.to_numeric(s["SnoDAR_snow_depth_Avg"], errors="coerce") * 100
    at6 = s[s["TIMESTAMP"].dt.strftime("%H:%M") == "06:00"]
    return (dict(zip(at6["TIMESTAMP"].dt.date, at6["temp_C"])),
            dict(zip(at6["TIMESTAMP"].dt.date, at6["hs_cm"])))


def pilot_06():
    """Pilot Peak air temp [C] nearest 06:00 local per day (CSV is degF, 15-min)."""
    p = pd.read_csv(paths.MET_PILOT)
    p["dtt"] = pd.to_datetime(p["datetime_mdt"])
    p["temp_C"] = (pd.to_numeric(p["temp_F"], errors="coerce") - 32.0) * 5.0 / 9.0
    p["d6"] = (p["dtt"].dt.hour * 60 + p["dtt"].dt.minute - 360).abs()
    sel = p.loc[p.groupby(p["dtt"].dt.date)["d6"].idxmin()]
    return dict(zip(sel["dtt"].dt.date, sel["temp_C"]))


def soil_06(dates):
    """Freeman SoilVUE temp [C] + liquid water [m3/m3] at 06:00, per depth, on `dates`."""
    cols = ["TIMESTAMP"] + [f"SoilVUE_{q}_{d}cm_Avg" for d in SOIL_DEPTHS for q in ("T", "VWC")]
    s = pd.read_csv(paths.MET_SOIL, usecols=cols)
    s["TIMESTAMP"] = pd.to_datetime(s["TIMESTAMP"])
    for c in cols[1:]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    at6 = s[s["TIMESTAMP"].dt.strftime("%H:%M") == "06:00"]
    by_date = {c: dict(zip(at6["TIMESTAMP"].dt.date, at6[c])) for c in cols[1:]}
    temp = {d: np.array([by_date[f"SoilVUE_T_{d}cm_Avg"].get(x, np.nan) for x in dates]) for d in SOIL_DEPTHS}
    water = {d: np.array([by_date[f"SoilVUE_VWC_{d}cm_Avg"].get(x, np.nan) for x in dates]) for d in SOIL_DEPTHS}
    return temp, water


def snowpack_thermograph(start=paths.D0, end=paths.D1):
    """Freeman DTC string -> continuous snowpack temperature field (height x time) between
    `start` and `end`. Returns (tnum, z_grid, G_masked, surface_height, timestamps)."""
    s = pd.read_csv(paths.MET_SOIL, usecols=["TIMESTAMP", "SnoDAR_snow_depth_Avg"] + DTC_COLS)
    s["TIMESTAMP"] = pd.to_datetime(s["TIMESTAMP"])
    s = s[(s["TIMESTAMP"] >= pd.Timestamp(start)) & (s["TIMESTAMP"] <= pd.Timestamp(end))].reset_index(drop=True)
    tnum = mdates.date2num(s["TIMESTAMP"].dt.to_pydatetime())
    hs = pd.to_numeric(s["SnoDAR_snow_depth_Avg"], errors="coerce").to_numpy()
    sensors = np.arange(1, 38)
    height = (DTC_GROUND - sensors) * DTC_SPACING
    keep = (sensors != DTC_DEAD) & (height >= -DTC_SPACING) & (height <= np.nanmax(hs) + 2 * DTC_SPACING)
    sensor_heights = height[keep]
    temps = s[[DTC_COLS[i - 1] for i in sensors[keep]]].apply(pd.to_numeric, errors="coerce").to_numpy()
    order = np.argsort(sensor_heights); sensor_heights, temps = sensor_heights[order], temps[:, order]
    z = np.arange(0, np.nanmax(hs) + DTC_SPACING, 0.01)
    # Every buried sensor, drawn at its own reading up to the snow surface. The wetness
    # criterion starts below wetness.SURFACE_SKIP_M, so the top few cm shown here are outside
    # it and can sit at the melting point while the pair still classifies dry (the Feb 7
    # overpass is exactly that). Showing the measurement beats extrapolating a cooler one over it.
    field = np.array([np.interp(z, sensor_heights, temps[k]) for k in range(len(tnum))])
    field[z[None, :] > hs[:, None]] = np.nan             # mask air above the surface
    return tnum, z, field, hs, s["TIMESTAMP"]


def load_met_series():
    """Every series the two met figures need, on the SNOTEL daily date axis."""
    t, swe, depth6_s, temp6_s = read_snotel()
    air_temp, snow_depth = freeman_06()
    temp6 = np.array([air_temp.get(d, np.nan) for d in t])
    depth6 = np.array([snow_depth.get(d, np.nan) for d in t])
    temp6_p = np.array([pilot_06().get(d, np.nan) for d in t])
    Tsoil, Vsoil = soil_06(t)
    doy0, doy1 = (dt.date.fromisoformat(x).timetuple().tm_yday for x in (paths.D0, paths.D1))
    tdates = [dt.date(paths.YEAR, 1, 1) + dt.timedelta(d - 1) for d in range(doy0, doy1 + 1)]
    vtec = np.array([vtec_overpass(d) for d in range(doy0, doy1 + 1)])
    return dict(t=t, swe=swe, depth6=depth6, depth6_s=depth6_s, temp6=temp6, temp6_s=temp6_s,
                temp6_p=temp6_p, Tsoil=Tsoil, Vsoil=Vsoil, tdates=tdates, vtec=vtec,
                thermo=snowpack_thermograph())


# ============================ met time series: draw ============================
def mark_acquisitions(ax, top=False, pad=1.0, nisar=None, lidar=None):
    """Vertical markers for NISAR acquisitions + lidar flights; labels on the top panel.
    `pad` scales the label heights above the axis (use <1 for a tighter composite). `nisar`
    and `lidar` are lists of dates; left out, the Fig. 2 pair and its two flights are marked."""
    nisar = [dt.date.fromisoformat(d) for d, _ in NISAR] if nisar is None else nisar
    lidar = [dt.date.fromisoformat(d) for d, _ in LIDAR] if lidar is None else lidar
    for date in nisar:
        ax.axvline(date, color=NISAR_COLOR, lw=1.1, alpha=0.85, zorder=1)
    for date in lidar:
        ax.axvline(date, color=LIDAR_COLOR, lw=2.1, alpha=0.85, ls=(0, (4, 3)), zorder=1)
    if not top:
        return
    y_nisar, y_arrow, y_pair, y_lidar = (1.0 + h * pad for h in (0.18, 0.21, 0.235, 0.29))
    for date, label in NISAR:
        ax.annotate(label, (dt.date.fromisoformat(date), y_nisar), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=12, color=NISAR_COLOR, fontweight="bold")
    for date, label in LIDAR:
        ax.annotate(label, (dt.date.fromisoformat(date), y_lidar), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=12, color=LIDAR_COLOR, fontweight="bold")
    ax.annotate("", xy=(dt.date(2026, 2, 17), y_arrow), xycoords=("data", "axes fraction"),
                xytext=(dt.date(2026, 2, 9), y_arrow), textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<->", color=NISAR_COLOR, lw=1.6))
    ax.annotate("InSAR Pair", (dt.date(2026, 2, 13), y_pair), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=11, color=NISAR_COLOR, fontweight="bold")


def render_time_series(data, panels, name):
    """Stack the given panel functions on a shared date axis, mark acquisitions, save."""
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14.5, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 2.6 * len(panels)),
                             sharex=True, constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    for i, panel in enumerate(panels):
        panel(axes[i], data)
        mark_acquisitions(axes[i], top=(i == 0))
    bottom = axes[-1]
    bottom.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    bottom.set_xlim(dt.date.fromisoformat(paths.D0), dt.date.fromisoformat(paths.D1))
    bottom.set_xlabel("2026")
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES_DIR / name, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {paths.FIGURES_DIR / name}")


def panel_air_temperature(ax, data):
    t, temp = data["t"], data["temp6"]
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.fill_between(t, 0, temp, where=temp >= 0, color="#d1495b", alpha=0.30, interpolate=True)
    ax.fill_between(t, 0, temp, where=temp < 0, color="#3f88c5", alpha=0.30, interpolate=True)
    ax.plot(t, data["temp6_s"], color="#b3001b", lw=1.8, ls=(0, (5, 2)), alpha=0.85, label="SNOTEL 637", zorder=2)
    ax.plot(t, data["temp6_p"], color="#3d348b", lw=2.0, label="Pilot Peak", zorder=2)
    ax.plot(t, temp, color="#b3001b", lw=2.4, marker="o", ms=4, label="Freeman", zorder=3)
    ax.set_ylabel("Air Temp (°C)")
    ax.set_ylim(-18, 5.5); ax.set_yticks([-15, -10, -5, 0, 5])
    ax.set_title("Air Temperature (06:00 local)", loc="left")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)


def acq_legend_handles():
    """NISAR (red) + Lidar (purple dashed) line proxies matching mark_acquisitions."""
    return [Line2D([], [], color=NISAR_COLOR, lw=1.1, label="NISAR"),
            Line2D([], [], color=LIDAR_COLOR, lw=2.1, ls=(0, (3, 2)), label="Lidar")]   # denser dash fills the handle


def panel_swe(ax, data, mark_acq=False):
    t, swe = data["t"], data["swe"]
    ax.fill_between(t, swe, 0, color="#4c8cbf", alpha=0.20)
    ax.plot(t, swe, color="#1f6aa5", lw=2.6, label="SWE: SNOTEL")
    ax.set_ylabel("SWE (mm)")
    ax.set_ylim(swe.min() - 35, swe.max() + 14)
    ax.set_title("Snow Water Equivalent", loc="left")
    depth_axis = ax.twinx()
    # line style carries the quantity (solid SWE, dashed depth), colour carries the station
    depth_axis.plot(t, data["depth6"], color="#588157", lw=2.0, ls=(0, (5, 2)), label="Depth: Freeman")
    depth_axis.plot(t, data["depth6_s"], color="#1f6aa5", lw=2.0, ls=(0, (5, 2)), alpha=0.85, label="Depth: SNOTEL")
    depth_axis.set_ylabel("Snow Depth (cm)", color="0.4"); depth_axis.tick_params(colors="0.4")
    depth_axis.spines["top"].set_visible(False)
    swe_handles, swe_labels = ax.get_legend_handles_labels()
    depth_handles, depth_labels = depth_axis.get_legend_handles_labels()
    extra = acq_legend_handles() if mark_acq else []
    depth_axis.legend(swe_handles + depth_handles + extra,
                      swe_labels + depth_labels + [h.get_label() for h in extra],
                      loc="lower right", fontsize=9, framealpha=0.9)


def panel_snowpack_temperature(ax, data, mark_acq=False):
    ax.grid(False)
    times, heights, temperature, surface, timestamps = data["thermo"]
    # linear scale over the cold range: 98% of the pack sits at or below freezing, and a
    # two-slope norm gave the top 2% half the colorbar. 0 C is the warm endpoint, so the
    # melting point reads off the end of the bar rather than a stretched midpoint.
    mesh = ax.pcolormesh(times, heights, temperature.T, cmap=snow_temp_colormap(),
                         norm=Normalize(vmin=SNOW_TEMP_VMIN, vmax=0), shading="nearest", zorder=0)
    ax.plot(timestamps, surface, color="k", lw=1.8, label="Snow Surface")
    ax.set_ylim(0, float(np.nanmax(surface)) + DTC_SPACING)
    ax.set_ylabel("Height above Ground (m)")
    ax.set_title("Freeman Snowpack Temperature", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    if mark_acq:
        handles += acq_legend_handles(); labels += ["NISAR", "Lidar"]
    ax.legend(handles, labels, loc="upper left", fontsize=9, framealpha=0.9, handlelength=2.8)
    cax = inset_axes(ax, width="1.8%", height="100%", loc="lower left",
                     bbox_to_anchor=(1.015, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    colorbar = ax.figure.colorbar(mesh, cax=cax, extend="both", ticks=[-8, -6, -4, -2, 0])
    colorbar.set_label("Snow Temp (°C)", fontsize=10)


def panel_soil(ax, data):
    t = data["t"]; colors = {5: "#8c510a", 10: "#bf812d", 20: "#dfc27d"}
    ax.axhline(0, color="0.55", lw=1, ls="--")
    for depth in SOIL_DEPTHS:
        ax.plot(t, data["Tsoil"][depth], color=colors[depth], lw=2.2, label=f"{depth} cm")
    ax.set_ylabel("Soil Temp (°C)"); ax.set_ylim(-5, 2)
    ax.set_title("Freeman Soil Temperature and Liquid Water", loc="left")
    ax.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9, title="Depth")
    water_axis = ax.twinx()
    for depth in SOIL_DEPTHS:
        water_axis.plot(t, data["Vsoil"][depth], color=colors[depth], lw=1.8, ls=":")
    water_axis.set_ylabel("Liquid Water [m³ m⁻³]", color="0.4"); water_axis.tick_params(colors="0.4")
    water_axis.set_ylim(-0.15, 0.06); water_axis.set_yticks([0, 0.05]); water_axis.spines["top"].set_visible(False)
    ax.annotate("Liquid Water ≈ 0 (frozen, all depths)", (dt.date(2026, 2, 13), 0.3),
                ha="center", va="bottom", fontsize=12, color="0.4", style="italic")


def panel_ionosphere(ax, data):
    dates, vtec = data["tdates"], data["vtec"]
    ax.fill_between(dates, vtec, 0, color="#2a9d8f", alpha=0.15)
    ax.plot(dates, vtec, color="#1d7d72", lw=2.4)
    for date, _ in NISAR:
        day = dt.date.fromisoformat(date); value = vtec[dates.index(day)]
        ax.plot(day, value, "o", color="#1d7d72", ms=9, mec="w", mew=1.4, zorder=4)
        ax.annotate(f"{value:.1f} TECU", (day, value), xytext=(7, 6), textcoords="offset points",
                    fontsize=12, color="#1d7d72", fontweight="bold")
    ax.set_ylabel("Ionospheric VTEC [TECU]"); ax.set_ylim(0, 12)
    ax.set_title("Ionospheric TEC (06:00 local)", loc="left")


# ============================ study-area map (fig 1) ============================
MAP_DX = 12.0                                          # contour-grid posting (m)
FOOT_PAD = 0.003                                       # lon/lat margin around the lidar footprint
MORES_LL = (-115.6667, 43.9333)                        # Idaho locator dot
SNOTEL_LL = (-115.66588, 43.932)                       # Mores Creek Summit #637, from the NRCS AWDB station record
MET_STATIONS = [(-115.7018528, 43.94091389, "Freeman Peak"),
                (-115.687, 43.960080, "Pilot Peak")]
NAIP_URL = ("https://imagery.nationalmap.gov/arcgis/rest/services/"
            "USGSNAIPImagery/ImageServer/exportImage")
LIDAR_OUTLINE_C, MET_C = LIDAR_COLOR, "white"   # lidar coverage outline on every map: the lidar purple; met stations
FIRE_C = "#ff5722"                             # Pioneer Fire colour, only used by the retired figures in old/


def _grid(aoi):
    """Regular UTM grid covering a lon/lat bbox at MAP_DX posting (north at top)."""
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform
    x0, y0 = to_utm(aoi[0], aoi[3]); x1, y1 = to_utm(aoi[2], aoi[1])
    xs = np.arange(min(x0, x1), max(x0, x1), MAP_DX)
    ys = np.arange(max(y0, y1), min(y0, y1), -MAP_DX)
    return xs, ys


def _forest_fraction(xs, ys):
    worldcover = _raster_on_grid(paths.WORLDCOVER, xs, ys, Resampling.nearest)
    return float(np.mean(worldcover == 10))                            # class 10 = trees


def lidar_footprint():
    """Clean outer outline of the NIVAL dHS coverage. Returns (mask, UTM extent, lon/lat bbox)."""
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
    feb22 = rioxarray.open_rasterio(paths.LIDAR_A22, masked=True).squeeze()
    feb07 = rioxarray.open_rasterio(paths.LIDAR_A07, masked=True).squeeze()
    change = (feb22 - feb07).where(lambda v: (v > -3) & (v < 3))
    valid = change.notnull().astype("float32").rio.write_crs(change.rio.crs)
    coverage = valid.rio.reproject(EPSG, resolution=100.0, resampling=Resampling.average)
    covered = np.nan_to_num(coverage.values) > 0.25
    covered = binary_fill_holes(binary_closing(covered, iterations=3))
    mask = gaussian_filter(covered.astype(float), 1.2)
    xs, ys = coverage.x.values, coverage.y.values
    extent = [xs[0], xs[-1], ys[-1], ys[0]]
    # bounds of the covered cells themselves, not of the raster they sit in, so the map hugs the footprint
    rows, cols = np.where(covered)
    half = 50.0                                               # half of the 100 m coverage cell
    minx, maxx = xs[cols.min()] - half, xs[cols.max()] + half
    miny, maxy = ys[rows.max()] - half, ys[rows.min()] + half
    to_lonlat = Transformer.from_crs(EPSG, 4326, always_xy=True).transform
    corners = [to_lonlat(minx, miny), to_lonlat(maxx, miny), to_lonlat(maxx, maxy), to_lonlat(minx, maxy)]
    bbox = (min(c[0] for c in corners), min(c[1] for c in corners),
            max(c[0] for c in corners), max(c[1] for c in corners))
    return mask, extent, bbox


def map_frame():
    """Lidar footprint -> AOI (+margin) -> contour grid + UTM extent for the map."""
    mask, lidar_extent, bbox = lidar_footprint()
    aoi = (bbox[0] - FOOT_PAD, bbox[1] - FOOT_PAD, bbox[2] + FOOT_PAD, bbox[3] + FOOT_PAD)
    xs, ys = _grid(aoi)
    extent = [xs[0], xs[-1], ys[-1], ys[0]]
    return dict(extent=extent, aoi=aoi, xs=xs, ys=ys, lidar_mask=mask, lidar_extent=lidar_extent)


def setup_map_axes(ax, frame=None):
    """Set `ax` up as a map in the Fig. 1 projection and bounds, so every map in the paper
    shares one frame: UTM 11N with north up, limits at the lidar footprint plus its margin,
    equal aspect, easting / northing labels. Returns the frame from map_frame(); draw
    rasters with `extent=frame["extent"]` (or their own UTM extent) and `origin="upper"`.
    Pass `frame` to reuse one map_frame() across several panels."""
    frame = map_frame() if frame is None else frame
    extent = frame["extent"]                       # [xmin, xmax, ymin, ymax] in UTM metres
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    ax.set_xlabel("UTM 11N easting (m)"); ax.set_ylabel("northing (m)")
    return frame


def fetch_naip(extent, max_px=4000):
    """USGS NAIP aerial over the UTM extent, cached to data/remote/. Returns rgb in 0..1."""
    if paths.BASEMAP_PNG.exists() and paths.BASEMAP_META.exists():
        if np.allclose(json.loads(paths.BASEMAP_META.read_text())["ext"], extent, atol=1.0):
            return mpimg.imread(paths.BASEMAP_PNG).astype("float32")
    xmin, xmax, ymin, ymax = extent
    dx, dy = xmax - xmin, ymax - ymin
    width, height = (max_px, round(max_px * dy / dx)) if dx >= dy else (round(max_px * dx / dy), max_px)
    params = dict(bbox=f"{xmin},{ymin},{xmax},{ymax}", bboxSR=EPSG, imageSR=EPSG,
                  size=f"{width},{height}", format="png", f="image")
    for attempt in range(4):                          # ride out transient 5xx gateway timeouts
        response = requests.get(NAIP_URL, params=params, timeout=180)
        if response.status_code == 200 and response.headers.get("content-type") == "image/png":
            break
        print(f"  NAIP retry {attempt + 1} (status {response.status_code})")
    response.raise_for_status()
    import io
    image = mpimg.imread(io.BytesIO(response.content), format="png").astype("float32")
    image[..., :3] = np.clip(image[..., :3] ** 0.7 * 1.08, 0, 1)      # gamma + lift (NAIP is dark)
    paths.BASEMAP_PNG.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(paths.BASEMAP_PNG, np.clip(image, 0, 1))
    paths.BASEMAP_META.write_text(json.dumps({"ext": list(extent)}))
    print(f"NAIP aerial {width}x{height} (~{dx / width:.1f} m/px)")
    return image


def pioneer_perimeter():
    """Pioneer Fire (2016) perimeter (UTM exterior rings + polygon) from the frozen geojson.
    No longer drawn on Fig. 1; kept for the retired figures in old/."""
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform, unary_union
    geojson = json.loads(paths.FIRE_GEOJSON.read_text())
    geometry = unary_union([shape(feature["geometry"]) for feature in geojson["features"]])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform
    geometry_utm = shp_transform(to_utm, geometry)
    polygons = geometry_utm.geoms if geometry_utm.geom_type == "MultiPolygon" else [geometry_utm]
    return [np.array(p.exterior.coords) for p in polygons], geometry_utm


def sar_geometry():
    """Flight azimuth, look azimuth, mean incidence [deg] from the GUNW LOS vectors."""
    import h5py
    with h5py.File(paths.gunw_product()) as gunw:
        radar_grid = gunw["science/LSAR/GUNW/metadata/radarGrid"]
        # the operational cube spans the whole frame; average only the nodes over the AOI so
        # the reported look geometry describes the study area, not 360 km of Idaho
        from pyproj import Transformer
        west, south, east, north = paths.AOI
        to_utm = Transformer.from_crs(4326, paths.EPSG, always_xy=True).transform
        eastings, northings = zip(*[to_utm(p, q) for p in (west, east) for q in (south, north)])
        cube_x, cube_y = radar_grid["xCoordinates"][:], radar_grid["yCoordinates"][:]
        jx = np.where((cube_x >= min(eastings)) & (cube_x <= max(eastings)))[0]
        iy = np.where((cube_y >= min(northings)) & (cube_y <= max(northings)))[0]
        sx, sy = (slice(iy[0], iy[-1] + 1), slice(jx[0], jx[-1] + 1)) if len(jx) and len(iy) \
            else (slice(None), slice(None))
        los_east = float(np.nanmean(radar_grid["losUnitVectorX"][:, sx, sy]))
        los_north = float(np.nanmean(radar_grid["losUnitVectorY"][:, sx, sy]))
    los_up = np.sqrt(max(0.0, 1.0 - los_east ** 2 - los_north ** 2))
    look_azimuth = np.degrees(np.arctan2(los_east, los_north)) % 360.0
    return (look_azimuth - 90.0) % 360.0, look_azimuth, np.degrees(np.arccos(los_up))


def sar_geometry_inset(ax, rect, track="077", scale=1.0):
    """Acquisition-geometry inset: flight track (black) + left-looking imaging arrow (blue).
    `scale` shrinks the text and line weights for a smaller box."""
    flight_az, look_az, incidence = sar_geometry()
    inset = ax.inset_axes(rect)
    inset.set_xlim(-1.25, 1.25); inset.set_ylim(-1.55, 1.25); inset.set_aspect("equal")
    inset.set_xticks([]); inset.set_yticks([]); inset.set_facecolor("#e9edf3"); inset.patch.set_alpha(0.92)
    for spine in inset.spines.values():
        spine.set_edgecolor("k"); spine.set_linewidth(1.0)
    direction = lambda az: (np.sin(np.radians(az)), np.cos(np.radians(az)))
    fx, fy = direction(flight_az)
    inset.plot([-fx * 0.85, fx * 0.85], [-fy * 0.85, fy * 0.85], color="k", lw=2.2 * scale, zorder=3)
    inset.annotate("", xy=(fx * 0.95, fy * 0.95), xytext=(fx * 0.6, fy * 0.6),
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=2.2 * scale, mutation_scale=13 * scale), zorder=3)
    lx, ly = direction(look_az + 180.0)        # imaging/look = sensor->ground (opposite the toward-sat LOS); left-looking here
    inset.annotate("", xy=(lx * 0.78, ly * 0.78), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=2.2 * scale, mutation_scale=13 * scale), zorder=4)
    inset.plot(0, 0, "o", color="k", ms=3 * scale, zorder=5)
    inset.text(fx * 1.04, fy * 1.10, "flight\ndirection", color="k", fontsize=7.5 * scale, fontweight="bold",
               ha="center", va="center", linespacing=0.9)
    inset.text(lx * 0.5 - 0.06, ly * 0.5 + 0.32, "imaging\ndirection", color="#1565c0", fontsize=7.5 * scale,
               fontweight="bold", ha="center", va="center", linespacing=0.9)
    inset.annotate("", xy=(-0.95, -0.7), xytext=(-0.95, -1.05),
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, mutation_scale=8 * scale))
    inset.text(-0.95, -0.52, "N", color="k", fontsize=9 * scale, fontweight="bold", ha="center", va="center")
    inset.text(0, -1.4, f"NISAR ASC {track}", color="k", fontsize=10 * scale,
               fontweight="bold", ha="center")
    return flight_az, look_az, incidence


def idaho_locator(ax):
    """Idaho state outline (Natural Earth) with a white dot on Mores Creek."""
    shapefile = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces")
    for record in shpreader.Reader(shapefile).records():
        if record.attributes.get("name") != "Idaho":
            continue
        geometry = record.geometry
        for polygon in (geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]):
            x, y = polygon.exterior.xy
            ax.fill(x, y, fc="0.82", ec="0.5", lw=0.8, alpha=0.9)        # light-grey state shape, no surrounding box
    import matplotlib.patheffects as patheffects
    ax.plot(*MORES_LL, "o", color="white", mec="k", mew=0.6, ms=7, zorder=5)   # white, as the SNOTEL star on the map
    ax.annotate("Mores Ck", MORES_LL, textcoords="offset points", xytext=(7, -2),
                fontsize=7, color="white", fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="k")])   # dark edge, to read on grey
    ax.set_aspect(1 / np.cos(np.radians(45)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_visible(False)                                          # no background box behind Idaho
    for spine in ax.spines.values():
        spine.set_visible(False)


def study_area(ax):
    """Fig-1 study-area map drawn into `ax`: elevation base map + lidar footprint +
    stations + geometry inset (composed by paper_figures.context_figure)."""
    frame = setup_map_axes(ax)                     # the shared map projection + bounds
    extent, aoi, xs, ys = frame["extent"], frame["aoi"], frame["xs"], frame["ys"]
    lidar_mask, lidar_extent = frame["lidar_mask"], frame["lidar_extent"]
    # clip the DEM a little wider than the frame, or reprojection leaves gaps along the map edge
    elevation = _elevation_on_grid(xs, ys, (aoi[0] - 0.01, aoi[1] - 0.01, aoi[2] + 0.01, aoi[3] + 0.01))
    forest = _forest_fraction(xs, ys)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform
    snotel_x, snotel_y = to_utm(*SNOTEL_LL)
    print(f"grid {elevation.shape} @ {MAP_DX:.0f} m | forest {100*forest:.0f}%")

    # elevation as the base map, in the terrain colours without the blue end and with only a
    # little of the green: the lowest ground starts at a light yellow-green
    from matplotlib.colors import LinearSegmentedColormap
    land = LinearSegmentedColormap.from_list("terrain_land", plt.cm.terrain(np.linspace(0.36, 1.0, 256)))
    base = ax.imshow(elevation, extent=extent, origin="upper", cmap=land, zorder=0)
    colorbar = ax.figure.colorbar(base, cax=ax.inset_axes([1.02, 0.2, 0.025, 0.6]))
    colorbar.set_label("Elevation (m)", fontsize=13); colorbar.ax.tick_params(labelsize=11)

    lo, hi = np.nanpercentile(elevation, [1, 99])
    levels = np.arange(np.ceil(lo / 100) * 100, hi, 100)
    ax.contour(elevation, levels=levels, extent=extent, origin="upper", colors="0.2", linewidths=0.6, alpha=0.4)
    ax.contour(lidar_mask, levels=[0.5], extent=lidar_extent, origin="upper", colors=LIDAR_OUTLINE_C, linewidths=2.6)

    ax.plot(snotel_x, snotel_y, "*", color="white", mec="k", mew=1.2, ms=22, zorder=6)
    ax.annotate("SNOTEL 637\n(Mores Ck Summit)", (snotel_x, snotel_y), textcoords="offset points",
                xytext=(10, 6), fontsize=12, fontweight="bold", color="k")
    for lon, lat, label in MET_STATIONS:
        x, y = to_utm(lon, lat)
        ax.plot(x, y, "^", color=MET_C, mec="k", mew=1.2, ms=13, zorder=6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(9, 4), fontsize=11,
                    fontweight="bold", color="k")

    # Fig. 1's map is its largest panel, so its axis text is set larger than the default
    ax.xaxis.label.set_size(13); ax.yaxis.label.set_size(13)
    ax.tick_params(labelsize=11)

    legend = [Line2D([], [], marker="*", color="none", mfc="white", mec="k", ms=15, label="SNOTEL 637"),
              Line2D([], [], color=LIDAR_OUTLINE_C, lw=2.6, label="Lidar Coverage"),
              Line2D([], [], marker="^", color="none", mfc=MET_C, mec="k", ms=11, label="met station")]
    ax.legend(handles=legend, loc="lower right", fontsize=12, framealpha=0.9)

    idaho_locator(ax.inset_axes([0.01, 0.755, 0.235, 0.235]))
    flight_az, look_az, incidence = sar_geometry_inset(ax, [0.012, 0.012, 0.257, 0.295], scale=0.95)
    print(f"SAR geometry: flight {flight_az:.0f}, look {look_az:.0f}, incidence {incidence:.0f} deg (ASC 077)")


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

def season_span():
    """(start, end) of the season the paper covers, as Fig. 3 draws it: the first track-077
    acquisition to a week past the last day of peak SNOTEL SWE."""
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"]).set_index("date")["wteq_in"]
    pairs = [g for g in local_gunws() if g[0] == TRACK and g[2] <= paths.FIG3_LAST]
    start, last = pd.Timestamp(min(g[1] for g in pairs)), pd.Timestamp(max(g[2] for g in pairs))
    return start, max(last, season[season == season.max()].index[-1] + pd.Timedelta(days=7))


def lidar_at_snotel(date, half_width_m=5.0):
    """NIVAL lidar snow depth at SNOTEL 637 on one flight date [m]: the median of the lidar
    cells within `half_width_m` of the station in easting and northing (a 10 m box)."""
    depth = rioxarray.open_rasterio(lidar_file(date), masked=True).squeeze()
    x, y = Transformer.from_crs(4326, depth.rio.crs, always_xy=True).transform(*SNOTEL_LL)
    box = depth.sel(x=slice(x - half_width_m, x + half_width_m), y=slice(y + half_width_m, y - half_width_m)).values
    return float(np.nanmedian(box))


def context_figure():
    """Study-area map on top; beneath it, across the whole season, snow depth at SNOTEL 637
    with the lidar depth at the station on each flight, and the Freeman snowpack temperature.
    Every track-077 acquisition and lidar flight of Fig. 3 is marked on both."""
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold"})
    start, end = season_span()
    # the acquisitions of Fig. 3: track-077 dates, and the lidar flights paired with them
    pairs = [g for g in local_gunws() if g[0] == TRACK and g[2] <= paths.FIG3_LAST]
    nisar_dates = sorted({pd.Timestamp(d).date() for g in pairs for d in g[1:3]})
    flights = sorted({d for p in PAIRS if p[0] == TRACK and p[2] <= paths.FIG3_LAST for d in p[3:5]})
    lidar_dates = [pd.Timestamp(d).date() for d in flights]

    fig = plt.figure(figsize=(10, 17.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[10.6, 3.1, 3.1], hspace=0.24)   # room for the map's axis label and the pair label
    ax_map = fig.add_subplot(gs[0])
    ax_depth = fig.add_subplot(gs[1])
    ax_temp = fig.add_subplot(gs[2], sharex=ax_depth)

    study_area(ax=ax_map)                                      # the existing map, drawn into our axis

    for ax in (ax_depth, ax_temp):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    # snow depth at the station, with the lidar depth there on each flight date
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"]).set_index("date")
    station_depth = season["snwd_in"] * 0.0254                 # inches -> metres at ingest
    if station_depth.max() > 6.0:
        raise ValueError("SNOTEL depth is not plausible in metres; check units")
    lidar_depth = [lidar_at_snotel(d) for d in flights]
    for d, value in zip(flights, lidar_depth):
        print(f"  lidar at SNOTEL {d}: {value:.2f} m | station {station_depth.get(pd.Timestamp(d), np.nan):.2f} m")
    ax_depth.fill_between(station_depth.index, station_depth.values, 0, color="#4c8cbf", alpha=0.20)
    ax_depth.plot(station_depth.index, station_depth.values, color="#1f6aa5", lw=2.6, zorder=3, label="SNOTEL 637")
    ax_depth.plot(lidar_dates, lidar_depth, "D", color=LIDAR_COLOR, mec="k", mew=0.9, ms=9, zorder=4,
                  label="Lidar at SNOTEL")
    ax_depth.set_ylabel("Snow Depth (m)"); ax_depth.set_ylim(0, None)
    ax_depth.set_title("Snow Depth at SNOTEL 637", loc="left")
    handles, labels = ax_depth.get_legend_handles_labels()
    extra = acq_legend_handles()
    ax_depth.legend(handles + extra, labels + [h.get_label() for h in extra], loc="upper left",
                    fontsize=9, framealpha=0.9)
    mark_acquisitions(ax_depth, nisar=nisar_dates, lidar=lidar_dates)

    # Freeman snowpack temperature through the same season
    panel_snowpack_temperature(ax_temp, dict(thermo=snowpack_thermograph(start, end)), mark_acq=False)
    mark_acquisitions(ax_temp, nisar=nisar_dates, lidar=lidar_dates)


    # shared date axis across the season
    ax_temp.set_xlim(start, end)
    ax_temp.xaxis.set_major_locator(mdates.MonthLocator())
    ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_temp.set_xlabel("date")
    plt.setp(ax_depth.get_xticklabels(), visible=False)

    # panel letters, matching the caption
    for ax, letter in ((ax_map, "a"), (ax_depth, "b"), (ax_temp, "c")):
        panel_letter(ax, letter, fontsize=14)

    out = paths.FIGURES_DIR / "fig1_context.png"
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_print(fig, out, 12)
    print("saved ->", out)


# =============================== figure 2: results ===============================
def _draw_map(ax, field, valid, elev, shade, caps, extent, posting, frame, contour_index, title, left=True):
    """One map in the shared Fig. 1 frame: hillshade and contours where `shade`, the field on
    top where `valid`, and the lidar footprint outlined. `extent` is the UTM extent of the
    arrays. Returns the field image for its colour bar."""
    cmap = cmocean.cm.ice.copy(); cmap.set_bad("none")         # nodata shows whatever is beneath
    setup_map_axes(ax, frame)                                  # Fig. 1 projection + bounds
    ax.set_facecolor("white")
    finite_elev = np.isfinite(elev)
    hillshade = LS.hillshade(np.where(finite_elev, elev, np.nanmedian(elev[finite_elev])),
                             vert_exag=1.5, dx=posting, dy=posting)
    ax.imshow(np.where(shade, hillshade, np.nan), extent=extent, cmap="gray", vmin=0, vmax=1, origin="upper")
    lo, hi = np.nanpercentile(elev[finite_elev], [2, 98])
    levels = np.arange(np.ceil(lo / contour_index) * contour_index, hi, contour_index)
    add_contours(ax, np.where(shade, elev, np.nan), extent, levels, contour_index * 5)
    image = ax.imshow(np.where(valid, field, np.nan), extent=extent, cmap=cmap,
                      vmin=caps[0], vmax=caps[1], alpha=0.85, origin="upper")
    # the lidar coverage outline, drawn as in Fig. 1
    ax.contour(frame["lidar_mask"], levels=[0.5], extent=frame["lidar_extent"], origin="upper",
               colors=LIDAR_OUTLINE_C, linewidths=2.6)
    ax.set_title(title, fontsize=13)
    if not left:                                               # the right map shares the left one's northing axis
        ax.set_ylabel(""); ax.set_yticklabels([])
    return image


TITLE_PAD = 22          # points between a Fig. 2 panel and its title: room for a line of dates


def results_figure():
    """Maps + density hexbin, all three panels on the delivered 80 m operational product."""
    product = retrieval.load_product()
    dswe, dhs, coh, elev = product["dswe"], product["dhs"], product["coherence"], product["elevation"]
    posting = float(abs(product["xs"][1] - product["xs"][0]))   # grid posting (m), from the product

    dswe = dswe * 1000                                         # metres -> mm, the unit NISAR SWE is shown in
    common = np.isfinite(dswe) & np.isfinite(dhs)
    r_all = np.corrcoef(dswe[common], dhs[common])[0, 1]
    high = common & (coh > 0.5)
    r_high = np.corrcoef(dswe[high], dhs[high])[0, 1]
    xlim = np.nanpercentile(dhs[common], [0.5, 99.5])
    ylim = np.nanpercentile(dswe[common], [0.5, 99.5])
    print(f"all r={r_all:.2f} n={int(common.sum())}; high-coh r={r_high:.2f} n={int(high.sum())}")

    # every panel on one grid: the maps read the same 80 m arrays the scatter does, drawn
    # north-up in the Fig. 1 map frame. xs / ys are pixel centres, so the raster's outer
    # edge sits half a posting beyond them.
    frame = map_frame()
    xs, ys = product["xs"], product["ys"]
    extent = [xs.min() - posting / 2, xs.max() + posting / 2, ys.min() - posting / 2, ys.max() + posting / 2]
    frame_aspect = (frame["extent"][3] - frame["extent"][2]) / (frame["extent"][1] - frame["extent"][0])

    # headline layout: two prominent maps + a compact density panel, packed tight by
    # constrained_layout so colorbars and labels never collide
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 5.4), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.07)                   # air between a map's colour bar label and the next panel's axis label
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax_sdswe, ax_sdhs, ax_hall = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # NISAR fills the frame wherever it has a value; the lidar covers only its footprint, so
    # the hillshade carries the rest of that panel
    has_dswe = np.isfinite(dswe)
    img_dswe = _draw_map(ax_sdswe, dswe, has_dswe, elev, has_dswe, (DSWE_VMIN * 1000, DSWE_VMAX * 1000), extent,
                         posting, frame, 100, SWE_CHANGE)
    img_dhs = _draw_map(ax_sdhs, dhs, common, elev, np.isfinite(elev), (DHS_VMIN, DHS_VMAX), extent,
                        posting, frame, 100, DEPTH_CHANGE, left=False)

    # the dates each map spans, in lighter type between its title and the map
    span = lambda a, b: (f"{pd.Timestamp(a):%b} {pd.Timestamp(a).day} to {pd.Timestamp(b):%b} {pd.Timestamp(b).day}, "
                         f"{pd.Timestamp(b).year}")
    for ax, (first, second) in ((ax_sdswe, (paths.NISAR_A, paths.NISAR_B)), (ax_sdhs, paths.LIDAR_DATES)):
        ax.set_title(ax.get_title(), fontsize=13, pad=TITLE_PAD)
        ax.text(0.5, 1.012, span(first, second), transform=ax.transAxes, ha="center", va="bottom", fontsize=11)

    # density of NISAR dSWE against NIVAL dHS, in the scatter style Fig. 3 shares
    hb = density_scatter(ax_hall, dhs, dswe, (xlim[0], xlim[1], ylim[0], ylim[1]))
    density_fit(ax_hall, dhs, dswe, "upper left")
    ax_hall.set_box_aspect(frame_aspect)                      # same box shape as the maps -> equal heights
    ax_hall.set_xlabel(DEPTH_CHANGE_LABEL); ax_hall.set_ylabel(SWE_CHANGE_LABEL)
    ax_hall.set_title(ax_hall.get_title(), pad=TITLE_PAD)      # level with the two map titles

    # colorbars attached to each panel; constrained_layout keeps them snug and overlap-free
    fig.colorbar(img_dswe, ax=ax_sdswe, fraction=0.05, pad=0.02).set_label(SWE_CHANGE_LABEL)
    fig.colorbar(img_dhs, ax=ax_sdhs, fraction=0.05, pad=0.02).set_label(DEPTH_CHANGE_LABEL)
    count_colorbar(fig.colorbar(hb, ax=ax_hall, fraction=0.05, pad=0.02))

    out = paths.FIGURES_DIR / "fig2_results.png"
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_print(fig, out, 18)
    print("saved ->", out)


# =============================== figure 3: SWE timeline ===============================
OURS_COLOR = "#e66100"         # marks the GUNWs we produced with Hyp3-Isce3
SWE_LIM_MM = 200.0             # timeline colour limit, either side of zero
LIDAR_LIM_M = 1.0              # timeline lidar depth-change colour limit, either side of zero
SCREEN_LIM_RAD = np.pi         # timeline colour limit of an atmospheric phase screen, either side of zero
SCREEN_LABEL = "Phase about scene median (rad)"
TOTAL_LIM_MM = 700.0           # timeline upper colour limit of the cumulative NISAR SWE map
DEPTH_LIM_M = 2.0              # timeline upper colour limit of the lidar snow depth map


def _snotel_swe():
    """SNOTEL 637 daily SWE in metres, converted from inches at ingest."""
    from nival.download_paper_data import fetch_snotel_season
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


def _corrected_phase(path):
    """One GUNW's 80 m unwrapped phase over the AOI with its ionosphere screen and its
    hydrostatic + wet troposphere screens removed [rad], and the pixel-centre coordinates of
    the clip. The screens are the GUNW's own layers; nisar_pytools samples the troposphere
    cubes at the terrain height, taken from the scene DEM (ellipsoidal heights, as the cubes use)."""
    from nisar_pytools.processing.atmospheric import correct_atmosphere
    with h5py.File(path) as f:
        g = f["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"]
        rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
        xs, ys = g["xCoordinates"][cols], g["yCoordinates"][rows]
        on_grid = lambda values: xr.DataArray(values.astype(float), dims=("y", "x"), coords={"y": ys, "x": xs})
        unwrapped, ionosphere = on_grid(g["unwrappedPhase"][rows, cols]), on_grid(g["ionospherePhaseScreen"][rows, cols])
        cubes = f["science/LSAR/GUNW/metadata/radarGrid"]
        hydrostatic, wet = cubes["hydrostaticTroposphericPhaseScreen"][:], cubes["wetTroposphericPhaseScreen"][:]
        heights, cube_x, cube_y = cubes["heightAboveEllipsoid"][:], cubes["xCoordinates"][:], cubes["yCoordinates"][:]
    dem = on_grid(retrieval._elevation_on_grid(xs, ys, paths.AOI))
    corrected = correct_atmosphere(unwrapped, hydrostatic, wet, heights, cube_x, cube_y, dem,
                                   ionosphere_screen=ionosphere)
    return corrected.values, xs, ys


def _troposphere_phase(path):
    """One GUNW's tropospheric phase over the AOI [rad]: the hydrostatic + wet screens sampled
    at terrain height, which is what _corrected_phase removes. Obtained as the unwrapped phase
    minus the same phase with only the troposphere taken out by nisar_pytools. Returns the
    screen and the pixel-centre coordinates of the clip."""
    from nisar_pytools.processing.atmospheric import correct_atmosphere
    with h5py.File(path) as f:
        g = f["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"]
        rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
        xs, ys = g["xCoordinates"][cols], g["yCoordinates"][rows]
        on_grid = lambda values: xr.DataArray(values.astype(float), dims=("y", "x"), coords={"y": ys, "x": xs})
        unwrapped = on_grid(g["unwrappedPhase"][rows, cols])
        cubes = f["science/LSAR/GUNW/metadata/radarGrid"]
        hydrostatic, wet = cubes["hydrostaticTroposphericPhaseScreen"][:], cubes["wetTroposphericPhaseScreen"][:]
        heights, cube_x, cube_y = cubes["heightAboveEllipsoid"][:], cubes["xCoordinates"][:], cubes["yCoordinates"][:]
    dem = on_grid(retrieval._elevation_on_grid(xs, ys, paths.AOI))
    without = correct_atmosphere(unwrapped, hydrostatic, wet, heights, cube_x, cube_y, dem)   # troposphere only
    return (unwrapped - without).values, xs, ys


def _ionosphere_phase(path):
    """One GUNW's ionospheric phase screen over the AOI [rad], as delivered, and the
    pixel-centre coordinates of the clip."""
    with h5py.File(path) as f:
        g = f["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"]
        rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
        return (g["ionospherePhaseScreen"][rows, cols].astype(float),
                g["xCoordinates"][cols], g["yCoordinates"][rows])


def _unwrapped_phase(path):
    """One GUNW's 80 m unwrapped phase over the AOI [rad], as delivered with no screen removed,
    and the pixel-centre coordinates of the clip."""
    with h5py.File(path) as f:
        g = f["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"]
        rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
        return (g["unwrappedPhase"][rows, cols].astype(float),
                g["xCoordinates"][cols], g["yCoordinates"][rows])


# the phase fields the timeline can show in place of SWE change: reader and row name
SCREENS = {"troposphere": (_troposphere_phase, "Troposphere"), "ionosphere": (_ionosphere_phase, "Ionosphere"),
           "unwrapped": (_unwrapped_phase, "Unwrapped phase")}


def _pi_ticks(colorbar, limit=np.pi):
    """Tick a phase colour bar at its limits, half its limits and zero, labelled in units of pi."""
    def name(value):
        k = value / np.pi
        if np.isclose(k, 0):
            return "0"
        sign, k = ("-" if k < 0 else ""), abs(k)
        if np.isclose(k, 0.5):
            return rf"${sign}\pi/2$"
        return rf"${sign}{'' if np.isclose(k, 1) else f'{k:g}'}\pi$"
    ticks = [-limit, -limit / 2, 0, limit / 2, limit]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([name(t) for t in ticks])


def _swe_change(path, ref, sec, snotel_swe):
    """One GUNW's SWE change over the AOI [m], referenced to SNOTEL 637.

    The corrected unwrapped phase has no absolute zero, so it is shifted by one constant, in
    phase units, until the phase at the SNOTEL equals what the station's own SWE change over
    the pair would produce there. Each pixel is then divided by its own sensitivity (Leinss
    2015, the same conversion as retrieval.build_product). The phase at the SNOTEL is the
    median of the 3 by 3 pixels (240 m) around it. Returns the field, its coordinates, and
    the reference numbers for logging."""
    from nisar_pytools.utils.local_incidence_angle import local_incidence_angle
    phase, xs, ys = _corrected_phase(path)
    # per-pixel local incidence angle from this GUNW's line-of-sight vectors and the scene DEM
    with h5py.File(path) as f:
        cubes = f["science/LSAR/GUNW/metadata/radarGrid"]
        los_x, los_y = cubes["losUnitVectorX"][:], cubes["losUnitVectorY"][:]
        los_z = np.sqrt(np.clip(1 - los_x ** 2 - los_y ** 2, 0, 1))
        heights, cube_x, cube_y = cubes["heightAboveEllipsoid"][:], cubes["xCoordinates"][:], cubes["yCoordinates"][:]
    elevation = xr.DataArray(retrieval._elevation_on_grid(xs, ys, paths.AOI), coords={"y": ys, "x": xs},
                             dims=("y", "x")).rio.write_crs(paths.EPSG)
    lia = np.asarray(local_incidence_angle(elevation, los_x, los_y, los_z, heights, cube_x, cube_y,
                                           epsg=paths.EPSG).values)
    kappa = 2 * np.pi / paths.LAMBDA * (1.59 + np.radians(lia) ** 2.5)     # radians per metre of SWE
    # the pixel holding the SNOTEL, and the phase its SWE change corresponds to there
    sx, sy = Transformer.from_crs(4326, paths.EPSG, always_xy=True).transform(*SNOTEL_LL)
    i, j = int(np.argmin(np.abs(ys - sy))), int(np.argmin(np.abs(xs - sx)))
    phase_at_snotel = np.nanmedian(phase[i - 1:i + 2, j - 1:j + 2])
    station_change = snotel_swe[pd.Timestamp(sec)] - snotel_swe[pd.Timestamp(ref)]
    swe_change = (phase - phase_at_snotel + kappa[i, j] * station_change) / kappa
    log = dict(station_mm=station_change * 1000, phase_3x3=phase_at_snotel, phase_1px=phase[i, j],
               kappa=kappa[i, j], lia=lia[i, j])
    return swe_change, xs, ys, (sx, sy), log


def timeline_figure(out, tracks=None, last=None, lidar_row=False, to_peak=False, totals=False,
                    print_width_cm=None, quantity="swe", swe_panel=True, heading=None,
                    screen_lim=SCREEN_LIM_RAD, total_lim=SCREEN_LIM_RAD):
    """The SWE change from every GUNW over the scene, one row per track, laid out along a
    shared time axis. Fig. 3 is the track-077 version with every extra switched on
    (swe_timeline_figure); old/figS1_all_tracks.py draws the plain all-track version. Production GUNWs, except where we have a GUNW from refocused
    RSLCs with hyp3-isce3: those replace the production one and are framed in orange (all of
    track 114 is ours).

    Each pair's map sits at the middle of its interval and is one 12-day repeat wide. A pair
    spanning more than one repeat gets arrows out to its two acquisition dates. Every map is in
    the Fig. 1 frame with the lidar footprint outlined, and shows the SWE change from the 80 m
    unwrapped phase (ionosphere and troposphere screens removed), referenced so the SNOTEL 637
    pixel carries the station's own SWE change over the pair, on one colour range. Dated vertical lines mark each track's acquisition
    dates, which is how a pair is identified. A panel underneath shows SNOTEL 637 SWE on the
    same time axis. Sized for a full page width: 12 in wide with 8 to 10 pt text.

    `quantity` picks what each pair's map shows: "swe" for the SWE change, or one or more of
    the atmospheric phase screens the SWE version removes ("troposphere", "ionosphere"), each
    as its own row of maps about the scene median; their `totals` is one map per screen, the
    sum over the pairs. "unwrapped" is the observed unwrapped phase, drawn the same way.
    `screen_lim` and `total_lim` are the colour limits [rad] of the pair maps and of the
    totals. `swe_panel=False` leaves out the SNOTEL SWE strip. `heading` replaces
    the figure's default heading.

    `out` is the output file. `tracks` keeps only those tracks and `last` ("YYYYMMDD") only
    pairs ending by that date. `print_width_cm` resizes the saved image to that printed width
    at 300 dpi, as the other paper figures are. `lidar_row` adds a row of NIVAL lidar snow-depth change, one map
    per census pair in pairs.py, placed on its two flight dates. `to_peak` runs the time axis
    on to a week past the SNOTEL's peak SWE, so the SWE panel shows the peak and its turn.
    `totals` (one track, pairs that chain end to end) adds a row under the SWE panel: the sum
    of the pairs' SWE change, the lidar snow depth of the last date, each in its own units, and
    a scatter of one against the other."""
    from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
    from matplotlib.patches import Polygon as MplPolygon
    quantities = (quantity,) if isinstance(quantity, str) else tuple(quantity)
    screens = quantities != ("swe",)                        # atmospheric screens instead of SWE change
    plt.rcdefaults()                                        # Figs. 1 and 2 set bold titles and sizes; start clean
    gunws = [g for g in local_gunws()                       # (track, ref, sec, path, refocused), date ordered
             if (tracks is None or g[0] in tracks) and (last is None or g[2] <= last)]
    tracks = sorted({g[0] for g in gunws})
    # orbit direction is the 7th field of a granule name (A or D), right after the track
    direction = {g[0]: {"A": "ascending", "D": "descending"}[g[3].name.split("_")[6]] for g in gunws}
    map_box = map_frame()                                   # one shared frame for every panel
    box_aspect = (map_box["extent"][3] - map_box["extent"][2]) / (map_box["extent"][1] - map_box["extent"][0])
    day = lambda stamp: pd.Timestamp(stamp)
    # SNOTEL 637 daily SWE. The daily record on its own: _snotel_swe() overlays the 06:00
    # overpass readings for February, which have missing days and would break the line.
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"])
    swe = pd.Series(season["wteq_in"].values * 0.0254, index=season["date"]).sort_index()   # inches -> metres at ingest
    if swe.max() > 3.0:
        raise ValueError("SNOTEL SWE is not plausible in metres; check units")
    t0, t1 = min(day(g[1]) for g in gunws), max(day(g[2]) for g in gunws)
    if to_peak:                                             # last day at peak SWE, plus a week to show the turn
        t1 = max(t1, swe[swe == swe.max()].index[-1] + pd.Timedelta(days=7))
    span_days = (t1 - t0).days
    # the lidar row: the census pairs on these tracks that have a product, with their two flight dates
    lidar_pairs = [p for p in PAIRS if lidar_row and p[0] in {g[0] for g in gunws}
                   and (last is None or p[2] <= last) and (PRODUCTS / f"{label(p)}.nc").exists()]
    lidar_pairs.sort(key=lambda p: p[3])
    line_c = "0.25"                                         # acquisition lines and their dates

    # figure geometry in inches: a 12-day slot is one panel wide, rows stack one per track
    width, left, right = 12.0, 0.6, 1.2
    slot = (width - left - right) * 12 / span_days          # width of 12 days on the time axis
    panel_w = 0.90 * slot
    panel_h = panel_w * box_aspect
    # the key goes in the empty stretch before the latest-starting track, if there is one;
    # otherwise it gets its own line under the title
    late = max(tracks, key=lambda t: min(day(g[1]) for g in gunws if g[0] == t))
    key_in_row = (min(day(g[1]) for g in gunws if g[0] == late) - t0).days >= 60
    row_h, top = panel_h + 0.50, 0.42 if key_in_row else 0.78   # each row: the maps, then the date labels beneath
    totals_h = 2.6 if totals else 0.0                       # the season-total maps and scatter under the SWE panel
    extra = totals_h + 1.0 if totals else 0.0
    swe_h = 1.3 if swe_panel else 0.0                       # the SWE panel, under the map rows
    bottom = (2.0 if swe_panel else 0.25) + extra           # the space the SWE panel and totals sit in
    rows = [(track, q) for q in quantities for track in tracks]
    height = top + bottom + row_h * (len(rows) + bool(lidar_pairs))
    x_of = lambda t: (left + (t - t0).days / span_days * (width - left - right)) / width   # date -> figure x

    snotel_swe = _snotel_swe()                              # metres, the series Table S1 reads
    sums = {}                                               # running sum of the pairs' fields per quantity, for `totals`
    fig = plt.figure(figsize=(width, height), dpi=300)

    def double_outline(ax, middle, y0):
        """Mark a map as the Fig. 2 pair: its frame in black plus a second line just outside."""
        for spine in ax.spines.values():
            spine.set_edgecolor("k"); spine.set_linewidth(1.0)
        pad = 0.022                                         # inches between the frame and the outer line
        fig.add_artist(Rectangle((middle - (panel_w / 2 + pad) / width, y0 - pad / height),
                                 (panel_w + 2 * pad) / width, (panel_h + 2 * pad) / height,
                                 transform=fig.transFigure, fill=False, edgecolor="k", lw=1.0))

    row_y, nisar_dates, lidar_dates = {}, set(), set()     # the dates drawn, for the SWE strip below
    for r, (track, q) in enumerate(rows):
        y0 = (height - top - row_h * (r + 1 + bool(lidar_pairs)) + 0.40) / height   # below the lidar row, if there is one
        row_y.setdefault(track, y0)
        row_name = f"{SCREENS[q][1]}\ntrack {track}" if screens else f"NISAR\nTrack {int(track)}"
        fig.text(0.2 / width, y0 + panel_h / height / 2, row_name, rotation=90,
                 ha="center", va="center", fontsize=9.5, linespacing=1.3)
        # every date a pair on this track starts or ends on, as a dated vertical line behind
        # the maps, so each pair's span can be read off
        row = [g for g in gunws if g[0] == track]
        nisar_dates |= {day(d) for g in row for d in g[1:3]}
        for date in sorted({day(d) for g in row for d in g[1:3]}):
            x, y_low = x_of(date), y0 - 0.07 / height
            fig.add_artist(Line2D([x, x], [y_low, y0 + panel_h / height], transform=fig.transFigure, zorder=-1,
                                  color=NISAR_COLOR, lw=1.1))
            fig.text(x, y_low - 0.03 / height, f"{date:%b} {date.day}", ha="center", va="top",
                     fontsize=8, color=line_c)
        for _, ref, sec, path, refocused in row:
            if screens:
                swe_change, xs, ys = SCREENS[q][0](path)        # radians; named for the SWE case it stands in for
                station_xy = None
                print(f"  {track} {ref}-{sec}: {q} median {np.nanmedian(swe_change):+.2f} rad, spread 1-99% "
                      f"{np.nanpercentile(swe_change - np.nanmedian(swe_change), 1):+.2f} to "
                      f"{np.nanpercentile(swe_change - np.nanmedian(swe_change), 99):+.2f} rad")
            else:
                swe_change, xs, ys, station_xy, log = _swe_change(path, ref, sec, snotel_swe)
                print(f"  {track} {ref}-{sec}: SNOTEL {log['station_mm']:+6.0f} mm | phase at SNOTEL {log['phase_3x3']:+7.2f} rad "
                      f"(3x3 median), {log['phase_1px']:+7.2f} (one pixel) | {log['kappa']:.1f} rad/m at LIA {log['lia']:.0f} deg | "
                      f"map 5-95% {np.nanpercentile(swe_change, 5) * 1000:+.0f} to {np.nanpercentile(swe_change, 95) * 1000:+.0f} mm")
            posting = float(abs(xs[1] - xs[0]))
            # xs / ys are pixel centres, so the raster edge sits half a posting beyond them
            extent = [xs.min() - posting / 2, xs.max() + posting / 2, ys.min() - posting / 2, ys.max() + posting / 2]
            middle = x_of(day(ref) + (day(sec) - day(ref)) / 2)
            ax = fig.add_axes([middle - panel_w / width / 2, y0, panel_w / width, panel_h / height])
            setup_map_axes(ax, map_box)                     # the Fig. 1 projection + bounds
            sums[q] = swe_change if q not in sums else sums[q] + swe_change
            if screens:
                image = ax.imshow(swe_change - np.nanmedian(swe_change), extent=extent, origin="upper",
                                  cmap="RdBu_r", vmin=-screen_lim, vmax=screen_lim)
            else:
                image = ax.imshow(swe_change * 1000, extent=extent, origin="upper",
                                  cmap="RdBu", vmin=-SWE_LIM_MM, vmax=SWE_LIM_MM)   # blue = gain
            ax.contour(map_box["lidar_mask"], levels=[0.5], extent=map_box["lidar_extent"],
                       origin="upper", colors=LIDAR_OUTLINE_C, linewidths=1.6)   # lidar coverage outline, Fig. 1's colour
            if station_xy is not None:
                ax.plot(*station_xy, "*", color="white", mec="k", mew=0.5, ms=6)   # the phase reference, as in Fig. 1
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
            for spine in ax.spines.values():                # a quiet frame unless the pair is singled out below
                spine.set_edgecolor("0.3"); spine.set_linewidth(0.5)
            if refocused:                                   # the GUNWs we produced: orange frame
                for spine in ax.spines.values():
                    spine.set_edgecolor(OURS_COLOR); spine.set_linewidth(2.0)
            if (track, ref, sec) == ANALYSED and not screens:   # the Fig. 2 pair
                double_outline(ax, middle, y0)
            # a pair longer than one repeat: arrows from the map out to its two dates
            if (day(sec) - day(ref)).days > 12:
                y_mid, gap = y0 + panel_h / height / 2, 0.05 / width
                for edge, end in ((middle - panel_w / width / 2 - gap, x_of(day(ref)) + gap),
                                  (middle + panel_w / width / 2 + gap, x_of(day(sec)) - gap)):
                    fig.add_artist(FancyArrowPatch((edge, y_mid), (end, y_mid), transform=fig.transFigure,
                                                   arrowstyle="->", mutation_scale=9, lw=1.0, color="k"))

    # NIVAL lidar snow-depth change, one map per census pair, in the same layout and as the top
    # row: each map sits at the middle of its two flight dates, which are drawn as dated lines,
    # so it lands roughly above the NISAR pair it is matched with
    if lidar_pairs:
        y0 = (height - top - row_h + 0.40) / height           # the top row
        fig.text(0.2 / width, y0 + panel_h / height / 2, "NIVAL Airborne\nLidar", rotation=90,
                 ha="center", va="center", fontsize=9.5, linespacing=1.3)
        lidar_dates = {day(d) for p in lidar_pairs for d in p[3:5]}
        # a map wider than the period it covers gets a span marker beneath it (below), so the
        # date lines and labels of this row sit lower to leave room for it
        narrow = [x_of(day(p[4])) - x_of(day(p[3])) < panel_w / width for p in lidar_pairs]
        stub = 0.24 if any(narrow) else 0.07
        for date in sorted(lidar_dates):
            x, y_low = x_of(date), y0 - stub / height
            fig.add_artist(Line2D([x, x], [y_low, y0 + panel_h / height], transform=fig.transFigure, zorder=-1,
                                  color=LIDAR_COLOR, lw=2.1, ls=(0, (4, 3))))
            fig.text(x, y_low - 0.03 / height, f"{date:%b} {date.day}", ha="center", va="top",
                     fontsize=8, color=line_c)
        # where each map goes: the middle of its two flights. Flights closer together than one
        # map width would overlap, so working back from the last pair, an earlier map slides
        # left until it clears the one after it. Later maps stay centred on their own period.
        middles, next_left = [], 2.0
        for pair in reversed(lidar_pairs):
            first, second = day(pair[3]), day(pair[4])
            middles.insert(0, min(x_of(first + (second - first) / 2), next_left - (panel_w / 2 + 0.04) / width))
            next_left = middles[0] - panel_w / width / 2
        for pair, middle in zip(lidar_pairs, middles):
            first, second = day(pair[3]), day(pair[4])
            ds = xr.open_dataset(PRODUCTS / f"{label(pair)}.nc")   # dhs: later flight minus earlier, on the 80 m grid [m]
            xs, ys = ds.x.values, ds.y.values
            posting = float(abs(xs[1] - xs[0]))
            extent = [xs.min() - posting / 2, xs.max() + posting / 2, ys.min() - posting / 2, ys.max() + posting / 2]
            ax = fig.add_axes([middle - panel_w / width / 2, y0, panel_w / width, panel_h / height])
            setup_map_axes(ax, map_box)                     # the Fig. 1 projection + bounds
            lidar_image = ax.imshow(ds["dhs"].values, extent=extent, origin="upper", cmap="RdBu",
                                    vmin=-LIDAR_LIM_M, vmax=LIDAR_LIM_M)   # blue = gain
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_edgecolor("0.3"); spine.set_linewidth(0.5)
            if pair[:3] == ANALYSED:                        # the lidar flights of the Fig. 2 pair
                double_outline(ax, middle, y0)
            # a map wider than the period it covers: a funnel from the map's bottom edge down to
            # a bar that runs from its first flight to its second, so the map's width is not
            # read as its span
            if x_of(second) - x_of(first) < panel_w / width:
                y_bar, inset = y0 - 0.19 / height, 0.02 / width
                x1, x2 = x_of(first) + inset, x_of(second) - inset
                fig.add_artist(MplPolygon([(middle - panel_w / width / 2, y0), (middle + panel_w / width / 2, y0),
                                           (x2, y_bar), (x1, y_bar)], closed=True, transform=fig.transFigure,
                                          facecolor=LIDAR_COLOR, alpha=0.22, edgecolor="none"))
                for top_x, bar_x in ((middle - panel_w / width / 2, x1), (middle + panel_w / width / 2, x2)):
                    fig.add_artist(Line2D([top_x, bar_x], [y0, y_bar], transform=fig.transFigure,
                                          color=LIDAR_COLOR, lw=0.9))
                fig.add_artist(Line2D([x1, x2], [y_bar, y_bar], transform=fig.transFigure, color=LIDAR_COLOR,
                                      lw=3.0, solid_capstyle="butt"))
            # arrows out to the two flight dates, where the date sits clear of this map and the
            # arrow would not run across a neighbouring map
            y_mid, gap = y0 + panel_h / height / 2, 0.05 / width
            for edge, end in ((middle - panel_w / width / 2 - gap, x_of(first) + gap),
                              (middle + panel_w / width / 2 + gap, x_of(second) - gap)):
                crosses = any(min(edge, end) < m + panel_w / width / 2 and max(edge, end) > m - panel_w / width / 2
                              for m in middles if m != middle)
                if abs(end - edge) > 0.12 / width and (end - edge) * (edge - middle) > 0 and not crosses:
                    fig.add_artist(FancyArrowPatch((edge, y_mid), (end, y_mid), transform=fig.transFigure,
                                                   arrowstyle="->", mutation_scale=9, lw=1.0, color="k"))

    # SNOTEL 637 SWE under the map rows, on the same time axis as the maps above it
    if swe_panel:
        ax_swe = fig.add_axes([x_of(t0), (0.5 + extra) / height, x_of(t1) - x_of(t0), swe_h / height])
        ax_swe.fill_between(swe.index, swe.values * 1000, 0, color="#4c8cbf", alpha=0.20, lw=0)   # Fig. 1's SWE colours
        ax_swe.plot(swe.index, swe.values * 1000, color="#1f6aa5", lw=1.5)
        ax_swe.set_xlim(t0, t1); ax_swe.set_ylim(0, None)
        ax_swe.xaxis.set_major_locator(mdates.MonthLocator())
        ax_swe.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_swe.set_xlabel("date", fontsize=9.5); ax_swe.set_ylabel("SNOTEL 637\nSWE (mm)", fontsize=9.5)
        ax_swe.tick_params(labelsize=8.5)
        ax_swe.grid(axis="y", color="0.85", lw=0.6); ax_swe.set_axisbelow(True)
        ax_swe.spines[["top", "right"]].set_visible(False)
        mark_acquisitions(ax_swe, nisar=sorted(nisar_dates), lidar=sorted(lidar_dates))   # as on Fig. 1

    # the colour bar sits beside the NISAR rows only; the lidar row, above them, has its own
    maps_low, maps_high = bottom / height, 1 - (top + row_h * bool(lidar_pairs)) / height
    cax = fig.add_axes([(width - right + 0.3) / width, maps_low + 0.12 * (maps_high - maps_low),
                        0.13 / width, 0.76 * (maps_high - maps_low)])
    colorbar = fig.colorbar(image, cax=cax, extend="both")
    # the bars beside single rows are short, so their labels break over two lines to stay
    # within their own row: the same words as SWE_CHANGE_LABEL and DEPTH_CHANGE_LABEL
    swe_two_lines = SWE_CHANGE.replace(" change", "\nchange") + f" ({SWE_UNIT})"
    colorbar.set_label(SCREEN_LABEL if screens else swe_two_lines, fontsize=9)
    if screens:
        _pi_ticks(colorbar, screen_lim)
    cax.tick_params(labelsize=8.5)
    cumulative = sums.get("swe")                            # the SWE total, for the season maps below
    if lidar_pairs:
        cax = fig.add_axes([(width - right + 0.3) / width, y0, 0.13 / width, panel_h / height])
        colorbar = fig.colorbar(lidar_image, cax=cax, extend="both")
        colorbar.set_label(f"{DEPTH}\nchange ({DEPTH_UNIT})", fontsize=9)
        cax.tick_params(labelsize=8.5)

    # screens version: one total per screen, the sum of that screen over the pairs
    if totals and screens:
        row = [g for g in gunws if g[0] == tracks[0]]
        assert len(tracks) == 1 and all(a[2] == b[1] for a, b in zip(row[:-1], row[1:])), \
            "totals need one track whose pairs chain end to end"
        nice_day = lambda stamp: f"{day(stamp):%b} {day(stamp).day}"
        map_w, gap_in = totals_h / box_aspect, 0.6
        x_start = left + (width - left - right - len(quantities) * map_w - (len(quantities) - 1) * gap_in) / 2
        for k, q in enumerate(quantities):
            relative = sums[q] - np.nanmedian(sums[q])
            print(f"  cumulative {q}: median {np.nanmedian(sums[q]):+.2f} rad, spread 1-99% "
                  f"{np.nanpercentile(relative, 1):+.2f} to {np.nanpercentile(relative, 99):+.2f} rad")
            x_map = x_start + k * (map_w + gap_in)
            ax = fig.add_axes([x_map / width, 0.3 / height, map_w / width, totals_h / height])
            setup_map_axes(ax, map_box)                     # the Fig. 1 projection + bounds
            total_image = ax.imshow(relative, extent=extent, origin="upper", cmap="RdBu_r",
                                    vmin=-total_lim, vmax=total_lim)
            ax.contour(map_box["lidar_mask"], levels=[0.5], extent=map_box["lidar_extent"],
                       origin="upper", colors=LIDAR_OUTLINE_C, linewidths=1.6)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_title(f"Cumulative {SCREENS[q][1].lower()},\n{nice_day(row[0][1])} to {nice_day(row[-1][2])}",
                         fontsize=9.5)
        # one colour bar for the totals, beside the last map
        cax = fig.add_axes([(x_map + map_w + 0.12) / width, (0.3 + 0.1 * totals_h) / height,
                            0.12 / width, 0.8 * totals_h / height])
        colorbar = fig.colorbar(total_image, cax=cax, extend="both")
        colorbar.set_label(SCREEN_LABEL, fontsize=9.5)
        _pi_ticks(colorbar, total_lim)
        cax.tick_params(labelsize=8.5)

    # two season totals under the SWE panel, each in what its instrument measures: the summed
    # NISAR SWE change, and the lidar snow depth of the last date. No density is assumed.
    if totals and not screens:
        row = [g for g in gunws if g[0] == tracks[0]]
        assert len(tracks) == 1 and all(a[2] == b[1] for a, b in zip(row[:-1], row[1:])), \
            "totals need one track whose pairs chain end to end"
        first_date, last_date = row[0][1], row[-1][2]
        grid = xr.DataArray(cumulative, coords={"y": ys, "x": xs}, dims=("y", "x")).rio.write_crs(paths.EPSG)
        depth = rioxarray.open_rasterio(lidar_file(last_date), masked=True).squeeze().rio.write_nodata(np.nan)
        depth = depth.rio.reproject_match(grid, resampling=Resampling.average).values   # metres, on the 80 m grid
        if np.nanmax(depth) > 10:
            raise ValueError("lidar snow depth is not plausible in metres; check units")
        print(f"  totals: {first_date} to {last_date} | NISAR cumulative median {np.nanmedian(cumulative) * 1000:.0f} mm | "
              f"lidar depth median {np.nanmedian(depth):.2f} m")
        nice_day = lambda stamp: f"{day(stamp):%b} {day(stamp).day}"
        # three equal boxes in a row: two maps, each with its colour bar in the gap to its right,
        # then the scatter, which also needs room on its left for its own axis labels
        map_w, gap_maps, gap_scatter, y_in = totals_h / box_aspect, 1.05, 1.5, 0.55
        x_start = left + (width - left - right - 3 * map_w - gap_maps - gap_scatter) / 2
        panels = ((cumulative * 1000, f"Cumulative {SWE_CHANGE},\n{nice_day(first_date)} to {nice_day(last_date)}",
                   TOTAL_LIM_MM, SWE_CHANGE_LABEL),
                  (depth, f"{DEPTH},\n{nice_day(last_date)}", DEPTH_LIM_M, DEPTH_LABEL))
        for k, (field, title, limit, unit) in enumerate(panels):
            x_map = x_start + k * (map_w + gap_maps)
            ax = fig.add_axes([x_map / width, y_in / height, map_w / width, totals_h / height])
            setup_map_axes(ax, map_box)                     # the Fig. 1 projection + bounds
            total_image = ax.imshow(field, extent=extent, origin="upper", cmap=cmocean.cm.ice, vmin=0, vmax=limit)
            ax.contour(map_box["lidar_mask"], levels=[0.5], extent=map_box["lidar_extent"],
                       origin="upper", colors=LIDAR_OUTLINE_C, linewidths=1.6)
            ax.plot(*station_xy, "*", color="white", mec="k", mew=0.5, ms=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_title(title, fontsize=9.5)
            # each map has its own colour bar: the two are in different units
            cax = fig.add_axes([(x_map + map_w + 0.12) / width, (y_in + 0.1 * totals_h) / height,
                                0.12 / width, 0.8 * totals_h / height])
            colorbar = fig.colorbar(total_image, cax=cax, extend="max")
            colorbar.set_label(unit, fontsize=9.5)
            cax.tick_params(labelsize=8.5)
        # the two totals against each other, pixel by pixel, where both exist; a density of
        # points on a log count scale, as in Fig. 2
        x_scatter = x_start + 2 * map_w + gap_maps + gap_scatter
        ax = fig.add_axes([x_scatter / width, y_in / height, map_w / width, totals_h / height])
        hexes = density_scatter(ax, depth, cumulative * 1000, (0, DEPTH_LIM_M * 1.3, 0, TOTAL_LIM_MM * 1.05))
        slope, intercept = density_fit(ax, depth, cumulative * 1000, "lower right", fontsize=8.5)
        print(f"  totals fit: slope {slope:.0f} kg/m3, intercept {intercept:+.0f} mm")
        ax.set_xlabel(DEPTH_LABEL, fontsize=9.5)
        ax.set_ylabel(f"Cumulative {SWE_CHANGE_LABEL}", fontsize=9.5)
        ax.title.set_fontsize(9.5)
        ax.tick_params(labelsize=8.5)
        cax = fig.add_axes([(x_scatter + map_w + 0.12) / width, (y_in + 0.1 * totals_h) / height,
                            0.12 / width, 0.8 * totals_h / height])
        colorbar = fig.colorbar(hexes, cax=cax)
        count_colorbar(colorbar)
        colorbar.ax.yaxis.label.set_size(9.5)
        cax.tick_params(which="both", labelsize=8.5)

    # one key for the whole figure
    place = (dict(loc="center left", bbox_to_anchor=(x_of(t0), row_y[late] + panel_h / height / 2)) if key_in_row
             else dict(loc="upper left", bbox_to_anchor=(x_of(t0) - 0.08 / width, 1 - 0.36 / height), ncol=6, columnspacing=1.2))
    handles = [Patch(facecolor="none", edgecolor=OURS_COLOR, lw=2.0, label="Hyp3-Isce3 produced GUNW"),
               Patch(facecolor="none", edgecolor="k", lw=1.0, label="Figure 2 Pair (double outline)"),
               Line2D([], [], color=LIDAR_OUTLINE_C, lw=1.6, label="lidar coverage"),
               Line2D([], [], marker="*", color="none", mfc="white", mec="k", mew=0.5, ms=8,
                      label="SNOTEL 637 (reference)"),
               Line2D([], [], color=NISAR_COLOR, lw=1.1, label="NISAR acquisition")]
    if lidar_pairs:
        handles.append(Line2D([], [], color=LIDAR_COLOR, lw=2.1, ls=(0, (3, 2)), label="lidar flight"))
    if screens:                                             # the screens version: refocused pairs, lidar outline, acquisitions
        handles = [Patch(facecolor="none", edgecolor=OURS_COLOR, lw=2.0, label="formed from the refocused Jan 14 image"),
                   handles[2], handles[4]]
    fig.legend(handles=handles, fontsize=8.5, frameon=False, handlelength=1.8, labelspacing=0.5, **place)
    default_heading = "NISAR atmospheric phase screens" if screens else SWE_CHANGE
    fig.text(x_of(t0), 1 - 0.12 / height, heading or default_heading, ha="left", va="top", fontsize=12,
             fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    if print_width_cm:
        _save_print(fig, out, print_width_cm)
    else:
        fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved ->", out)


def swe_timeline_figure():
    """Fig. 3, split in two as a test: the pair timeline here, the season totals in
    _season_totals_figure(), which this calls with the summed NISAR SWE change.

    Top row, the NISAR SWE change of every track 077 pair from Nov 3 to Mar 15: the maps are
    evenly spaced, and a funnel ties each one down to its two acquisition dates on the thin
    SNOTEL SWE strip beneath. Bottom row, the NIVAL lidar snow-depth change of the flights
    matched to each NISAR pair in pairs.py: each map sits under its NISAR pair, with a funnel
    up to its two flight dates on the bottom of the same strip. The Dec 3 to Dec 13 lidar pair
    is left out."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.patches import Polygon as MplPolygon
    plt.rcdefaults()                                        # Figs. 1 and 2 set bold titles and sizes; start clean
    day = pd.Timestamp
    nice_day = lambda stamp: f"{day(stamp):%b} {day(stamp).day}"

    # the pairs, the lidar flights matched to them, and the SWE record
    gunws = [g for g in local_gunws() if g[0] == TRACK and g[2] <= paths.FIG3_LAST]
    assert all(a[2] == b[1] for a, b in zip(gunws[:-1], gunws[1:])), "the pairs must chain end to end"
    lidar_pairs = sorted((p for p in PAIRS if p[0] == TRACK and p[2] <= paths.FIG3_LAST
                          and (PRODUCTS / f"{label(p)}.nc").exists()), key=lambda p: p[3])
    lidar_pairs = [p for p in lidar_pairs if p[3] != "20251203"]   # the Dec 3 to Dec 13 flights are not shown
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"])
    swe = pd.Series(season["wteq_in"].values * 0.0254, index=season["date"]).sort_index()   # inches -> metres at ingest
    if swe.max() > 3.0:
        raise ValueError("SNOTEL SWE is not plausible in metres; check units")
    t0, t1 = season_span()
    snotel_swe = _snotel_swe()                              # metres, the series Table S1 reads
    map_box = map_frame()                                   # one shared frame for every map
    box_aspect = (map_box["extent"][3] - map_box["extent"][2]) / (map_box["extent"][1] - map_box["extent"][0])

    # figure geometry in inches, measured from the top edge
    width, left, right = 12.0, 0.75, 1.25
    usable = width - left - right
    span_days = (t1 - t0).days
    pitch = usable / len(gunws)                             # NISAR: one evenly spaced column per pair
    map_w = 0.88 * pitch
    map_h = map_w * box_aspect
    lidar_w, lidar_h = map_w, map_h                         # lidar maps match the NISAR maps they sit over
    top, funnel_h, strip_h = 0.66, 0.36, 0.75                # the top holds the key and the NISAR date row
    y_nisar = top + map_h                                   # bottom of the NISAR maps
    y_strip_top = y_nisar + funnel_h
    y_strip = y_strip_top + strip_h                         # bottom of the thin SWE strip
    y_lidar_top = y_strip + funnel_h                        # room for the lidar funnels
    y_lidar = y_lidar_top + lidar_h                         # bottom of the lidar maps
    height = y_lidar + 0.34                                 # room for the lidar date row
    fx = lambda inches: inches / width                      # inches from the left -> figure fraction
    fy = lambda inches: 1 - inches / height                 # inches from the top -> figure fraction
    x_of = lambda t: fx(left + (day(t) - t0).days / span_days * usable)   # date -> figure x on the time axis
    column_x = lambda k: left + (k + 0.5) * pitch           # centre of NISAR column k [in]

    fig = plt.figure(figsize=(width, height), dpi=300)

    def draw_map(x_centre, y_bottom, w, h, field, extent, limit):
        """One pair-change map centred at x_centre [in], in the Fig. 1 frame. Blue is gain."""
        ax = fig.add_axes([fx(x_centre - w / 2), fy(y_bottom), fx(w), h / height])
        setup_map_axes(ax, map_box)
        image = ax.imshow(field, extent=extent, origin="upper", cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for spine in ax.spines.values():                    # a quiet frame
            spine.set_edgecolor("0.3"); spine.set_linewidth(0.5)
        return ax, image

    def funnel(x_centre, w, y_map_edge, y_axis_edge, first, second, colour):
        """Shaded funnel from a map's edge to the two dates it spans on the time axis."""
        corners = [(fx(x_centre - w / 2), fy(y_map_edge)), (fx(x_centre + w / 2), fy(y_map_edge)),
                   (x_of(second), fy(y_axis_edge)), (x_of(first), fy(y_axis_edge))]
        fig.add_artist(MplPolygon(corners, closed=True, transform=fig.transFigure, facecolor=colour, alpha=0.18,
                                  edgecolor="none", zorder=-2))
        for a, b in ((corners[0], corners[3]), (corners[1], corners[2])):
            fig.add_artist(Line2D([a[0], b[0]], [a[1], b[1]], transform=fig.transFigure, color=colour, lw=0.8, zorder=-1))

    # NISAR row, on top, each map tied down to its two acquisition dates on the SWE strip
    cumulative = None
    for k, (track, ref, sec, path, ours) in enumerate(gunws):
        swe_change, xs, ys, station_xy, log = _swe_change(path, ref, sec, snotel_swe)
        print(f"  {ref}-{sec}: SNOTEL {log['station_mm']:+6.0f} mm | map 5-95% {np.nanpercentile(swe_change, 5) * 1000:+.0f} "
              f"to {np.nanpercentile(swe_change, 95) * 1000:+.0f} mm")
        cumulative = swe_change if cumulative is None else cumulative + swe_change
        posting = float(abs(xs[1] - xs[0]))
        # xs / ys are pixel centres, so the raster edge sits half a posting beyond them
        extent = [xs.min() - posting / 2, xs.max() + posting / 2, ys.min() - posting / 2, ys.max() + posting / 2]
        # show NISAR only inside the lidar footprint: look the footprint mask up at each pixel centre
        mask, (m_x0, m_x1, m_y0, m_y1) = map_box["lidar_mask"], map_box["lidar_extent"]
        cols = np.floor((xs - m_x0) / (m_x1 - m_x0) * mask.shape[1]).astype(int)
        rows = np.floor((m_y1 - ys) / (m_y1 - m_y0) * mask.shape[0]).astype(int)
        on_mask = ((rows >= 0) & (rows < mask.shape[0]))[:, None] & ((cols >= 0) & (cols < mask.shape[1]))[None, :]
        inside = on_mask & (mask[np.clip(rows, 0, mask.shape[0] - 1)][:, np.clip(cols, 0, mask.shape[1] - 1)] > 0.5)
        ax, nisar_image = draw_map(column_x(k), y_nisar, map_w, map_h, np.where(inside, swe_change * 1000, np.nan),
                                   extent, SWE_LIM_MM)
        ax.plot(*station_xy, "*", color="white", mec="k", mew=0.5, ms=7)   # the phase reference, as in Fig. 1
        funnel(column_x(k), map_w, y_nisar, y_strip_top, ref, sec, NISAR_COLOR)

    # lidar row, at the bottom: each map sits under the NISAR pair it is matched with, and a funnel
    # runs from its top edge up to its two flight lines on the time axis
    nisar_dates = sorted({day(d) for g in gunws for d in g[1:3]})
    flight_dates = sorted({day(d) for p in lidar_pairs for d in p[3:5]})
    column_of = {g[:3]: k for k, g in enumerate(gunws)}     # NISAR pair -> its column
    for pair in lidar_pairs:
        centre = column_x(column_of[pair[:3]])
        ds = xr.open_dataset(PRODUCTS / f"{label(pair)}.nc")   # dhs: later flight minus earlier, on the 80 m grid [m]
        ax, lidar_image = draw_map(centre, y_lidar, lidar_w, lidar_h, ds["dhs"].values, extent, LIDAR_LIM_M)
        funnel(centre, lidar_w, y_lidar_top, y_strip, pair[3], pair[4], LIDAR_COLOR)

    # the time axis: a thin SNOTEL 637 SWE strip between the rows, with every acquisition and flight marked
    ax_swe = fig.add_axes([fx(left), fy(y_strip), fx(usable), strip_h / height])
    # SNOTEL SWE as a light grey line, with a fine grid of light grey speckles beneath it
    peak_mm = swe.max() * 1000
    spacing = 0.03                                          # speckle spacing on the page [in]
    grid_day = np.arange(0, span_days, span_days * spacing / usable)   # days after t0
    grid_mm = np.arange(0, peak_mm, peak_mm * spacing / strip_h)[1:]
    speck_day, speck_mm = [values.ravel() for values in np.meshgrid(grid_day, grid_mm)]
    curve_mm = np.interp(speck_day, (swe.index - t0).days, swe.values * 1000)
    under = speck_mm < curve_mm
    ax_swe.scatter(t0 + pd.to_timedelta(speck_day[under], unit="D"), speck_mm[under], s=0.5, color="0.72", lw=0)
    ax_swe.plot(swe.index, swe.values * 1000, color="0.6", lw=1.5)
    ax_swe.set_xlim(t0, t1); ax_swe.set_ylim(0, None)
    # a tick at each month start; only January and March are labelled, above the axis
    ax_swe.set_xticks([day("2025-12-01"), day("2026-01-01"), day("2026-02-01"), day("2026-03-01")])
    ax_swe.set_xticklabels(["", "Jan 2026", "", "Mar 2026"])
    ax_swe.tick_params(axis="x", direction="in", pad=-12.5)
    ax_swe.set_ylabel("SNOTEL 637\nSWE (mm)", fontsize=9.5)   # the month ticks need no axis label
    ax_swe.set_yticks([0, 150, 300])
    ax_swe.tick_params(labelsize=9.5)
    ax_swe.grid(axis="y", color="0.85", lw=0.6); ax_swe.set_axisbelow(True)
    ax_swe.spines[["top", "right"]].set_visible(False)
    mark_acquisitions(ax_swe, nisar=nisar_dates, lidar=flight_dates)   # as on Fig. 1

    # the dates of each row, written between its maps with an arrow across each map: NISAR
    # acquisitions above the NISAR row, lidar flights below the lidar row
    first_lidar_column = column_of[lidar_pairs[0][:3]]
    for dates, first_column, y, colour in ((nisar_dates, 0, top - 0.14, NISAR_COLOR),
                                           (flight_dates, first_lidar_column, y_lidar + 0.15, LIDAR_COLOR)):
        edges = [left + (first_column + k) * pitch for k in range(len(dates))]   # the gaps between the maps [in]
        for date, x in zip(dates, edges):
            fig.text(fx(x), fy(y), nice_day(date), ha="center", va="center", fontsize=9.5, color=colour)
        for x0, x1 in zip(edges[:-1], edges[1:]):
            fig.add_artist(FancyArrowPatch((fx(x0 + 0.3), fy(y)), (fx(x1 - 0.3), fy(y)), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=9, color=colour, lw=1.0))

    # row names and colour bars. The bars are one map tall, so their labels break over two lines.
    for text, y_bottom, h in ((f"NISAR Track {int(TRACK)}\nSWE Change", y_nisar, map_h),
                              ("NIVAL Airborne\nLidar Snow\nDepth Change", y_lidar, lidar_h)):
        fig.text(fx(0.34), fy(y_bottom - h / 2), text, rotation=90, ha="center", va="center", fontsize=10.5,
                 linespacing=1.25)
    for image, y_bottom, h, text in ((lidar_image, y_lidar, lidar_h, f"{DEPTH}\nchange ({DEPTH_UNIT})"),
                                     (nisar_image, y_nisar, map_h, SWE_CHANGE.replace(" change", "\nchange") + f" ({SWE_UNIT})")):
        cax = fig.add_axes([fx(width - right + 0.3), fy(y_bottom), fx(0.13), h / height])
        colorbar = fig.colorbar(image, cax=cax, extend="both")
        colorbar.set_label(text, fontsize=10); cax.tick_params(labelsize=9.5)

    # one key for the whole figure, in a single row along the top
    fig.legend(handles=[Line2D([], [], marker="*", color="none", mfc="white", mec="k", mew=0.5, ms=9,
                               label="SNOTEL 637"),
                        Line2D([], [], color=NISAR_COLOR, lw=1.1, label="NISAR acquisition"),
                        Line2D([], [], color=LIDAR_COLOR, lw=2.1, ls=(0, (3, 2)), label="lidar flight")],
               loc="upper left", bbox_to_anchor=(fx(left - 0.08), 1 - 0.08 / height), ncol=3, columnspacing=1.4,
               fontsize=9.5, frameon=False, handlelength=1.6, borderaxespad=0)

    out = paths.FIGURES_DIR / "fig3a_pair_timeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_print(fig, out, 18)
    print("saved ->", out)
    plt.close(fig)
    _season_totals_figure(cumulative, xs, ys, station_xy, extent, gunws[0][1], gunws[-1][2], map_box)


def _season_totals_figure(cumulative, xs, ys, station_xy, extent, first_date, last_date, map_box):
    """The other half of the split Fig. 3: the summed NISAR SWE change [m] on the grid xs / ys,
    the lidar snow depth of the last date, and one against the other."""
    nice_day = lambda stamp: f"{pd.Timestamp(stamp):%b} {pd.Timestamp(stamp).day}"
    box_aspect = (map_box["extent"][3] - map_box["extent"][2]) / (map_box["extent"][1] - map_box["extent"][0])
    # figure geometry in inches, measured from the top edge: three equal boxes across the width,
    # with the gaps that hold their colour bars and the scatter's axis labels
    width = 12.0
    gap_maps, gap_scatter, totals_left, totals_right = 1.0, 1.62, 0.15, 0.9
    total_w = (width - totals_left - totals_right - gap_maps - gap_scatter) / 3
    totals_h = total_w * box_aspect
    y_totals = 0.5 + totals_h                               # bottom of the panels, under their titles
    height = y_totals + 0.48
    fx = lambda inches: inches / width                      # inches from the left -> figure fraction
    fy = lambda inches: 1 - inches / height                 # inches from the top -> figure fraction
    fig = plt.figure(figsize=(width, height), dpi=300)

    # season totals, each in what its instrument measures: the summed NISAR SWE change, and the
    # lidar snow depth of the last date. No density is assumed.
    grid = xr.DataArray(cumulative, coords={"y": ys, "x": xs}, dims=("y", "x")).rio.write_crs(paths.EPSG)
    depth = rioxarray.open_rasterio(lidar_file(last_date), masked=True).squeeze().rio.write_nodata(np.nan)
    depth = depth.rio.reproject_match(grid, resampling=Resampling.average).values   # metres, on the 80 m grid
    if np.nanmax(depth) > 10:
        raise ValueError("lidar snow depth is not plausible in metres; check units")
    print(f"  totals: {first_date} to {last_date} | NISAR cumulative median {np.nanmedian(cumulative) * 1000:.0f} mm | "
          f"lidar depth median {np.nanmedian(depth):.2f} m")
    # three equal boxes in a row: two maps, each with its colour bar in the gap to its right,
    # then the scatter, which also needs room on its left for its own axis labels
    panels = ((cumulative * 1000, f"Cumulative NISAR SWE,\n{nice_day(first_date)} to {nice_day(last_date)}",
               TOTAL_LIM_MM, SWE_CHANGE_LABEL),
              (depth, f"{DEPTH},\n{nice_day(last_date)}", DEPTH_LIM_M, DEPTH_LABEL))
    for k, (field, title, limit, unit) in enumerate(panels):
        x_map = totals_left + k * (total_w + gap_maps)
        ax = fig.add_axes([fx(x_map), fy(y_totals), fx(total_w), totals_h / height])
        setup_map_axes(ax, map_box)                         # the Fig. 1 projection + bounds
        total_image = ax.imshow(field, extent=extent, origin="upper", cmap=cmocean.cm.ice, vmin=0, vmax=limit)
        ax.contour(map_box["lidar_mask"], levels=[0.5], extent=map_box["lidar_extent"], origin="upper",
                   colors=LIDAR_OUTLINE_C, linewidths=1.6)
        ax.plot(*station_xy, "*", color="white", mec="k", mew=0.5, ms=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title(title, fontsize=10.5)
        panel_letter(ax, "ab"[k], fontsize=11)
        # each map has its own colour bar: the two are in different units
        cax = fig.add_axes([fx(x_map + total_w + 0.12), fy(y_totals - 0.1 * totals_h), fx(0.12), 0.8 * totals_h / height])
        colorbar = fig.colorbar(total_image, cax=cax, extend="max")
        colorbar.set_label(unit, fontsize=10.5); cax.tick_params(labelsize=9.5)
    # the two totals against each other, pixel by pixel, where both exist
    x_scatter = totals_left + 2 * total_w + gap_maps + gap_scatter
    ax = fig.add_axes([fx(x_scatter), fy(y_totals), fx(total_w), totals_h / height])
    hexes = density_scatter(ax, depth, cumulative * 1000, (0, DEPTH_LIM_M * 1.3, 0, TOTAL_LIM_MM * 1.05))
    slope, intercept = density_fit(ax, depth, cumulative * 1000, "lower right", fontsize=9.5)
    print(f"  totals fit: slope {slope:.0f} kg/m3, intercept {intercept:+.0f} mm")
    ax.set_xlabel(DEPTH_LABEL, fontsize=10.5); ax.set_ylabel(f"Cumulative {SWE_CHANGE_LABEL}", fontsize=10.5)
    ax.title.set_fontsize(10.5); ax.tick_params(labelsize=9.5)
    panel_letter(ax, "c", fontsize=11)
    cax = fig.add_axes([fx(x_scatter + total_w + 0.12), fy(y_totals - 0.1 * totals_h), fx(0.12), 0.8 * totals_h / height])
    colorbar = fig.colorbar(hexes, cax=cax)
    count_colorbar(colorbar)
    colorbar.ax.yaxis.label.set_size(10.5); cax.tick_params(which="both", labelsize=9.5)

    out = paths.FIGURES_DIR / "fig3b_season_totals.png"
    _save_print(fig, out, 18)
    print("saved ->", out)
    plt.close(fig)


if __name__ == "__main__":
    context_figure()
    results_figure()
    swe_timeline_figure()

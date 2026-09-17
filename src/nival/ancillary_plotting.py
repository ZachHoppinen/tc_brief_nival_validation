"""Ancillary plotting: the study-area map, the met-series readers, and the panel
builders that paper_figures.context_figure() composes into fig1.

These give context, not the headline result, so correctness here is less critical
than in validation_plotting.py. The panel builders (panel_swe, panel_air_temperature,
panel_snowpack_temperature, panel_soil, panel_ionosphere) and the render_time_series
stacker are a reusable met-figure toolkit.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import cartopy.io.shapereader as shpreader
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor)
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from rasterio.enums import Resampling

from . import paths, wetness
from .retrieval import _elevation_on_grid, _raster_on_grid

EPSG = paths.EPSG
NISAR_COLOR, LIDAR_COLOR = "#c81d4e", "#6a4c93"


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


def snowpack_thermograph():
    """Freeman DTC string -> continuous snowpack temperature field (height x time).
    Returns (tnum, z_grid, G_masked, surface_height, timestamps)."""
    s = pd.read_csv(paths.MET_SOIL, usecols=["TIMESTAMP", "SnoDAR_snow_depth_Avg"] + DTC_COLS)
    s["TIMESTAMP"] = pd.to_datetime(s["TIMESTAMP"])
    s = s[(s["TIMESTAMP"] >= paths.D0) & (s["TIMESTAMP"] <= paths.D1)].reset_index(drop=True)
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
def mark_acquisitions(ax, top=False, pad=1.0):
    """Vertical markers for the NISAR pair + lidar flights; labels on the top panel.
    `pad` scales the label heights above the axis (use <1 for a tighter composite)."""
    for date, _ in NISAR:
        ax.axvline(dt.date.fromisoformat(date), color=NISAR_COLOR, lw=1.8, zorder=1)
    for date, _ in LIDAR:
        ax.axvline(dt.date.fromisoformat(date), color=LIDAR_COLOR, lw=2.6, ls=(0, (4, 3)), zorder=1)
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
    return [Line2D([], [], color=NISAR_COLOR, lw=1.8, label="NISAR"),
            Line2D([], [], color=LIDAR_COLOR, lw=2.6, ls=(0, (3, 2)), label="Lidar")]   # denser dash fills the handle


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
    ax.legend(handles, labels, loc="lower left", fontsize=9, framealpha=0.9, handlelength=2.8)
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
FOOT_PAD = 0.004                                       # lon/lat margin around the lidar footprint
MORES_LL = (-115.6667, 43.9333)                        # Idaho locator dot
SNOTEL_LL = (-115.6667, 43.9333)                       # Mores Creek Summit #637
MET_STATIONS = [(-115.7018528, 43.94091389, "Freeman Peak"),
                (-115.687, 43.960080, "Pilot Peak")]
NAIP_URL = ("https://imagery.nationalmap.gov/arcgis/rest/services/"
            "USGSNAIPImagery/ImageServer/exportImage")
FIRE_C, MET_C = "#ff5722", "white"


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
    minx, miny, maxx, maxy = coverage.rio.bounds()
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
    """Pioneer Fire (2016) perimeter (UTM exterior rings + polygon) from the frozen geojson."""
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


def sar_geometry_inset(ax, rect, track="077"):
    """Acquisition-geometry inset: flight track (black) + left-looking imaging arrow (blue)."""
    flight_az, look_az, incidence = sar_geometry()
    inset = ax.inset_axes(rect)
    inset.set_xlim(-1.25, 1.25); inset.set_ylim(-1.55, 1.25); inset.set_aspect("equal")
    inset.set_xticks([]); inset.set_yticks([]); inset.set_facecolor("#e9edf3"); inset.patch.set_alpha(0.92)
    for spine in inset.spines.values():
        spine.set_edgecolor("k"); spine.set_linewidth(1.0)
    direction = lambda az: (np.sin(np.radians(az)), np.cos(np.radians(az)))
    fx, fy = direction(flight_az)
    inset.plot([-fx * 0.85, fx * 0.85], [-fy * 0.85, fy * 0.85], color="k", lw=2.2, zorder=3)
    inset.annotate("", xy=(fx * 0.95, fy * 0.95), xytext=(fx * 0.6, fy * 0.6),
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=2.2, mutation_scale=13), zorder=3)
    lx, ly = direction(look_az + 180.0)        # imaging/look = sensor->ground (opposite the toward-sat LOS); left-looking here
    inset.annotate("", xy=(lx * 0.78, ly * 0.78), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=2.2, mutation_scale=13), zorder=4)
    inset.plot(0, 0, "o", color="k", ms=3, zorder=5)
    inset.text(fx * 1.04, fy * 1.10, "flight\ndirection", color="k", fontsize=7.5, fontweight="bold",
               ha="center", va="center", linespacing=0.9)
    inset.text(lx * 0.5 - 0.06, ly * 0.5 + 0.32, "imaging\ndirection", color="#1565c0", fontsize=7.5,
               fontweight="bold", ha="center", va="center", linespacing=0.9)
    inset.annotate("", xy=(-0.95, -0.7), xytext=(-0.95, -1.05),
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, mutation_scale=8))
    inset.text(-0.95, -0.52, "N", color="k", fontsize=9, fontweight="bold", ha="center", va="center")
    inset.text(0, -1.4, f"NISAR ASC {track}", color="k", fontsize=10,
               fontweight="bold", ha="center")
    return flight_az, look_az, incidence


def idaho_locator(ax):
    """Idaho state outline (Natural Earth) with a red dot on Mores Creek."""
    shapefile = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces")
    for record in shpreader.Reader(shapefile).records():
        if record.attributes.get("name") != "Idaho":
            continue
        geometry = record.geometry
        for polygon in (geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]):
            x, y = polygon.exterior.xy
            ax.fill(x, y, fc="0.82", ec="0.5", lw=0.8, alpha=0.9)        # light-grey state shape, no surrounding box
    ax.plot(*MORES_LL, "o", color="red", mec="k", mew=0.6, ms=7, zorder=5)
    ax.annotate("Mores Ck", MORES_LL, textcoords="offset points", xytext=(7, -2),
                fontsize=7, color="red", fontweight="bold")
    ax.set_aspect(1 / np.cos(np.radians(45)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_visible(False)                                          # no background box behind Idaho
    for spine in ax.spines.values():
        spine.set_visible(False)


def study_area(ax):
    """Fig-1 study-area map drawn into `ax`: NAIP basemap + lidar footprint + fire +
    stations + geometry inset (composed by paper_figures.context_figure)."""
    from shapely.geometry import box as shapely_box
    frame = map_frame()
    extent, aoi, xs, ys = frame["extent"], frame["aoi"], frame["xs"], frame["ys"]
    lidar_mask, lidar_extent = frame["lidar_mask"], frame["lidar_extent"]
    elevation = _elevation_on_grid(xs, ys, aoi)
    forest = _forest_fraction(xs, ys)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform
    snotel_x, snotel_y = to_utm(*SNOTEL_LL)
    print(f"grid {elevation.shape} @ {MAP_DX:.0f} m | forest {100*forest:.0f}%")

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    ax.imshow(fetch_naip(extent), extent=extent, origin="upper", zorder=0)

    lo, hi = np.nanpercentile(elevation, [1, 99])
    levels = np.arange(np.ceil(lo / 100) * 100, hi, 100)
    ax.contour(elevation, levels=levels, extent=extent, origin="upper", colors="white", linewidths=0.7, alpha=0.45)
    ax.contour(lidar_mask, levels=[0.5], extent=lidar_extent, origin="upper", colors="cyan", linewidths=1.8)

    fire_rings, fire_polygon = pioneer_perimeter()
    burned = fire_polygon.intersection(shapely_box(extent[0], extent[2], extent[1], extent[3]))
    for polygon in (burned.geoms if burned.geom_type == "MultiPolygon" else [burned]):
        if not polygon.is_empty:
            ax.add_patch(Polygon(np.array(polygon.exterior.coords), closed=True, facecolor=FIRE_C,
                                 edgecolor="none", alpha=0.16, zorder=1.5))
    for ring in fire_rings:
        ax.plot(ring[:, 0], ring[:, 1], color=FIRE_C, lw=2.4, ls="--", zorder=3)
    frame_box = shapely_box(extent[0], extent[2], extent[1], extent[3])
    print(f"Pioneer burn covers {100*burned.area/frame_box.area:.0f}% of frame")

    ax.plot(snotel_x, snotel_y, "*", color="yellow", mec="k", mew=1.2, ms=22, zorder=6)
    ax.annotate("SNOTEL 637\n(Mores Ck Summit)", (snotel_x, snotel_y), textcoords="offset points",
                xytext=(10, 6), fontsize=12, fontweight="bold", color="white")
    for lon, lat, label in MET_STATIONS:
        x, y = to_utm(lon, lat)
        ax.plot(x, y, "^", color=MET_C, mec="k", mew=1.2, ms=13, zorder=6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(9, 4), fontsize=11,
                    fontweight="bold", color="white")

    ax.set_xlabel("UTM 11N easting (m)", fontsize=13); ax.set_ylabel("northing (m)", fontsize=13)
    ax.tick_params(labelsize=11)

    legend = [Line2D([], [], marker="*", color="none", mfc="yellow", mec="k", ms=15, label="SNOTEL 637"),
              Line2D([], [], color="cyan", lw=1.8, label="Lidar Coverage"),
              Line2D([], [], color=FIRE_C, lw=2.4, ls="--", label="Pioneer Fire (2016)"),
              Line2D([], [], marker="^", color="none", mfc=MET_C, mec="k", ms=11, label="met station")]
    ax.legend(handles=legend, loc="lower right", fontsize=12, framealpha=0.9)

    idaho_locator(ax.inset_axes([0.01, 0.73, 0.25, 0.25]))
    flight_az, look_az, incidence = sar_geometry_inset(ax, [0.015, 0.012, 0.27, 0.31])
    print(f"SAR geometry: flight {flight_az:.0f}, look {look_az:.0f}, incidence {incidence:.0f} deg (ASC 077)")

"""Quantitative support for the NISAR dSWE vs NIVAL dHS paper: one script -> one markdown report.

`main()` prints every number the manuscript cites, each block under the claim it supports.
Everything runs on the operational 80 m product of the Fig. 2 pair. In run order:

  ASSUMPTIONS   -- physical constants + modelling choices, pulled from the live code
  LIDAR GATE    -- fraction of native-resolution lidar dHS the |dHS| outlier gate removes
  WRAPPING      -- lidar dHS -> phase: fringes spanned + snaphu fringes added (little wrapping)
  HEADLINE      -- r / r(deramp) / RMSE / density on the operational GUNW, by coherence gate
  SCALE         -- kilometer-scale (low-pass) vs sub-kilometer (residual) correlation
  OPENNESS/BURN -- r + coherence by canopy fraction and MTBS severity (less canopy -> better)
  LOCAL r       -- local r vs coherence / incidence / canopy, + per-stratum medians
  TERRAIN, DRIFT-CONTROL, WIND, LIDAR-WINDOW, MET, IONOSPHERE, TROPOSPHERE -- conditions + controls
  PAIR TABLE    -- every track-077 pair: NISAR SWE change, lidar depth change, SNOTEL SWE change, r

Run:  conda run -n nival python src/nival/paper_stats.py   (writes outputs/paper_stats.md)
"""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # add src/ so `nival` imports

import numpy as np
import h5py
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.stats import spearmanr

from nival import paths, retrieval
from nival.retrieval import local_correlation          # shared local-r metric (fig3 uses the same)
from nival.paper_figures import (_bilinear as _bilin, _elevation_on_grid,
                                      _raster_on_grid, read_ionex)

# Everything runs on the one operational product (80 m posting). The windows are specified
# in METRES and converted, so the physical scales match the earlier multilook analysis:
# a ~400 m canopy-fraction window and a 440 m local-correlation window.
POSTING_M = 80.0
WINDOW = int(round(400.0 / POSTING_M))            # 5 px = 400 m canopy-fraction window
LOCAL_R_WINDOW = 440.0 / POSTING_M                # 5.5 px = 440 m local-r window (float sigma)
SNOW_DENSITY, WATER_DENSITY = retrieval.SNOW_DENSITY, retrieval.WATER_DENSITY
DHS_GATE_M = retrieval.DHS_GATE_M                 # the same gate build_product applies
OPEN_NAMES = ["forest", "open", "burn-low", "burn-mod", "burn-high"]

# nuisance delay -> equivalent dSWE (Leinss kappa at 40 deg; iono 16.8 rad/TECU)
LAMBDA, INC = paths.LAMBDA, np.radians(40.0)
KAPPA = (2 * np.pi / LAMBDA) * (1.59 + INC ** 2.5)
MM_PER_RAD = 1000.0 / KAPPA
MM_PER_TECU = 16.8 * MM_PER_RAD
DSWE_PER_M_ZTD = (4 * np.pi / LAMBDA / np.cos(INC)) * MM_PER_RAD
OVERPASS_UTC_H = 12 + 46 / 60 + 36 / 3600


HH = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
RG = "science/LSAR/GUNW/metadata/radarGrid"


def fig1_met():
    """Fig-1 context numbers from the met series (SWE change, dawn temps, quiet iono)."""
    import datetime as dt
    from nival.paper_figures import load_met_series
    m = load_met_series()
    t, td = list(m["t"]), list(m["tdates"])
    iA, iB = t.index(dt.date(2026, 2, 7)), t.index(dt.date(2026, 2, 19))
    jA, jB = td.index(dt.date(2026, 2, 7)), td.index(dt.date(2026, 2, 19))
    print("\n=== MET CONTEXT (SWE / dawn temp / VTEC) ===")
    print(f"  SWE A->B (SNOTEL):       {m['swe'][iA]:.0f} -> {m['swe'][iB]:.0f} mm  (+{m['swe'][iB]-m['swe'][iA]:.0f})")
    print(f"  Freeman 06:00 air temp:  A {m['temp6'][iA]:+.1f} C   B {m['temp6'][iB]:+.1f} C")
    print(f"  GIM VTEC (overpass):     A {m['vtec'][jA]:.1f}   B {m['vtec'][jB]:.1f} TECU   (quiet)")


def ionosphere_stats(xs, ys):
    """IGS GIM at the dawn overpass: the inter-pass TEC change set against the site's normal
    same-hour 12-day variability over the campaign month. Only the CHANGE matters for a
    differential measurement, and because both passes are at the same local time the diurnal cycle
    differences out -- so we compare the pair change to the distribution of matched-hour 12-day
    changes. (GIM is far too coarse -- ~2.5x5 deg, interpolated -- to resolve or bound the small-
    scale TEC gradients that would affect a 10 km scene, so we characterize background variability
    rather than a mm bound.) Then the split-spectrum screen consistency-with-zero (guarded)."""
    from pyproj import Transformer
    print("\n=== IONOSPHERE ===")
    tr = Transformer.from_crs(paths.EPSG, 4326, always_xy=True)
    lon0, lat0 = tr.transform(float(np.mean(xs)), float(np.mean(ys)))        # AOI center

    # campaign IONEX stack at the AOI center -> TEC[day, epoch] on a common 2-hourly epoch grid
    rows, doys, grid_ep = [], [], None
    for doy in range(32, 61):                                               # Feb 1 (DOY 32) -> Mar 1 (DOY 60)
        p = paths.IONEX_DIR / f"IGS0OPSFIN_2026{doy:03d}0000_01D_02H_GIM.INX"
        if not p.exists():
            continue
        ep, la, lo, tec = read_ionex(p)
        s = np.array([_bilin(la, lo, tec[k], lat0, lon0) for k in range(len(ep))])
        if grid_ep is None:
            grid_ep = np.array(ep)
        rows.append(np.interp(grid_ep, ep, s))                             # resample onto the common epoch grid
        doys.append(doy)
    M, doys = np.array(rows), np.array(doys)
    over_col = int(np.argmin(np.abs(grid_ep - OVERPASS_UTC_H)))            # nearest epoch to the dawn overpass

    # the pair itself, at the dawn overpass hour
    iA, iB = int(np.where(doys == 38)[0][0]), int(np.where(doys == 50)[0][0])
    vA = np.interp(OVERPASS_UTC_H, grid_ep, M[iA]); vB = np.interp(OVERPASS_UTC_H, grid_ep, M[iB])
    print(f"  overpass VTEC (dawn): A (Feb7) {vA:.1f}  B (Feb19) {vB:.1f} TECU  -> pair change {abs(vB - vA):.2f} TECU")

    # background: matched-hour 12-day |dTEC| by hour of day -> dawn is the quiet part of the diurnal cycle
    ch12 = np.abs(M[12:] - M[:-12])                                        # [pair, epoch]
    per_hour = np.median(ch12, axis=0)
    dawn, worst = per_hour[over_col], per_hour.max()
    print(f"  12-day matched-hour |dTEC| over the month ({ch12.shape[0]} pairs/hour):")
    print(f"    dawn overpass hour  median {dawn:.2f} TECU   worst hour (afternoon) median {worst:.2f}  "
          f"-> dawn {worst / dawn:.1f}x quieter")
    pair, pooled = abs(vB - vA), ch12.ravel()
    print(f"    all hours pooled    median {np.median(pooled):.2f}   max {pooled.max():.1f} TECU")
    print(f"  -> the {pair:.2f} TECU pair change is {100*(1-pair/np.median(pooled)):.0f}% below the all-hours median "
          f"({100*np.mean(pooled<=pair):.0f}th pct of {pooled.size} matched 12-day changes);")
    print(f"     below even the typical dawn value ({dawn:.2f} TECU) -- the dawn overpass puts it in the quiet tail")

    if not paths.gunw_product().exists():
        print("  split-spectrum screen:   GUNW unavailable (reprocessing) -> compute when it lands")
        return
    with h5py.File(paths.gunw_product()) as f:
        grp = f[HH]
        rows, cols = retrieval._aoi_window(grp["xCoordinates"][:], grp["yCoordinates"][:])
        scr = grp["ionospherePhaseScreen"][rows, cols]
        unc = grp["ionospherePhaseScreenUncertainty"][rows, cols]
    if np.count_nonzero(np.nan_to_num(scr)) == 0:
        print("  split-spectrum screen:   ALL ZERO in this product -> compute on the FULL run when it lands")
    else:
        msk = np.isfinite(scr) & (scr != 0) & np.isfinite(unc) & (unc > 0)
        s, u = scr[msk], unc[msk]; chi2 = np.mean(((s - s.mean()) / u) ** 2)
        print(f"  split-spectrum screen:   std {s.std():.2f} rad vs unc {np.median(u):.2f} -> chi2 vs zero {chi2:.2f}  "
              f"({'consistent with zero' if chi2 < 1 else 'structure above noise'})")


def troposphere_stats(xs, ys, dswe, elev):
    """RAiDER hydro+wet differential screen: its dSWE-elevation slope vs the snow ramp."""
    print("\n=== TROPOSPHERE (RAiDER, from the GUNW screens) ===")
    if not paths.gunw_product().exists():
        print("  RAiDER tropo: GUNW unavailable (reprocessing) -> compute when it lands")
        return
    with h5py.File(paths.gunw_product()) as f:
        cube = f[f"{RG}/hydrostaticTroposphericPhaseScreen"][:] + f[f"{RG}/wetTroposphericPhaseScreen"][:]
        H = f[f"{RG}/heightAboveEllipsoid"][:]; CX = f[f"{RG}/xCoordinates"][:]; CY = f[f"{RG}/yCoordinates"][:]
    if np.count_nonzero(np.nan_to_num(cube)) == 0:
        print("  RAiDER tropo screens ALL ZERO (full run still processing) -> compute when it lands")
        return
    se, sp = [], []
    inside_y = np.where((CY >= ys.min()) & (CY <= ys.max()))[0]
    inside_x = np.where((CX >= xs.min()) & (CX <= xs.max()))[0]
    for i in inside_y:
        iy = int(np.argmin(np.abs(ys - CY[i])))
        for j in inside_x:
            e = elev[iy, int(np.argmin(np.abs(xs - CX[j])))]
            if np.isfinite(e):
                se.append(e); sp.append(np.interp(e, H, cube[:, i, j]))
    if len(se) < 10:
        print("  RAiDER tropo: fewer than 10 cube nodes inside the AOI -> slope not estimated")
        return
    se, sp = np.array(se), np.array(sp)
    tslope = np.polyfit(se, (sp - sp.mean()) * MM_PER_RAD, 1)[0] * 100
    g = np.isfinite(dswe) & np.isfinite(elev)
    ramp = np.polyfit(elev[g], dswe[g] * 1000, 1)[0] * 100
    print(f"  RAiDER tropo dSWE slope = {tslope:+.2f} mm/100 m   (snow ramp {ramp:+.1f} mm/100 m)")
    print(f"  -> troposphere is {abs(tslope/ramp)*100:.0f}% of the snow ramp")


def mtbs_severity(xs, ys):
    """MTBS burn-severity class on the grid (0 bkgd, 1 unburned-low, 2 low, 3 moderate, 4 high,
    5 greenness, 6 mask) from the frozen raster paths.MTBS (USFS MTBS CONUS)."""
    return np.asarray(_raster_on_grid(paths.MTBS, xs, ys, Resampling.nearest))


def load_lia(xs, ys):
    """Per-pixel local incidence angle [deg] on the product grid. Returns None if the GUNW
    is absent. The DEM is the gap-free NISAR DEM (_elevation_on_grid) -- the same DEM
    isce3 geocodes against; it spans the full scene, so local_incidence_angle returns no
    nodata holes. See [[lidar-reproject-nodata-gotcha]]."""
    if not paths.gunw_product().exists():
        return None
    from nisar_pytools.utils.local_incidence_angle import local_incidence_angle
    ref = xr.DataArray(np.zeros((len(ys), len(xs))), coords={"y": ys, "x": xs}, dims=("y", "x")).rio.write_crs(paths.EPSG)
    dem_on = ref.copy(data=_elevation_on_grid(xs, ys, paths.AOI))
    with h5py.File(paths.gunw_product()) as f:
        m = f[RG]
        lx, ly = m["losUnitVectorX"][:], m["losUnitVectorY"][:]
        lz = np.sqrt(np.clip(1 - lx ** 2 - ly ** 2, 0, 1))
        h, xq, yq = m["heightAboveEllipsoid"][:], m["xCoordinates"][:], m["yCoordinates"][:]
    return np.asarray(local_incidence_angle(dem_on, lx, ly, lz, h, xq, yq, epsg=paths.EPSG).values)


def scale_decomposition(cutoff_m=1000.0):   # sigma = cutoff/2 = the 500 m the paper quotes
    """Split dSWE and dHS by spatial scale and correlate each part -> the kilometer-scale vs
    sub-kilometer distinction the paper draws. A NaN-aware Gaussian low-pass (sigma = cutoff/2)
    gives the kilometer-scale (coarse) component; the residual is the sub-kilometer part. Reports
    the headline split at `cutoff_m` plus a sweep (coarse r rises, residual r falls with cutoff)."""
    p = retrieval.load_product()
    dswe, dhs = p["dswe"], p["dhs"]
    posting = float(abs(p["xs"][1] - p["xs"][0]))
    valid = np.isfinite(dswe) & np.isfinite(dhs)

    def lowpass(x, sigma):                                   # NaN-aware Gaussian smooth (normalize by validity)
        num = gaussian_filter(np.where(valid, np.nan_to_num(x), 0.0), sigma, mode="constant")
        den = gaussian_filter(valid.astype(float), sigma, mode="constant")
        return num / np.where(den > 0.05, den, np.nan)

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[m], b[m])[0, 1]

    print("\n=== SCALE DECOMPOSITION (operational product, Gaussian low-pass) ===")
    print(f"  whole-scene r (all scales) = {corr(dswe, dhs):.2f}")
    print(f"  {'cutoff':>8}{'r kilometer (>L)':>18}{'r sub-km (<L)':>15}")
    for L in (250, 500, 1000, 2000):
        sigma = (L / posting) / 2.0                          # Gaussian sigma ~ half the cutoff wavelength
        lo_s, lo_h = lowpass(dswe, sigma), lowpass(dhs, sigma)
        print(f"  {L:>5.0f} m{corr(lo_s, lo_h):>18.2f}{corr(dswe - lo_s, dhs - lo_h):>15.2f}")
    sigma = (cutoff_m / posting) / 2.0
    lo_s, lo_h = lowpass(dswe, sigma), lowpass(dhs, sigma)
    print(f"  -> at the {cutoff_m:.0f} m cutoff (Gaussian sigma {cutoff_m/2:.0f} m): "
          f"kilometer-scale r = {corr(lo_s, lo_h):.2f}, "
          f"sub-kilometer residual r = {corr(dswe - lo_s, dhs - lo_h):.2f}")


def deramp_r(a, b, e):
    """r between a and b after regressing elevation `e` out of each (the sub-pixel agreement)."""
    g = np.isfinite(a) & np.isfinite(b) & np.isfinite(e)
    da = a[g] - np.polyval(np.polyfit(e[g], a[g], 1), e[g])
    db = b[g] - np.polyval(np.polyfit(e[g], b[g], 1), e[g])
    return np.corrcoef(da, db)[0, 1]


def local_std(A, w):
    """Windowed standard deviation of A (sub-pixel variability)."""
    v = np.isfinite(A).astype(float); A0 = np.where(v > 0, np.nan_to_num(A), 0.0)
    c = uniform_filter(v, w, mode="constant")
    m = uniform_filter(A0, w, mode="constant") / np.where(c > 0.5, c, np.nan)
    m2 = uniform_filter(A0 * A0, w, mode="constant") / np.where(c > 0.5, c, np.nan)
    return np.sqrt(np.clip(m2 - m ** 2, 0, None))


def drift_depth_stats(dswe, dhs, coh, fcf):
    """NEGATIVE CONTROL: does deep / wind-drifted snow decorrelate the radar? NO -- coherence is
    flat-to-rising with depth (and with depth in the open it CLIMBS), so the deep-snow retrieval
    drop is a forest confound (+ a residual unwrapping/sub-pixel-washout term at high coherence),
    not wind-drift decorrelation."""
    print("\n=== DRIFT / DEEP-SNOW NEGATIVE CONTROL ===")
    sig = local_std(dhs, WINDOW)
    lr = local_correlation(dswe, dhs, WINDOW)
    g = np.isfinite(dhs) & np.isfinite(coh) & np.isfinite(sig)
    print(f"  Spearman(coh, dHS magnitude) = {spearmanr(coh[g], dhs[g]).statistic:+.2f}   "
          f"Spearman(coh, dHS variability) = {spearmanr(coh[g], sig[g]).statistic:+.2f}   (both ~0 => NOT drift)")
    print(f"  local r by depth, open (fcf<0.3) vs forest (fcf>=0.7):")
    print(f"    {'dHS cm':>9}{'open r':>8}{'open coh':>9}{'forest r':>10}{'forest coh':>12}")
    gg = g & np.isfinite(lr)
    for lo, hi in [(.0, .3), (.3, .5), (.5, .7), (.7, 3)]:
        b = gg & (dhs >= lo) & (dhs < hi); o, fr = b & (fcf < 0.3), b & (fcf >= 0.7)
        print(f"    {lo*100:3.0f}-{hi*100:<5.0f}{np.nanmedian(lr[o]):>8.2f}{np.nanmedian(coh[o]):>9.2f}"
              f"{np.nanmedian(lr[fr]):>10.2f}{np.nanmedian(coh[fr]):>12.2f}")
    print("  -> coherence does NOT fall with depth (rises in the open); forest >> open at every depth;")
    print("     residual open-depth r drop occurs at HIGH coherence => unwrapping/washout, not drift.")


def wind_stats():
    """Wind context for the drift control: the valley study site (Freeman) stayed calm all period;
    only the fully-exposed high ridge (Pilot Peak) saw gusts to ~38 mph -- modest for an exposed
    ridgeline and confined to it, so AOI-scale wind redistribution was minimal. The headline the
    paper cites ("winds low, < 3.3 m/s") is the Freeman sustained-wind max over the InSAR pair
    interval itself (Feb 7 -> Feb 19), reported in m/s up front."""
    import pandas as pd
    print("\n=== WIND (drift context) ===")
    t0, t1 = pd.Timestamp(paths.D0), pd.Timestamp(paths.D1)
    fr = pd.read_csv(paths.MET_SOIL, usecols=["TIMESTAMP", "WS_ms_S_WVT", "WS_ms_Max"])
    ft = pd.to_datetime(fr["TIMESTAMP"], errors="coerce")
    ws_ms = pd.to_numeric(fr["WS_ms_S_WVT"], errors="coerce")         # native m/s (sustained, vector-avg)
    gust_ms = pd.to_numeric(fr["WS_ms_Max"], errors="coerce")         # native m/s (15-min peak gust)
    ws, gust = ws_ms * 2.237, gust_ms * 2.237                         # m/s -> mph for the context lines below

    # PAIR-WINDOW headline (m/s): the InSAR interval is Feb 7 -> Feb 19, both at the 05:46 MST dawn pass
    p0 = pd.to_datetime(paths.NISAR_A) + pd.Timedelta(hours=5, minutes=46)
    p1 = pd.to_datetime(paths.NISAR_B) + pd.Timedelta(hours=5, minutes=46)
    mp0 = (ft >= p0) & (ft <= p1)
    print(f"  PAIR WINDOW (Feb7 05:46 -> Feb19 05:46, the {(p1-p0).days}-day InSAR interval): Freeman sustained "
          f"wind mean {ws_ms[mp0].mean():.1f} m/s, max {ws_ms[mp0].max():.1f} m/s (peak gust {gust_ms[mp0].max():.1f} m/s)")
    print(f"     -> the '< 3.3 m/s' the paper cites = this sustained-wind max; well below the ~5 m/s fresh-snow")
    print(f"        saltation threshold for the whole interval, so no wind redistribution between acquisitions")

    m = (ft >= t0) & (ft <= t1)
    print(f"  Freeman (valley study site): mean {ws[m].mean():.1f} mph, max gust {gust[m].max():.1f} mph "
          f"-> below the ~11-16 mph (5-7 m/s) fresh-snow transport threshold => little drift at the AOI")
    pl = pd.read_csv(paths.MET_PILOT)
    pt = pd.to_datetime(pl["datetime_mdt"], errors="coerce")
    pw, pg = pd.to_numeric(pl["wind_mph"], errors="coerce"), pd.to_numeric(pl["gust_mph"], errors="coerce")
    mp = (pt >= t0) & (pt <= t1)
    print(f"  Pilot Peak (fully-exposed ridge ~2480 m): mean {pw[mp].mean():.1f} mph, max gust {pg[mp].max():.0f} mph "
          f"-- modest for a completely exposed ridgeline, and confined to it")
    # full-winter record (the whole station deployment, not just the Feb1-Mar1 campaign window)
    print("  FULL WINTER (whole station record):")
    fgi = gust.idxmax()
    print(f"    Freeman: {ft.min():%Y-%m-%d} -> {ft.max():%Y-%m-%d}  mean {ws.mean():.1f} mph, "
          f"max gust {gust.max():.1f} mph (on {ft[fgi]:%Y-%m-%d %H:%M})")
    pgi = pg.idxmax()
    print(f"    Pilot:   {pt.min():%Y-%m-%d} -> {pt.max():%Y-%m-%d}  mean {pw.mean():.1f} mph, "
          f"max gust {pg.max():.0f} mph (on {pt[pgi]:%Y-%m-%d %H:%M})")
    print("  -> wind is not a confound: the AOI itself was calm, and the windy ridge shows no")
    print("     drift-decorrelation signature (coherence flat with depth/variability, above).")


def lidar_window_swe():
    """Quantify the 3 extra days the 15-day lidar pair (Feb7->Feb22) carries over the 12-day radar
    pair (Feb7->Feb19), the limitation line in the paper. Co-located SNOTEL 637 06:00 SWE (matches
    the dawn overpass): the extra window is small in SWE next to the 12-day accumulation."""
    import pandas as pd
    print("\n=== LIDAR vs RADAR WINDOW (3 extra lidar days) ===")
    swe = pd.read_csv(paths.SNOTEL_SNAP).set_index("date")["swe_in_06"] * 25.4   # in -> mm
    a, b, c = swe["2026-02-07"], swe["2026-02-19"], swe["2026-02-22"]            # radar A, radar B, lidar B
    radar, extra = b - a, c - b
    print(f"  SNOTEL 637 SWE (06:00):  Feb07 {a:.0f}  Feb19 {b:.0f}  Feb22 {c:.0f} mm")
    print(f"  radar pair (Feb07->19, 12 d): +{radar:.0f} mm    extra lidar days (Feb19->22, 3 d): +{extra:.0f} mm")
    print(f"  -> the 3 extra days add only {extra/radar*100:.0f}% of the 12-day accumulation (small); the 12-day "
          f"+{radar:.0f} mm also matches the scene dSWE signal")


def pair_table():
    """One row per track-077 pair of the pair timeline figure: the NISAR SWE change, the lidar
    snow depth change of the flights matched to it in pairs.py, the SNOTEL 637 SWE change over
    each of the two date spans, the days from each NISAR acquisition to its lidar flight, the
    SNOTEL 637 air temperature at each NISAR acquisition and its positive degree days, the variance and
    the 25th and 75th percentiles of the lidar map, and the pixel correlation (Pearson and Spearman) and fit slope of the two maps.

    NISAR is the field the timeline draws (paper_figures._swe_change: atmosphere screens
    removed, referenced to SNOTEL 637), lidar is the gated 80 m dHS of the pair's product. Map
    values are medians over the pixels where both exist. The Dec 3 to Dec 13 flights are left
    out, as in the figure, so that pair has no lidar columns."""
    import pandas as pd
    from nival.build_multipair import PRODUCTS
    from nival.download_paper_data import TRACK, fetch_snotel_air_temp, local_gunws
    from nival.pairs import PAIRS, label
    from nival.paper_figures import _snotel_swe, _swe_change
    snotel = _snotel_swe()                                    # metres, keyed by date
    station = lambda a, b: (snotel[pd.Timestamp(b)] - snotel[pd.Timestamp(a)]) * 1000   # mm
    # SNOTEL 637 daily snow depth [m], converted from inches at ingest, for the station's own accumulation density
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"]).set_index("date")
    station_depth = season["snwd_in"] * 0.0254
    if station_depth.max() > 10:
        raise ValueError("SNOTEL snow depth is not plausible in metres; check units")
    depth_change = lambda a, b: station_depth[pd.Timestamp(b)] - station_depth[pd.Timestamp(a)]   # m
    # SWE change over depth change at the station across the NISAR pair [kg/m3]; no value unless depth rose by 5 cm or more
    station_density = lambda a, b: f"{station(a, b) / depth_change(a, b):.0f}" if depth_change(a, b) >= 0.05 else "-"
    nice = lambda a, b: f"{pd.Timestamp(a):%b} {pd.Timestamp(a).day} to {pd.Timestamp(b):%b} {pd.Timestamp(b).day}"
    flights = {p[:3]: p[3:5] for p in PAIRS if p[3] != "20251203"}
    # hourly air temperature [degC, UTC], read at the 12:46:36 UTC overpass by linear interpolation between the hours
    if not paths.SNOTEL_AIR_TEMP.exists():
        fetch_snotel_air_temp()
    air = pd.read_csv(paths.SNOTEL_AIR_TEMP, index_col=0, parse_dates=True)["air_temp_c"].dropna()
    hours = air.index.values.astype("datetime64[s]").astype(float)
    at_pass = lambda d: np.interp((pd.Timestamp(d) + pd.Timedelta(hours=OVERPASS_UTC_H)).timestamp(), hours, air.values)
    temps = lambda a, b: f"{at_pass(a):+.1f}, {at_pass(b):+.1f}"
    # positive degree days [degC d] between two overpasses: hourly air temperature above 0 degC, summed and divided by 24
    passing = lambda d: pd.Timestamp(d) + pd.Timedelta(hours=OVERPASS_UTC_H)
    degree_days = lambda start, end: air[(air.index > start) & (air.index <= end)].clip(lower=0).sum() / 24
    pdd = lambda a, b: f"{degree_days(passing(a), passing(b)):.1f}"
    print("\n=== PAIR TABLE (track 077, every pair of the pair timeline figure) ===")
    # days from each NISAR acquisition to its lidar flight, first and second date (positive: lidar flew later)
    offset = lambda nisar, lidar: (pd.Timestamp(lidar) - pd.Timestamp(nisar)).days
    print(f"{'NISAR pair':>17}{'lidar flights':>18}{'lidar - NISAR d':>17}{'air temp C':>14}{'PDD degC d':>16}{'NISAR dSWE mm':>15}{'lidar dHS m':>13}{'% dHS>0':>9}"
          f"{'dHS var m2':>12}{'dHS 25-75% m':>16}{'SNOTEL mm (NISAR)':>19}{'SNOTEL dHS m':>14}{'SNOTEL kg/m3':>14}{'SNOTEL mm (lidar)':>19}{'r':>7}{'Spearman':>10}{'slope kg/m3':>13}{'n':>7}")
    dates, matched, flown = set(), 0, set()                   # NISAR acquisitions, pairs with lidar, lidar flight dates
    for track, ref, sec, path, _ in local_gunws():
        if track != TRACK or sec > paths.FIG3_LAST:
            continue
        dates |= {ref, sec}
        dswe, xs, ys, _, _ = _swe_change(path, ref, sec, snotel)   # metres
        row = f"{nice(ref, sec):>17}"
        product = PRODUCTS / f"{label((track, ref, sec))}.nc"
        if (track, ref, sec) not in flights or not product.exists():
            print(row + f"{'-':>18}{'-':>17}{temps(ref, sec):>14}{pdd(ref, sec):>16}{np.nanmedian(dswe) * 1000:>+15.0f}{'-':>13}{'-':>9}{'-':>12}{'-':>16}{station(ref, sec):>+19.0f}{depth_change(ref, sec):>+14.2f}{station_density(ref, sec):>14}{'-':>19}{'-':>7}{'-':>10}{'-':>13}{'-':>7}")
            continue
        first, second = flights[(track, ref, sec)]
        matched += 1
        flown |= {first, second}
        ds = xr.open_dataset(product)
        assert np.allclose(ds.x.values, xs) and np.allclose(ds.y.values, ys), "NISAR and lidar maps share one grid"
        dhs = ds["dhs"].values                                # metres, later flight minus earlier
        both = np.isfinite(dswe) & np.isfinite(dhs)
        r = np.corrcoef(dswe[both], dhs[both])[0, 1]
        rank_r = spearmanr(dswe[both], dhs[both]).statistic
        slope = np.polyfit(dhs[both], dswe[both] * 1000, 1)[0]   # mm SWE per m depth = kg/m3, the plain fit Fig. 2 draws
        days = f"{offset(ref, first):+d}, {offset(sec, second):+d}"
        q25, q75 = np.percentile(dhs[both], [25, 75])
        print(row + f"{nice(first, second):>18}{days:>17}{temps(ref, sec):>14}{pdd(ref, sec):>16}{np.median(dswe[both]) * 1000:>+15.0f}{np.median(dhs[both]):>+13.2f}{100 * (dhs[both] > 0).mean():>9.1f}"
              f"{dhs[both].var():>12.4f}{f'{q25:+.2f}, {q75:+.2f}':>16}{station(ref, sec):>+19.0f}{depth_change(ref, sec):>+14.2f}{station_density(ref, sec):>14}{station(first, second):>+19.0f}{r:>+7.2f}{rank_r:>+10.2f}{slope:>+13.0f}{int(both.sum()):>7d}")
    # the two kinds of NISAR pair: every pair in the cumulative sum, and those with a matched lidar pair
    day = lambda d: f"{pd.Timestamp(d):%b} {pd.Timestamp(d).day}"
    cycle = pd.date_range(min(dates), max(dates), freq="12D").strftime("%Y%m%d")
    missed = [d for d in cycle if d not in dates]
    print(f"  NISAR pairs in the cumulative sum: {len(dates) - 1}, from {len(dates)} acquisitions {day(min(dates))} to {day(max(dates))}")
    print(f"  NISAR pairs with a matched lidar pair: {matched}, from {len(flown)} lidar flights ({', '.join(day(d) for d in sorted(flown))})")
    print(f"  12-day cycle dates in that span with no acquisition: {len(missed)} of {len(cycle)} ({', '.join(day(d) for d in missed)})")


def swe_mm_at(field, i, j, half):
    """Median of a field [m] over the (2 half + 1) square of pixels centred on (i, j), in mm."""
    return float(np.nanmedian(field[i - half:i + half + 1, j - half:j + half + 1]) * 1000)


def cumulative_stats():
    """The cumulative comparison: the seven track-077 SWE changes from Nov 3 to Mar 15 summed pixel by pixel
    (paper_figures._swe_change, as the season totals figure sums them), against the lidar snow depth of the
    Mar 15 flight averaged onto the 80 m grid. Prints the correlation and the fit over every pixel where both
    exist, then the same within quartiles of local incidence angle, canopy fraction, elevation and the mean coherence of
    the seven pairs. No masks."""
    import pandas as pd
    from nival.download_paper_data import TRACK, lidar_file, local_gunws
    from nival.paper_figures import _snotel_swe, _swe_change
    snotel = _snotel_swe()
    print("\n=== CUMULATIVE (track 077, Nov 3 to Mar 15) vs lidar snow depth of Mar 15 ===")
    cumulative, coherence, pairs = None, None, 0
    for track, ref, sec, path, _ in local_gunws():
        if track != TRACK or sec > paths.FIG3_LAST:
            continue
        swe_change, xs, ys, station_xy, _ = _swe_change(path, ref, sec, snotel)   # metres
        with h5py.File(path) as f:
            g = f[HH]
            rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
            coh = g["coherenceMagnitude"][rows, cols].astype(float)
        cumulative = swe_change if cumulative is None else cumulative + swe_change
        coherence = coh if coherence is None else coherence + coh
        pairs += 1
    coherence = coherence / pairs                             # mean coherence of the pairs in the sum
    grid = xr.DataArray(cumulative, coords={"y": ys, "x": xs}, dims=("y", "x")).rio.write_crs(paths.EPSG)
    depth = rioxarray.open_rasterio(lidar_file(paths.FIG3_LAST), masked=True).squeeze().rio.write_nodata(np.nan)
    depth = depth.rio.reproject_match(grid, resampling=Resampling.average).values   # metres, on the 80 m grid
    if np.nanmax(depth) > 10:
        raise ValueError("lidar snow depth is not plausible in metres; check units")
    lia = load_lia(xs, ys)
    # canopy fraction in the same 400 m window the other statistics use
    fcf = uniform_filter((_raster_on_grid(paths.WORLDCOVER, xs, ys, Resampling.nearest) == 10).astype(float), size=WINDOW)
    both = np.isfinite(cumulative) & np.isfinite(depth)
    swe_mm = cumulative * 1000

    def line(label, sel):
        r = np.corrcoef(swe_mm[sel], depth[sel])[0, 1]
        slope, intercept = np.polyfit(depth[sel], swe_mm[sel], 1)   # mm SWE per m depth = kg/m3
        print(f"{label:>34}{r:>+7.2f}{slope:>+13.0f}{intercept:>+14.0f}{np.median(swe_mm[sel]):>12.0f}{np.median(depth[sel]):>12.2f}{int(sel.sum()):>7d}")
        return r, slope, int(sel.sum())

    print(f"  pairs summed: {pairs}; SNOTEL 637 SWE on Mar 15: {snotel[pd.Timestamp(paths.FIG3_LAST)] * 1000:.0f} mm")
    # the cumulative map at the station: its own pixel and the 3 by 3 pixels the referencing uses
    sx, sy = station_xy
    i, j = int(np.argmin(np.abs(ys - sy))), int(np.argmin(np.abs(xs - sx)))
    print(f"  cumulative NISAR SWE at the SNOTEL: pixel {swe_mm_at(cumulative, i, j, 0):.0f} mm, 3 by 3 median {swe_mm_at(cumulative, i, j, 1):.0f} mm, "
          f"3 by 3 range {np.nanmin(cumulative[i - 1:i + 2, j - 1:j + 2]) * 1000:.0f} to {np.nanmax(cumulative[i - 1:i + 2, j - 1:j + 2]) * 1000:.0f} mm; "
          f"lidar depth at that pixel {depth[i, j]:.2f} m")
    print(f"{'subset':>34}{'r':>7}{'slope kg/m3':>13}{'intercept mm':>14}{'med SWE mm':>12}{'med depth m':>12}{'n':>7}")
    line("all pixels", both)
    elevation = retrieval._elevation_on_grid(xs, ys, paths.AOI)   # metres, the scene DEM on the grid
    # the supplement table of the splits: one row per quartile, in the order the text reports them
    table = ["\\begin{tabular}{llrrr}", "\\hline",
             "Split by & Quartile range & $r$ & Slope (kg\\,m$^{-3}$) & Pixels \\\\", "\\hline"]
    names = {"canopy fraction": ("Forest canopy fraction", "{:.2f} to {:.2f}"), "incidence angle": ("Local incidence angle ($^{{\\circ}}$)", "{:.0f} to {:.0f}"),
             "mean coherence": ("Mean coherence", "{:.2f} to {:.2f}"), "elevation": ("Elevation (m)", "{:.0f} to {:.0f}")}
    rows = {}
    for name, field, unit in (("incidence angle", lia, "deg"), ("canopy fraction", fcf, ""), ("mean coherence", coherence, ""),
                              ("elevation", elevation, "m")):
        edges = np.percentile(field[both], [0, 25, 50, 75, 100])
        for k in range(4):
            upper = field <= edges[k + 1] if k == 3 else field < edges[k + 1]
            r, slope, n = line(f"{name} {edges[k]:.2f} to {edges[k + 1]:.2f} {unit}".strip(), both & (field >= edges[k]) & upper)
            rows.setdefault(name, []).append((names[name][1].format(edges[k], edges[k + 1]), r, slope, n))
    for name in ("canopy fraction", "incidence angle", "mean coherence", "elevation"):
        for k, (span, r, slope, n) in enumerate(rows[name]):
            table.append(f"{names[name][0] if k == 0 else ''} & {span} & {r:.2f} & {slope:.0f} & {n} \\\\")
        table.append("\\hline")
    table.append("\\end{tabular}")
    out = paths.OUTPUTS / "table_s2.tex"
    out.write_text("\n".join(table) + "\n")
    print(f"  wrote {out.name}")


def reference_sensitivity():
    """How the phase reference changes the correlations the paper reports. Each pair's SWE change is
    (phase - c) / kappa with one constant c in phase. The paper takes c from SNOTEL 637. Here c is also taken
    from a least-squares fit to the lidar depth change over the lidar pixels (phase / kappa = c / kappa +
    density * dHS, which needs no density), and set to zero for no reference. Prints the pairwise r for each,
    then the cumulative r against the Mar 15 lidar depth with the lidar-fitted pairs summed (pairs without
    lidar keep the SNOTEL reference). No masks."""
    import pandas as pd
    from nival.build_multipair import PRODUCTS
    from nival.download_paper_data import TRACK, lidar_file, local_gunws
    from nival.pairs import PAIRS, label
    from nival.paper_figures import _corrected_phase, _snotel_swe, _swe_change
    snotel = _snotel_swe()
    matched = {p[:3] for p in PAIRS if p[3] != "20251203"}    # the five pairs of Table 1
    nice = lambda a, b: f"{pd.Timestamp(a):%b} {pd.Timestamp(a).day} to {pd.Timestamp(b):%b} {pd.Timestamp(b).day}"
    print("\n=== PHASE REFERENCE: SNOTEL vs lidar fit vs none (track 077) ===")
    print(f"{'pair':>18}{'r SNOTEL':>10}{'r lidar fit':>13}{'r none':>8}")
    totals = {"SNOTEL": None, "lidar fit": None, "none": None}
    for track, ref, sec, path, _ in local_gunws():
        if track != TRACK or sec > paths.FIG3_LAST:
            continue
        swe, xs, ys, _, log = _swe_change(path, ref, sec, snotel)
        phase, _, _ = _corrected_phase(path)
        # per-pixel sensitivity [rad per m of SWE] from the local incidence angle, as _swe_change computes it
        kappa = 2 * np.pi / LAMBDA * (1.59 + np.radians(load_lia(xs, ys)) ** 2.5)
        fitted = swe                                          # pairs without lidar keep the SNOTEL reference
        product = PRODUCTS / f"{label((track, ref, sec))}.nc"
        if (track, ref, sec) in matched and product.exists():
            dhs = xr.open_dataset(product)["dhs"].values
            ok = np.isfinite(dhs) & np.isfinite(swe) & np.isfinite(kappa)
            design = np.column_stack([1 / kappa[ok], dhs[ok]])
            c_fit = np.linalg.lstsq(design, (phase / kappa)[ok], rcond=None)[0][0]
            fitted = (phase - c_fit) / kappa
            r = lambda field: np.corrcoef(field[ok], dhs[ok])[0, 1]
            print(f"{nice(ref, sec):>18}{r(swe):>+10.2f}{r(fitted):>+13.2f}{r(phase / kappa):>+8.2f}")
        for name, field in (("SNOTEL", swe), ("lidar fit", fitted), ("none", phase / kappa)):
            totals[name] = field if totals[name] is None else totals[name] + field
    grid = xr.DataArray(totals["SNOTEL"], coords={"y": ys, "x": xs}, dims=("y", "x")).rio.write_crs(paths.EPSG)
    depth = rioxarray.open_rasterio(lidar_file(paths.FIG3_LAST), masked=True).squeeze().rio.write_nodata(np.nan)
    depth = depth.rio.reproject_match(grid, resampling=Resampling.average).values
    print(f"{'cumulative':>18}{'r':>8}{'slope kg/m3':>13}{'intercept mm':>14}{'median mm':>11}")
    for name, field in totals.items():
        ok = np.isfinite(depth) & np.isfinite(field)
        slope, intercept = np.polyfit(depth[ok], field[ok] * 1000, 1)
        print(f"{name:>18}{np.corrcoef(field[ok], depth[ok])[0, 1]:>+8.3f}{slope:>13.0f}{intercept:>+14.0f}{np.median(field[ok]) * 1000:>11.0f}")


def lidar_vs_snotel_depth():
    """Lidar snow depth at SNOTEL 637 (paper_figures.lidar_at_snotel, the Fig. 1 diamonds) against the
    station's own snow depth on each flight date: r and RMSE over all twelve NIVAL flights."""
    import pandas as pd
    from nival.paper_figures import lidar_at_snotel
    print("\n=== LIDAR vs SNOTEL snow depth at the station, by flight ===")
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"]).set_index("date")
    station = season["snwd_in"] * 0.0254                      # metres
    used = sorted(p.name.split("_")[-3] for p in paths.LIDAR_DIR.glob("*_SD_v01.tif"))     # every flight on disk
    lidar = np.array([lidar_at_snotel(d) for d in used])
    point = np.array([station[pd.Timestamp(d)] for d in used])
    for d, a, b in zip(used, lidar, point):
        print(f"  {d}: lidar {a:.2f} m | station {b:.2f} m")
    print(f"  n {len(used)}, r {np.corrcoef(lidar, point)[0, 1]:.2f}, RMSE {np.sqrt(((lidar - point) ** 2).mean()):.2f} m, "
          f"mean difference {(lidar - point).mean():+.2f} m")


def refocus_coherence():
    """Mean 80 m coherence of the track-077 pairs we rebuilt from the refocused Jan 14 acquisition:
    the production GUNW against our GUNW from the refocused RSLC. Means are over every finite
    pixel of the AOI clip, and over the pixels where the pair's lidar depth change exists."""
    from nival.build_multipair import PRODUCTS
    from nival.download_paper_data import TRACK, local_gunws, pair_of
    from nival.pairs import label
    print("\n=== REFOCUSED PAIRS: mean coherence, production vs refocused (track 077) ===")
    production = {pair_of(f): f for f in paths.GUNW_OPERATIONAL.parent.glob("NISAR_L2_PR_GUNW_*.h5")}

    def coherence(path):
        # the 80 m coherence layer over the AOI and the pixel-centre coordinates of the clip
        with h5py.File(path) as f:
            g = f[HH]
            rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
            return g["coherenceMagnitude"][rows, cols].astype(float), g["xCoordinates"][cols], g["yCoordinates"][rows]

    print(f"{'pair':>17}{'product':>12}{'mean coh AOI':>14}{'mean coh lidar px':>19}{'n lidar px':>12}")
    for track, ref, sec, path, refocused in local_gunws():
        if track != TRACK or not refocused or sec > paths.FIG3_LAST:
            continue
        ds = xr.open_dataset(PRODUCTS / f"{label((track, ref, sec))}.nc")
        lidar = np.isfinite(ds["dhs"].values)
        for name, source in (("production", production[(track, ref, sec)]), ("refocused", path)):
            coh, xs, ys = coherence(source)
            assert np.allclose(ds.x.values, xs) and np.allclose(ds.y.values, ys), "coherence and lidar share one grid"
            inside = lidar & np.isfinite(coh)
            print(f"{f'{ref} {sec}':>17}{name:>12}{np.nanmean(coh):>14.2f}{coh[inside].mean():>19.2f}{int(inside.sum()):>12d}")


def lidar_gate_stats():
    """How much of the native-resolution lidar the |dHS| outlier gate removes.

    build_product applies the gate at the native 0.5 m posting, before averaging onto the
    product grid, so the fraction has to be counted there rather than on the saved product.
    Reported over the pixels where both flights returned a depth."""
    print("\n=== LIDAR OUTLIER GATE ===")
    later = rioxarray.open_rasterio(paths.LIDAR_A22, masked=True).squeeze()
    earlier = rioxarray.open_rasterio(paths.LIDAR_A07, masked=True).squeeze()
    change = (later - earlier).values
    both = np.isfinite(change)
    gated = np.abs(change[both]) > DHS_GATE_M
    print(f"  |dHS| > {DHS_GATE_M:.1f} m gate removes {100 * gated.mean():.3f}% "
          f"({gated.sum():,} of {both.sum():,}) of the lidar depth-change pixels "
          f"at the native 0.5 m posting")
    print(f"  gated values run to {np.abs(change[both]).max():.0f} m, far outside any real "
          f"15-day snow change, so the gate is removing voids and canopy returns")
    print("  (applied before averaging onto the 20 and 80 m product grids)")


def wrapping_stats(dswe, dhs, lia):
    """Fringes the lidar dHS signal spans + snaphu fringes added -> little phase wrapping."""
    print("\n=== WRAPPING ===")
    if lia is None:
        print("  GUNW unavailable (reprocessing) -> needs LIA + unwrapped phase; re-run when it lands")
        return
    rms = 2 * np.pi / paths.LAMBDA * (1.59 + np.radians(lia) ** 2.5)   # rad per m SWE (per-pixel LIA)
    exp = rms * (dhs * SNOW_DENSITY / WATER_DENSITY)                   # lidar dHS -> expected InSAR phase
    gg = np.isfinite(exp)
    span = np.nanpercentile(exp[gg], 98) - np.nanpercentile(exp[gg], 2)
    cm_per_fringe = (2 * np.pi / np.nanmedian(rms[gg])) * WATER_DENSITY / SNOW_DENSITY * 100   # dHS for one 2pi
    print(f"  lidar-expected phase spans {span:.1f} rad = {span/(2*np.pi):.1f} fringes (p2-p98);  "
          f"one fringe = {cm_per_fringe:.0f} cm of snow  (so the scene wraps {span/(2*np.pi):.1f}x)")
    # per-pixel wrapping: |lidar-expected phase| in fringes -- does the SWE signal itself exceed
    # one 2pi fringe? (same quantity as WHY IT WORKED item 2; the paper's "within one fringe" line)
    fringes = np.abs(exp[gg]) / (2 * np.pi)
    print(f"  lidar |expected phase|: median {np.nanmedian(fringes):.2f} fringe; "
          f"{100 * np.mean(fringes < 1):.0f}% of pixels within one fringe")
    # snaphu fringe count. In the operational product the unwrapped phase is geocoded at
    # 80 m while the wrapped interferogram ships at 20 m, so the two are not on one grid:
    # complex-average the wrapped field 4x4 to the unwrapped posting before differencing.
    # That extra look is what the unwrap stage itself applied, so the comparison is like for
    # like, but it is a reconstruction rather than a direct readout of snaphu's own step.
    with h5py.File(paths.gunw_product()) as f:
        grp = f[HH]
        rows, cols = retrieval._aoi_window(grp["xCoordinates"][:], grp["yCoordinates"][:])
        unw = grp["unwrappedPhase"][rows, cols]
        ux, uy = grp["xCoordinates"][cols], grp["yCoordinates"][rows]
        wgrp = f["science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH"]
        wrows, wcols = retrieval._aoi_window(wgrp["xCoordinates"][:], wgrp["yCoordinates"][:])
        cpx = wgrp["wrappedInterferogram"][wrows, wcols]
        wx, wy = wgrp["xCoordinates"][wcols], wgrp["yCoordinates"][wrows]
    if np.count_nonzero(unw) == 0:
        print("  snaphu fringe %: unwrappedPhase ALL ZERO -> compute when the product lands")
        return
    looked = (xr.DataArray(cpx, dims=("y", "x"), coords={"y": wy, "x": wx})
              .coarsen(y=4, x=4, boundary="trim").mean()
              .interp(y=uy, x=ux, method="nearest").values)
    wrp = np.angle(looked)
    g = np.isfinite(unw) & np.isfinite(wrp) & (unw != 0)
    k = np.round((unw - wrp) / (2 * np.pi)); k = k - np.round(np.nanmedian(k[g]))
    two = int(np.sum(np.abs(k[g]) >= 2))
    print(f"  snaphu added a fringe (|k|>=1) to {100 * np.mean(np.abs(k[g]) >= 1):.0f}% of pixels "
          f"(|k|>=2 on {two} px = {100 * two / g.sum():.2f}%, max |k| = {int(np.nanmax(np.abs(k[g])))})")


def burn_fcf_stats(dswe, dhs, coh, fcf, sev):
    """Do BURN and FOREST-CANOPY-FRACTION each matter? scene r in the 2x2."""
    burned = np.isin(sev, [1, 2, 3, 4])
    print("\n=== BURN x FOREST-CANOPY-FRACTION ===")
    print(f"{'group':>26}{'med coh':>9}{'r':>7}{'n':>9}")
    for lab, m in [("unburned forest (fcf>=.5)", (~burned) & (fcf >= 0.5)),
                   ("unburned open   (fcf<.5)", (~burned) & (fcf < 0.5)),
                   ("burned   forest (fcf>=.5)", burned & (fcf >= 0.5)),
                   ("burned   open   (fcf<.5)", burned & (fcf < 0.5))]:
        r, _, _, _, n = metrics(dswe, dhs, m)
        mc = np.nanmedian(coh[m & np.isfinite(coh)]) if m.sum() else np.nan
        print(f"{lab:>26}{mc:>9.2f}{r:>7.2f}{n:>9d}")


def driver_local_r(dswe, dhs, coh, lia, fcf, sev, window):
    """Local (windowed) r vs coherence, LIA, canopy fraction -> binned medians + Spearman,
    then the per-stratum local-r distribution (median / 10th pct / % of windows r<0) for the
    strata the paper sentence names: open-or-burned (low canopy), dense forest, steep LIA.
    `window` is the local-r window in pixels (held at the ~420 m physical scale for the posting)."""
    lr = local_correlation(dswe, dhs, window)
    print("\n=== LOCAL r vs DRIVERS (coherence, LIA, canopy fraction) ===")
    drivers = [("coherence", coh), ("canopy frac", fcf)]
    if lia is not None:
        drivers.insert(1, ("LIA (deg)", lia))
    for name, D in drivers:
        m = np.isfinite(lr) & np.isfinite(D)
        rho = spearmanr(lr[m], D[m]).statistic
        edges = np.nanpercentile(D[m], np.linspace(0, 100, 6))
        meds = [np.nanmedian(lr[m & (D >= lo) & (D < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
        cens = [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
        print(f"  local r vs {name:12} Spearman {rho:+.2f}   binned: " +
              " ".join(f"{c:.2f}:{v:+.2f}" for c, v in zip(cens, meds)))

    # per-stratum local-r distribution -> the "highest in open/burned, toward zero & locally
    # negative under forest / steep LIA" sentence. open-or-burned = low canopy (mostly burn-thinned);
    # the negative tail (% windows r<0) is what "locally negative" cites.
    forest = (sev == 0) & (fcf >= 0.5)
    open_or_burned = ((sev == 0) & (fcf < 0.5)) | np.isin(sev, [1, 2, 3, 4])
    strata = [("open or burned (low canopy)", open_or_burned), ("dense forest (canopy>=.5)", forest)]
    if lia is not None:
        strata.append(("shallow LIA (<50 deg)", lia <= 50))   # the abstract's "below 50 deg" claim
        strata.append(("steep LIA (>50 deg)", lia > 50))       # 50 deg ~ scene LIA upper quartile (48 deg)
    print("  local r by stratum (median / 10th pct / % of windows r<0):")
    for name, M in strata:
        v = lr[M & np.isfinite(lr)]
        print(f"    {name:30} {np.nanmedian(v):+.2f} / {np.nanpercentile(v, 10):+.2f} / "
              f"{100 * np.mean(v < 0):3.0f}%   n {v.size}")


def terrain_stats(dswe, dhs, coh, elev, posting):
    """Aspect / elevation / slope stratification of local r, dHS, dSWE, coherence.
    Aspect is the dominant geometric control: NISAR ASC-077 is left-looking with the
    satellite to the EAST, so E/NE-facing slopes face the sensor (foreshortened, well
    imaged) -> high coherence + retrieval, while W/SW-facing slopes face away (layover /
    shadow) -> poor. (The deepest dHS sits on the W/N lee slopes, which also face away,
    so 'deep snow retrieves worse' partly aliases with aspect.) Elevation: snow climbs
    steeply, but local r falls at the highest elevations while coherence stays flat ->
    that loss is sub-pixel/wrapping, not decorrelation."""
    lr = local_correlation(dswe, dhs, WINDOW)
    gy, gx = np.gradient(elev, posting)
    aspect = np.degrees(np.arctan2(-gx, gy)) % 360    # azimuth the slope FACES: 0=N 90=E 180=S 270=W
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    print("\n=== TERRAIN: aspect / elevation / slope ===")

    def table(title, bins):
        print(f"  by {title}: {'bin':>12}{'local_r':>9}{'dHS m':>8}{'dSWE m':>9}{'coh':>7}{'slope':>7}{'n':>8}")
        for lab, m in bins:
            g = m & np.isfinite(dhs) & np.isfinite(dswe)
            if g.sum() < 50:
                continue
            print(f"  {'':>12}{lab:>12}{np.nanmedian(lr[g]):>9.2f}{np.nanmedian(dhs[g]):>8.2f}"
                  f"{np.nanmedian(dswe[g]):>9.3f}{np.nanmedian(coh[g]):>7.2f}{np.nanmedian(slope[g]):>7.0f}{int(g.sum()):>8d}")

    table("ASPECT faced", [(lab, ((aspect - lo) % 360) < (hi - lo)) for lab, lo, hi in
          [("N", -22.5, 22.5), ("NE", 22.5, 67.5), ("E", 67.5, 112.5), ("SE", 112.5, 157.5),
           ("S", 157.5, 202.5), ("SW", 202.5, 247.5), ("W", 247.5, 292.5), ("NW", 292.5, 337.5)]])
    eq = np.nanpercentile(elev[np.isfinite(elev)], [0, 20, 40, 60, 80, 100])
    table("ELEVATION m", [(f"{lo:.0f}-{hi:.0f}", (elev >= lo) & (elev < hi)) for lo, hi in zip(eq[:-1], eq[1:])])
    print("  Spearman vs elevation / slope:")
    for nm, F in [("local_r", lr), ("dHS", dhs), ("dSWE", dswe), ("coh", coh)]:
        g = np.isfinite(F) & np.isfinite(elev) & np.isfinite(slope)
        print(f"    {nm:>8}: vs elev {spearmanr(F[g], elev[g]).statistic:+.2f}   "
              f"vs slope {spearmanr(F[g], slope[g]).statistic:+.2f}")


def met_snow():
    """6. Freeman SNOW temperatures at both overpasses (dry if all < 0) + depth change."""
    import datetime as dt
    import matplotlib.dates as mdates
    from nival.paper_figures import snowpack_thermograph, freeman_06
    tnum, _, field, hs, _ = snowpack_thermograph()
    print("\n=== FREEMAN SNOWPACK TEMPERATURE at the overpasses (dry if all < 0) ===")
    for lbl, day in [("A Feb 7", "2026-02-07"), ("B Feb 19", "2026-02-19")]:
        k = int(np.argmin(np.abs(tnum - mdates.date2num(dt.datetime.fromisoformat(day + "T06:00")))))
        insnow = field[k][np.isfinite(field[k])]
        print(f"  {lbl}: in-snow temp {insnow.min():+.1f} .. {insnow.max():+.1f} C, "
              f"mean {insnow.mean():+.1f} C, median {np.median(insnow):+.1f} C  "
              f"(all below 0: {bool((insnow < 0).all())}),  HS {hs[k]*100:.0f} cm")
    air, depth = freeman_06()
    dA, dB = depth.get(dt.date(2026, 2, 7), np.nan), depth.get(dt.date(2026, 2, 19), np.nan)
    print(f"  Freeman SnoDAR depth A->B: {dA:.0f} -> {dB:.0f} cm  (+{dB-dA:.0f})")


def metrics(dswe, dhs, mask=None):
    """r, RMSE(mm), bias(mm), density(kg/m3), n over finite pixels (optionally within mask).
    dSWE referenced to lidar by a constant; correlation is reference-free."""
    g = np.isfinite(dswe) & np.isfinite(dhs)
    if mask is not None:
        g &= mask
    if g.sum() < 30:
        return np.nan, np.nan, np.nan, np.nan, int(g.sum())
    lidar = dhs * SNOW_DENSITY / WATER_DENSITY
    c = np.median(lidar[g] - dswe[g]); nisar = dswe + c
    r = np.corrcoef(nisar[g], dhs[g])[0, 1]
    rmse = np.sqrt(np.mean((nisar[g] - lidar[g]) ** 2)) * 1000
    bias = np.mean((nisar[g] - lidar[g])) * 1000
    dens = np.polyfit(dhs[g], nisar[g], 1)[0] * 1000   # same plain fit as fig2's scatter line
    return r, rmse, bias, dens, int(g.sum())


class _Tee:
    """Write to several streams at once -- so main()'s prints go live to the console AND
    into a capture buffer we render to markdown."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for stream in self._streams:
            stream.write(text)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _to_markdown(console_text):
    """Turn the captured console output into a markdown file the manuscript can pull numbers
    from: each `=== SECTION ===` becomes a `## heading`, the aligned tables/prose under it go
    in a code fence (monospace keeps the columns lined up)."""
    out = ["# NISAR dSWE vs NIVAL dHS -- supporting statistics", ""]
    block = []

    def flush():
        while block and not block[0].strip():
            block.pop(0)
        while block and not block[-1].strip():
            block.pop()
        if block:
            out.extend(["```", *(line.rstrip() for line in block), "```", ""])
        block.clear()

    for line in console_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("===") and stripped.endswith("==="):
            flush()
            out.extend([f"## {stripped.strip('= ').strip()}", ""])
        else:
            block.append(line)
    flush()
    return "\n".join(out)


def assumptions():
    """Emit every physical constant + modelling assumption the retrieval and these stats rely on,
    pulled from the live constants so the report can't drift from the code."""
    print("\n=== ASSUMPTIONS / CONSTANTS ===")
    print(f"  NISAR L-band wavelength lambda      = {paths.LAMBDA:.3f} m  (f = 1.26 GHz)")
    print(f"  new-snow density (anchor + wrapping + RMSE) = {SNOW_DENSITY:.0f} kg/m3   (water {WATER_DENSITY:.0f} kg/m3)")
    print(f"  phase -> SWE: Leinss (2015) kappa(theta) = (2pi/lambda)(1.59 + theta^2.5),")
    print(f"               per-pixel LOCAL incidence theta (GUNW LOS + NISAR DEM, not flat 40 deg)")
    print(f"               kappa(40 deg) = {KAPPA:.1f} rad/m SWE  ({MM_PER_RAD:.2f} mm SWE per rad)")
    print(f"  dHS = HS(Feb22) - HS(Feb07), outlier-gated |dHS| < 1.5 m, averaged to the GUNW posting")
    print(f"  dSWE phase: the GUNW's ionosphere and hydrostatic + wet troposphere screens removed;")
    print(f"               referenced in phase units so the SNOTEL 637 pixel carries the station's SWE change")
    print(f"  interferogram: OPERATIONAL NISAR L2 GUNW (PROVISIONAL V1, PGE R05.02.3),")
    print(f"               {POSTING_M:.0f} m unwrapped posting; crossmul 5x6, phase-unwrap 13x16 looks")
    print(f"  local r: Gaussian-weighted window, sigma = window/4, {LOCAL_R_WINDOW*POSTING_M:.0f} m physical "
          f"({LOCAL_R_WINDOW:.1f} px)")
    print(f"  canopy fraction window {WINDOW} px = {WINDOW*POSTING_M:.0f} m")
    print(f"  coherence split for r(coh>.5)        = 0.50")
    print(f"  nuisance-delay conversions use 40 deg NOMINAL incidence:")
    print(f"               ionosphere 16.8 rad/TECU -> {MM_PER_TECU:.1f} mm SWE per TECU")
    print(f"               troposphere {DSWE_PER_M_ZTD:.0f} mm SWE per m ZTD  (= 4pi/lambda/cos40 * mm-per-rad)")
    print(f"  projection EPSG = {paths.EPSG} (UTM 11N);  dawn overpass = {OVERPASS_UTC_H:.4f} h UTC (05:46 MST)")


def main():
    assumptions()                                                            # 0
    d = retrieval.load_product()
    dswe, dhs, coh, elev = d["dswe"], d["dhs"], d["coherence"], d["elevation"]
    xs, ys = d["xs"], d["ys"]
    posting = float(abs(xs[1] - xs[0]))
    lia = load_lia(xs, ys)
    fcf = uniform_filter((_raster_on_grid(paths.WORLDCOVER, xs, ys, Resampling.nearest) == 10).astype(float),
                         WINDOW, mode="constant")
    sev = mtbs_severity(xs, ys)

    lidar_gate_stats()                                                        # 0b
    wrapping_stats(dswe, dhs, lia)                                            # 1

    # --- 2. HEADLINE: the operational L2 GUNW as delivered (80 m unwrapped posting).
    #     r(deramp) = elevation regressed out of both fields = the sub-pixel agreement left
    #     after the topographic ramp. Coherence gating sharpens r without moving the retrieved
    #     density, which is the check that the gate is not selecting an unrepresentative subset. ---
    print("\n=== HEADLINE (operational L2 GUNW, 80 m posting) ===")
    print(f"{'gate':>12}{'r':>7}{'r(deramp)':>11}{'RMSE mm':>9}{'density':>9}{'n':>9}{'% of valid':>12}")
    total = int((np.isfinite(dswe) & np.isfinite(dhs)).sum())
    for label, mask in [("all pixels", None), ("coh > 0.2", coh > 0.2), ("coh > 0.3", coh > 0.3),
                        ("coh > 0.4", coh > 0.4), ("coh > 0.5", coh > 0.5)]:
        r, rmse, bias, dens, n = metrics(dswe, dhs, mask=mask)
        sub = (dswe, dhs) if mask is None else (np.where(mask, dswe, np.nan), np.where(mask, dhs, np.nan))
        print(f"{label:>12}{r:>7.2f}{deramp_r(sub[0], sub[1], elev):>11.2f}{rmse:>9.0f}"
              f"{dens:>9.0f}{n:>9d}{n / total:>11.1%}")

    scale_decomposition()                                                    # 2b: kilometer vs sub-kilometer

    burn_fcf_stats(dswe, dhs, coh, fcf, sev)                                  # 3

    # --- 3b. OPENNESS class (fig3 synthesis): r + coherence climb as canopy falls ---
    opn = np.full(sev.shape, np.nan)
    opn[(sev == 0) & (fcf >= 0.5)] = 0; opn[(sev == 0) & (fcf < 0.5)] = 1
    opn[np.isin(sev, [1, 2])] = 2; opn[sev == 3] = 3; opn[sev == 4] = 4
    print("\n=== OPENNESS class (fig3): r + coherence climb as canopy falls ===")
    print(f"{'class':>11}{'med coh':>9}{'r':>7}{'RMSE mm':>9}{'n':>9}")
    for k, name in enumerate(OPEN_NAMES):
        m = opn == k; r, rmse, _, _, n = metrics(dswe, dhs, m)
        mc = np.nanmedian(coh[m & np.isfinite(coh)]) if m.sum() else np.nan
        print(f"{name:>11}{mc:>9.2f}{r:>7.2f}{rmse:>9.0f}{n:>9d}")
    print("=== MTBS severity gradient ===")
    for cls, lab in [(1, "unburned-low"), (2, "low"), (3, "moderate"), (4, "high")]:
        m = sev == cls; n = int((m & np.isfinite(dswe) & np.isfinite(dhs)).sum())
        if n > 30:
            print(f"  {lab:>13}: med coh {np.nanmedian(coh[m&np.isfinite(coh)]):.2f}  r {metrics(dswe,dhs,m)[0]:+.2f}  n {n}")
    print("=== CONTROL: openness r within elevation quartiles (not an elevation artifact) ===")
    g0 = np.isfinite(opn) & np.isfinite(elev) & np.isfinite(dswe) & np.isfinite(dhs)
    eq = np.nanpercentile(elev[g0], [0, 25, 50, 75, 100])
    print("  " + f"{'elev band':>13}" + "".join(f"{n:>11}" for n in OPEN_NAMES))
    for lo, hi in zip(eq[:-1], eq[1:]):
        band = g0 & (elev >= lo) & (elev < hi)
        print(f"  {lo:6.0f}-{hi:<6.0f}" + "".join(
            (f"{metrics(dswe,dhs,band&(opn==k))[0]:>11.2f}"
             if np.isfinite(metrics(dswe,dhs,band&(opn==k))[0]) else f"{'-':>11}") for k in range(5)))

    # 4/5 LOCAL r on the same operational product Fig 3 draws, 440 m Gaussian window
    driver_local_r(dswe, dhs, coh, lia, fcf, sev, LOCAL_R_WINDOW)             # 4/5 (Fig-3 consistent)
    terrain_stats(dswe, dhs, coh, elev, posting)                             # aspect / elevation / slope
    drift_depth_stats(dswe, dhs, coh, fcf)                                   # drift ruled out
    wind_stats()                                                             # wind: AOI calm, ridge only to 38
    lidar_window_swe()                                                       # 3 extra lidar days vs radar pair
    pair_table()                                                             # every timeline pair: NISAR, lidar, SNOTEL, r
    cumulative_stats()                                                       # the summed SWE against Mar 15 lidar depth, with quartile splits
    reference_sensitivity()                                                  # SNOTEL vs lidar vs no phase reference
    lidar_vs_snotel_depth()                                                  # lidar and station snow depth at the SNOTEL
    refocus_coherence()                                                      # mean coherence of the refocused pairs, before and after

    fig1_met(); met_snow()                                                   # snowpack temps, SWE, depth
    ionosphere_stats(xs, ys); troposphere_stats(xs, ys, dswe, elev)          # nuisance delays


def write_report():
    """Run main() and save everything it prints as outputs/paper_stats.md."""
    capture = io.StringIO()
    console = sys.stdout
    sys.stdout = _Tee(console, capture)            # live to console + captured for markdown
    try:
        main()
    finally:
        sys.stdout = console
    report = paths.OUTPUTS / "paper_stats.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_to_markdown(capture.getvalue()))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    write_report()

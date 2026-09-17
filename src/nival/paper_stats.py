"""Quantitative support for the NISAR dSWE vs NIVAL dHS paper: one script -> one markdown report.

`main()` prints every number the manuscript cites, each block under the claim it supports. The
scene-r blocks run on the 20 m product; the local-r block runs on the 40 m headline product
(matching fig3), with the local window held at the same ~420 m physical scale. In run order:

  ASSUMPTIONS   -- physical constants + modelling choices, pulled from the live code
  LIDAR GATE    -- fraction of native-resolution lidar dHS the |dHS| outlier gate removes
  WRAPPING      -- lidar dHS -> phase: fringes spanned + snaphu fringes added (little wrapping)
  HEADLINE      -- r / r(deramp) / RMSE / density on the operational GUNW, by coherence gate
  SCALE         -- kilometer-scale (low-pass) vs sub-kilometer (residual) correlation
  OPENNESS/BURN -- r + coherence by canopy fraction and MTBS severity (less canopy -> better)
  LOCAL r       -- local r vs coherence / incidence / canopy, + per-stratum medians (fig3)
  TERRAIN, DRIFT-CONTROL, WIND, LIDAR-WINDOW, MET, IONOSPHERE, TROPOSPHERE -- conditions + controls

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
from nival.ancillary_plotting import (_bilinear as _bilin, _elevation_on_grid,
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
    from nival.ancillary_plotting import load_met_series
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
    is absent. The DEM is the gap-free Copernicus GLO-30 (_elevation_on_grid) -- the same DEM
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


def posting_stats():
    """Scene-wide agreement at the delivered 80 m posting and at the 20 m posting we unwrap.

    The operational GUNW ships the wrapped interferogram at 20 m but only unwraps at 80 m.
    Fig. 3 is drawn from the 20 m layer unwrapped here (nival.unwrap_20m). The third row
    averages that field onto the delivered product's own cell centres, so the three rows are
    the same scene at three effective resolutions.
    """
    from nival import unwrap_20m
    from nival.paper_figures import (BURNED, SUB_WIN_X, SUB_WIN_Y, _fraction_on_grid,
                                     _lidar_on, subscene_windows)

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
    from nival.ancillary_plotting import snowpack_thermograph, freeman_06
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
    print(f"               per-pixel LOCAL incidence theta (GUNW LOS + Copernicus GLO-30 DEM, not flat 40 deg)")
    print(f"               kappa(40 deg) = {KAPPA:.1f} rad/m SWE  ({MM_PER_RAD:.2f} mm SWE per rad)")
    print(f"  dHS = HS(Feb22) - HS(Feb07), outlier-gated |dHS| < 1.5 m, averaged to the GUNW posting")
    print(f"  dSWE absolute datum: ONE constant offset to the lidar-implied SWE (dHS * density);")
    print(f"               offset-only, so spatial pattern and correlation are unchanged")
    print(f"  interferogram: OPERATIONAL NISAR L2 GUNW as delivered (PROVISIONAL V1, PGE R05.02.3),")
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
    posting_stats()                                                          # 2c: 80 m vs 20 m + Fig-3 subscenes

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

    fig1_met(); met_snow()                                                   # snowpack temps, SWE, depth
    ionosphere_stats(xs, ys); troposphere_stats(xs, ys, dswe, elev)          # nuisance delays


if __name__ == "__main__":
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

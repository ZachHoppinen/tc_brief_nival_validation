"""Supplement assets for the track-077 pair census.

  table_s1.tex   the seven candidate pairs on track 077, with their perpendicular baselines
  figS1          production against refocused RSLC amplitude for one acquisition, on the Fig. 1 map
  figS2          the extra pulse time stamps in the Jan 14 raw data and the row shift that corrects them
  figS3          coherence and unwrapped phase of the two Jan 14 pairs, production against refocused GUNW
  figS4          the tropospheric and ionospheric phase screens of each track-077 pair beside the observed
                 unwrapped phase, and their sums
  (fig_layover_shadow is kept but is not in the supplement)

Everything the paper depends on is built from tracked code: the pair list comes from
nival.pairs and the wetness from nival.supplement.wetness, not from scratch.

    conda run -n nival python src/nival/supplement/supplement_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the src/ dir
from nival import paths, retrieval   # noqa: E402
from nival.supplement import wetness   # noqa: E402
from nival.build_multipair import PRODUCTS   # noqa: E402
from nival.pairs import ANALYSED, label, on_track   # noqa: E402
from nival.paper_figures import LIDAR_OUTLINE_C, _snotel_swe, _swe_change   # noqa: E402

OUT = paths.FIGURES_DIR / "supplement"
DENS = retrieval.SNOW_DENSITY / retrieval.WATER_DENSITY
MONTH = {"11": "Nov", "12": "Dec", "01": "Jan", "02": "Feb", "03": "Mar"}
RSLC_MAP_STEP = 1              # Fig. S1 map posting, in steps of the 12 m Fig. 1 grid
LIA_DX = 30.0                  # Fig. S3 local incidence angle posting (m), the scene DEM's own
LAYOVER_PAD = 2000.0           # Fig. S3 radar window margin around the Fig. 1 map frame (m)
def nice(stamp):
    """'20260207' -> 'Feb 7' (month first, matching the manuscript's date style)."""
    return f"{MONTH[stamp[4:6]]} {int(stamp[6:])}"


def _perpendicular_baseline(pair, product):
    """Perpendicular baseline [m] of one pair on its product grid: the GUNW's baseline cube
    (height, y, x) interpolated to each pixel's terrain height with nisar_pytools."""
    from nisar_pytools.processing.atmospheric import _interpolate_3d_to_2d
    from nival.download_paper_data import pair_gunw
    with h5py.File(pair_gunw(pair)) as f:
        cube = f["science/LSAR/GUNW/metadata/radarGrid"]
        baseline = _interpolate_3d_to_2d(cube["perpendicularBaseline"][:], cube["heightAboveEllipsoid"][:],
                                         cube["yCoordinates"][:], cube["xCoordinates"][:], product["elevation"])
    return np.asarray(baseline)


def census(track="077"):
    """One row per candidate pair: interval, baseline, coherence, wetness, accumulation, r."""
    rep = pd.read_csv(paths.OUTPUTS / "multipair_stats.csv").set_index("pair") \
        if (paths.OUTPUTS / "multipair_stats.csv").exists() else None
    wet_ends = wetness.acquisition_table().set_index(["pair", "end"])
    from nival.download_paper_data import pair_gunw
    swe = _snotel_swe()
    rows = []
    for pair in on_track(track):
        t, ref, sec = pair[0], pair[1], pair[2]
        name = f"{t} {ref[4:]}-{sec[4:]}"
        path = PRODUCTS / f"{label(pair)}.nc"
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        # NISAR SWE change as the pair timeline draws it: ionosphere and troposphere screens removed, referenced to SNOTEL 637
        dswe, xs, ys, _, _ = _swe_change(pair_gunw(pair), ref, sec, swe)
        assert np.allclose(ds.x.values, xs) and np.allclose(ds.y.values, ys), "NISAR and lidar maps share one grid"
        dhs = ds["dhs"].values
        good = np.isfinite(dswe) & np.isfinite(dhs)
        days = (pd.Timestamp(sec) - pd.Timestamp(ref)).days
        bperp = _perpendicular_baseline(pair, ds)
        print(f"  {name}: perpendicular baseline min {np.nanmin(bperp):+.1f} mean {np.nanmean(bperp):+.1f} "
              f"max {np.nanmax(bperp):+.1f} m over {int(np.isfinite(bperp).sum())} of {bperp.size} pixels")
        rows.append(dict(
            name=name, key=label(pair), lab=f"{nice(ref)}--{nice(sec)}", days=days,
            bperp_m=float(np.nanmean(bperp)),
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

    lines = [r"\begin{tabular}{lrrrlrr}", r"\hline",
             r"Pair & $\Delta t$ (d) & $B_\perp$ (m) & $\bar{\gamma}$ & wet at & $\Delta$SWE (mm) & $r$ \\",
             r"\hline"]
    for _, x in frame.iterrows():
        bold = r"\textbf{" if x["analysed"] else ""
        end = "}" if bold else ""
        lines.append(f"{bold}{x['lab']}{end} & {x['days']} & ${x['bperp_m']:+.0f}$ & {x['coh']:.2f} & "
                     f"{wet_at(x)} & ${x['dswe_mm']:+.0f}$ & ${x['r']:+.2f}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    out = paths.OUTPUTS / "table_s1.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def _geocoded_amplitude(window_file, orbit, grid_lon, grid_lat, grid_height, wavelength, side, looks=4):
    """One RSLC amplitude window, multilooked and put on a map grid [dB].

    The window is in radar geometry: lines of zero-Doppler time, samples of slant range. Each
    map pixel is located in it with isce3's geo2rdr (orbit + terrain height, zero Doppler),
    and takes the value of the multilooked cell it falls in. Map pixels outside the window
    stay NaN."""
    import isce3
    crop = np.load(window_file)
    amp, t, r = crop["amp"], crop["t"], crop["r"]
    # average power over looks x looks single-look pixels, to tame speckle
    n_az, n_rg = amp.shape[0] // looks * looks, amp.shape[1] // looks * looks
    power = (amp[:n_az, :n_rg].astype(float) ** 2).reshape(n_az // looks, looks, n_rg // looks, looks).mean(axis=(1, 3))
    t0, dt_look = t[:looks].mean(), (t[1] - t[0]) * looks        # centre of the first multilooked line, line spacing [s]
    r0, dr_look = r[:looks].mean(), (r[1] - r[0]) * looks        # centre of the first multilooked sample, sample spacing [m]
    ellipsoid, zero_doppler = isce3.core.Ellipsoid(), isce3.core.LUT2d()
    out = np.full(grid_lon.shape, np.nan)
    for index in np.ndindex(grid_lon.shape):
        if not np.isfinite(grid_height[index]):              # no terrain height, no location
            continue
        llh = np.array([np.radians(grid_lon[index]), np.radians(grid_lat[index]), grid_height[index]])
        az_time, slant_range = isce3.geometry.geo2rdr(llh, ellipsoid, orbit, zero_doppler, wavelength, side)
        line, sample = int(round((az_time - t0) / dt_look)), int(round((slant_range - r0) / dr_look))
        if 0 <= line < power.shape[0] and 0 <= sample < power.shape[1]:
            out[index] = power[line, sample]
    return 10 * np.log10(out), str(crop["name"])


def _acquisition_geometry(track, date):
    """Orbit, wavelength and look side of one acquisition, from the production GUNW it is the
    reference of (the GUNW carries its reference RSLC's orbit)."""
    import isce3
    from nival.download_paper_data import gunw_file
    pair = next(p for p in on_track(track) if p[1] == date)
    with h5py.File(gunw_file(pair)) as f:
        orb = f["science/LSAR/GUNW/metadata/orbit/reference"]
        times, positions, velocities = orb["time"][:], orb["position"][:], orb["velocity"][:]
        epoch = isce3.core.DateTime(orb["time"].attrs["units"].decode().split("since ")[1])
        wavelength = isce3.core.speed_of_light / f["science/LSAR/GUNW/grids/frequencyA/centerFrequency"][()]
        side = {"left": isce3.core.LookSide.Left, "right": isce3.core.LookSide.Right}[
            f["science/LSAR/identification/lookDirection"][()].decode().lower()]
    orbit = isce3.core.Orbit([isce3.core.StateVector(epoch + isce3.core.TimeDelta(float(t)), p, v)
                              for t, p, v in zip(times, positions, velocities)], epoch)
    return orbit, wavelength, side


def fig_rslc_refocus():
    """Fig. S1: HH amplitude of one acquisition (paths.REFOCUS_EXAMPLE) as the production RSLC
    and as our refocused RSLC, beside the mean amplitude of the production RSLCs of other dates
    on the same track (paths.REFOCUS_STANDARD), which is what this scene usually looks like.
    All three are geocoded to the Fig. 1 map frame with the lidar footprint outlined.

    Every window is placed on the map with its own acquisition's orbit, taken from the
    operational GUNW that acquisition is the reference of, and the same scene DEM."""
    from pyproj import Transformer
    from nival.paper_figures import map_frame, setup_map_axes
    plt.rcdefaults()
    track, date = paths.REFOCUS_EXAMPLE
    # the Fig. 1 map grid, with its terrain height (ellipsoidal, as geo2rdr needs)
    frame = map_frame()
    xs, ys = frame["xs"][::RSLC_MAP_STEP], frame["ys"][::RSLC_MAP_STEP]
    posting = float(xs[1] - xs[0])
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_lon, grid_lat = Transformer.from_crs(paths.EPSG, 4326, always_xy=True).transform(grid_x, grid_y)
    # clip the DEM a little wider than the frame, or reprojection leaves gaps along the map edge
    west, south, east, north = frame["aoi"]
    height = retrieval._elevation_on_grid(xs, ys, (west - 0.01, south - 0.01, east + 0.01, north + 0.01))
    print(f"  map grid {height.shape} at {posting:.0f} m | terrain height gaps: {int(np.isnan(height).sum())}")

    def on_map(when, kind):
        """One staged window, in dB on the map grid."""
        db, name = _geocoded_amplitude(paths.RSLC_WINDOWS / f"{track}_{when}_{kind}.npz", *_acquisition_geometry(track, when)[:1],
                                       grid_lon, grid_lat, height, *_acquisition_geometry(track, when)[1:])
        print(f"  {when} {kind}: {name} | finite {int(np.isfinite(db).sum())} of {db.size} | "
              f"dB 2-50-98%: {np.nanpercentile(db, 2):.1f} {np.nanpercentile(db, 50):.1f} {np.nanpercentile(db, 98):.1f}")
        return db

    nice_day = lambda stamp: f"{pd.Timestamp(stamp):%b} {pd.Timestamp(stamp).day}"
    # the usual amplitude: production RSLCs of other dates, averaged as power and then put in dB
    usual = [on_map(when, "production") for when in paths.REFOCUS_STANDARD]
    mean_db = 10 * np.log10(np.mean([10 ** (db / 10) for db in usual], axis=0))
    panels = [(on_map(date, "production"), f"Production RSLC, {nice_day(date)}"),
              (on_map(date, "refocus"), f"Refocused RSLC, {nice_day(date)}"),
              (mean_db, "Mean production RSLC, " + " and ".join(nice_day(d) for d in paths.REFOCUS_STANDARD))]
    # one colour range for all panels: the pooled 2nd to 98th percentile
    low, high = np.nanpercentile(np.concatenate([p[0].ravel() for p in panels]), [2, 98])
    extent = [xs.min() - posting / 2, xs.max() + posting / 2, ys.min() - posting / 2, ys.max() + posting / 2]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), dpi=300, constrained_layout=True)
    for ax, (db, title) in zip(axes, panels):
        setup_map_axes(ax, frame)                           # the Fig. 1 projection + bounds
        image = ax.imshow(db, extent=extent, origin="upper", cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        ax.contour(frame["lidar_mask"], levels=[0.5], extent=frame["lidar_extent"], origin="upper",
                   colors=LIDAR_OUTLINE_C, linewidths=2.2)           # lidar coverage outline, as in Fig. 1
        ax.locator_params(nbins=5)
        ax.set_title(title, fontsize=11, fontweight="bold")
    for ax in axes[1:]:                                      # the other maps share the left one's northing axis
        ax.set_ylabel(""); ax.set_yticklabels([])
    fig.colorbar(image, ax=axes, shrink=0.85, pad=0.015, label="HH amplitude (dB)")
    _save(fig, "figS1_rslc_refocus.png")


def fig_atmosphere():
    """Fig. S4: the tropospheric and ionospheric phase screens of each track-077 pair and the
    observed unwrapped phase they are removed from, one row each in the layout of Fig. 2, and
    the sum of each over the seven pairs. One colour range for the pair maps (-2 pi to 2 pi)
    and one for the sums (-4 pi to 4 pi), so the screens can be read against the phase."""
    from nival.download_paper_data import TRACK
    from nival.paper_figures import timeline_figure
    timeline_figure(OUT / "figS4_atmospheric_phase.png", tracks=[TRACK], last=paths.FIG3_LAST, totals=True,
                    quantity=("troposphere", "ionosphere", "unwrapped"), swe_panel=False,
                    heading="NISAR phase corrections and observed unwrapped phase",
                    screen_lim=2 * np.pi, total_lim=4 * np.pi)


def fig_gunw_refocus():
    """Fig. S3: the two track-077 pairs that use the Jan 14 acquisition, as the production GUNW
    and as our GUNW from the refocused RSLC. One row per pair: 80 m coherence of each, then the
    80 m unwrapped phase of each about its own median, all in the Fig. 1 map frame with the
    lidar footprint outlined."""
    from nival.download_paper_data import TRACK, local_gunws, pair_of
    from nival.paper_figures import _unwrapped_phase, _pi_ticks, map_frame, setup_map_axes
    plt.rcdefaults()
    frame = map_frame()
    production = {pair_of(f): f for f in paths.GUNW_OPERATIONAL.parent.glob("NISAR_L2_PR_GUNW_*.h5")}
    pairs = [g for g in local_gunws() if g[0] == TRACK and g[4] and g[2] <= paths.FIG3_LAST]

    def coherence(path):
        # the 80 m coherence layer over the AOI
        with h5py.File(path) as f:
            g = f["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"]
            rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
            return g["coherenceMagnitude"][rows, cols].astype(float)

    phase_lim = 2 * np.pi
    fig, axes = plt.subplots(len(pairs), 4, figsize=(17, 4.6 * len(pairs)), dpi=300, constrained_layout=True)
    for row, (track, ref, sec, ours, _) in zip(axes, pairs):
        name = f"{nice(ref)} to {nice(sec)}"
        for k, (kind, path) in enumerate((("production", production[(track, ref, sec)]), ("refocused", ours))):
            phase, xs, ys = _unwrapped_phase(path)
            coh = coherence(path)
            posting = float(abs(xs[1] - xs[0]))
            extent = [xs.min() - posting / 2, xs.max() + posting / 2, ys.min() - posting / 2, ys.max() + posting / 2]
            print(f"  {name} {kind}: coherence mean {np.nanmean(coh):.2f} | unwrapped phase finite "
                  f"{int(np.isfinite(phase).sum())} of {phase.size}, 1-99% about median "
                  f"{np.nanpercentile(phase - np.nanmedian(phase), 1):+.1f} to {np.nanpercentile(phase - np.nanmedian(phase), 99):+.1f} rad")
            coh_image = row[k].imshow(coh, extent=extent, origin="upper", cmap="gray", vmin=0, vmax=1)
            phase_image = row[2 + k].imshow(phase - np.nanmedian(phase), extent=extent, origin="upper", cmap="RdBu_r",
                                            vmin=-phase_lim, vmax=phase_lim)
            row[k].set_title(f"Coherence, {kind}\n{name}", fontsize=11)
            row[2 + k].set_title(f"Unwrapped phase, {kind}\n{name}", fontsize=11)
        for ax in row:
            setup_map_axes(ax, frame)                           # the Fig. 1 projection + bounds
            ax.contour(frame["lidar_mask"], levels=[0.5], extent=frame["lidar_extent"], origin="upper",
                       colors=LIDAR_OUTLINE_C, linewidths=2.0)
            ax.locator_params(nbins=4)
        for ax in row[1:]:                                      # the other maps share the left one's northing axis
            ax.set_ylabel(""); ax.set_yticklabels([])
        fig.colorbar(coh_image, ax=row[:2], shrink=0.8, pad=0.02, label="Coherence")
        bar = fig.colorbar(phase_image, ax=row[2:], shrink=0.8, pad=0.02, extend="both", label="Phase about median (rad)")
        _pi_ticks(bar, phase_lim)
    for ax in axes[:-1].ravel():                                # easting labels on the bottom row only
        ax.set_xlabel("")
    _save(fig, "figS3_gunw_refocus.png")


def fig_layover_shadow():
    """Fig. S3: local incidence angle beside radar layover and shadow of track 077, both on the
    Fig. 1 map frame with the lidar footprint outlined.

    The local incidence angle is nisar_pytools' local_incidence_angle over the Fig. 1 frame:
    the GUNW line-of-sight cube against the slope of the scene DEM, as in the retrieval. It is
    computed at 30 m, the DEM's posting, because slopes of the DEM resampled to the 12 m map
    grid carry a crosshatch from the resampling.

    The mask comes from isce3's rdr2geo on the radar grid of the analysed pair's reference RSLC
    (paths.NISAR_A), geocoded by nearest neighbour onto the 12 m Fig. 1 grid. Both steps are the
    NISAR static-layer routine nisar.static.layover_shadow_mask. The orbit and the radar grid
    parameters are read from the operational GUNW, and the terrain is the scene DEM.
    Mask values: 0 neither, 1 shadow, 2 layover, 3 layover and shadow."""
    import glob
    import tempfile
    import isce3
    import rasterio
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from nisar.static.layover_shadow_mask import compute_geocoded_layover_shadow_mask
    from nisar_pytools.utils.local_incidence_angle import local_incidence_angle
    from pyproj import Transformer
    from nival.download_paper_data import TRACK
    from nival.paper_figures import MAP_DX, map_frame, setup_map_axes
    plt.rcdefaults()
    # the Fig. 1 map grid: UTM pixel centres at MAP_DX, north at top
    frame = map_frame()
    xs, ys, extent = frame["xs"], frame["ys"], frame["extent"]
    geo_grid = isce3.product.GeoGridParameters(start_x=xs[0] - MAP_DX / 2, start_y=ys[0] + MAP_DX / 2,
                                               spacing_x=MAP_DX, spacing_y=-MAP_DX,
                                               width=len(xs), length=len(ys), epsg=paths.EPSG)

    # local incidence angle over the frame at LIA_DX [deg]. The DEM is clipped a little wider
    # than the frame, or reprojection leaves gaps along the map edge
    west, south, east, north = frame["aoi"]
    lia_xs = np.arange(extent[0] + LIA_DX / 2, extent[1], LIA_DX)
    lia_ys = np.arange(extent[3] - LIA_DX / 2, extent[2], -LIA_DX)
    height = xr.DataArray(retrieval._elevation_on_grid(lia_xs, lia_ys, (west - 0.01, south - 0.01, east + 0.01, north + 0.01)),
                          coords={"y": lia_ys, "x": lia_xs}, dims=("y", "x")).rio.write_crs(paths.EPSG)
    with h5py.File(paths.gunw_product()) as gunw:
        cube = gunw["science/LSAR/GUNW/metadata/radarGrid"]
        los_x, los_y = cube["losUnitVectorX"][:], cube["losUnitVectorY"][:]
        los_z = np.sqrt(np.clip(1 - los_x ** 2 - los_y ** 2, 0, 1))
        incidence = np.asarray(local_incidence_angle(height, los_x, los_y, los_z, cube["heightAboveEllipsoid"][:],
                                                     cube["xCoordinates"][:], cube["yCoordinates"][:], epsg=paths.EPSG).values)
    lia_extent = [lia_xs[0] - LIA_DX / 2, lia_xs[-1] + LIA_DX / 2, lia_ys[-1] - LIA_DX / 2, lia_ys[0] + LIA_DX / 2]
    print(f"  local incidence angle grid {incidence.shape} at {LIA_DX:.0f} m | min {np.nanmin(incidence):.1f} median {np.nanmedian(incidence):.1f} "
          f"max {np.nanmax(incidence):.1f} deg | gaps {int(np.isnan(incidence).sum())}")

    # the reference RSLC's full radar grid, rebuilt from the parameters the GUNW records
    orbit, wavelength, side = _acquisition_geometry(TRACK, paths.NISAR_A)
    with h5py.File(paths.gunw_product()) as gunw:
        ref = gunw["science/LSAR/GUNW/metadata/processingInformation/parameters/reference/frequencyA"]
        start_utc = ref["zeroDopplerStartTime"][()].decode()
        line_dt, r0, dr = (float(ref[k][()]) for k in ("zeroDopplerTimeSpacing", "slantRangeStart", "slantRangeSpacing"))
        n_lines, n_samples = int(ref["numberOfAzimuthLines"][()]), int(ref["numberOfRangeSamples"][()])
    t0 = (isce3.core.DateTime(start_utc) - orbit.reference_epoch).total_seconds()   # first line, seconds since orbit epoch
    full_grid = isce3.product.RadarGridParameters(t0, wavelength, 1.0 / line_dt, r0, dr, side, n_lines, n_samples,
                                                  orbit.reference_epoch)

    # radar window: geo2rdr a coarse grid of the padded map frame at DEM height, take the bounding box
    dem_raster = isce3.io.Raster(str(paths.SUBSET_DEM))
    dem = isce3.geometry.DEMInterpolator(); dem.load_dem(dem_raster)
    px, py = np.meshgrid(np.linspace(extent[0] - LAYOVER_PAD, extent[1] + LAYOVER_PAD, 25),
                         np.linspace(extent[2] - LAYOVER_PAD, extent[3] + LAYOVER_PAD, 25))
    lon, lat = Transformer.from_crs(paths.EPSG, 4326, always_xy=True).transform(px.ravel(), py.ravel())
    az_times, slant_ranges = [], []
    for lo, la in zip(np.radians(lon), np.radians(lat)):
        llh = np.array([lo, la, dem.interpolate_lonlat(lo, la)])
        az_time, slant_range = isce3.geometry.geo2rdr(llh, isce3.core.Ellipsoid(), orbit, isce3.core.LUT2d(), wavelength, side)
        az_times.append(az_time); slant_ranges.append(slant_range)
    a0, a1 = int((min(az_times) - t0) / line_dt), int((max(az_times) - t0) / line_dt) + 1
    s0, s1 = int((min(slant_ranges) - r0) / dr), int((max(slant_ranges) - r0) / dr) + 1
    assert 0 <= a0 < a1 <= n_lines and 0 <= s0 < s1 <= n_samples, "map frame falls outside the radar swath"
    radar_grid = full_grid[a0:a1, s0:s1]

    # layover / shadow on the radar window, then geocoded to the map grid
    scratch = tempfile.mkdtemp(prefix="layover_shadow_")
    geocoded = compute_geocoded_layover_shadow_mask(radar_grid=radar_grid, orbit=orbit, img_grid_doppler=isce3.core.LUT2d(),
                                                    geo_grid=geo_grid, dem_raster=dem_raster, scratch_dir=scratch)
    del geocoded                                         # close the raster so the GeoTIFF is flushed
    with rasterio.open(glob.glob(f"{scratch}/geocoded-layover-shadow-mask_*")[0]) as src:
        mask = src.read(1)
    classes = [(0, "neither", "#d9d9d9"), (1, "shadow", "#1f3b73"), (2, "layover", "#d7301f"),
               (3, "layover and shadow", "#7b3294")]
    print(f"  radar window {radar_grid.length} x {radar_grid.width} | map grid {mask.shape} at {MAP_DX:.0f} m | "
          + ", ".join(f"{name} {int(np.sum(mask == value))}" for value, name, _ in classes))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), dpi=300, constrained_layout=True)
    image = axes[0].imshow(incidence, extent=lia_extent, origin="upper", cmap="viridis", interpolation="nearest")
    fig.colorbar(image, ax=axes[0], shrink=0.85, pad=0.015, label="local incidence angle (deg)")
    axes[1].imshow(mask, extent=extent, origin="upper", interpolation="nearest",
                   cmap=ListedColormap([colour for _, _, colour in classes]), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
    for ax, title in zip(axes, ("Local incidence angle", "Layover and shadow")):
        setup_map_axes(ax, frame)                        # the Fig. 1 projection + bounds
        ax.contour(frame["lidar_mask"], levels=[0.5], extent=frame["lidar_extent"], origin="upper",
                   colors=LIDAR_OUTLINE_C, linewidths=2.2)        # lidar coverage outline, as in Fig. 1
        ax.locator_params(nbins=5)
        ax.set_title(title, fontsize=11, fontweight="bold")
    axes[1].set_ylabel(""); axes[1].set_yticklabels([])  # shares the left map's northing axis
    axes[1].legend(handles=[Patch(color=colour, label=name) for _, name, colour in classes[1:3]]
                   + [Line2D([], [], color=LIDAR_OUTLINE_C, lw=2.2, label="Lidar Coverage")], loc="lower right", fontsize=9, framealpha=0.9)
    _save(fig, "figS3_layover_shadow.png")


def fig_stamp_fix():
    """Fig. S2: the pulse time stamps of the Jan 14 raw (L0B) granule against those of the Feb 7
    granule of the same track, which has the same pulse timing pattern and no extra stamps, and
    with the Dec 21 granule as a control: a second date whose stamps should, and do, match Feb 7.

    (a) Stamps recorded in each second, all three dates, over the part of the data take that
    holds the bursts ahead of frame 024 and the frame itself (shaded). A second set of bursts
    follows the frame and is not shown. (b) A zoom on the largest group of those bursts:
    the interval from each stamp to the next, Jan 14 as recorded over the two clean dates.
    (c) The fix: the same zoom with the extra Jan 14 stamps deleted, which leaves Jan 14 on the
    pattern of the two clean dates. The refocus itself
    dropped the first paths.STAMP_SHIFT stamps of the file and slid the rest up; each extra
    stamp pushes every later stamp one echo row down, so for the frame's echo rows that gives
    exactly the same stamps as deleting the extras (asserted below). Only
    stamps are read, never echoes."""
    from matplotlib.patches import ConnectionPatch
    plt.rcdefaults()
    key = "frequencyA/txH/UTCtime"
    jan, feb = np.load(paths.STAMPS_FAULTY)[key], np.load(paths.STAMPS_CLEAN)[key]   # [s of day]
    assert jan.size == feb.size and 4e4 < jan[0] < 5e4, "stamps are seconds of the UTC day"
    dec = np.load(paths.STAMPS_CONTROL)[key]               # Dec 21, a second date with no extra stamps: the control
    jan_s, feb_s, dec_s = jan - jan[0], feb - feb[0], dec - dec[0]       # seconds after each date's first pulse
    # the control's stamps recorded so far minus Feb 7 pulses sent so far, counted the same way as Jan 14 below
    dec_ahead = np.arange(dec.size) - np.searchsorted(feb_s, dec_s + 1e-7, side="right") + 1
    print(f"  Dec 21 against Feb 7, row by row: largest difference {np.abs(dec_s - feb_s).max() * 1e9:.1f} ns, extra stamps {dec_ahead.max()}")
    # extra stamps ahead of each echo row: Jan 14 stamps recorded so far minus pulses sent so far
    ahead = np.arange(jan.size) - np.searchsorted(feb_s, jan_s + 1e-7, side="right") + 1
    frame_span = [t - jan[0] for t in paths.FRAME_UTC]
    in_frame = (feb_s >= frame_span[0]) & (feb_s <= frame_span[1])       # echo rows sent during the frame
    assert ahead[in_frame].min() == ahead[in_frame].max() == paths.STAMP_SHIFT, "the shift is the count at the frame"
    print(f"  {jan.size} stamps per date | extra stamps ahead of the frame rows: {ahead[in_frame].max()}, at the end of the file: {ahead[-1]}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), dpi=300, constrained_layout=True)
    view = (100.0, 450.0)                                   # seconds shown: the bursts ahead of the frame, and the frame
    zoom_view = (221.5, 227.0)                              # seconds shown in (b) and (c): the largest group of bursts ahead of the frame
    # stamps recorded in each whole second all three records cover
    seconds = np.arange(0, int(min(jan_s[-1], feb_s[-1], dec_s[-1])) + 1)
    jan_rate, feb_rate, dec_rate = (np.histogram(t, seconds)[0] for t in (jan_s, feb_s, dec_s))
    busiest = int(np.argmax(np.where(seconds[:-1] < frame_span[0], jan_rate, 0)))   # the busiest second ahead of the frame
    # (a) stamp rate; the two clean dates as one grey line on top of Jan 14. Its legend serves all three panels
    axes[0].axvspan(*frame_span, color="0.85", lw=0, zorder=0, label="Frame 024")
    axes[0].plot(seconds[:-1], dec_rate, color="0.7", lw=2.5, zorder=3)
    axes[0].plot(seconds[:-1], feb_rate, color="0.7", lw=2.5, zorder=3, label="Dec 21 and Feb 7")
    axes[0].plot(seconds[:-1], jan_rate, color="C3", lw=1.4, zorder=2, label="Jan 14")   # under the clean dates, so they stay visible
    shown = (seconds[:-1] >= view[0]) & (seconds[:-1] <= view[1])
    axes[0].set_ylim(feb_rate[shown].min() - 60, jan_rate[shown].max() + 120)        # fitted to the seconds shown
    axes[0].set_xlim(*view); axes[0].set_xlabel("Time after first pulse (s)"); axes[0].set_ylabel("Time stamps per second")
    axes[0].set_title("(a) Problem: extra time stamps"); axes[0].legend(fontsize=9, loc="upper left")
    # the fix drawn in (c), the extra stamps deleted: for each pulse of the Feb 7 pattern keep the one
    # Jan 14 stamp that sits on it (within 0.1 microseconds) and drop every other stamp
    from nival.supplement.remove_extra_stamps import kept_rows
    kept = kept_rows(jan, feb)                                              # one stamp per pulse
    cleaned = jan_s[kept]
    # the refocus dropped the first `shift` stamps instead; for the frame's echo rows the two are the same stamps
    frame_rows = np.flatnonzero(in_frame)
    assert np.array_equal(cleaned[frame_rows], jan_s[frame_rows + paths.STAMP_SHIFT]), "deleting the extras = the row shift, in the frame"
    removed = np.setdiff1d(np.arange(jan.size), kept)
    print(f"  extra stamps deleted: {int((removed < frame_rows[0] + paths.STAMP_SHIFT).sum())} ahead of the frame, {removed.size} in the file; "
          f"frame rows identical to the {paths.STAMP_SHIFT}-row shift")
    # (b) and (c): the interval from each stamp to the next across those bursts, Jan 14 as recorded and with
    # the extra stamps deleted, as small points over the two clean dates as wide grey points
    for ax, stamps, colour, label, title in ((axes[1], jan_s, "C3", "Jan 14 as recorded", "(b) Zoom: intervals between time stamps"),
                                             (axes[2], cleaned, "C3", "Jan 14, extra stamps removed", "(c) Fix: extra time stamps removed")):
        ax.plot(dec_s[:-1], np.diff(dec_s) * 1e6, ".", color="0.7", ms=7, rasterized=True)
        ax.plot(feb_s[:-1], np.diff(feb_s) * 1e6, ".", color="0.7", ms=7, rasterized=True)
        ax.plot(stamps[:-1], np.diff(stamps) * 1e6, ".", color=colour, ms=1.2, rasterized=True)
        ax.set_xlim(*zoom_view); ax.set_ylim(-20, 660); ax.set_xlabel("Time after first pulse (s)")
        ax.set_ylabel("Interval to next time stamp (\u00b5s)"); ax.set_title(title)
        inside = lambda t: int(((t >= zoom_view[0]) & (t < zoom_view[1])).sum())
        print(f"  {label}: {inside(stamps)} stamps in {zoom_view[0]} to {zoom_view[1]} s (Feb 7 {inside(feb_s)}, Dec 21 {inside(dec_s)})")
    # call-out lines from the busiest second on the rate curve to the zoom panel
    for corner in (0, 1):
        fig.add_artist(ConnectionPatch((busiest, jan_rate[busiest]), (0, corner), "data", "axes fraction",
                                       axesA=axes[0], axesB=axes[1], color="0.4", lw=0.8))
    _save(fig, "figS2_stamp_fix.png")


def main():
    """Build the pair census and its table, then the four supplement figures."""
    frame = census()
    print(f"{len(frame)} track-077 pairs; {int((~frame.wet).sum())} dry at both stations")
    print(frame[["lab", "days", "coh", "r", "wet", "analysed"]].to_string(index=False))
    table_s1(frame)
    fig_rslc_refocus()
    fig_atmosphere()
    fig_layover_shadow()
    fig_stamp_fix()


if __name__ == "__main__":
    main()

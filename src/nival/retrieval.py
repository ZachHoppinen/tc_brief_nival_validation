"""NISAR phase + NIVAL lidar -> the dSWE/dHS product on the GUNW grid.

This is the one load-bearing computation. `build_product()` runs it top-to-bottom
and writes a single NetCDF; every figure reads that grid via `load_product()`.
Everything is in metres.

  dSWE [m] = (phase - median) / (k * (1.59 + theta^2.5))     Leinss (2015),
             k = 2*pi/wavelength  (== uavsar_pytools.swe_from_phase_leinss)
  dHS  [m] = (snow depth Feb22 - snow depth Feb07) resampled to the grid

The interferogram is the OPERATIONAL NISAR L2 GUNW for the pair (paths.GUNW_OPERATIONAL),
read as delivered at its 80 m unwrapped posting; `_aoi_window` clips the full frame to the AOI.

dSWE is then shifted by one constant so the scene-median dSWE matches the lidar-implied
SWE (dHS * SNOW_DENSITY / WATER_DENSITY) -- an absolute anchor for display. It is a pure
offset, so the spatial pattern and any correlation with dHS are unchanged.

theta is the per-pixel local incidence angle from the tested nisar_pytools tool
(GUNW line-of-sight vectors + the Copernicus GLO-30 DEM, not a flat 40 deg). Everything
else is plain numpy / rioxarray / xarray.
"""

from __future__ import annotations

import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter

from . import paths

# new-snow density: co-located SNOTEL 637 06:00 SWE/depth change over the radar pair
# (Feb 7->19) = 56 mm SWE / 38 cm depth = 147 kg/m3. SWE depth = HS depth * SNOW_DENSITY / WATER_DENSITY.
# Sets the absolute dSWE reference level below (a pure offset) and the lidar->phase wrapping conversion.
SNOW_DENSITY, WATER_DENSITY = 147.0, 1000.0

# lidar outlier gate: |dHS| above this over the 15-day lidar interval is voids / canopy
# returns, not snow. paper_stats reports the fraction it removes, so the manuscript number
# and the gate applied here can never diverge.
DHS_GATE_M = 1.5

# the scene DEM: public Copernicus GLO-30 (staged over the AOI by generate_gunw, the same
# DEM isce3 geocodes against). It spans the full GUNW grid gap-free, so local_incidence_angle
# covers the whole coherent scene. Shared by the figure/stats DEM readers too.
REGION_DEM = paths.SUBSET_DEM


def _raster_on_grid(path, xs, ys, resampling, clip_aoi=None):
    """Reproject + regrid a raster onto the regular UTM grid defined by (xs, ys)."""
    reference = xr.DataArray(np.zeros((len(ys), len(xs))), coords={"y": ys, "x": xs},
                             dims=("y", "x")).rio.write_crs(paths.EPSG)
    raster = rioxarray.open_rasterio(path, masked=True).squeeze()
    if clip_aoi is not None:
        raster = raster.rio.clip_box(*clip_aoi, crs="EPSG:4326")
    return np.asarray(raster.rio.reproject(paths.EPSG).rio.reproject_match(reference, resampling=resampling).values)


def _elevation_on_grid(xs, ys, aoi):
    """Copernicus GLO-30 elevation on the grid -- a gap-free DEM for local_incidence_angle."""
    return _raster_on_grid(REGION_DEM, xs, ys, Resampling.bilinear, clip_aoi=aoi)


def _aoi_window(xs, ys):
    """Row/column slices of a geocoded GUNW grid covering the AOI. A no-op for our own GUNWs
    (geocoded to the AOI box); it is what keeps a full-frame operational granule -- the whole
    ~360 x 356 km frame -- from being read in whole."""
    from pyproj import Transformer
    west, south, east, north = paths.AOI
    to_utm = Transformer.from_crs(4326, paths.EPSG, always_xy=True).transform
    eastings, northings = zip(*[to_utm(a, b) for a in (west, east) for b in (south, north)])
    cols = np.where((xs >= min(eastings)) & (xs <= max(eastings)))[0]
    rows = np.where((ys >= min(northings)) & (ys <= max(northings)))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def build_product(gunw_path=None, lidar_a=None, lidar_b=None, out_path=None):
    """Compute dSWE/dHS/coherence/elevation on the GUNW grid and save one NetCDF.
    Defaults read the operational L2 GUNW for the Feb 7 -> Feb 19 pair and the Feb 7 / Feb 22
    lidar flights. `gunw_path`, `lidar_a`/`lidar_b` and `out_path` override those, so any
    other pair runs through the identical computation."""
    # nisar_pytools is needed only for this InSAR step -- deferred so load_product()
    # and the figure modules import without it.
    from nisar_pytools.utils.local_incidence_angle import local_incidence_angle
    # --- read the GUNW: unwrapped phase, coherence, line-of-sight vectors + heights ---
    with h5py.File(gunw_path or paths.gunw_product()) as gunw_file:
        UNWRAPPED_INTERFEROGRAM_GROUP = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
        unwrapped_interferogram = gunw_file[UNWRAPPED_INTERFEROGRAM_GROUP]
        RADAR_GRID_GROUP = "science/LSAR/GUNW/metadata/radarGrid"
        radar_grid = gunw_file[RADAR_GRID_GROUP]
        rows, cols = _aoi_window(unwrapped_interferogram["xCoordinates"][:],
                                 unwrapped_interferogram["yCoordinates"][:])
        phase = xr.DataArray(
            unwrapped_interferogram["unwrappedPhase"][rows, cols], dims=("y", "x"),
            coords={"y": unwrapped_interferogram["yCoordinates"][rows],
                    "x": unwrapped_interferogram["xCoordinates"][cols]},
        ).rio.write_crs(paths.EPSG)
        coherence = unwrapped_interferogram["coherenceMagnitude"][rows, cols]
        line_of_sight_east = radar_grid["losUnitVectorX"][:]
        line_of_sight_north = radar_grid["losUnitVectorY"][:]
        line_of_sight_up = np.sqrt(np.clip(1 - line_of_sight_east ** 2 - line_of_sight_north ** 2, 0, 1))
        radar_grid_heights = radar_grid["heightAboveEllipsoid"][:]
        radar_grid_x = radar_grid["xCoordinates"][:]
        radar_grid_y = radar_grid["yCoordinates"][:]

    # --- Copernicus DEM on the phase grid -> per-pixel local incidence angle [degrees].
    #     The same DEM isce3 geocodes against; it spans the full coherent scene, so LIA
    #     (and thus dSWE) cover every pixel, matching the LIA paper_stats uses. ---
    elevation = xr.DataArray(
        _elevation_on_grid(phase.x.values, phase.y.values, paths.AOI),
        coords={"y": phase.y.values, "x": phase.x.values}, dims=("y", "x"),
    ).rio.write_crs(paths.EPSG)
    incidence_angle_degrees = np.asarray(local_incidence_angle(
        elevation, line_of_sight_east, line_of_sight_north, line_of_sight_up,
        radar_grid_heights, radar_grid_x, radar_grid_y, epsg=paths.EPSG).values)

    # --- phase -> dSWE [m], Leinss (2015). First strip the arbitrary unwrapping offset
    #     (InSAR phase has no absolute zero) in PHASE units -- it must come out here, before
    #     the per-pixel kappa divide, or it smears into an incidence-angle ramp. ---
    phase_relative = phase.values - np.nanmedian(phase.values)
    radians_per_metre_swe = 2 * np.pi / paths.LAMBDA * (1.59 + np.radians(incidence_angle_degrees) ** 2.5)
    dswe_metres = phase_relative / radians_per_metre_swe

    # --- NIVAL lidar -> dHS [m] on the same grid (masked=True reads nodata as NaN) ---
    snow_depth_later = rioxarray.open_rasterio(lidar_b or paths.LIDAR_A22, masked=True).squeeze()
    snow_depth_earlier = rioxarray.open_rasterio(lidar_a or paths.LIDAR_A07, masked=True).squeeze()
    snow_depth_change = snow_depth_later - snow_depth_earlier
    # outlier gate: |dHS| > DHS_GATE_M over 15 days is lidar voids / canopy / flight mismatch
    snow_depth_change = snow_depth_change.where(
        (snow_depth_change > -DHS_GATE_M) & (snow_depth_change < DHS_GATE_M))
    snow_depth_change = snow_depth_change.rio.write_nodata(np.nan)   # so resampling fills gaps with NaN, not a sentinel
    dhs_metres = snow_depth_change.rio.reproject_match(phase, resampling=Resampling.average).values

    # --- set the absolute SWE datum (a different d.o.f. than the phase offset above): shift
    #     dSWE by one constant so its scene median matches the lidar-implied SWE (dHS *
    #     new-snow density). A pure offset -- pattern + correlation unchanged. ---
    both_finite = np.isfinite(dswe_metres) & np.isfinite(dhs_metres)
    lidar_swe = dhs_metres * SNOW_DENSITY / WATER_DENSITY
    reference = np.nanmedian(lidar_swe[both_finite] - dswe_metres[both_finite])
    dswe_metres = dswe_metres + reference

    # --- assemble + save the one product ---
    product = xr.Dataset(
        {"dswe": (("y", "x"), dswe_metres),
         "dhs": (("y", "x"), dhs_metres),
         "coherence": (("y", "x"), coherence),
         "elevation": (("y", "x"), elevation.values),
         "lia": (("y", "x"), incidence_angle_degrees)},   # the per-pixel angle already used for kappa
        coords={"y": phase.y.values, "x": phase.x.values},
        attrs={"crs_epsg": paths.EPSG, "posting_m": float(abs(phase.x.values[1] - phase.x.values[0]))})
    destination = out_path or paths.product()
    destination.parent.mkdir(parents=True, exist_ok=True)
    product.to_netcdf(destination)
    print(f"product -> {destination}  ({product.sizes['y']} x {product.sizes['x']})")
    return product


def load_product():
    """Read the saved product -> dict(dswe[m], dhs[m], coherence, elevation[m], xs, ys; lia[deg] if present)."""
    if not paths.product().exists():
        raise SystemExit(f"missing {paths.product()} -- run python src/nival/run_workflow.py")
    product = xr.open_dataset(paths.product())
    out = {"dswe": product["dswe"].values,
           "dhs": product["dhs"].values,
           "coherence": product["coherence"].values,
           "elevation": product["elevation"].values,
           "xs": product["x"].values,
           "ys": product["y"].values}
    if "lia" in product:                                  # local incidence angle (deg), added for fig 3
        out["lia"] = product["lia"].values
    return out


def local_correlation(a, b, window):
    """Locally-weighted Pearson r between two fields on a Gaussian window (sigma = window/4).
    The one shared metric behind fig3's local-r map and paper_stats' local-r numbers -- imported
    by both so they can never diverge. A Gaussian (vs a flat box) weighting avoids the blocky
    window artifacts that, once the swath is rotated, smear into diagonal stripes."""
    sigma = window / 4.0
    valid = (np.isfinite(a) & np.isfinite(b)).astype(float)
    af, bf = np.where(valid > 0, np.nan_to_num(a), 0.0), np.where(valid > 0, np.nan_to_num(b), 0.0)
    smooth = lambda x: gaussian_filter(x, sigma, mode="constant")
    count = smooth(valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = smooth(af) / count, smooth(bf) / count
        cov = smooth(af * bf) / count - ma * mb
        va, vb = smooth(af * af) / count - ma ** 2, smooth(bf * bf) / count - mb ** 2
        r = cov / np.sqrt(np.clip(va, 1e-9, None) * np.clip(vb, 1e-9, None))
    r[count < 0.3] = np.nan
    return r

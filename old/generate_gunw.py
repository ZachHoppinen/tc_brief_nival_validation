"""Generate the Mores GUNWs (20/40/60/80 m) by running isce3's NISAR InSAR workflow.

JPL's operational runconfig (the mountain_glaciers parameter set that made the
operational L2 GUNW) is used VERBATIM except for two things:
  1. resolution -> 20/40/60/80 m (crossmul + phase-unwrap looks, geocode posting)
  2. the geocode bounding box -> the Mores AOI

The RSLCs are CROPPED to the AOI in radar coordinates FIRST (each full frame ->
a small <name>_sub.h5), so every radar-domain step -- crossmul, filtering, and
especially snaphu phase-unwrapping -- runs on a ~10 km patch instead of the whole
~370x330 km frame. (Running snaphu on the full frame took >12 h and ~99 % of the
work was outside the AOI.) The crop comes from hyp3-isce3's `crop_rslc`, which
geo2rdrs the AOI corners into each acquisition's grid. The single cropped pair is
shared across all four resolutions -- only the multilook looks + geocode posting
change per level (the coherence-vs-resolution sweep).

The DEM is the PUBLIC Copernicus GLO-30 (via dem_stitcher, no credentials) over the AOI
(+ a buffer for the crop margin), on the WGS84 ellipsoid isce3 expects. The orbit /
ECMWF-troposphere / TEC / water-mask ancillary are self-fetched fresh from NASA Earthdata
(stage_ancillary.fetch_gunw_ancillary -> outputs/gunw/ancillary/) and override the
runconfig's paths; only the runconfig template (data/rslc/runconfig.rc.yaml) is hand-staged.
Each level's GUNW -> outputs/gunw/<level>/output/GUNW_product.h5.

    conda run -n nival python old/generate_gunw.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nival import paths   # noqa: E402

# multilook sweep for the LEGACY self-processing path only: the paper reads the operational
# L2 GUNW (paths.GUNW_OPERATIONAL), so nothing downstream depends on these levels.
# label -> (crossmul azimuth_looks, range_looks, geocode posting [m])
GUNW_LEVELS = {
    "20m": dict(az=4, rg=4, posting=20.0),
    "40m": dict(az=9, rg=8, posting=40.0),
    "60m": dict(az=13, rg=12, posting=60.0),
    "80m": dict(az=18, rg=15, posting=80.0),
}

SUBSET_DEM = paths.SUBSET_DEM                              # shared across all levels (also the scene DEM for LIA)
# cropped RSLCs live beside the full frames (data/rslc/) so they survive a run-dir wipe
RSLC_REF_SUB = paths.RSLC_DIR / f"{paths.RSLC_REF.stem}_sub.h5"
RSLC_SEC_SUB = paths.RSLC_DIR / f"{paths.RSLC_SEC.stem}_sub.h5"
CROP_LOOKS = 4             # crop-window snap (the finest level's looks); shared crop for every level
DEM_BUFFER_DEG = 0.1       # pad the AOI so the DEM covers the cropped swath (crop adds a 512-sample margin)

# run the full operational pipeline (this isce3 build needs an explicit run_steps);
# every step on -> ionosphere, troposphere, solid-earth tides all run as JPL had them
FULL_STEPS = dict(
    bandpass_insar=True, rdr2geo=True, geo2rdr=True, prepare_insar_hdf5=True,
    coarse_resample=True, dense_offsets=True, offsets_product=True, rubbersheet=True,
    fine_resample=True, crossmul=True, filter_interferogram=True, unwrap=True,
    geocode=True, ionosphere=True, troposphere=True, solid_earth_tides=True, baseline=True,
)


def run_level(level, looks, aoi_corners, anc, h5_prep, insar, InsarRunConfig):
    """Build the runconfig for one resolution and run the InSAR workflow on the subsets.
    `anc` is the self-fetched ancillary dict from stage_ancillary.fetch_gunw_ancillary()."""
    out_dir = paths.GUNW_DIR / level
    out_dir.mkdir(parents=True, exist_ok=True)
    top_left_easting, top_left_northing, bottom_right_easting, bottom_right_northing = aoi_corners

    # load JPL's operational runconfig fresh and bind each group we touch to a clear name
    config = yaml.safe_load(paths.OPERATIONAL_RUNCONFIG.read_text())
    groups = config["runconfig"]["groups"]
    input_files = groups["input_file_group"]
    ancillary = groups["dynamic_ancillary_file_group"]
    processing = groups["processing"]
    output_paths = groups["product_path_group"]

    # inputs: the CROPPED RSLC subsets + the AOI DEM (both shared across levels)
    input_files["reference_rslc_file"] = str(RSLC_REF_SUB)
    input_files["secondary_rslc_file"] = str(RSLC_SEC_SUB)
    ancillary["dem_file"] = str(SUBSET_DEM)
    # self-fetched ancillary -> override the template runconfig's stale paths
    ancillary["water_mask_file"] = str(anc["watermask"])
    ancillary["tec_file"] = str(anc["tec"])
    ancillary["orbit_files"]["reference_orbit_file"] = str(anc["ref_orbit"])
    ancillary["orbit_files"]["secondary_orbit_file"] = str(anc["sec_orbit"])
    ancillary["troposphere_weather_model_files"]["reference_troposphere_file"] = str(anc["ref_tropo"])
    ancillary["troposphere_weather_model_files"]["secondary_troposphere_file"] = str(anc["sec_tropo"])

    # this level's resolution: crossmul + phase-unwrap looks and geocode posting
    processing["crossmul"].update(azimuth_looks=looks["az"], range_looks=looks["rg"])
    processing["phase_unwrap"].update(azimuth_looks=looks["az"], range_looks=looks["rg"])
    processing["geocode"]["output_posting"]["A"] = dict(x_posting=looks["posting"], y_posting=looks["posting"])

    # geocode (and the radar-grid metadata cubes) the Mores AOI box (trims the crop margin)
    for geocoded in (processing["geocode"], processing["radar_grid_cubes"]):
        geocoded["top_left"] = dict(x_abs=top_left_easting, y_abs=top_left_northing)
        geocoded["bottom_right"] = dict(x_abs=bottom_right_easting, y_abs=bottom_right_northing)

    # outputs -> per-level dir; GUNW lands at <out_dir>/output/GUNW_product.h5, no move needed
    output_paths["product_path"] = str(out_dir / "output")
    output_paths["scratch_path"] = str(out_dir / "scratch")
    output_paths["sas_output_file"] = str(out_dir / "output" / "product.h5")
    output_paths["qa_output_dir"] = str(out_dir / "qa")
    groups["logging"]["path"] = str(out_dir / "scratch" / "insar.log")

    runconfig_path = out_dir / "runconfig.yaml"
    runconfig_path.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"running {level} (az{looks['az']} rg{looks['rg']}, {looks['posting']:.0f} m posting) ...")
    run_config = InsarRunConfig(argparse.Namespace(run_config_path=str(runconfig_path), log_file=False))
    _, isce3_output_paths = h5_prep.get_products_and_paths(run_config.cfg)
    insar.run(cfg=run_config.cfg, out_paths=isce3_output_paths, run_steps=FULL_STEPS)
    gunw = paths.GUNW_DIR / level / "output" / "GUNW_product.h5"
    print(f"  DONE -> {gunw}" if gunw.exists() else f"  ran but no GUNW at {gunw}")


def main():
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid                                # RAiDER (troposphere) still calls np.trapz
    import rasterio
    from dem_stitcher import stitch_dem
    from nisar.products.readers import SLC
    from nisar.workflows import h5_prep, insar
    from nisar.workflows.insar_runconfig import InsarRunConfig
    from hyp3_isce3.crop_rslc import aoi_to_radar_window, crop_rslc, get_polarizations
    from nival.stage_ancillary import fetch_gunw_ancillary

    paths.GUNW_DIR.mkdir(parents=True, exist_ok=True)
    west, south, east, north = paths.AOI

    # 1. stage a small DEM over the AOI (+ buffer) from the PUBLIC Copernicus GLO-30
    #    (dem_stitcher, no credentials), on the WGS84 ellipsoid isce3 wants
    if not SUBSET_DEM.exists():
        print("staging AOI Copernicus GLO-30 DEM (ellipsoid) ...")
        dem_bbox = [west - DEM_BUFFER_DEG, south - DEM_BUFFER_DEG, east + DEM_BUFFER_DEG, north + DEM_BUFFER_DEG]
        dem_array, dem_profile = stitch_dem(dem_bbox, dem_name="glo_30", dst_ellipsoidal_height=True)
        with rasterio.open(SUBSET_DEM, "w", **dem_profile) as dst:
            dst.write(dem_array, 1)

    # 2. crop each full RSLC frame to the AOI in RADAR coordinates -> a small <name>_sub.h5
    #    (once; the same cropped pair feeds every resolution). The window snap uses the finest
    #    looks; the geocode re-grids each level onto the UTM posting, so coarser levels align too.
    for source, subset in [(paths.RSLC_REF, RSLC_REF_SUB), (paths.RSLC_SEC, RSLC_SEC_SUB)]:
        if subset.exists():
            print(f"  skip (exists): {subset.name}")
            continue
        print(f"cropping {source.name} -> {subset.name} ...")
        slc = SLC(hdf5file=str(source))
        window = aoi_to_radar_window(slc, list(paths.AOI), SUBSET_DEM, az_looks=CROP_LOOKS, rg_looks=CROP_LOOKS)
        crop_rslc(source, subset, window, get_polarizations(slc))

    # 3. fetch the orbit / ECMWF troposphere / TEC / water-mask ancillary fresh from Earthdata
    #    (cached in outputs/gunw/ancillary/)
    print("staging GUNW ancillary (orbits / troposphere / TEC / water mask) ...")
    anc = fetch_gunw_ancillary()

    # 4. the Mores AOI as UTM 11N geocode corners (top-left = NW, bottom-right = SE)
    to_utm = Transformer.from_crs(4326, paths.EPSG, always_xy=True).transform
    top_left_easting, top_left_northing = to_utm(west, north)
    bottom_right_easting, bottom_right_northing = to_utm(east, south)
    aoi_corners = (top_left_easting, top_left_northing, bottom_right_easting, bottom_right_northing)

    # 5. run the InSAR workflow once per resolution on the shared subsets
    for level, looks in GUNW_LEVELS.items():
        run_level(level, looks, aoi_corners, anc, h5_prep, insar, InsarRunConfig)


if __name__ == "__main__":
    main()

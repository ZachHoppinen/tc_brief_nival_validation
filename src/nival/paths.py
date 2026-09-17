"""Central, repo-relative paths for the NIVAL validation repo.

Everything is anchored to the repo root (the parent of `src/`), so the repo is
self-contained and portable: move the folder, and every path still resolves.
No absolute paths into the original nisar_swe tree appear anywhere downstream.
"""

from __future__ import annotations

from pathlib import Path

# repo root = two levels up from this file (src/nival/paths.py -> repo/)
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"            # inputs
OUTPUTS = ROOT / "outputs"      # generated data products (GUNW, dSWE/dHS NetCDF)
FIGURES_DIR = ROOT / "figures"  # the rendered figure PNGs
PAPER_FIGURES_DIR = ROOT.parent / "nival_paper" / "figures"  # the TC paper repo (sibling checkout), if present

# --- generated outputs ---
GUNW_DIR = OUTPUTS / "gunw"          # legacy self-processed GUNWs (generate_gunw.py only)
PRODUCTS_DIR = OUTPUTS / "retrieval"  # retrieval.build_product writes dswe_dhs_operational.nc
SUBSET_DEM = GUNW_DIR / "dem_subset.tif"  # AOI Copernicus GLO-30 staged by generate_gunw;
#                                           the scene DEM for LIA too (retrieval + maps)

# --- input data (assumed present under data/) ---
RSLC_DIR = DATA / "rslc"        # the two full RSLC granules
LIDAR_DIR = DATA / "lidar"      # NIVAL snow-depth tifs
MET_DIR = DATA / "met"          # Freeman + Pilot on-site station CSVs
REMOTE_DIR = DATA / "remote"    # frozen SNOTEL / IONEX / NAIP / fire-perimeter snapshots

# --- the operational NISAR L2 GUNW for this pair ----------------------------
# ASF collection NISAR_L2_GUNW_PROVISIONAL_V1, PGE R05.02.3. This IS the analysis product:
# the paper evaluates the operational interferogram as delivered, not a self-processed one.
# 80 m unwrapped posting (crossmul 5x6, phase-unwrap 13x16), full frame -- retrieval clips
# the read to the AOI. Fetch with src/nival/download_gunw.py.
GUNW_OPERATIONAL = DATA / "gunw_operational" / (
    "NISAR_L2_PR_GUNW_012_077_A_024_013_4000_SH_"
    "20260207T124619_20260207T124654_20260219T124619_20260219T124654_P05023_N_F_J_001.h5")


def gunw_product() -> Path:
    """The operational L2 GUNW the whole analysis reads."""
    return GUNW_OPERATIONAL


def product() -> Path:
    """The dSWE/dHS common-grid NetCDF built from the operational GUNW."""
    return PRODUCTS_DIR / "dswe_dhs_operational.nc"


# --- lidar acquisitions bracketing the NISAR Feb7->Feb19 pair ---------------
# dHS = HS(Feb22) - HS(Feb07); Feb22 is the lidar flight closest to NISAR B (Feb19).
LIDAR_A07 = LIDAR_DIR / "NIVAL_MCS_Lidar_20260207_SD_v01.tif"
LIDAR_A22 = LIDAR_DIR / "NIVAL_MCS_Lidar_20260222_SD_v01.tif"
WORLDCOVER = REMOTE_DIR / "worldcover_class_mores.tif"
MTBS = REMOTE_DIR / "mtbs_severity_mores.tif"           # USFS MTBS burn-severity class (0-6)

# --- on-site met station CSVs (local-only; copied by hand into data/met/, not fetchable) ---
MET_PILOT = MET_DIR / "pilot_weather_clean.csv"          # Pilot Peak high-elev air temp (degF)
MET_SOIL = MET_DIR / "freeman_minfifteen_clean.csv"      # full Freeman export: air temp + SnoDAR
                                                         # depth + DTC string + SoilVUE + barometer

# --- frozen API snapshots (written by remote/fetch_snapshots.py; no network at figure time) ---
SNOTEL_SNAP = REMOTE_DIR / "snotel_637_06local.csv"      # SNOTEL 637 SWE/temp/depth at 06:00
SNOTEL_SEASON = REMOTE_DIR / "snotel_637_season.csv"     # SNOTEL 637 daily depth, whole WY2026
SEASON_D0, SEASON_D1 = "2025-10-01", "2026-04-30"        # the wetness census window
IONEX_DIR = REMOTE_DIR / "ionex"                         # IGS GIM .INX files (one per day-of-year)
BASEMAP_PNG = REMOTE_DIR / "naip_basemap.png"            # USGS NAIP RGB mosaic over the AOI
BASEMAP_META = REMOTE_DIR / "naip_basemap.json"          # its UTM extent
FIRE_GEOJSON = REMOTE_DIR / "pioneer_fire_2016.geojson"  # NIFC Pioneer Fire perimeter

# --- physics / geometry constants (Leinss 2015, NISAR L-band) ---------------
LAMBDA = 0.238          # NISAR L-band carrier wavelength (m)
EPSG = 32611            # UTM zone 11N (study-area projection)

# --- Mores Creek AOI (the radar-grid crop window for the SAS prep) -----------
AOI = (-115.745, 43.898, -115.620, 43.993)    # WGS84 (W, S, E, N)

# --- the InSAR pair: NISAR ASC-077 frame-024, Feb7 (ref) -> Feb19 (sec) ------
# full L1 RSLC granules, local-only: hard-copied by hand into data/rslc/ (26 GB each, not fetched).
RSLC_REF = RSLC_DIR / "NISAR_L1_PR_RSLC_012_077_A_024_4005_DHDH_A_20260207T124619_20260207T124654_X05013_N_F_J_001.h5"
RSLC_SEC = RSLC_DIR / "NISAR_L1_PR_RSLC_013_077_A_024_4005_DHDH_A_20260219T124619_20260219T124654_X05016_P_F_J_001.h5"

# self-fetched GUNW-processing ancillary (precise orbits / ECMWF troposphere / TEC / water
# mask), staged fresh from NASA Earthdata by stage_ancillary.fetch_gunw_ancillary() -- one
# sub-folder per role.
GUNW_ANC_DIR = GUNW_DIR / "ancillary"

# the master runconfig template -- a hand-staged input alongside the RSLCs (data/rslc/), used
# ONLY by the legacy generate_gunw.py self-processing path. The operational mountain_glaciers
# parameter set, taken from a different-track GUNW. The paper itself now reads the operational
# L2 GUNW for this pair (GUNW_OPERATIONAL) rather than any self-processed interferogram.
OPERATIONAL_RUNCONFIG = RSLC_DIR / "runconfig.rc.yaml"

# --- key dates (the figures annotate these) ---------------------------------
NISAR_A, NISAR_B = "20260207", "20260219"     # the InSAR pair acquisitions
LIDAR_DATES = ("20260207", "20260222")        # lidar flights nearest A and B

# --- met campaign window + on-site geometry (shared by the met figures and the
#     SNOTEL / IONEX ancillary fetches; dawn pass 12:46:36 UTC = 05:46 MST)
D0, D1 = "2026-02-01", "2026-03-01"
YEAR = 2026
SITE_LAT, SITE_LON = 43.93, -115.66

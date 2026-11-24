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

# --- generated outputs ---
GUNW_DIR = OUTPUTS / "gunw"          # holds the scene DEM (SUBSET_DEM)
PRODUCTS_DIR = OUTPUTS / "retrieval"  # retrieval.build_product writes dswe_dhs_operational.nc
SUBSET_DEM = GUNW_DIR / "dem_subset.tif"  # AOI NISAR DEM staged by download_paper_data;
#                                           the scene DEM for LIA too (retrieval + maps)

# --- input data (assumed present under data/) ---
LIDAR_DIR = DATA / "lidar"      # NIVAL snow-depth tifs
MET_DIR = DATA / "met"          # Freeman + Pilot on-site station CSVs
REMOTE_DIR = DATA / "remote"    # frozen SNOTEL / IONEX / NAIP / fire-perimeter snapshots

# --- the operational NISAR L2 GUNW for this pair ----------------------------
# ASF collection NISAR_L2_GUNW_PROVISIONAL_V1, PGE R05.02.3. This IS the analysis product:
# the paper evaluates the operational interferogram as delivered, not a self-processed one.
# 80 m unwrapped posting (crossmul 5x6, phase-unwrap 13x16), full frame -- retrieval clips
# the read to the AOI. Fetch with src/nival/download_paper_data.py.
GUNW_OPERATIONAL = DATA / "gunw_operational" / (
    "NISAR_L2_PR_GUNW_012_077_A_024_013_4000_SH_"
    "20260207T124619_20260207T124654_20260219T124619_20260219T124654_P05023_N_F_J_001.h5")


# GUNW crops we produced ourselves from refocused RSLCs, over the lidar domain + 10 km. They
# cannot be downloaded: copied by hand into data/gunw_refocused/. Fig. 3 reads the two track-077 ones.
GUNW_REFOCUSED = DATA / "gunw_refocused"


# Radar-geometry HH amplitude windows over the AOI, cut from production RSLCs (streamed from
# ASF) and from our refocused RSLC (on our S3 bucket). Staged by hand in data/rslc_windows/ as
# <track>_<date>_{production,refocus}.npz; Fig. S1 reads them.
RSLC_WINDOWS = DATA / "rslc_windows"
REFOCUS_EXAMPLE = ("077", "20260114")            # the acquisition Fig. S1 shows as production and as refocused
REFOCUS_STANDARD = ("20251221", "20260207")      # production RSLCs of the same track averaged as the usual amplitude
L0B_STAMPS = DATA / "l0b_stamps"                 # pulse time stamps of the track-077 raw (L0B) granules, one npz per granule
STAMPS_FAULTY = L0B_STAMPS / "NISAR_L0_PR_RRSD_010_077_A_156S_20260114T124024_20260114T125436_P05023_F_J_002.npz"
STAMPS_CLEAN = L0B_STAMPS / "NISAR_L0_PR_RRSD_012_077_A_156S_20260207T124025_20260207T125437_P05023_F_J_002.npz"
STAMPS_CONTROL = L0B_STAMPS / "NISAR_L0_PR_RRSD_008_077_A_156S_20251221T124023_20251221T125435_P05023_F_J_002.npz"   # Dec 21, a second clean date
STAMP_SHIFT = 2169                               # rows the Jan 14 stamps were moved back before refocusing
FRAME_UTC = (45978.0, 46013.0)                   # frame 024 on Jan 14, 12:46:18 to 12:46:53 UTC [s of day]


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
SNOTEL_PTEMP = REMOTE_DIR / "snotel_637_ptemp.csv.gz"    # SNOTEL 637 hourly snow temperature profile (degC)
SNOTEL_AIR_TEMP = REMOTE_DIR / "snotel_637_air_temp.csv"   # SNOTEL 637 hourly air temperature (degC, UTC)
SEASON_D0, SEASON_D1 = "2025-10-01", "2026-04-30"        # the wetness census window
IONEX_DIR = REMOTE_DIR / "ionex"                         # IGS GIM .INX files (one per day-of-year)
BASEMAP_PNG = REMOTE_DIR / "naip_basemap.png"            # USGS NAIP RGB mosaic over the AOI
BASEMAP_META = REMOTE_DIR / "naip_basemap.json"          # its UTM extent
FIRE_GEOJSON = REMOTE_DIR / "pioneer_fire_2016.geojson"  # NIFC Pioneer Fire perimeter

# --- physics / geometry constants (Leinss 2015, NISAR L-band) ---------------
LAMBDA = 299_792_458 / 1.239e9   # wavelength (m) at the GUNW frequencyA centre frequency, 1239 MHz: 0.242 m
EPSG = 32611            # UTM zone 11N (study-area projection)

# --- Mores Creek AOI (the radar-grid crop window for the SAS prep) -----------
AOI = (-115.745, 43.898, -115.620, 43.993)    # WGS84 (W, S, E, N)

# --- key dates (the figures annotate these) ---------------------------------
FIG3_LAST = "20260315"                        # Fig. 3 runs through the pair ending on this date
NISAR_A, NISAR_B = "20260207", "20260219"     # the InSAR pair acquisitions
LIDAR_DATES = ("20260207", "20260222")        # lidar flights nearest A and B

# --- met campaign window + on-site geometry (shared by the met figures and the
#     SNOTEL / IONEX ancillary fetches; dawn pass 12:46:36 UTC = 05:46 MST)
D0, D1 = "2026-02-01", "2026-03-01"
YEAR = 2026
SITE_LAT, SITE_LON = 43.93, -115.66

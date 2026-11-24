"""Fetch every remote input the paper needs, fresh, into data/.

One script, run once. Pulls:
  NIVAL lidar snow depth  -- NSIDC NIVAL_MCS_Lidar  (DOI 10.5067/DPFDH2M49DQG)
  SNOTEL 637              -- USDA AWDB REST          -> snotel_637_06local.csv
  ionosphere (IGS GIM)    -- CDDIS                   -> ionex/IGS0OPSFIN_*.INX
  Pioneer Fire perimeter  -- NIFC                    -> pioneer_fire_2016.geojson
  WorldCover 2021         -- ESA (AWS)               -> worldcover_class_mores.tif
  MTBS burn severity      -- USFS ImageServer        -> mtbs_severity_mores.tif
  NAIP aerial basemap     -- USGS ImageServer        -> naip_basemap.png
  NISAR DEM               -- earthaccess (NISAR_DEM)  -> outputs/gunw/dem_subset.tif
  operational L2 GUNW     -- ASF (2.5 GB)            -> data/gunw_operational/
  Fig. 3 pair GUNWs       -- ASF (2.3 GB each): every production GUNW of track 077 from
                             Nov 3 to Mar 15         -> data/gunw_operational/
  Fig. 3 pair lidar       -- NSIDC (0.3 GB each): the flights bracketing every track-077
                             pair in pairs.py        -> data/lidar/
  SNOTEL 637 season       -- USDA AWDB REST          -> snotel_637_season.csv

Fig. 3 also reads two Hyp3-Isce3 GUNW crops in data/gunw_refocused/ (Dec 21 to Jan 14 and
Jan 14 to Feb 7). We produced those ourselves, so they are staged by hand like the met CSVs.

The GUNW is CMR collection NISAR_L2_GUNW_PROVISIONAL_V1: the analysis product, evaluated as
delivered. Needs a NASA Earthdata Login (earthaccess + ~/.netrc for CDDIS).
Idempotent: existing files are skipped.

    conda run -n nival python src/nival/download_paper_data.py
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import earthaccess
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the src/ dir
from nival import paths   # noqa: E402
from nival.pairs import on_track   # noqa: E402

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
SNOTEL_STATION = "637:ID:SNTL"
GUNW_COLLECTION = "NISAR_L2_GUNW_PROVISIONAL_V1"


# ------------------------------ NSIDC rasters ------------------------------
def fetch_nsidc(short_name, must_contain, dst, count=400):
    """Download the ONE granule file whose name contains all `must_contain` substrings from
    NSIDC collection `short_name`, saving it to `dst`. Targeted single-file download: each NIVAL
    granule bundles SD/DTM/DSM/CHM (~1.2 GB), so we GET only the matched URL via earthaccess's
    authenticated session (Earthdata netrc login) instead of pulling the whole granule."""
    if dst.exists():
        print(f"  skip (exists): {dst.relative_to(paths.ROOT)}")
        return
    earthaccess.login()                                         # ensure the store/session exists (idempotent)
    urls = [url for g in earthaccess.search_data(short_name=short_name, count=count)
            for url in g.data_links() if all(s in url.split("/")[-1] for s in must_contain)]
    if not urls:
        print(f"  NOT FOUND in {short_name}: {must_contain}")
        return
    url = urls[0]
    print(f"  downloading {url.split('/')[-1]} -> {dst.relative_to(paths.ROOT)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    session = earthaccess.get_requests_https_session()          # reuses the earthaccess Earthdata login
    tmp = dst.with_suffix(dst.suffix + ".part")
    with session.get(url, stream=True) as response:
        response.raise_for_status()
        with open(tmp, "wb") as out:
            for chunk in response.iter_content(1 << 20):        # stream in 1 MB chunks
                out.write(chunk)
    tmp.replace(dst)                                            # atomic: only a complete download lands at dst
    print(f"  wrote {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)")


# ------------------------------ SNOTEL 637 ------------------------------
def _awdb(code, duration):
    data = _get_json(f"{AWDB}/data?stationTriplets={SNOTEL_STATION}&elements={code}"
                     f"&duration={duration}&beginDate={paths.D0}&endDate={paths.D1}")[0]["data"][0]["values"]
    return data


def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_snotel():
    if paths.SNOTEL_SNAP.exists():
        print(f"  skip (exists): {paths.SNOTEL_SNAP.name}")
        return
    daily = lambda code: {dt.date.fromisoformat(v["date"]): v["value"] for v in _awdb(code, "DAILY")}
    at_0600 = lambda code: {dt.date.fromisoformat(v["date"][:10]): v["value"]
                            for v in _awdb(code, "HOURLY")
                            if v["date"].endswith(" 06:00") and v["value"] is not None}
    swe_daily, swe_0600 = daily("WTEQ"), at_0600("WTEQ")
    temp_0600, depth_daily = at_0600("TOBS"), daily("SNWD")
    rows = [dict(date=d, swe_in_daily=swe_daily.get(d), swe_in_06=swe_0600.get(d),
                 tobs_F_06=temp_0600.get(d), snwd_in_daily=depth_daily.get(d))
            for d in sorted(swe_daily)]
    paths.SNOTEL_SNAP.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.SNOTEL_SNAP, index=False)
    print(f"  wrote {paths.SNOTEL_SNAP.name} ({len(rows)} days)")


# ------------------------------ ionosphere ------------------------------
def fetch_ionex():
    paths.IONEX_DIR.mkdir(parents=True, exist_ok=True)
    doy0, doy1 = (dt.date.fromisoformat(x).timetuple().tm_yday for x in (paths.D0, paths.D1))
    cookie = paths.IONEX_DIR / "cddis_cookie"
    for doy in range(doy0, doy1 + 1):
        name = f"IGS0OPSFIN_{paths.YEAR}{doy:03d}0000_01D_02H_GIM.INX"
        inx, gz = paths.IONEX_DIR / name, paths.IONEX_DIR / (name + ".gz")
        if inx.exists():
            continue
        url = f"https://cddis.nasa.gov/archive/gnss/products/ionex/{paths.YEAR}/{doy:03d}/{name}.gz"
        subprocess.run(["curl", "-s", "-L", "-n", "-c", str(cookie), "-b", str(cookie),
                        "-o", str(gz), url], check=True)
        with gzip.open(gz, "rb") as fin, open(inx, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz.unlink()
        print(f"  ionex doy {doy}")


# ------------------------------ study-area map layers ------------------------------
def fetch_fire():
    """Pioneer Fire (2016) perimeter from the NIFC interagency history service -> geojson."""
    if paths.FIRE_GEOJSON.exists():
        print(f"  skip (exists): {paths.FIRE_GEOJSON.name}")
        return
    url = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
           "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query")
    params = dict(where="INCIDENT='Pioneer' AND FIRE_YEAR='2016'", outFields="INCIDENT",
                  returnGeometry="true", outSR="4326", f="geojson")
    geojson = requests.get(url, params=params, timeout=120).json()
    paths.FIRE_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    paths.FIRE_GEOJSON.write_text(json.dumps(geojson))
    print(f"  wrote {paths.FIRE_GEOJSON.name} ({len(geojson.get('features', []))} features)")


def fetch_worldcover():
    """ESA WorldCover 2021 (10 m) clipped to the AOI -> class raster (class 10 = trees)."""
    if paths.WORLDCOVER.exists():
        print(f"  skip (exists): {paths.WORLDCOVER.name}")
        return
    import rioxarray  # noqa: F401
    tile = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            "ESA_WorldCover_10m_2021_v200_N42W117_Map.tif")     # 3-deg tile covering Mores
    worldcover = rioxarray.open_rasterio(f"/vsicurl/{tile}").squeeze().rio.clip_box(*paths.AOI)
    paths.WORLDCOVER.parent.mkdir(parents=True, exist_ok=True)
    worldcover.rio.to_raster(paths.WORLDCOVER)
    print(f"  wrote {paths.WORLDCOVER.name}")


def fetch_dem():
    """NISAR DEM over the AOI (+0.1 deg buffer) -> the scene DEM. This is the DEM the NISAR products are
    processed against (collection NISAR_DEM on Earthdata): ellipsoidal heights, pixel centres on whole arc-seconds."""
    if paths.SUBSET_DEM.exists():
        print(f"  skip (exists): {paths.SUBSET_DEM.name}")
        return
    import rasterio
    from rasterio.merge import merge
    from rasterio.windows import from_bounds
    west, south, east, north = paths.AOI
    bounds = (west - 0.1, south - 0.1, east + 0.1, north + 0.1)
    earthaccess.login()
    tiles = earthaccess.download(earthaccess.search_data(short_name="NISAR_DEM", bounding_box=bounds),
                                 str(paths.REMOTE_DIR / "nisar_dem"))
    # mosaic the geographic 1-degree tiles on their own pixel lattice, then cut on whole pixels: no height is resampled or moved
    sources = [rasterio.open(t) for t in sorted(str(t) for t in tiles if str(t).endswith(".tif"))]
    sources = [src for src in sources if src.crs.to_epsg() == 4326]
    mosaic, transform = merge(sources)
    window = from_bounds(*bounds, transform=transform).round_offsets().round_lengths()
    rows, cols = window.toslices()
    dem = mosaic[0][rows, cols]
    profile = sources[0].profile | dict(height=dem.shape[0], width=dem.shape[1], driver="GTiff",
                                        transform=rasterio.windows.transform(window, transform))
    if not (dem.min() > -500 and dem.max() < 9000):
        raise ValueError("NISAR DEM heights are not plausible in metres; check units")
    paths.SUBSET_DEM.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(paths.SUBSET_DEM, "w", **profile) as dst:
        dst.write(dem, 1)
    print(f"  wrote {paths.SUBSET_DEM.name}")


def fetch_mtbs():
    """USFS MTBS thematic burn severity clipped to the AOI -> class raster.
    0=background, 1=unburned-low, 2=low, 3=moderate, 4=high, 5=greenness, 6=mask.
    Pulled live from the USFS MTBS CONUS ImageServer exportImage (no auth)."""
    if paths.MTBS.exists():
        print(f"  skip (exists): {paths.MTBS.name}")
        return
    from pyproj import Transformer
    w, s, e, n = paths.AOI                                  # WGS84 AOI -> UTM bounds (all 4 corners)
    to_utm = Transformer.from_crs(4326, paths.EPSG, always_xy=True).transform
    es, ns = zip(*[to_utm(a, b) for a in (w, e) for b in (s, n)])
    minx, maxx, miny, maxy = min(es), max(es), min(ns), max(ns)
    nx, ny = int((maxx - minx) / 30) + 1, int((maxy - miny) / 30) + 1   # ~30 m native MTBS posting
    url = ("https://imagery.geoplatform.gov/iipp/rest/services/"
           "Fire_Aviation/USFS_EDW_MTBS_CONUS/ImageServer/exportImage")
    params = dict(bbox=f"{minx},{miny},{maxx},{maxy}", bboxSR=paths.EPSG, imageSR=paths.EPSG,
                  size=f"{nx},{ny}", format="tiff", f="image", interpolation="RSP_NearestNeighbor")
    response = requests.get(url, params=params, timeout=120); response.raise_for_status()
    paths.MTBS.parent.mkdir(parents=True, exist_ok=True)
    paths.MTBS.write_bytes(response.content)
    print(f"  wrote {paths.MTBS.name}")


def fetch_naip():
    """USGS NAIP aerial over the study-area frame -> cached png (imported lazily to avoid
    pulling cartopy/matplotlib when only the other data is wanted)."""
    from nival import paper_figures
    paper_figures.fetch_naip(paper_figures.map_frame()["extent"])


# ------------------------------ operational GUNW ------------------------------
def fetch_gunw(dst=None):
    """Download the one GUNW granule named by paths.GUNW_OPERATIONAL, streaming it to disk."""
    dst = Path(dst) if dst else paths.GUNW_OPERATIONAL
    if dst.exists():
        print(f"  skip (exists): {dst.relative_to(paths.ROOT)}")
        return dst

    earthaccess.login()                                         # Earthdata netrc login (idempotent)
    # granule_name is a CMR pattern match, so the trailing * covers a granule_ur
    # that carries the .h5 extension as well as one that does not
    results = earthaccess.search_data(short_name=GUNW_COLLECTION, granule_name=f"{dst.stem}*", count=10)
    urls = [url for g in results for url in g.data_links() if url.endswith(dst.name)]
    if not urls:
        raise SystemExit(f"NOT FOUND in {GUNW_COLLECTION}: {dst.name}")

    print(f"  downloading {dst.name} -> {dst.relative_to(paths.ROOT)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    session = earthaccess.get_requests_https_session()          # reuses the earthaccess login
    tmp = dst.with_suffix(dst.suffix + ".part")
    with session.get(urls[0], stream=True) as response:
        response.raise_for_status()
        with open(tmp, "wb") as out:
            for chunk in response.iter_content(1 << 20):        # stream in 1 MB chunks
                out.write(chunk)
    tmp.replace(dst)                                            # atomic: only a complete file lands at dst
    print(f"  wrote {dst.name} ({dst.stat().st_size / 1e9:.1f} GB)")
    return dst


# ------------------------------ the pairs of Fig. 3 and the census ------------------------------
TRACK = "077"                  # the paper's track; ascending, frame 024 covers Mores Creek


def gunw_pattern(pair):
    """Granule-name pattern of the operational GUNW for one (track, ref, sec, ...) pair."""
    track, ref, sec = pair[:3]
    return f"NISAR_L2_PR_GUNW_*_{track}_A_024_*_{ref}T*_{sec}T*"


def gunw_file(pair):
    """The pair's GUNW on disk, or None if it has not been downloaded."""
    found = sorted(paths.GUNW_OPERATIONAL.parent.glob(gunw_pattern(pair) + ".h5"))
    return found[-1] if found else None


def lidar_file(date):
    """Where the NIVAL snow-depth raster of one flight date lives."""
    return paths.LIDAR_DIR / f"NIVAL_MCS_Lidar_{date}_SD_v01.tif"


def pair_of(name):
    """(track, ref date, sec date) parsed from a GUNW granule or file name."""
    parts = Path(name).name.split("_")
    return parts[5], parts[11][:8], parts[13][:8]


def in_window(sec):
    """True if a pair's second date falls inside the census window in paths.py."""
    return sec <= paths.SEASON_D1.replace("-", "")


def local_gunws():
    """Every GUNW on disk whose pair ends inside the census window, as
    (track, ref, sec, path, refocused) in track then date order. A refocused crop in
    data/gunw_refocused/ replaces the production GUNW of the same pair, and adds the pairs
    (all of track 114) that have no production GUNW."""
    found = {pair_of(f): (f, False) for f in paths.GUNW_OPERATIONAL.parent.glob("NISAR_L2_PR_GUNW_*.h5")}
    found.update({pair_of(f): (f, True) for f in paths.GUNW_REFOCUSED.glob("*_crop.h5")})
    return sorted((*pair, *value) for pair, value in found.items() if in_window(pair[2]))


def pair_gunw(pair):
    """The GUNW the paper uses for one (track, ref, sec, ...) pair: our refocused one where
    there is one, otherwise the production one. None if neither is on disk."""
    return next((g[3] for g in local_gunws() if g[:3] == tuple(pair[:3])), None)


def fetch_gunws(tracks=None, last=None):
    """Every production GUNW on ASF that covers the AOI and whose pair ends inside the census
    window. `tracks` keeps only those tracks and `last` ("YYYYMMDD") only pairs ending by that
    date. Granule names carry processing stamps we cannot predict, so they are found by a
    spatial and temporal search."""
    earthaccess.login()
    results = earthaccess.search_data(short_name=GUNW_COLLECTION, bounding_box=paths.AOI,
                                      temporal=(paths.SEASON_D0, paths.SEASON_D1), count=2000)
    latest = {}
    for name in sorted(g["umm"]["GranuleUR"].removesuffix(".h5") for g in results):
        track, _, sec = pair_of(name)
        if in_window(sec) and (tracks is None or track in tracks) and (last is None or sec <= last):
            latest[pair_of(name)] = name              # sorted, so a later product counter replaces an earlier one
    print(f"  {len(latest)} production GUNWs over the AOI, tracks {sorted({k[0] for k in latest})}")
    for pair, name in sorted(latest.items()):
        if gunw_file(pair):
            print(f"  skip (exists): {gunw_file(pair).name}")
        else:
            fetch_gunw(paths.GUNW_OPERATIONAL.parent / f"{name}.h5")


def fetch_pair_lidar():
    """The two NIVAL snow-depth rasters bracketing every census pair."""
    dates = sorted({date for pair in on_track(TRACK) for date in pair[3:5]})
    for date in dates:
        fetch_nsidc("NIVAL_MCS_Lidar", [date, "SD", ".tif"], lidar_file(date))


def fetch_snotel_season():
    """SNOTEL 637 daily snow depth and SWE across the whole water year.

    fetch_snotel() freezes only the February window the headline pair sits in; Fig. 3 and
    the wetness census need the whole season.
    """
    if paths.SNOTEL_SEASON.exists():
        print(f"  skip (exists): {paths.SNOTEL_SEASON.name}")
        return
    def series(code):
        url = (f"{AWDB}/data?stationTriplets={SNOTEL_STATION}&elements={code}&duration=DAILY"
               f"&beginDate={paths.SEASON_D0}&endDate={paths.SEASON_D1}")
        return {v["date"]: v["value"] for v in _get_json(url)[0]["data"][0]["values"]}

    depth, swe = series("SNWD"), series("WTEQ")
    rows = [dict(date=d, snwd_in=depth.get(d), wteq_in=swe.get(d)) for d in sorted(depth)]
    paths.SNOTEL_SEASON.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.SNOTEL_SEASON, index=False)
    print(f"  wrote {paths.SNOTEL_SEASON.name} ({len(rows)} days)")


def fetch_snotel_air_temp():
    """SNOTEL 637 hourly air temperature across the whole water year, in degrees C with UTC times."""
    if paths.SNOTEL_AIR_TEMP.exists():
        print(f"  skip (exists): {paths.SNOTEL_AIR_TEMP.name}")
        return
    url = (f"{AWDB}/data?stationTriplets={SNOTEL_STATION}&elements=TOBS&duration=HOURLY"
           f"&beginDate={paths.SEASON_D0}&endDate={paths.SEASON_D1}")
    series = _get_json(url)[0]["data"][0]
    assert series["stationElement"]["storedUnitCode"] == "degF"
    degf = pd.Series({v["date"]: v.get("value") for v in series["values"]}, dtype=float)
    # the station reports in UTC-8 all year (its dataTimeZone); shift to UTC and degF -> degC at ingest
    degc = (degf - 32.0) * 5.0 / 9.0
    degc.index = pd.to_datetime(degc.index) + pd.Timedelta(hours=8)
    if not (degc.min() > -50 and degc.max() < 45):
        raise ValueError("SNOTEL air temperature is not plausible in degrees C; check units")
    paths.SNOTEL_AIR_TEMP.parent.mkdir(parents=True, exist_ok=True)
    degc.rename("air_temp_c").rename_axis("time_utc").to_csv(paths.SNOTEL_AIR_TEMP)
    print(f"  wrote {paths.SNOTEL_AIR_TEMP.name} ({len(degc)} hours)")


def main():
    earthaccess.login()
    print("NIVAL lidar snow depth (Feb7 + Feb22)...")
    fetch_nsidc("NIVAL_MCS_Lidar", ["20260207", "SD", ".tif"], paths.LIDAR_A07)
    fetch_nsidc("NIVAL_MCS_Lidar", ["20260222", "SD", ".tif"], paths.LIDAR_A22)
    print("SNOTEL 637...")
    fetch_snotel()
    print("ionosphere (IGS GIM)...")
    fetch_ionex()
    print("Pioneer Fire perimeter (NIFC)...")
    fetch_fire()
    print("WorldCover (ESA)...")
    fetch_worldcover()
    print("NISAR DEM (Earthdata)...")
    fetch_dem()
    print("MTBS burn severity (USFS)...")
    fetch_mtbs()
    print("NAIP aerial basemap (USGS)...")
    fetch_naip()
    print("operational L2 GUNW (ASF)...")
    fetch_gunw()
    print(f"operational GUNWs for the track-{TRACK} pairs of Fig. 3 (ASF)...")
    fetch_gunws(tracks=(TRACK,), last=paths.FIG3_LAST)
    print(f"NIVAL lidar snow depth for the track-{TRACK} pairs (NSIDC)...")
    fetch_pair_lidar()
    print("SNOTEL 637 season depth + SWE...")
    fetch_snotel_season()
    print("done.")


if __name__ == "__main__":
    main()

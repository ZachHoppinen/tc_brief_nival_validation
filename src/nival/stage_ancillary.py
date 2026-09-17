"""Fetch all the remote ancillary data the pipeline needs, fresh, into data/.

One script, run once. Pulls:
  NIVAL lidar snow depth  -- NSIDC NIVAL_MCS_Lidar  (DOI 10.5067/DPFDH2M49DQG)
  SNOTEL 637              -- USDA AWDB REST          -> snotel_637_06local.csv
  ionosphere (IGS GIM)    -- CDDIS                   -> ionex/IGS0OPSFIN_*.INX
  Pioneer Fire perimeter  -- NIFC                    -> pioneer_fire_2016.geojson
  WorldCover 2021         -- ESA (AWS)               -> worldcover_class_mores.tif
  MTBS burn severity      -- USFS ImageServer        -> mtbs_severity_mores.tif
  NAIP aerial basemap     -- USGS ImageServer        -> naip_basemap.png
  GUNW ancillary          -- NASA Earthdata (hyp3)   -> outputs/gunw/ancillary/<role>/
                             (precise orbits, ECMWF troposphere, TEC, water mask)

Needs a NASA Earthdata Login (earthaccess + ~/.netrc for CDDIS / GES DISC).
Idempotent: existing files are skipped.

    conda run -n nival python src/nival/stage_ancillary.py
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
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

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
SNOTEL_STATION = "637:ID:SNTL"


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


def fetch_snotel_season():
    """SNOTEL 637 daily snow depth across the whole water year.

    fetch_snotel() freezes only the February window the headline pair sits in; the
    supplement's wetness census needs every acquisition from November to March.
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
    from nival import ancillary_plotting
    ancillary_plotting.fetch_naip(ancillary_plotting.map_frame()["extent"])


def fetch_gunw_ancillary():
    """Fetch the GUNW-processing ancillary the SAS needs -- precise orbits, ECMWF troposphere,
    TEC, water mask -- fresh from NASA Earthdata (the same earthaccess login as the rest), into
    outputs/gunw/ancillary/<role>/ (cached per role). Idempotent: a
    role whose folder already holds a file is skipped. Returns {role: Path} so generate_gunw can
    override the runconfig template's stale paths."""
    from hyp3_isce3.process import get_orbit, get_tec, get_tropo, get_watermask
    earthaccess.login()          # the get_* fetchers need an Earthdata session; self-contained so a
    #                              cold generate_gunw (which calls this directly) works, not just main()
    ref, sec = paths.RSLC_REF.stem, paths.RSLC_SEC.stem
    # role -> a zero-arg fetch (each get_* downloads into the cwd and returns the filename)
    jobs = {"ref_orbit": lambda: get_orbit(ref), "sec_orbit": lambda: get_orbit(sec),
            "ref_tropo": lambda: get_tropo(ref), "sec_tropo": lambda: get_tropo(sec),
            "tec": lambda: get_tec(ref),
            "watermask": lambda: get_watermask(str(paths.RSLC_REF), list(paths.AOI))}
    # each role's file lands under <folder>/data/<hash>/ (the get_* download pattern); the moved
    # troposphere seeds sit directly in <folder>. rglob the role's extension finds either.
    ext = {"ref_orbit": "*.xml", "sec_orbit": "*.xml", "ref_tropo": "*.nc", "sec_tropo": "*.nc",
           "tec": "*.json"}
    resolved = {}
    for role, fetch in jobs.items():
        folder = paths.GUNW_ANC_DIR / role
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.iterdir()):
            print(f"  skip (exists): {role}")
        else:
            cwd = Path.cwd()
            try:
                os.chdir(folder)                       # the get_* functions download into the cwd
                fetch()
            finally:
                os.chdir(cwd)
            print(f"  fetched {role}")
        if role == "watermask":
            # get_watermask stages a hierarchical VRT tree (EPSG4326.vrt -> EPSG4326/N40_50/
            # band.vrt -> ...) but only materializes the flat AOI .tif tiles, not the nested
            # sub-VRTs -- so handing isce3's geocode any of those top VRTs makes its gdal.Warp
            # fail on a missing nested source. Build one flat VRT directly over the downloaded
            # tiles instead: self-contained, references only files that exist, warps cleanly.
            from osgeo import gdal
            tiles = sorted(str(p) for p in folder.rglob("*.tif"))
            flat_vrt = folder / "watermask.vrt"
            gdal.BuildVRT(str(flat_vrt), tiles)
            resolved[role] = flat_vrt
        else:
            matches = sorted(folder.rglob(ext[role]))
            resolved[role] = matches[0] if matches else next(p for p in sorted(folder.rglob("*")) if p.is_file())
    return resolved


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
    print("MTBS burn severity (USFS)...")
    fetch_mtbs()
    print("NAIP aerial basemap (USGS)...")
    fetch_naip()
    print("GUNW ancillary (orbits / ECMWF troposphere / TEC / water mask)...")
    fetch_gunw_ancillary()
    print("done.")


if __name__ == "__main__":
    main()

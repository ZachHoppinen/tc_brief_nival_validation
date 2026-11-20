"""Fetch what the supplement needs beyond the main paper's data, fresh, into data/.

  SNOTEL 637 snow temperature profile -- USDA AWDB REST, no login: hourly PTEMP at every
                                         sensor height        -> remote/snotel_637_ptemp.csv.gz

The Freeman profile, the other station the wetness census reads, is in the tracked
data/met/ CSVs and needs no download. Idempotent: an existing file is skipped.

    conda run -n nival python src/nival/supplement/download_supplement_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the src/ dir
from nival import paths   # noqa: E402
from nival.download_paper_data import AWDB, SNOTEL_STATION, _get_json   # noqa: E402

IN_TO_M = 0.0254
def fetch_snotel_profile():
    """SNOTEL 637 hourly snow temperature profile in degrees C; one column per sensor,
    named by its height above ground in metres."""
    if paths.SNOTEL_PTEMP.exists():
        print(f"  skip (exists): {paths.SNOTEL_PTEMP.name}")
        return
    # `PTEMP` alone returns no rows: the height wildcard is required
    url = (f"{AWDB}/data?stationTriplets={SNOTEL_STATION}&elements=PTEMP:*:*&duration=HOURLY"
           f"&beginDate={paths.SEASON_D0}&endDate={paths.SEASON_D1}"
           "&returnFlags=false&returnOriginalValues=false&returnSuspectData=false")
    columns = {}
    for series in _get_json(url)[0]["data"]:
        element = series["stationElement"]
        assert element["storedUnitCode"] == "degF"
        degf = pd.Series({v["date"]: v.get("value") for v in series["values"]}, dtype=float)
        # convert at ingest: inches -> metres for the sensor height, degF -> degC for the reading
        columns[round(element["heightDepth"] * IN_TO_M, 4)] = (degf - 32.0) * 5.0 / 9.0
    profile = pd.DataFrame(columns).sort_index(axis=1)
    profile.index = pd.to_datetime(profile.index)
    profile = profile.sort_index()
    # sanity check on the converted values; 99.0 C is the API's unflagged fill code
    # (210.2 degF), which the readers mask along with anything outside -50..65 C
    real = profile.where(profile != 99.0)
    if not (real.min().min() > -60 and real.max().max() < 70):
        raise ValueError("SNOTEL profile temperatures are not plausible in degrees C; check units")
    paths.SNOTEL_PTEMP.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(paths.SNOTEL_PTEMP)
    print(f"  wrote {paths.SNOTEL_PTEMP.name} ({len(profile)} hours, {profile.shape[1]} sensors, "
          f"{profile.columns.min():.2f} to {profile.columns.max():.2f} m)")


def main():
    print("SNOTEL 637 snow temperature profile...")
    fetch_snotel_profile()
    print("done.")


if __name__ == "__main__":
    main()

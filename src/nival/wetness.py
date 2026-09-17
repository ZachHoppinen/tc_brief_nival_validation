"""Snowpack wetness at each NISAR acquisition, from the two instrumented profiles.

A snowpack holding liquid water sits at the melting point, so wetness is read straight off
the temperature profile. Two details matter:

* The snow-ground interface is held near 0 C all winter by ground heat, so a criterion
  that includes the basal layer calls almost every acquisition wet. We look only at the
  top ``TOP_M`` of the pack, which is also the part an L-band wave interacts with. The
  classification is unchanged for any depth between 0.2 and 0.6 m.
* Both records are read at their native hourly step and sampled at the overpass, not
  averaged over the day: what matters is liquid water present when the radar observes.

SNOTEL 637 sits at 1880 m and Freeman at 2392 m, near the bottom and top of the scene's
1477 to 2451 m range, so a wet reading at either places the wet line inside the scene.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nival import paths   # noqa: E402

MELT_TOL_C = 0.1          # sensor resolution either side of the melting point
TOP_M = 0.5               # depth below the snow surface the criterion looks at
SURFACE_SKIP_M = 0.05     # the top few cm track air, not the pack: analysis starts below this
PLAUSIBLE_C = (-50.0, 65.0)

#: dawn overpass in local time; track 114 images the previous evening
OVERPASS_LOCAL_H = {"077": 5 + 46 / 60, "149": 5 + 37 / 60, "114": 19 + 46 / 60}
EVENING_TRACKS = {"114"}

SNOTEL_ELEV_M, FREEMAN_ELEV_M = 1880.0, 2392.0
SNOTEL_PROFILE = Path.home() / "Documents/nisar_swe/data/snotels/ptemp/637_ID.csv.gz"

# Freeman DTC string: 4 in spacing, ground at sensor 35, sensor 20 reads a constant 0 C
DTC_SPACING_M, DTC_GROUND_SENSOR, DTC_DEAD = 0.1016, 35, {20}


def date_of(stamp: str) -> dt.date:
    return dt.date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))


def overpass(track: str, date: dt.date) -> pd.Timestamp:
    """Local timestamp of the acquisition."""
    day = date - dt.timedelta(days=1) if track in EVENING_TRACKS else date
    return pd.Timestamp(day) + pd.Timedelta(hours=OVERPASS_LOCAL_H[track])


def freeman_profile():
    """Freeman DTC as (hourly profile, hourly depth); columns are height above ground [m]."""
    raw = pd.read_csv(paths.MET_SOIL, parse_dates=["TIMESTAMP"], low_memory=False)
    raw = raw.set_index("TIMESTAMP").sort_index()
    columns, heights = [], []
    for i in range(1, 38):
        name = "DTC1_Avg" if i == 1 else f"DTC_Avg({i})"
        if name not in raw.columns or i in DTC_DEAD:
            continue
        height = (DTC_GROUND_SENSOR - i) * DTC_SPACING_M
        if -0.05 <= height <= 3.0:
            columns.append(name)
            heights.append(round(height, 4))
    frame = raw[columns].apply(pd.to_numeric, errors="coerce")
    frame.columns = heights
    frame = frame.reindex(sorted(frame.columns), axis=1)
    frame = frame.mask((frame < PLAUSIBLE_C[0]) | (frame > PLAUSIBLE_C[1]))
    depth = pd.to_numeric(raw["SnoDAR_snow_depth_Avg"], errors="coerce")
    depth = depth.where((depth >= 0) & (depth < 5))
    return frame.resample("1h").mean(), depth.resample("1h").mean()


def snotel_profile():
    """SNOTEL 637 as (hourly profile, daily depth); columns are height above ground [m]."""
    frame = pd.read_csv(SNOTEL_PROFILE, index_col=0, parse_dates=True)
    frame.columns = [float(c) for c in frame.columns]
    frame = frame.reindex(sorted(frame.columns), axis=1)
    frame = frame.mask((frame < PLAUSIBLE_C[0]) | (frame > PLAUSIBLE_C[1]))
    from nival.stage_ancillary import fetch_snotel_season
    if not paths.SNOTEL_SEASON.exists():
        fetch_snotel_season()
    season = pd.read_csv(paths.SNOTEL_SEASON, parse_dates=["date"])
    depth = pd.Series(season["snwd_in"].values * 0.0254,        # inches -> metres at ingest
                      index=season["date"]).sort_index()
    if depth.max() > 6.0:
        raise ValueError(f"SNOTEL depth {depth.max():.1f} m is not plausible; check units")
    return frame, depth


def top_of_pack_c(frame, depth, when, top_m: float = TOP_M) -> float:
    """Warmest temperature in the top `top_m` of the pack at the hour nearest `when`.

    NaN if the profile or the snow depth is missing there. A pack shallower than `top_m`
    falls back to its topmost buried sensor.
    """
    idx = frame.index.get_indexer([when], method="nearest", tolerance=pd.Timedelta("2h"))
    if idx[0] < 0:
        return np.nan
    hs = depth.reindex([frame.index[idx[0]]], method="nearest").iloc[0]
    if not np.isfinite(hs):
        return np.nan
    heights = np.asarray(frame.columns, float)
    buried = (heights >= 0.0) & (heights <= hs - SURFACE_SKIP_M)   # snow, not soil and not air
    values = pd.to_numeric(frame.iloc[idx[0]].values[buried], errors="coerce")
    order = np.argsort(heights[buried])
    values, hb = values[order], heights[buried][order]
    top = values[hb >= hs - top_m]
    top = top[np.isfinite(top)]
    if top.size == 0:
        finite = values[np.isfinite(values)]
        top = finite[-1:]
    return float(np.max(top)) if top.size else np.nan


def acquisition_table(pairs=None) -> pd.DataFrame:
    """One row per pair endpoint: top-of-pack temperature and wet flag at both stations."""
    from nival.pairs import PAIRS
    pairs = PAIRS if pairs is None else pairs
    snotel, snotel_depth = snotel_profile()
    freeman, freeman_depth = freeman_profile()
    rows = []
    for track, ref, sec, *_ in pairs:
        for end, stamp in (("ref", ref), ("sec", sec)):
            when = overpass(track, date_of(stamp))
            s = top_of_pack_c(snotel, snotel_depth, when)
            f = top_of_pack_c(freeman, freeman_depth, when)
            rows.append(dict(pair=f"{track} {ref[4:]}-{sec[4:]}", track=track, end=end,
                             date=date_of(stamp),
                             snotel_top_c=s, snotel_wet=bool(np.isfinite(s) and s >= -MELT_TOL_C),
                             freeman_top_c=f, freeman_wet=bool(np.isfinite(f) and f >= -MELT_TOL_C)))
    frame = pd.DataFrame(rows)
    frame["wet"] = frame["snotel_wet"] | frame["freeman_wet"]
    return frame


def pair_table(pairs=None) -> pd.DataFrame:
    """One row per pair: wet if either station is at the melting point at either end."""
    ends = acquisition_table(pairs)
    out = []
    for pair, grp in ends.groupby("pair", sort=False):
        s_wet, f_wet = bool(grp.snotel_wet.any()), bool(grp.freeman_wet.any())
        out.append(dict(pair=pair, track=grp.track.iloc[0], snotel_wet=s_wet,
                        freeman_wet=f_wet, wet=s_wet or f_wet,
                        state=("both wet" if s_wet and f_wet else
                               "SNOTEL wet, Freeman dry" if s_wet else
                               "Freeman wet, SNOTEL dry" if f_wet else "both dry")))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ends = acquisition_table()
    print(f"wet = top {TOP_M:.1f} m of the pack within {MELT_TOL_C} C of melting at the overpass\n")
    t77 = ends[ends.track == "077"]
    print(f"{'pair':>16} {'end':>4} {'date':>12} {'SNOTEL C':>9} {'Freeman C':>10}  state")
    for _, r in t77.iterrows():
        flag = ("SNOTEL " if r.snotel_wet else "") + ("Freeman" if r.freeman_wet else "")
        print(f"{r['pair']:>16} {r['end']:>4} {str(r['date']):>12} {r['snotel_top_c']:>9.2f} "
              f"{r['freeman_top_c']:>10.2f}  {flag or 'dry'}")
    pairs = pair_table()
    dry = pairs[(pairs.track == "077") & (~pairs.wet)]
    print(f"\ntrack 077: {int((pairs.track == '077').sum())} pairs, "
          f"{len(dry)} dry at both stations at both acquisitions -> {list(dry.pair)}")

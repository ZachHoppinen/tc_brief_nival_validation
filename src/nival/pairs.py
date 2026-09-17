"""The candidate NISAR pairs over Mores Creek Summit and the lidar flights around them.

The supplement's pair census is built from this list, so it lives here rather than in
scratch: anything a figure or table in the paper depends on has to be tracked.

    (track, ref, sec, lidar_a, lidar_b)
"""

PAIRS = [
    ("077", "20260207", "20260219", "20260207", "20260222"),
    ("077", "20260219", "20260315", "20260222", "20260315"),
    ("077", "20251221", "20260114", "20251220", "20260110"),
    ("149", "20251226", "20260107", "20251220", "20260110"),
    ("149", "20260212", "20260308", "20260207", "20260303"),
    ("077", "20260114", "20260207", "20260110", "20260207"),
    ("149", "20260119", "20260131", "20260126", "20260131"),
    ("149", "20251202", "20260107", "20251203", "20260110"),
    ("077", "20251209", "20251221", "20251213", "20251220"),
    ("149", "20260131", "20260212", "20260131", "20260207"),
    ("149", "20260107", "20260119", "20260110", "20260126"),
    ("077", "20251115", "20251209", "20251203", "20251213"),
    ("114", "20251130", "20251224", "20251203", "20251220"),
    ("114", "20251224", "20260117", "20251220", "20260110"),
    ("077", "20260315", "20260327", "20260315", "20260404"),
]

#: the pair the paper analyses
ANALYSED = ("077", "20260207", "20260219")


def label(pair):
    """'077_20260207_20260219' for the first three fields of a pair row."""
    return f"{pair[0]}_{pair[1]}_{pair[2]}"


def on_track(track):
    """Pairs on one track, in date order."""
    return sorted((p for p in PAIRS if p[0] == track), key=lambda p: p[1])

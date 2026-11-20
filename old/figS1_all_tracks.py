"""RETIRED: the all-track SWE-change timeline that was Fig. S1 -> old/figures/supplement/figS1_swe_change.png.

The same drawing function as the paper's Fig. 3 (nival.paper_figures.timeline_figure), run with
no track or date limit and none of Fig. 3's extras: one row per track (077, 114, 149), every
GUNW from November through April. Fig. 3 keeps the track-077 part of it.

It needs more data than the paper pipeline fetches: every production GUNW over the AOI on
every track (about 30 GB more, fetched below), and all twelve Hyp3-Isce3 GUNW crops staged by
hand in data/gunw_refocused/.

    conda run -n nival python old/figS1_all_tracks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nival.download_paper_data import fetch_gunws   # noqa: E402
from nival.paper_figures import timeline_figure   # noqa: E402

if __name__ == "__main__":
    print("operational GUNWs over the AOI, all tracks (ASF)...")
    fetch_gunws()                                    # no track or date limit
    timeline_figure(Path(__file__).resolve().parent / "figures" / "supplement" / "figS1_swe_change.png")

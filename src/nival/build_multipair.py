"""Build the dSWE/dHS product of every track-077 pair -> outputs/retrieval_multipair/.

Each pair goes through retrieval.build_product(), the same computation as the analysed
pair, with its own operational GUNW and the two lidar flights pairs.py lists for it. The
lidar row of Fig. 3 and the supplement's pair census and Table S1 read these files. Fetch
the inputs first with download_paper_data.py. Idempotent: existing products are skipped.

    conda run -n nival python src/nival/build_multipair.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the src/ dir
from nival import paths, retrieval   # noqa: E402
from nival.download_paper_data import TRACK, gunw_file, lidar_file   # noqa: E402
from nival.pairs import label, on_track   # noqa: E402

PRODUCTS = paths.OUTPUTS / "retrieval_multipair"


def main():
    for pair in on_track(TRACK):
        out = PRODUCTS / f"{label(pair)}.nc"
        if out.exists():
            print(f"  skip (exists): {out.relative_to(paths.ROOT)}")
            continue
        gunw = gunw_file(pair)
        if gunw is None:
            raise SystemExit(f"missing GUNW for {label(pair)}\n"
                             "  fetch it with: python src/nival/download_paper_data.py")
        # pair = (track, ref, sec, lidar_a, lidar_b): the lidar flights bracket the radar pair
        retrieval.build_product(gunw_path=gunw, lidar_a=lidar_file(pair[3]),
                                lidar_b=lidar_file(pair[4]), out_path=out)


if __name__ == "__main__":
    main()

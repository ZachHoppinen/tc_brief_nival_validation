"""Build the analysis product from the operational NISAR L2 GUNW.

  operational GUNW + NIVAL lidar -> dSWE/dHS   retrieval.build_product()
                                             -> outputs/retrieval/dswe_dhs_operational.nc

The interferogram is the operational L2 GUNW as delivered by ASF
(NISAR_L2_GUNW_PROVISIONAL_V1, PGE R05.02.3), not a self-processed one. Download it first:

    conda run -n nival python src/nival/download_gunw.py

then

    conda run -n nival python src/nival/run_workflow.py

Then unwrap the delivered 20 m wrapped layer that Fig. 3 is drawn from:

    conda run -n nival python src/nival/unwrap_20m.py

Render the figures afterwards with src/nival/paper_figures.py.

The legacy self-processing path and its 20/40/60/80 m multilook sweep have been retired to
old/generate_gunw.py; nothing in the analysis reads its products.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the src/ dir
from nival import paths, retrieval   # noqa: E402

if __name__ == "__main__":
    if not paths.gunw_product().exists():
        raise SystemExit(f"missing {paths.gunw_product()}\n"
                         "  fetch it with: python src/nival/download_gunw.py")
    print(f"operational GUNW: {paths.gunw_product().name}")
    retrieval.build_product()
    print(f"done -> {paths.product()}")

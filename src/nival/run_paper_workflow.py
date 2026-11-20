"""START HERE. The whole paper pipeline in one script: download, process, stats, figures,
supplement.

  1. download   download_paper_data.main()   lidar, SNOTEL, IONEX, fire, WorldCover, DEM, MTBS, NAIP, the
                                             operational L2 GUNW of the Fig. 2 pair, and for Fig. 3 every
                                             track-077 GUNW from Nov 3 to Mar 15 with its lidar flights -> data/
  2. process    retrieval.build_product()    GUNW 80 m phase + lidar -> outputs/retrieval/dswe_dhs_operational.nc
                build_multipair.main()       the same product for every track-077 pair -> outputs/retrieval_multipair/
  3. stats      paper_stats.write_report()   every number the manuscript cites -> outputs/paper_stats.md
  4. figures    paper_figures.*_figure()     the three paper figures -> figures/
  5. supplement download_supplement_data.main()  SNOTEL temperature profile -> data/
                supplement_figures.main()    table_s1.tex -> outputs/, figS2 -> figures/supplement/

The interferograms are the operational L2 GUNWs as delivered by ASF
(NISAR_L2_GUNW_PROVISIONAL_V1, PGE R05.02.3), except for the pairs we produced ourselves with
Hyp3-Isce3 from refocused RSLCs: those crops are staged by hand in data/gunw_refocused/, like
the met CSVs in data/met/. The downloads skip anything already on disk, so a rerun only redoes
the processing, stats and figures. Step 1 fetches about 13 GB.

    conda run -n nival python src/nival/run_paper_workflow.py

Each step's module also runs on its own (python src/nival/<module>.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the src/ dir
from nival import (build_multipair, download_paper_data, paper_figures, paper_stats,   # noqa: E402
                   paths, retrieval)
from nival.supplement import download_supplement_data, supplement_figures   # noqa: E402

if __name__ == "__main__":
    print("=== 1/5 download ===")
    download_paper_data.main()             # all remote ancillary data, the GUNWs and lidar of Figs. 2 and 3

    print("=== 2/5 process ===")
    retrieval.build_product()              # the 80 m dSWE/dHS product of the Fig. 2 pair
    build_multipair.main()                 # the same product for every track-077 pair

    print("=== 3/5 stats ===")
    paper_stats.write_report()

    print("=== 4/5 figures ===")
    paper_figures.context_figure()         # fig1_context.png
    paper_figures.results_figure()         # fig2_results.png
    paper_figures.swe_timeline_figure()    # fig3_swe_timeline.png

    print("=== 5/5 supplement ===")
    download_supplement_data.main()        # SNOTEL temperature profile
    supplement_figures.main()              # table_s1.tex, figS2
    print(f"done -> {paths.OUTPUTS} and {paths.FIGURES_DIR}")

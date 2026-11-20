# tc_brief_nival_validation

Self-contained reproduction of the **NISAR L-band InSAR dSWE vs NIVAL lidar dHS**
comparison at Mores Creek Summit, Idaho (NISAR ASC-077, 7 -> 19 Feb 2026).

The analysis reads the **operational NISAR L2 GUNW** for this pair as delivered by ASF
(`NISAR_L2_GUNW_PROVISIONAL_V1`, PGE R05.02.3, 80 m unwrapped posting). Fig. 3 follows the
same track through the season, Nov 3 to Mar 15, and for two of its seven pairs uses GUNWs we
produced ourselves with Hyp3-Isce3 from refocused RSLCs. NISAR phase-derived SWE change (dSWE) tracks the airborne-lidar snow-depth
change (dHS) across an 8 km mountain scene, and agreement rises as low-coherence pixels are
excluded while the retrieved density stays put.

| coherence gate | r | RMSE | density | n | % of valid |
|---|---|---|---|---|---|
| all pixels | 0.82 | 19 mm | 150 | 4,976 | 100% |
| > 0.2 | 0.93 | 12 mm | 157 | 2,891 | 58% |
| > 0.4 | 0.96 |  9 mm | 161 | 1,004 | 20% |
| > 0.5 | 0.96 |  9 mm | 162 |   463 |  9% |

## Install (two steps)

conda for the geospatial + isce3 stack, then one source-only package with `pip --no-deps`
(conda already provides its deps; a plain pip install can pull an uncompilable PyPI gdal and a
numpy 2 that breaks isce3):

```bash
conda env create -f environment.yml && conda activate nival
pip install --no-deps nisar_pytools
python -c "import isce3, snaphu, nisar.workflows.unwrap, nisar_pytools.utils.local_incidence_angle; print('env OK')"
```

## Credentials

The fresh downloads need this auth file in your home directory (public sources -- SNOTEL,
WorldCover, MTBS, NAIP, Pioneer Fire -- need nothing):

- **`~/.netrc`** -- NASA Earthdata Login; covers earthaccess (NIVAL lidar, the GUNW)
  and CDDIS (IONEX ionosphere). One line, then `chmod 600 ~/.netrc`:
  ```
  machine urs.earthdata.nasa.gov login <user> password <pass>
  ```
  Register at <https://urs.earthdata.nasa.gov>.

## Run

```bash
python src/nival/run_paper_workflow.py           # download -> process -> stats -> figures -> supplement
```

Start here: `run_paper_workflow.py` runs the five stages in order. Each stage's module also runs on its own:

```bash
python src/nival/download_paper_data.py      # 1. fetch remote data: ancillary layers, the track-077 GUNWs and lidar (about 13 GB)
python src/nival/build_multipair.py          # 2. one dSWE/dHS product per track-077 pair
python src/nival/paper_stats.py              # 3. every number the manuscript cites -> outputs/paper_stats.md
python src/nival/paper_figures.py            # 4. the three paper figures -> figures/
python src/nival/supplement/supplement_figures.py   # 5. table S1, figS2
```

Everything is fetched fresh from its authoritative source **except** two hand-staged inputs
that are not distributed anywhere: the on-site met CSVs under `data/met/` (tracked), and the
Hyp3-Isce3 GUNW crops under `data/gunw_refocused/` (not tracked: copy them in by hand; Fig. 3
needs the two track-077 ones). `data/` is gitignored except the met CSVs.

## Layout

```
src/nival/
  run_paper_workflow.py   START HERE: the whole pipeline -- download, process, stats, figures, supplement
  paths.py                every path + campaign config, repo-relative (portable)
  pairs.py                the candidate NISAR pairs and the lidar flights around them
  download_paper_data.py  RUN: fetch the ancillary layers, the track-077 GUNWs and their lidar flights
  retrieval.py            operational GUNW + lidar -> dSWE/dHS/coherence/elevation NetCDF (load-bearing)
  build_multipair.py      RUN: that product for every track-077 pair -> outputs/retrieval_multipair/
  paper_stats.py          RUN: products -> outputs/paper_stats.md
  paper_figures.py        RUN: the three paper figures (plus draw helpers, met readers, study-area map)
  supplement/             download_supplement_data.py (RUN: SNOTEL temperature profile),
                          supplement_figures.py (RUN: table S1, figS2), wetness.py
old/       retired code and figures: the 20 m unwrap and the 20 m Fig. 3, the all-track timeline, press figures
outputs/   retrieval/dswe_dhs_operational.nc (the analysis product), retrieval_multipair/ (one per pair)
figures/   fig1_context, fig2_results, fig3_swe_timeline and supplement/ (the deliverables)
```

## Processing

The analysis product is the operational L2 GUNW as delivered: 5x6 looks for the wrapped
interferogram (20 m), unwrapped at 13x16 looks and geocoded to 80 m, with the ionosphere,
troposphere and solid-earth-tide screens carried as separate layers. Those screens are
**not** applied to the delivered unwrapped phase, despite the `*CorrectionApplied` metadata
flags reading `True`. The Fig. 2 analysis and Table S1 apply none of them. `retrieval.py`
clips the full ~360 x 356 km frame to the Mores AOI on read.

Fig. 3 is the exception. Each pair's unwrapped phase has its ionosphere and
troposphere screens removed (`nisar_pytools` `correct_atmosphere`), is shifted so the SNOTEL 637
pixel carries the phase of the station's own SWE change over that pair, and is then converted
to SWE change with the same per-pixel sensitivity as the retrieval.

An earlier version of this analysis self-processed the same pair from the RSLCs with isce3
over a 20/40/60/80 m multilook sweep (`generate_gunw.py`, now only in the git history).
Its 80 m output reproduces the operational product closely (r = 0.84 vs
0.82, median coherence 0.275 vs 0.263, pixel-matched r = 0.92 with a median difference of
0.1 mm SWE), so the operational granule was adopted and the sweep dropped.

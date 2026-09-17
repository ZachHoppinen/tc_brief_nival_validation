# tc_brief_nival_validation

Self-contained reproduction of the **NISAR L-band InSAR dSWE vs NIVAL lidar dHS**
comparison at Mores Creek Summit, Idaho (NISAR ASC-077, 7 -> 19 Feb 2026).

The analysis reads the **operational NISAR L2 GUNW** for this pair as delivered by ASF
(`NISAR_L2_GUNW_PROVISIONAL_V1`, PGE R05.02.3, 80 m unwrapped posting). Nothing is
self-processed. NISAR phase-derived SWE change (dSWE) tracks the airborne-lidar snow-depth
change (dHS) across an 8 km mountain scene, and agreement rises as low-coherence pixels are
excluded while the retrieved density stays put.

| coherence gate | r | RMSE | density | n | % of valid |
|---|---|---|---|---|---|
| all pixels | 0.82 | 19 mm | 150 | 4,976 | 100% |
| > 0.2 | 0.93 | 12 mm | 157 | 2,891 | 58% |
| > 0.4 | 0.96 |  9 mm | 161 | 1,004 | 20% |
| > 0.5 | 0.96 |  9 mm | 162 |   463 |  9% |

## Install (two steps)

conda for the geospatial + isce3 stack, then two source-only packages with `pip --no-deps`
(conda already provides their deps; a plain pip install pulls an uncompilable PyPI gdal and a
numpy 2 that breaks isce3):

```bash
conda env create -f environment.yml && conda activate nival
pip install --no-deps nisar_pytools \
  "hyp3_isce3 @ git+https://github.com/ASFHyP3/hyp3-isce3.git@740350a"   # crop_rslc lives on this commit
python -c "import nisar.workflows.insar, hyp3_isce3.crop_rslc, nisar_pytools.utils.local_incidence_angle; print('env OK')"
```

## Credentials

The fresh downloads need these auth files in your home directory (public sources -- SNOTEL,
WorldCover, MTBS, NAIP, Pioneer Fire -- need nothing):

- **`~/.netrc`** -- NASA Earthdata Login; covers earthaccess (NIVAL lidar, precise orbits, TEC,
  water mask) and CDDIS (IONEX ionosphere). One line, then `chmod 600 ~/.netrc`:
  ```
  machine urs.earthdata.nasa.gov login <user> password <pass>
  ```
  Register at <https://urs.earthdata.nasa.gov>.
- **`~/.ecmwfapirc`** (ECMWF) or **`~/.cdsapirc`** (Copernicus CDS) -- API key for the RAiDER
  tropospheric weather model used in the GUNW. Keys: <https://api.ecmwf.int/v1/key/> or
  <https://cds.climate.copernicus.eu/>.
- **`~/.urs_cookies`** (empty file) + **`~/.dodsrc`** (`HTTP.COOKIEJAR=~/.urs_cookies`) -- the
  Earthdata OPeNDAP cookie jar that RAiDER reads when pulling the weather model.

## Run

```bash
python src/nival/stage_ancillary.py          # fetch remote data (lidar, SNOTEL, IONEX, fire, WorldCover, MTBS, NAIP)
python src/nival/download_gunw.py            # the operational L2 GUNW for the pair (2.5 GB, from ASF)
python src/nival/run_workflow.py             # GUNW + lidar -> dSWE/dHS product
python src/nival/paper_figures.py            # the three paper figures -> figures/
```

Everything is fetched fresh from its authoritative source **except** the on-site met CSVs,
which are not distributed anywhere and are assumed present under `data/met/`. `data/` is
gitignored except those CSVs and the small frozen API snapshots in `data/remote/`. The two
26 GB RSLC granules and the runconfig template are needed only for the legacy
self-processing path, not for the analysis.

## Layout

```
src/nival/
  paths.py            every path + campaign config, repo-relative (portable)
  stage_ancillary.py  RUN: fetch all remote / GUNW-ancillary data
  download_gunw.py    RUN: fetch the operational L2 GUNW granule (2.5 GB)
  retrieval.py        operational GUNW + lidar -> dSWE/dHS/coherence/elevation NetCDF (load-bearing)
  run_workflow.py     RUN: build the product from the operational GUNW -> outputs/
  paper_figures.py    RUN: product -> the three figures
  validation_plotting.py / ancillary_plotting.py   draw helpers, study-area map, met readers
outputs/   retrieval/dswe_dhs_operational.nc (the analysis product)
figures/   the paper figures, supplement/ and the NISAR_PRESS images (the deliverables)
```

## Processing

The analysis product is the operational L2 GUNW as delivered: 5x6 looks for the wrapped
interferogram (20 m), unwrapped at 13x16 looks and geocoded to 80 m, with the ionosphere,
troposphere and solid-earth-tide screens carried as separate layers. Those screens are
**not** applied to the delivered unwrapped phase, despite the `*CorrectionApplied` metadata
flags reading `True`, and we apply none of them. `retrieval.py` clips the full ~360 x 356 km
frame to the Mores AOI on read.

`old/generate_gunw.py` is kept for reference only. It self-processes the same pair from the RSLCs
with isce3 over a 20/40/60/80 m multilook sweep, which is how an earlier version of this
analysis worked. Its 80 m output reproduces the operational product closely (r = 0.84 vs
0.82, median coherence 0.275 vs 0.263, pixel-matched r = 0.92 with a median difference of
0.1 mm SWE), so the operational granule was adopted and the sweep dropped.

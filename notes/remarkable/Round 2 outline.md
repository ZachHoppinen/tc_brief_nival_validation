# Round 2 outline

reMarkable: `/Reviewing/tc-brief-nival/Round 2 outline`, modified 2026-11-22 13:37. One page, transcribed from the page image. `[?]` marks a word I could not read.

## Intro

Solid

## Campaign Summary - Fig 1

Nival = lidar flights to evaluate nisar for SWE retrieval by measuring snow depth with helicopter lidar over mores creek summit aiming to coincide with track 77 Nisar overpasses.

Study area descriptions - in idaho, intermountain snow climate, mix of open, and evergreens. burn in eastern 1/2 in 2016.

Winter 2025-26 - historically warm. Rain in early season, Jan - March good accumulation w/ dryish snow. Flights began on Dec XX and aligned w/ in 4 days of overpasses as wx allows.

## Methods

### InSAR

Mostly provisional GUNWs from ASF. One acquisition on Jan XX had extra time steps that have been noted as a known issue (Release Notes YY). We remove the extra L0b timestamps to align w/ correct raw pulses and refocus this acquisition and form the two GUNWs w/ this refocused slc (Appendix A). Coherence improved from 0.08 mean for both production GUNWs to XX and YY mean.

We remove the provided ionospheric, atmospheric correction layers (App. B).

### SWE retrieval

- Equ. + LIA for ΔSWE
- Phase referencing
- Cumulative for SWE

### Comparison to lidar

- Lidar is averaged to 80 m
- masked outliers
- differenced
- march 15 used as "peak SWE"
- split by LIA, vegetation, elev for cumulative
- compare in time w/ (1) lidar ΔHS, (2) snotel temps @ acquisition, (3) snotel degree days to pairwise r.

## Results

Fig 2 -

Table 1 - Nisar pairs w/ lidar dates, lidar ΔHS, snotel acquisition temps, snotel degree days, pairwise r.

Fig 3 - spatial break down by LIA, FCF, elev

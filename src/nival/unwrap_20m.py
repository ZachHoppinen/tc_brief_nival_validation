"""Unwrap the operational GUNW's delivered 20 m wrapped interferogram with SNAPHU.

The operational product ships the wrapped interferogram at 20 m but only unwraps at 80 m.
Unwrapping the delivered layer here gives a finer view of the same operational phase with
no reprocessing from the RSLCs, and is what Fig. 3 is drawn from.

    conda run -n isce3 python src/nival/unwrap_20m.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import snaphu
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nival import paths, retrieval   # noqa: E402

WRAPPED = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH"
CROSSMUL_LOOKS = (5, 6)        # range, azimuth: what the delivered wrapped product carries
# operational SNAPHU settings, from nisar/workflows/defaults/insar.yaml
COST, INIT, MIN_CC_FRAC, GRAD_WINDOW = "smooth", "mcf", 0.01, (7, 7)


def effective_looks():
    """ISCE3's own estimate, n_e = k_r k_a d_r d_a / (rho_r rho_a). For the delivered 5 by 6
    look interferogram this is 18.6, not the nominal 30, because the single-look samples are
    oversampled relative to the resolution."""
    from nisar.products.readers import SLC
    from nisar.workflows.unwrap import get_effective_looks
    slc = SLC(hdf5file=str(paths.RSLC_REF))
    swath = "/science/LSAR/RSLC/swaths/frequencyA"
    with h5py.File(paths.RSLC_REF) as f:
        rg_bw = f[f"{swath}/processedRangeBandwidth"][()]
        az_bw = f[f"{swath}/processedAzimuthBandwidth"][()]
        rg_spac = f[f"{swath}/slantRangeSpacing"][()] * CROSSMUL_LOOKS[0]
        az_spac = f[f"{swath}/sceneCenterAlongTrackSpacing"][()] * CROSSMUL_LOOKS[1]
    return float(get_effective_looks(slc, slc.getOrbit(), rg_spac, az_spac,
                                     rg_bw, az_bw, freq="A"))


def product(filtered: bool = False) -> Path:
    """The unfiltered product is what the paper reads; the filtered one is for press art."""
    name = "dswe_delivered_20m_filtered.nc" if filtered else "dswe_delivered_20m.nc"
    return paths.PRODUCTS_DIR / name


def goldstein(z, alpha=0.5, patch=32):
    """Goldstein adaptive filter: each patch's spectrum raised to a power set by its own
    smoothness, recombined with a Hanning taper.

    Applied to the wrapped phase before unwrapping it suppresses speckle while leaving
    fringe structure intact. Against the lidar it raises the scene-wide correlation from
    0.75 to 0.80, and the result is flat for any alpha between 0.3 and 1.0, so the gain is
    the filtering itself rather than a tuned exponent.
    """
    out = np.zeros_like(z)
    wgt = np.zeros(z.shape, float)
    step = patch // 2
    win = np.outer(np.hanning(patch), np.hanning(patch))
    kernel = np.ones((3, 3)) / 9
    for i in range(0, z.shape[0] - patch + 1, step):
        for j in range(0, z.shape[1] - patch + 1, step):
            spec = np.fft.fft2(z[i:i + patch, j:j + patch])
            sm = np.fft.fftshift(np.abs(spec))
            sm = np.abs(np.fft.ifft2(np.fft.fft2(sm) * np.fft.fft2(kernel, s=sm.shape)))
            sm = np.fft.ifftshift(sm)
            filtered = spec * (sm / (sm.max() or 1)) ** alpha
            out[i:i + patch, j:j + patch] += np.fft.ifft2(filtered) * win
            wgt[i:i + patch, j:j + patch] += win
    return np.where(wgt > 0, out / np.where(wgt > 0, wgt, 1), z)


def build(alpha=None):
    with h5py.File(paths.gunw_product()) as f:
        g = f[WRAPPED]
        rows, cols = retrieval._aoi_window(g["xCoordinates"][:], g["yCoordinates"][:])
        igram = g["wrappedInterferogram"][rows, cols]
        corr = g["coherenceMagnitude"][rows, cols]
        xs, ys = g["xCoordinates"][cols], g["yCoordinates"][rows]
    nlooks = effective_looks()
    print(f"delivered wrapped AOI: {igram.shape} at 20 m, median coherence "
          f"{np.nanmedian(corr):.2f}, effective looks {nlooks:.1f}")

    mask = np.isfinite(igram) & (igram != 0) & np.isfinite(corr)
    if alpha is not None:
        igram = goldstein(np.where(mask, igram, 0).astype(np.complex64), alpha=alpha)
        print(f"Goldstein filter applied, alpha = {alpha}")
    unw, conncomp = snaphu.unwrap(np.where(mask, igram, 0).astype(np.complex64),
                                  np.where(mask, np.nan_to_num(corr), 0).astype(np.float32),
                                  nlooks=nlooks, cost=COST, init=INIT, mask=mask,
                                  min_conncomp_frac=MIN_CC_FRAC,
                                  phase_grad_window=GRAD_WINDOW)
    unw = np.where(mask, np.asarray(unw, float), np.nan)
    cc = np.asarray(conncomp)
    # label 0 is SNAPHU's unassigned class, not a component; it is kept in the product
    # (it is a coherence mask, not an error) but reported separately
    counts = np.bincount(cc[mask])
    largest = counts[1:].argmax() + 1 if counts[1:].size else 0
    print(f"snaphu: {cc.max()} connected components, "
          f"{100 * np.mean(cc[mask] == largest):.0f}% of valid pixels in the largest, "
          f"{100 * np.mean(cc[mask] == 0):.0f}% unassigned")

    # phase -> dSWE. The incidence angle is computed on the 20 m grid from the Copernicus
    # DEM rather than upsampled from the 80 m product: theta enters Eq. 1 as theta^2.5 and
    # this product exists to show structure below the 80 m posting.
    from nisar_pytools.utils.local_incidence_angle import local_incidence_angle
    with h5py.File(paths.gunw_product()) as f:
        rg = f["science/LSAR/GUNW/metadata/radarGrid"]
        los_x, los_y = rg["losUnitVectorX"][:], rg["losUnitVectorY"][:]
        los_z = np.sqrt(np.clip(1 - los_x ** 2 - los_y ** 2, 0, 1))
        heights, qx, qy = rg["heightAboveEllipsoid"][:], rg["xCoordinates"][:], rg["yCoordinates"][:]
    elevation = xr.DataArray(retrieval._elevation_on_grid(xs, ys, paths.AOI),
                             coords={"y": ys, "x": xs}, dims=("y", "x")).rio.write_crs(paths.EPSG)
    lia = np.asarray(local_incidence_angle(elevation, los_x, los_y, los_z, heights,
                                           qx, qy, epsg=paths.EPSG).values)
    kappa = 2 * np.pi / paths.LAMBDA * (1.59 + np.radians(lia) ** 2.5)
    # one global constant cannot remove a per-component 2-pi ambiguity, so report the
    # component medians: a component a whole fringe off would show up here as ~1.0
    ref = np.nanmedian(unw[mask & (cc == largest)])
    for label in np.unique(cc[mask]):
        offset = (np.nanmedian(unw[mask & (cc == label)]) - ref) / (2 * np.pi)
        print(f"  component {label}: {offset:+.2f} fringe of the largest, "
              f"{100 * np.mean(cc[mask] == label):.0f}% of valid pixels")
    dswe = (unw - np.nanmedian(unw)) / kappa * 1000

    out = xr.Dataset({"dswe": (("y", "x"), dswe), "coherence": (("y", "x"), corr),
                      "lia": (("y", "x"), lia)},
                     coords={"y": ys, "x": xs},
                     attrs={"crs_epsg": paths.EPSG, "posting_m": 20.0,
                            "nlooks": nlooks, "dswe_units": "mm", "lia_units": "deg",
                            "source": "operational GUNW wrappedInterferogram, SNAPHU here "
                                      "with the operational cost mode, initialisation and "
                                      "connected-component settings"})
    path = product(filtered=alpha is not None)
    out.to_netcdf(path)
    print(f"wrote {path.relative_to(paths.ROOT)}")
    return out


if __name__ == "__main__":
    build()                 # the paper product, unfiltered
    build(alpha=0.5)        # the press product, Goldstein filtered

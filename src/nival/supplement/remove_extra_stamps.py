"""Remove the extra pulse time stamps from the raw (L0B) data of a NISAR acquisition.

The Jan 14 track-077 L0B holds extra time stamps. Each one pushes every later stamp one echo row
down, so the echoes carry the wrong times and the image does not focus. A clean acquisition of the
same track has the same pulse timing pattern, counted from its first pulse. We keep the one faulty
stamp that sits on each pulse of that pattern and drop the rest, so row i again carries the stamp
of pulse i. The rows left without a stamp at the end of the file are filled at the mean interval,
to keep the arrays increasing.

Stamps are read from the small npz files in data/l0b_stamps (UTCtime, radarTime per transmit group).

    python src/nival/supplement/remove_extra_stamps.py <faulty.npz> <clean.npz> <out.npz> [l0b.h5]

With an L0B file given, its UTCtime and radarTime are also overwritten in place.
"""
import sys

import h5py
import numpy as np

TOLERANCE_S = 1e-7             # a stamp is on a pulse when it is within 0.1 microseconds of it


def kept_rows(faulty_utc, clean_utc):
    """Rows of the faulty stamps that sit on a pulse of the clean pattern, one per pulse, in order."""
    faulty_s, clean_s = faulty_utc - faulty_utc[0], clean_utc - clean_utc[0]      # seconds after the first pulse
    # the clean pulse nearest each faulty stamp
    nearest = np.clip(np.searchsorted(clean_s, faulty_s), 1, clean_s.size - 1)
    nearest = np.where(np.abs(faulty_s - clean_s[nearest - 1]) < np.abs(faulty_s - clean_s[nearest]), nearest - 1, nearest)
    offset = np.abs(faulty_s - clean_s[nearest])
    on_pulse = np.flatnonzero(offset < TOLERANCE_S)
    # where two stamps sit on one pulse, keep the closer one
    closest_first = on_pulse[np.argsort(offset[on_pulse], kind="stable")]
    return np.sort(closest_first[np.unique(nearest[closest_first], return_index=True)[1]])


def corrected(values, kept):
    """One stamp array with the extra rows removed, at its original length. The rows left over at
    the end continue at the mean interval of the last 1000 kept stamps."""
    out = np.empty_like(values)
    out[:kept.size] = values[kept]
    missing = values.size - kept.size
    step = (out[kept.size - 1] - out[kept.size - 1001]) / 1000
    out[kept.size:] = (out[kept.size - 1] + step * np.arange(1, missing + 1)).astype(values.dtype)
    return out


def main(faulty_npz, clean_npz, out_npz, l0b=None):
    faulty, clean = np.load(faulty_npz), np.load(clean_npz)
    out = {key: faulty[key] for key in faulty.files}
    for key in [k for k in faulty.files if k.endswith("/UTCtime")]:
        kept = kept_rows(faulty[key], clean[key])
        for name in ("UTCtime", "radarTime"):
            out[key.replace("UTCtime", name)] = corrected(faulty[key.replace("UTCtime", name)], kept)
        print(f"{key.rsplit('/', 1)[0]}: {faulty[key].size - kept.size} extra stamps removed of {faulty[key].size}")
    np.savez_compressed(out_npz, **out)
    print(f"wrote {out_npz}")
    if l0b:
        # write the corrected stamps into the L0B, leaving every other record alone
        with h5py.File(l0b, "r+") as f:
            for key in [k for k in out if k.endswith(("/UTCtime", "/radarTime"))]:
                f[f"science/LSAR/RRSD/swaths/{key}"][...] = out[key]
        print(f"updated {l0b}")


if __name__ == "__main__":
    main(*sys.argv[1:])

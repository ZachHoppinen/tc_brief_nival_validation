"""Fetch the operational NISAR L2 GUNW granule the analysis reads, into data/gunw_operational/.

One granule, ~2.5 GB, from ASF via NASA Earthdata (CMR collection NISAR_L2_GUNW_PROVISIONAL_V1).
This IS the analysis product -- the paper evaluates the operational interferogram as delivered.
Needs a NASA Earthdata Login (~/.netrc). Idempotent: an existing file is skipped.

    conda run -n nival python src/nival/download_gunw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import earthaccess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the src/ dir
from nival import paths   # noqa: E402

COLLECTION = "NISAR_L2_GUNW_PROVISIONAL_V1"


def fetch_gunw(dst=None):
    """Download the one GUNW granule named by paths.GUNW_OPERATIONAL, streaming it to disk."""
    dst = Path(dst) if dst else paths.GUNW_OPERATIONAL
    if dst.exists():
        print(f"  skip (exists): {dst.relative_to(paths.ROOT)}")
        return dst

    earthaccess.login()                                         # Earthdata netrc login (idempotent)
    # granule_name is a CMR pattern match, so the trailing * covers a granule_ur
    # that carries the .h5 extension as well as one that does not
    results = earthaccess.search_data(short_name=COLLECTION, granule_name=f"{dst.stem}*", count=10)
    urls = [url for g in results for url in g.data_links() if url.endswith(dst.name)]
    if not urls:
        raise SystemExit(f"NOT FOUND in {COLLECTION}: {dst.name}")

    print(f"  downloading {dst.name} -> {dst.relative_to(paths.ROOT)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    session = earthaccess.get_requests_https_session()          # reuses the earthaccess login
    tmp = dst.with_suffix(dst.suffix + ".part")
    with session.get(urls[0], stream=True) as response:
        response.raise_for_status()
        with open(tmp, "wb") as out:
            for chunk in response.iter_content(1 << 20):        # stream in 1 MB chunks
                out.write(chunk)
    tmp.replace(dst)                                            # atomic: only a complete file lands at dst
    print(f"  wrote {dst.name} ({dst.stat().st_size / 1e9:.1f} GB)")
    return dst


if __name__ == "__main__":
    fetch_gunw()

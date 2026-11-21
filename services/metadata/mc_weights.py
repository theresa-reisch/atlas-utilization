"""
MC event-weight normalisation for ATLAS Open Data (2024r-pp, mc20 PHYSLITE).

Every MC event gets

    EventWeight = mcEventWeights[0] * PileupWeight_NOSYS * norm(DSID)

    norm(DSID)  = sigma[pb] * 1000 * kFactor * genFiltEff * L[fb^-1] / sumw_used(DSID)

so that a histogram filled with EventWeight is in expected events for the
luminosity L.

sumw_used is the DSID's sumOfWeights (from AMI via atlasopenmagic) scaled by
the fraction of that DSID's files this run processes. When every file of a
DSID is processed this is exact; for partial runs (tests, FILES_PER_JOB
limits) it assumes the files of a DSID carry roughly equal weight sums.

The sample metadata (cross-sections, sum of weights, file lists) is fetched
once per run and cached as JSON next to metadata_cache.json.
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections import Counter
from contextlib import redirect_stdout
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

SAMPLE_INFO_FILENAME = "mc_sample_info.json"

# EventInfo branches in mc20 PHYSLITE (verified on 2024r-pp DSID 301204).
MC_EVENT_WEIGHTS_BRANCH = "EventInfoAuxDyn.mcEventWeights"
PILEUP_WEIGHT_BRANCH = "EventInfoAuxDyn.PileupWeight_NOSYS"

_NORM_FIELDS = ("cross_section_pb", "genFiltEff", "kFactor", "sumOfWeights")


def _basename(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def fetch_sample_info(release: str) -> dict:
    """Fetch per-DSID normalisation inputs and a file -> DSID map from atlasopenmagic."""
    import atlasopenmagic as atom

    with redirect_stdout(io.StringIO()):
        atom.set_release(release)
        all_meta = atom.get_all_metadata()

    samples: Dict[str, dict] = {}
    file_to_dsid: Dict[str, str] = {}
    for info in all_meta.values():
        dsid = str(info.get("dataset_number") or "")
        if not dsid.isdigit() or dsid in samples:
            continue  # skips the "data" entry and the physics_short aliases
        files = info.get("file_list") or []
        samples[dsid] = {
            "physics_short": info.get("physics_short"),
            "cross_section_pb": info.get("cross_section_pb"),
            "genFiltEff": info.get("genFiltEff"),
            "kFactor": info.get("kFactor"),
            "sumOfWeights": info.get("sumOfWeights"),
            "nEvents": info.get("nEvents"),
            "n_files": len(files),
        }
        for f in files:
            file_to_dsid[_basename(f)] = dsid

    logger.info(
        f"Fetched MC sample info for release {release}: "
        f"{len(samples)} DSIDs, {len(file_to_dsid)} files"
    )
    return {"release": release, "samples": samples, "file_to_dsid": file_to_dsid}


def load_or_fetch_sample_info(cache_path: str, release: str) -> dict:
    """Load the cached sample info, or fetch it and write it atomically."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            info = json.load(f)
        if info.get("release") == release:
            logger.info(f"Loaded MC sample info from {cache_path}")
            return info
        logger.warning(
            f"{cache_path} is for release {info.get('release')!r}, not {release!r}; refetching"
        )

    info = fetch_sample_info(release)
    tmp = f"{cache_path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(info, f)
    os.replace(tmp, cache_path)  # atomic: concurrent array jobs never see a partial file
    logger.info(f"Saved MC sample info to {cache_path}")
    return info


class McNormalizer:
    """Per-file normalisation factor (xsec * k * eff * lumi / sumw_used)."""

    def __init__(
        self,
        sample_info: dict,
        luminosity_fb: float,
        processed_urls: Iterable[str],
    ):
        self.samples = sample_info["samples"]
        self.file_to_dsid = sample_info["file_to_dsid"]
        self.luminosity_fb = float(luminosity_fb)

        n_processed = Counter(
            self.file_to_dsid.get(_basename(u)) for u in processed_urls
        )
        n_processed.pop(None, None)

        self._norm: Dict[str, float] = {}
        for dsid, n_proc in n_processed.items():
            s = self.samples[dsid]
            missing = [k for k in _NORM_FIELDS if s.get(k) in (None, "")]
            if missing:
                logger.warning(f"DSID {dsid} ({s.get('physics_short')}): missing {missing}; its files will be skipped")
                continue
            sumw = float(s["sumOfWeights"])
            n_files = int(s["n_files"]) or 1
            sumw_used = sumw * min(1.0, n_proc / n_files)
            if sumw_used == 0:
                logger.warning(f"DSID {dsid}: sumOfWeights is 0; its files will be skipped")
                continue
            xsec_fb = float(s["cross_section_pb"]) * 1000.0
            self._norm[dsid] = (
                xsec_fb * float(s["kFactor"]) * float(s["genFiltEff"]) * self.luminosity_fb / sumw_used
            )

        partial = sum(1 for d, n in n_processed.items() if n < int(self.samples[d]["n_files"]))
        logger.info(
            f"MC normalisation for L = {self.luminosity_fb} fb^-1: {len(self._norm)} DSIDs "
            f"({partial} only partially processed: sumw scaled by file fraction)"
        )

    def dsid_for_url(self, url: str) -> Optional[str]:
        return self.file_to_dsid.get(_basename(url))

    def norm_for_url(self, url: str) -> float:
        """Normalisation factor for one file; raises if it cannot be determined."""
        dsid = self.dsid_for_url(url)
        if dsid is None:
            raise ValueError(f"No DSID known for MC file {url}")
        if dsid not in self._norm:
            raise ValueError(f"No normalisation for DSID {dsid} (file {url})")
        return self._norm[dsid]

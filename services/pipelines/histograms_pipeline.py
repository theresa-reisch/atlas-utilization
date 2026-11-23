"""
Histogram creation pipeline.

Builds ROOT histograms from processed invariant-mass arrays.
Supports BumpNet-compatible naming and concurrent file writing.
"""
import logging
import sys
import os
import fcntl
import time
import re
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
import ROOT
import math
from services.pipelines.im_pipeline import (
    MC_WEIGHT_SUFFIX, SIGMA_SUFFIX, DSID_SUFFIX, is_sibling_signature,
)
from services.storage.sqlite_shards import list_signatures, iter_arrays_for_signature

# Fixed histogram range eliminates the range pre-scan and ensures that all
# independently produced batch histograms have merge-compatible bin edges.
FIXED_MASS_MIN_GEV = 0.0
FIXED_MASS_MAX_GEV = 10000.0

# Companion objects written next to each mass histogram ROI_<base>_width_<w>
# for the post-merge resolution rebinning (services/pipelines/rebin_pipeline.py):
#   RESO_<base>                  TProfile of the per-event sigma_m vs mass
#   PROC_<group>_<base>_width_<w> mass histogram of one process group (MC only)
RESOLUTION_PROFILE_PREFIX = "RESO_"
PROCESS_HIST_PREFIX = "PROC_"


def _bin_is_empty(hist: ROOT.TH1, b: int) -> bool:
    # A weighted bin whose weights cancel has zero content but a nonzero error.
    return hist.GetBinContent(b) == 0 and hist.GetBinError(b) == 0


def slice_empty_bins(hist: ROOT.TH1) -> ROOT.TH1:
    """Return a new histogram holding only the bins from the first to the last non-empty one.

    This changes the stored axis, not just the display range. Bin edges
    stay on the original grid, so bin width is unchanged. Interior empty
    bins are kept. An all-empty histogram is returned as is.
    """
    if hist.InheritsFrom("TProfile"):
        # A profile stores sums per bin; copying contents would corrupt it.
        return hist
    nbins = hist.GetNbinsX()
    filled = [b for b in range(1, nbins + 1) if not _bin_is_empty(hist, b)]
    if not filled:
        return hist
    first, last = filled[0], filled[-1]

    add_directory = ROOT.TH1.AddDirectoryStatus()
    ROOT.TH1.AddDirectory(False)
    try:
        sliced = type(hist)(
            hist.GetName(), hist.GetTitle(), last - first + 1,
            hist.GetBinLowEdge(first), hist.GetBinLowEdge(last + 1),
        )
    finally:
        ROOT.TH1.AddDirectory(add_directory)
    # Errors are copied only for Sumw2 histograms: SetBinError would enable
    # Sumw2 on an unweighted one, whose errors are sqrt(content) anyway.
    weighted = hist.GetSumw2N() > 0
    if weighted:
        sliced.Sumw2()

    # Bins dropped below/above the slice are empty, so under/overflow carry over unchanged.
    for new_b, old_b in enumerate([0, *range(first, last + 1), nbins + 1]):
        sliced.SetBinContent(new_b, hist.GetBinContent(old_b))
        if weighted:
            sliced.SetBinError(new_b, hist.GetBinError(old_b))
    sliced.SetEntries(hist.GetEntries())
    return sliced


def slice_empty_bins_in_file(root_filepath: str) -> int:
    """Replace every histogram in a ROOT file with its ``slice_empty_bins`` version."""
    root_file = ROOT.TFile(root_filepath, "UPDATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Failed to open ROOT file for empty-bin trimming: {root_filepath}")

    sliced = 0
    try:
        histograms = []
        for key in root_file.GetListOfKeys():
            obj = key.ReadObj()
            if obj.InheritsFrom("TH1"):
                histograms.append(obj)

        for hist in histograms:
            root_file.cd()
            slice_empty_bins(hist).Write("", ROOT.TObject.kOverwrite)
            sliced += 1
    finally:
        root_file.Close()
    return sliced


def _new_histogram(name: str, nbins: int, weighted: bool) -> ROOT.TH1:
    """
    Fixed-range histogram; double precision when filled with MC weights.

    TH1F keeps float32 bin contents, so a small weight added to a populated
    bin is rounded away (a rare process on top of a large background simply
    vanishes). Weighted histograms therefore use TH1D with Sumw2 so both the
    contents and the errors (sqrt(sum w^2)) are exact.
    """
    if weighted:
        hist = ROOT.TH1D(name, name, nbins, FIXED_MASS_MIN_GEV, FIXED_MASS_MAX_GEV)
        hist.Sumw2()
        return hist
    return ROOT.TH1F(name, name, nbins, FIXED_MASS_MIN_GEV, FIXED_MASS_MAX_GEV)


def _fill_mass(hist: ROOT.TH1, value: float, weight: float = 1.0) -> None:
    """Fill a fixed-range histogram, including the declared upper endpoint."""
    mass = float(value)
    if mass == FIXED_MASS_MAX_GEV:
        mass = math.nextafter(mass, FIXED_MASS_MIN_GEV)
    hist.Fill(mass, weight)


def _clip_to_range(masses) -> np.ndarray:
    """Masses as float64, the declared upper endpoint moved just inside the range."""
    x = np.asarray(masses, dtype=np.float64)
    return np.where(x == FIXED_MASS_MAX_GEV, math.nextafter(FIXED_MASS_MAX_GEV, FIXED_MASS_MIN_GEV), x)


class _HistogramSet:
    """
    The histograms of one output name: the mass histogram per bin width plus,
    when requested, the sigma_m profile and the per-process-group histograms.

    ``process_groups`` maps a group name to the DSIDs in it. Process-group
    histograms are only made when the events carry DSIDs (MC).
    """

    def __init__(self, base_name: str, bin_widths_gev: List[float], weighted: bool,
                 profile_bin_width_gev: Optional[float] = None,
                 process_groups: Optional[Dict[str, List[int]]] = None):
        self.base_name = base_name
        self.bin_widths_gev = list(bin_widths_gev)
        self.mass = [
            _new_histogram(f"ROI_{base_name}_width_{w}", _nbins(w), weighted)
            for w in self.bin_widths_gev
        ]
        self.profile = None
        if profile_bin_width_gev:
            name = f"{RESOLUTION_PROFILE_PREFIX}{base_name}"
            self.profile = ROOT.TProfile(
                name, name, _nbins(profile_bin_width_gev), FIXED_MASS_MIN_GEV, FIXED_MASS_MAX_GEV
            )
            self.profile.SetDirectory(0)
        self.groups = {
            g: np.asarray(sorted(int(d) for d in dsids), dtype=np.int64)
            for g, dsids in (process_groups or {}).items()
        }
        self.process: Dict[str, List[ROOT.TH1]] = {}
        self.has_data = False

    def fill(self, masses, weights=None, sigmas=None, dsids=None) -> None:
        if len(masses) == 0:
            return
        self.has_data = True
        _fill_chunk(self.mass, masses, weights)
        x = _clip_to_range(masses)
        w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
        if self.profile is not None and sigmas is not None:
            # |w|: the profile is the mean sigma_m at a given mass; negative
            # generator weights would only make that mean unstable.
            self.profile.FillN(len(x), x, np.asarray(sigmas, dtype=np.float64), np.abs(w))
        if dsids is not None:
            dsids = np.asarray(dsids, dtype=np.int64)
            for group, members in self.groups.items():
                sel = np.isin(dsids, members)
                if not sel.any():
                    continue
                if group not in self.process:
                    self.process[group] = [
                        _new_histogram(f"{PROCESS_HIST_PREFIX}{group}_{self.base_name}_width_{bw}",
                                       _nbins(bw), True)
                        for bw in self.bin_widths_gev
                    ]
                for hist in self.process[group]:
                    hist.FillN(int(sel.sum()), x[sel], w[sel])

    def all_histograms(self) -> List[ROOT.TH1]:
        out = list(self.mass)
        if self.profile is not None and self.profile.GetEntries() > 0:
            out.append(self.profile)
        for hists in self.process.values():
            out.extend(hists)
        return out


def _nbins(bin_width: float) -> int:
    return max(1, math.ceil((FIXED_MASS_MAX_GEV - FIXED_MASS_MIN_GEV) / bin_width))


def _fill_chunk(histograms: List[ROOT.TH1], masses, weights=None) -> None:
    """Fill every histogram with a chunk of masses and, when given, their per-event MC weights."""
    x = _clip_to_range(masses)
    if len(x) == 0:
        return
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    for hist in histograms:
        hist.FillN(len(x), x, w)


def create_histograms(histograms_config: Dict, file_list: Optional[List[str]] = None):
    logger = _init_logging()

    input_dir = histograms_config["input_dir"]
    output_dir = histograms_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    bin_width_gev = histograms_config["bin_width_gev"]
    if isinstance(bin_width_gev, (int, float)):
        bin_widths_gev = [bin_width_gev]
    else:
        bin_widths_gev = bin_width_gev
    use_bumpnet_naming = histograms_config.get("use_bumpnet_naming", False)
    exclude_outliers = histograms_config.get("exclude_outliers", False)
    apply_peak_removal = histograms_config.get("apply_peak_removal_at_histogram_level", False)
    is_distributed_batch = (
        histograms_config.get("batch_job_index") is not None
        and histograms_config.get("total_batch_jobs") is not None
    )
    # Batch outputs keep the full grid for hadd; the merge step slices them afterwards.
    trim_before_write = histograms_config.get("trim_empty_bins", False) and not is_distributed_batch

    if file_list is not None:
        sqlite_files = [f for f in file_list if f.endswith(".sqlite")]
        if sqlite_files:
            logger.info(f"Using explicit file list with {len(sqlite_files)} SQLite processed shard file(s)")
            existing_sqlite = []
            for filename in sqlite_files:
                file_path = os.path.join(input_dir, filename)
                if os.path.exists(file_path):
                    existing_sqlite.append(filename)
                else:
                    logger.warning(f"File {filename} not found in {input_dir}, skipping")
            if not existing_sqlite:
                logger.warning(f"None of the {len(sqlite_files)} specified SQLite files exist in {input_dir}")
                return
            return _create_histograms_from_sqlite(
                existing_sqlite,
                input_dir,
                histograms_config,
                logger,
                apply_peak_removal=apply_peak_removal,
            )

        im_array_files = [f for f in file_list if f.endswith(".npy")]
        logger.info(f"Using explicit file list with {len(im_array_files)} IM array files")

        existing_files = []
        for filename in im_array_files:
            file_path = os.path.join(input_dir, filename)
            if os.path.exists(file_path):
                existing_files.append(filename)
            else:
                logger.warning(f"File {filename} not found in {input_dir}, skipping")
        im_array_files = existing_files

        if not im_array_files:
            logger.warning(f"None of the {len(file_list)} specified files exist in {input_dir}")
            return
    else:
        if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
            logger.warning(f"Input directory '{input_dir}' is empty or doesn't exist.")
            return

        sqlite_files = [f for f in os.listdir(input_dir) if f.endswith(".sqlite")]
        if sqlite_files:
            return _create_histograms_from_sqlite(
                sorted(sqlite_files),
                input_dir,
                histograms_config,
                logger,
                apply_peak_removal=apply_peak_removal,
            )

        im_array_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
        if not im_array_files:
            logger.warning(f"No .npy files found in {input_dir}")
            return

        batch_job_index = histograms_config.get("batch_job_index")
        total_batch_jobs = histograms_config.get("total_batch_jobs")
        if batch_job_index is not None and total_batch_jobs is not None:
            im_array_files = _get_batch_files(im_array_files, batch_job_index, total_batch_jobs)
            logger.info(f"Batch {batch_job_index}/{total_batch_jobs}: Processing {len(im_array_files)} files")

    if exclude_outliers:
        before_count = len(im_array_files)
        im_array_files = [f for f in im_array_files if "_outliers" not in f]
        excluded_count = before_count - len(im_array_files)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} outlier files (exclude_outliers=true)")

    total_im_arrays = len(im_array_files)
    logger.info(f"Found {total_im_arrays} IM array files to process")

    single_output_file = histograms_config.get("single_output_file", False)
    output_filename = histograms_config.get("output_filename", "all_histograms.root")

    if use_bumpnet_naming:
        logger.info("Using BumpNet naming mode: grouping files by signature and merging")
        _process_im_arrays_bumpnet(
            input_dir, output_dir, bin_widths_gev, logger, im_array_files,
            single_output_file=single_output_file,
            output_filename=output_filename,
            apply_peak_removal=apply_peak_removal,
            trim_before_write=trim_before_write,
        )
    else:
        logger.info("Using standard naming mode")
        _process_im_arrays_standard(
            input_dir, output_dir, bin_widths_gev, logger, im_array_files,
            single_output_file=single_output_file,
            output_filename=output_filename,
            apply_peak_removal=apply_peak_removal,
            trim_before_write=trim_before_write,
        )


def _create_histograms_from_sqlite(
    sqlite_files: List[str],
    input_dir: str,
    histograms_config: Dict,
    logger: logging.Logger,
    apply_peak_removal: bool = False,
):
    output_dir = histograms_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)


    bin_width_gev = histograms_config["bin_width_gev"]
    bin_widths_gev = [bin_width_gev] if isinstance(bin_width_gev, (int, float)) else bin_width_gev
    use_bumpnet_naming = histograms_config.get("use_bumpnet_naming", False)
    exclude_outliers = histograms_config.get("exclude_outliers", False)
    single_output_file = histograms_config.get("single_output_file", False)
    output_filename = histograms_config.get("output_filename", "all_histograms.root")
    extras = {
        "profile_bin_width_gev": histograms_config.get("resolution_profile_bin_width_gev"),
        "process_groups": histograms_config.get("process_groups"),
    }

    db_paths = [os.path.join(input_dir, f) for f in sqlite_files]

    # Split SQLite files across histogram batch jobs
    batch_job_index = histograms_config.get("batch_job_index")
    total_batch_jobs = histograms_config.get("total_batch_jobs")
    trim_before_write = histograms_config.get("trim_empty_bins", False) and (
        batch_job_index is None or total_batch_jobs is None
    )
    if batch_job_index is not None and total_batch_jobs is not None:
        sqlite_files = _get_batch_files(
            sorted(sqlite_files), batch_job_index, total_batch_jobs
        )
        db_paths = [os.path.join(input_dir, f) for f in sqlite_files]
        logger.info(
            f"Batch {batch_job_index}/{total_batch_jobs}: "
            f"processing SQLite files: {sqlite_files}"
        )
    
    signatures = set()
    for db_path in db_paths:
        signatures.update(list_signatures(db_path))
    signatures = sorted(signatures)

    # Per-event sibling arrays (MC weights, sigma_m, DSID) are consumed during
    # the fill; they are never histogrammed as data. MC weights turn on Sumw2
    # so bin errors are weighted errors.
    weighted = any(s.endswith(MC_WEIGHT_SUFFIX) for s in signatures)
    signatures = [s for s in signatures if not is_sibling_signature(s)]
    if weighted:
        logger.info("MC event weights present: filling weighted histograms with Sumw2")

    if exclude_outliers:
        before = len(signatures)
        signatures = [s for s in signatures if not s.endswith("_outliers")]
        logger.info(f"Excluded {before - len(signatures)} outlier signatures (exclude_outliers=true)")

    logger.info(
        f"Creating histograms from {len(signatures)} signatures across {len(sqlite_files)} SQLite file(s)"
    )
    if not signatures:
        logger.warning("No signatures found after filtering; skipping histogram creation")
        return

    if use_bumpnet_naming:
        grouped = _group_signatures_by_bumpnet(signatures)
        logger.info(f"Grouped {len(signatures)} signatures into {len(grouped)} unique histogram signatures")
        if single_output_file:
            root_filepath = os.path.join(output_dir, output_filename)
            hist_count = 0
            for bumpnet_name, group_sigs in grouped.items():
                hists = _create_merged_histograms_from_sqlite_signatures(
                    group_sigs, db_paths, bumpnet_name, bin_widths_gev, logger,
                    weighted=weighted, **extras,
                )
                if hists:
                    if trim_before_write:
                        hists = [slice_empty_bins(hist) for hist in hists]
                    _write_hists_to_shared_file(hists, root_filepath, logger)
                    hist_count += len(hists)
            logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
        else:
            for bumpnet_name, group_sigs in grouped.items():
                hists = _create_merged_histograms_from_sqlite_signatures(
                    group_sigs, db_paths, bumpnet_name, bin_widths_gev, logger,
                    weighted=weighted, **extras,
                )
                if hists:
                    if trim_before_write:
                        hists = [slice_empty_bins(hist) for hist in hists]
                    root_filename = f"{bumpnet_name}_hists.root"
                    root_filepath = os.path.join(output_dir, root_filename)
                    root_file = ROOT.TFile(root_filepath, "RECREATE")
                    for hist in hists:
                        hist.Write()
                    root_file.Close()
    else:
        if single_output_file:
            root_filepath = os.path.join(output_dir, output_filename)
            hist_count = 0
            for signature in signatures:
                hists = _create_histograms_for_signature(
                    signature, db_paths, bin_widths_gev, logger, apply_peak_removal,
                    weighted=weighted, **extras,
                )
                if hists:
                    if trim_before_write:
                        hists = [slice_empty_bins(hist) for hist in hists]
                    _write_hists_to_shared_file(hists, root_filepath, logger)
                    hist_count += len(hists)
            logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
        else:
            for signature in signatures:
                hists = _create_histograms_for_signature(
                    signature, db_paths, bin_widths_gev, logger, apply_peak_removal,
                    weighted=weighted, **extras,
                )
                if hists:
                    if trim_before_write:
                        hists = [slice_empty_bins(hist) for hist in hists]
                    root_filename = f"{signature}_hists.root"
                    root_filepath = os.path.join(output_dir, root_filename)
                    root_file = ROOT.TFile(root_filepath, "RECREATE")
                    for hist in hists:
                        hist.Write()
                    root_file.Close()


def _group_signatures_by_bumpnet(signatures: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for signature in signatures:
        # signature format: <prefix>_FS_<fs>_IM_<im>[_main|_outliers]
        cleaned = signature
        if cleaned.endswith("_main"):
            cleaned = cleaned[:-5]
        elif cleaned.endswith("_outliers"):
            cleaned = cleaned[:-9]

        # match = re.search(r"_FS_(\d+e_\d+m_\d+j_\d+g)_IM_([emjg\d]+)$", cleaned)
        match = re.search(r"_FS_([0-9emjgtb_]+)_IM_([emjgtb\d]+)$", cleaned)
        if not match:
            continue
        fs_str, im_str = match.groups()
        bumpnet_name = _convert_to_bumpnet_name(fs_str, im_str)
        groups[bumpnet_name].append(signature)
    return dict(groups)


def _iter_weighted_signature_chunks(signature: str, db_paths: List[str], logger: logging.Logger):
    """
    Yield ``(masses, weights, sigmas, dsids)`` per stored chunk of ``signature``.

    The last three are the aligned chunks of the ``_mcw``, ``_sigm`` and
    ``_dsid`` sibling signatures, or None when the signature has no such
    sibling. A sibling that does not line up chunk-for-chunk is ignored (with
    a warning) rather than misapplied.
    """
    for db_path in db_paths:
        chunks = [a for a in iter_arrays_for_signature(db_path, signature) if len(a) > 0]
        siblings = []
        for suffix in (MC_WEIGHT_SUFFIX, SIGMA_SUFFIX, DSID_SUFFIX):
            sib = [a for a in iter_arrays_for_signature(db_path, signature + suffix) if len(a) > 0]
            aligned = bool(sib) and len(sib) == len(chunks) and all(
                len(w) == len(c) for w, c in zip(sib, chunks)
            )
            if sib and not aligned:
                logger.warning(
                    f"{signature}: {suffix} in {os.path.basename(db_path)} does not align "
                    "with the masses; ignoring it"
                )
            siblings.append(sib if aligned else None)
        for i, chunk in enumerate(chunks):
            yield (chunk, *(sib[i] if sib is not None else None for sib in siblings))


def _create_histograms_for_signature(
    signature: str,
    db_paths: List[str],
    bin_widths_gev: List[float],
    logger: logging.Logger,
    apply_peak_removal: bool = False,
    weighted: bool = False,
    profile_bin_width_gev: Optional[float] = None,
    process_groups: Optional[Dict[str, List[int]]] = None,
) -> List[ROOT.TH1]:
    hset = _HistogramSet(signature, bin_widths_gev, weighted,
                         profile_bin_width_gev, process_groups)
    for chunk, weights, sigmas, dsids in _iter_weighted_signature_chunks(signature, db_paths, logger):
        hset.fill(chunk, weights, sigmas, dsids)

    if not hset.has_data:
        return []

    if apply_peak_removal:
        for hist in hset.mass:
            _apply_peak_removal_to_histogram(hist)

    return hset.all_histograms()


def _create_merged_histograms_from_sqlite_signatures(
    signatures: List[str],
    db_paths: List[str],
    hist_name_base: str,
    bin_widths_gev: List[float],
    logger: logging.Logger,
    apply_peak_removal: bool = False,
    weighted: bool = False,
    profile_bin_width_gev: Optional[float] = None,
    process_groups: Optional[Dict[str, List[int]]] = None,
) -> List[ROOT.TH1]:
    if 'cat' not in hist_name_base and 'hCat' not in hist_name_base:
        raise ValueError(
            f"Invalid histogram name base '{hist_name_base}': must contain 'cat'"
        )

    hset = _HistogramSet(hist_name_base, bin_widths_gev, weighted,
                         profile_bin_width_gev, process_groups)
    for signature in signatures:
        for chunk, weights, sigmas, dsids in _iter_weighted_signature_chunks(signature, db_paths, logger):
            hset.fill(chunk, weights, sigmas, dsids)

    if not hset.has_data:
        return []

    if apply_peak_removal:
        for hist in hset.mass:
            _apply_peak_removal_to_histogram(hist)

    return hset.all_histograms()

def _group_im_files_by_signature(im_files: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    unmatched_files = []
    for filename in im_files:
        match = re.search(r'_FS_([0-9emjgtb_]+)_IM_([emjgtb\d]+)', filename)
        if match:
            fs_str, im_str = match.groups()
            bumpnet_name = _convert_to_bumpnet_name(fs_str, im_str)
            groups[bumpnet_name].append(filename)
        else:
            unmatched_files.append(filename)

    if unmatched_files:
        print(f"WARNING: {len(unmatched_files)} files didn't match FS/IM pattern and will be skipped")
    return dict(groups)


def _convert_to_bumpnet_name(fs_str: str, im_str: str) -> str:
    """
    Convert FS and IM strings to a BumpNet histogram name.

    IM string is now index-based (e.g. "e0e1j0") so it passes through
    directly as the combo part — no conversion needed.

    FS string remains count-based (e.g. "2e_0m_3j_0g") and is formatted
    as before (e.g. "2ex_0mx_3jx_0gx").

    Examples:
      fs="2e_0m_3j_0g"  im="e0j0"   → mass_e0j0_cat_2ex_0mx_3jx_0gx
      fs="2e_0m_3j_0g"  im="e0e1j0" → mass_e0e1j0_cat_2ex_0mx_3jx_0gx
      fs="2e_0m_3j_0g"  im="e1j0"   → mass_e1j0_cat_2ex_0mx_3jx_0gx  (sub-leading)
      fs="1e_0m_0j_0g_1bx"  im="b0e0"   → mass_b0e0_cat_1ex_0mx_0jx_0gx_1bx for btag case

    The regex in _group_signatures_by_bumpnet that feeds this function
    also needs to be updated.
    """
    # IM part is already index-based — use directly as combo
    combo = im_str if im_str else "none"

    # FS part: count-based "2e_0m_3j_0g" → "2ex_0mx_3jx_0gx"
    #fs_particles = re.findall(r'(\d+)([emjg])', fs_str)
    fs_particles = re.findall(r'(\d+)([emjgtb])', fs_str)
    fs_formatted  = "_".join(f"{c}{p}x" for c, p in fs_particles)

    result = f"mass_{combo}_cat_{fs_formatted}"

    if 'cat' not in result and 'hCat' not in result:
        raise ValueError(
            f"Generated histogram name '{result}' doesn't contain 'cat' — "
            "BumpNet incompatible"
        )
    return result

def _process_im_arrays_bumpnet(
    im_arrays_dir: str, output_dir: str, bin_widths_gev: list,
    logger: logging.Logger, im_array_files: List[str],
    single_output_file: bool = False,
    output_filename: str = "all_histograms.root",
    apply_peak_removal: bool = False,
    trim_before_write: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    grouped_files = _group_im_files_by_signature(im_array_files)
    logger.info(f"Grouped {len(im_array_files)} files into {len(grouped_files)} unique signatures")

    if single_output_file:
        root_filepath = os.path.join(output_dir, output_filename)
        hist_count = 0

        for bumpnet_name, matching_files in grouped_files.items():
            logger.info(f"Processing {bumpnet_name}: merging {len(matching_files)} files")
            hists = _create_merged_histograms_streaming(
                matching_files,
                im_arrays_dir,
                bumpnet_name,
                bin_widths_gev,
                logger,
                apply_peak_removal=apply_peak_removal,
            )
            if hists:
                if trim_before_write:
                    hists = [slice_empty_bins(hist) for hist in hists]
                _write_hists_to_shared_file(hists, root_filepath, logger)
                hist_count += len(hists)

        logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
    else:
        for bumpnet_name, matching_files in grouped_files.items():
            logger.info(f"Processing {bumpnet_name}: merging {len(matching_files)} files")
            hists = _create_merged_histograms_streaming(
                matching_files,
                im_arrays_dir,
                bumpnet_name,
                bin_widths_gev,
                logger,
                apply_peak_removal=apply_peak_removal,
            )
            if hists:
                if trim_before_write:
                    hists = [slice_empty_bins(hist) for hist in hists]
                root_filename = f"{bumpnet_name}_hists.root"
                root_filepath = os.path.join(output_dir, root_filename)
                root_file = ROOT.TFile(root_filepath, "RECREATE")
                for hist in hists:
                    hist.Write()
                root_file.Close()
                logger.debug(f"Saved {len(hists)} histograms to {root_filepath}")


def _create_merged_histograms_streaming(
    files: List[str], directory: str, hist_name_base: str,
    bin_widths_gev: list, logger: logging.Logger, apply_peak_removal: bool = False
) -> List[ROOT.TH1F]:
    histograms = []
    for bin_width in bin_widths_gev:
        nbins = max(1, math.ceil((FIXED_MASS_MAX_GEV - FIXED_MASS_MIN_GEV) / bin_width))
        hist_name = f"ROI_{hist_name_base}_width_{bin_width}"
        if 'cat' not in hist_name_base and 'hCat' not in hist_name_base:
            logger.error(
                f"CRITICAL: Histogram name base '{hist_name_base}' doesn't contain 'cat' or 'hCat' "
                "- this will cause UnboundLocalError in BumpNet!"
            )
            raise ValueError(
                f"Invalid histogram name base '{hist_name_base}': must contain 'cat' for BumpNet compatibility"
            )
        hist = ROOT.TH1F(
            hist_name, hist_name, nbins,
            FIXED_MASS_MIN_GEV, FIXED_MASS_MAX_GEV,
        )
        histograms.append(hist)

    total_entries = 0
    for f in files:
        try:
            arr = np.load(os.path.join(directory, f))
            total_entries += len(arr)
            for hist in histograms:
                for val in arr:
                    _fill_mass(hist, val)
            del arr
        except Exception as e:
            logger.warning(f"Error filling from {f}: {e}")
            continue

    if total_entries == 0:
        logger.warning(f"No valid data found for {hist_name_base}")
        return []

    logger.debug(
        f"{hist_name_base}: {total_entries} entries, fixed range "
        f"[{FIXED_MASS_MIN_GEV:.2f}, {FIXED_MASS_MAX_GEV:.2f}]"
    )

    if apply_peak_removal:
        for hist in histograms:
            _apply_peak_removal_to_histogram(hist)
    return histograms


def _process_im_arrays_standard(
    im_arrays_dir: str, output_dir: str, bin_widths_gev: list,
    logger: logging.Logger, im_array_files: Optional[List[str]] = None,
    single_output_file: bool = False,
    output_filename: str = "all_histograms.root",
    apply_peak_removal: bool = False,
    trim_before_write: bool = False,
):
    if im_array_files is None:
        im_array_files = [f for f in os.listdir(im_arrays_dir) if f.endswith(".npy")]

    os.makedirs(output_dir, exist_ok=True)

    if single_output_file:
        root_filepath = os.path.join(output_dir, output_filename)
        hist_count = 0

        for im_array_filename in im_array_files:
            hists = _make_histograms_single_file(
                im_array_filename,
                im_arrays_dir,
                bin_widths_gev,
                logger,
                apply_peak_removal=apply_peak_removal,
            )
            if hists:
                if trim_before_write:
                    hists = [slice_empty_bins(hist) for hist in hists]
                _write_hists_to_shared_file(hists, root_filepath, logger)
                hist_count += len(hists)

        logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
    else:
        for im_array_filename in im_array_files:
            hists = _make_histograms_single_file(
                im_array_filename,
                im_arrays_dir,
                bin_widths_gev,
                logger,
                apply_peak_removal=apply_peak_removal,
            )
            if trim_before_write:
                hists = [slice_empty_bins(hist) for hist in hists]
            _save_hists(hists, output_dir, im_array_filename, logger)


def _make_histograms_single_file(
    im_array_filename: str, im_arrays_dir: str,
    bin_widths_gev: list, logger: logging.Logger, apply_peak_removal: bool = False
):
    im_array = np.load(os.path.join(im_arrays_dir, im_array_filename))
    hists = []
    for bin_width in bin_widths_gev:
        hist = _create_histogram_single_array(im_array_filename, im_array, bin_width)
        if apply_peak_removal:
            _apply_peak_removal_to_histogram(hist)
        hists.append(hist)
    return hists


def _apply_peak_removal_to_histogram(hist: ROOT.TH1F) -> None:
    """Zero all bins strictly before the rightmost highest bin."""
    nbins = hist.GetNbinsX()
    if nbins <= 0:
        return
    max_count = hist.GetMaximum()
    if max_count <= 0:
        return

    peak_bin_idx = None
    for bin_idx in range(nbins, 0, -1):
        if hist.GetBinContent(bin_idx) == max_count:
            peak_bin_idx = bin_idx
            break

    if peak_bin_idx is None or peak_bin_idx <= 1:
        return

    for bin_idx in range(1, peak_bin_idx):
        hist.SetBinContent(bin_idx, 0.0)
        hist.SetBinError(bin_idx, 0.0)


def _create_histogram_single_array(im_array_filename, im_array, bin_width) -> ROOT.TH1F:
    nbins = max(1, math.ceil((FIXED_MASS_MAX_GEV - FIXED_MASS_MIN_GEV) / bin_width))
    hist_name = f"ROI_{im_array_filename}_width_{bin_width}"
    hist = ROOT.TH1F(
        hist_name, hist_name, nbins,
        FIXED_MASS_MIN_GEV, FIXED_MASS_MAX_GEV,
    )

    for mass in im_array:
        _fill_mass(hist, mass)
    return hist


def _save_hists(hists: List[ROOT.TH1F], output_dir: str,
                im_array_filename: str, logger: logging.Logger) -> None:
    if not hists:
        logger.warning(f"No histograms to save for {im_array_filename}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = im_array_filename.replace(".npy", "")
    root_filename = f"{base_name}_hists.root"
    root_filepath = os.path.join(output_dir, root_filename)

    root_file = ROOT.TFile(root_filepath, "RECREATE")
    for hist in hists:
        hist.Write()
    root_file.Close()
    logger.debug(f"Saved {len(hists)} histograms to {root_filepath}")


def _write_hists_to_shared_file(
    hists: List[ROOT.TH1F], root_filepath: str,
    logger: logging.Logger, max_retries: int = 10, retry_delay: float = 0.5
) -> None:
    if not hists:
        return

    lock_filepath = root_filepath + ".lock"

    for attempt in range(max_retries):
        try:
            lock_file = open(lock_filepath, 'w')
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                if os.path.exists(root_filepath):
                    root_file = ROOT.TFile(root_filepath, "UPDATE")
                else:
                    root_file = ROOT.TFile(root_filepath, "RECREATE")

                if not root_file or root_file.IsZombie():
                    logger.error(f"Failed to open ROOT file: {root_filepath}")
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                    return

                for hist in hists:
                    hist.Write()
                root_file.Close()

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                logger.debug(f"Wrote {len(hists)} histograms to {root_filepath}")
                return

            except BlockingIOError:
                lock_file.close()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to acquire lock after {max_retries} attempts for {root_filepath}")
                    return

        except Exception as e:
            logger.error(f"Error writing to shared ROOT file: {e}")
            if 'lock_file' in locals() and not lock_file.closed:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                except Exception:
                    pass
            return


def _get_batch_files(files: List[str], batch_index: int, total_batches: int) -> List[str]:
    batch_index = int(batch_index)
    total_batches = int(total_batches)
    total_files = len(files)
    files_per_batch = total_files // total_batches
    start_idx = (batch_index - 1) * files_per_batch

    if batch_index == total_batches:
        end_idx = total_files
    else:
        end_idx = start_idx + files_per_batch
    return files[start_idx:end_idx]


def _init_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

"""
Resolution-based rebinning of the merged mass histograms.

Mirrors the WLCG HistogramProduction step ``scripts/rebinStackHistograms.py``
(ANA-EXOT-2023-18) on resonex histograms:

1. sigma_m(m): Gaussian-process fit (RBF kernel) of the ``RESO_<base>``
   profile, which the histogram step fills with the per-event sigma_m from the
   public object-resolution table. Fewer than 10 usable profile points: the
   constant mean.
2. Full binning: from the first filled bin, each next edge is the previous
   edge + sigma_m(edge) / rho, at least one base bin, snapped to the base grid
   of the fine input histogram.
3. Region of interest (ROI), the part BumpNet sees:
   - same-flavour dilepton masses (ee, mumu): edges above m_Z + 15 GeV
     (106.18) and at or above the maximum bin (WLCG ``binning_after_Z``);
   - otherwise from the maximum bin to the last filled bin, with two
     overrides that need the process-group histograms of an MC run:
     * 1 lepton + 1 jet (j or b) and N_top > sqrt(N_tot): start >= 175 GeV;
     * e + mu and N_ztautau > sqrt(N_tot): start at the maximum bin above
       80 GeV.
4. ``mode: fit`` saves both binnings per histogram; ``mode: cached`` reuses
   them (e.g. data with the binning of the matching MC run).

Deviations from the WLCG script: the profile points are selected by their
effective entries instead of yerr > 0.01 (sigma_m here is computed per event,
so the error on the mean is small exactly where the statistics are best); the
ROI includes the last filled bin; there is no charge split, so the Z veto
applies to all same-flavour dilepton pairs; no even/odd or systematics
handling (resonex has neither).

Output histograms in ``output_filename`` (next to the merged file):
``rebinned_<base>`` (full binning) and ``ROI_<base>`` (ROI binning).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.pipelines.histograms_pipeline import (
    PROCESS_HIST_PREFIX,
    RESOLUTION_PROFILE_PREFIX,
)

logger = logging.getLogger(__name__)

TOP_GROUP = "top"
ZTAUTAU_GROUP = "ztautau"
_MASS_NAME = re.compile(r"^ROI_(?P<base>.+)_width_(?P<width>[0-9.]+)$")


@dataclass(frozen=True)
class RebinSettings:
    mode: str = "fit"                     # "fit" or "cached"
    base_bin_width_gev: float = 1.0
    binning_dir: str = "rebin"
    output_filename: str = "rebinned.root"
    rho: float = 1.0                      # bin width = sigma_m / rho
    gpr_length_scale: float = 30.0
    gpr_length_scale_bounds: Tuple[float, float] = (20.0, 500.0)
    gpr_alpha_floor: float = 1.0          # alpha = max(yerr^2, floor), as in WLCG
    min_profile_entries: float = 5.0
    min_gpr_points: int = 10
    z_veto_edge_gev: float = 91.18 + 15.0
    top_roi_start_gev: float = 175.0
    ztautau_cutoff_gev: float = 80.0
    # The binning ends where the expected background above it drops below this
    # many events (weighted, i.e. at the run's luminosity); 0 = last filled bin.
    min_tail_events: float = 1.0


@dataclass
class RebinResult:
    base: str
    full_edges: List[float]
    roi_edges: List[float]
    notes: List[str] = field(default_factory=list)
    sigma_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (mass, fitted sigma_m)


# --------------------------------------------------------------------------- #
# name parsing / topology
# --------------------------------------------------------------------------- #

def combination_objects(base: str) -> Dict[str, int]:
    """Object counts of the mass combination in a histogram base name.

    Works for BumpNet names (``mass_e0j0_cat_...``) and signature names
    (``..._FS_..._IM_e0j0[_main]``).
    """
    m = re.search(r"mass_([a-z0-9]+)_cat_", base) or re.search(r"_IM_([a-z0-9]+)", base)
    combo = m.group(1) if m else ""
    counts = {k: 0 for k in "emjbgt"}
    for letter in re.findall(r"([a-z])\d+", combo):
        if letter in counts:
            counts[letter] += 1
    return counts


def is_same_flavour_dilepton(objects: Dict[str, int]) -> bool:
    total = sum(objects.values())
    return total == 2 and (objects["e"] == 2 or objects["m"] == 2)


def is_1l_1jet(objects: Dict[str, int]) -> bool:
    n_lep = objects["e"] + objects["m"]
    b, j = objects["b"], objects["j"]
    return n_lep == 1 and ((b == 1 and j == 0) or (b == 0 and j == 1))


def is_emu(objects: Dict[str, int]) -> bool:
    return objects["e"] == 1 and objects["m"] == 1


# --------------------------------------------------------------------------- #
# histogram helpers (ROOT objects are only touched through these)
# --------------------------------------------------------------------------- #

def _contents(hist) -> np.ndarray:
    return np.array([hist.GetBinContent(b) for b in range(1, hist.GetNbinsX() + 1)])


def _low_edges(hist) -> np.ndarray:
    return np.array([hist.GetBinLowEdge(b) for b in range(1, hist.GetNbinsX() + 2)])


def _filled_range(hist) -> Optional[Tuple[float, float]]:
    """(low edge of the first, upper edge of the last) non-empty bin."""
    contents = _contents(hist)
    errors = np.array([hist.GetBinError(b) for b in range(1, hist.GetNbinsX() + 1)])
    filled = np.nonzero((contents != 0) | (errors != 0))[0]
    if len(filled) == 0:
        return None
    edges = _low_edges(hist)
    return float(edges[filled[0]]), float(edges[filled[-1] + 1])


def tail_end(hist, min_tail_events: float) -> Optional[float]:
    """Lowest grid edge above which the expected content is below ``min_tail_events``.

    The tail above it (including the overflow) holds fewer than ``min_tail_events``
    expected events. None when the whole histogram holds fewer.
    """
    contents = _contents(hist)
    overflow = float(hist.GetBinContent(hist.GetNbinsX() + 1))
    # tail[k] = content of bins k..end + overflow
    tail = np.concatenate([np.cumsum(contents[::-1])[::-1], [0.0]]) + overflow
    above = np.nonzero(tail >= min_tail_events)[0]
    if len(above) == 0:
        return None
    return float(_low_edges(hist)[above[-1] + 1])


def _integral(hist) -> float:
    return float(hist.Integral(0, hist.GetNbinsX() + 1)) if hist else 0.0


# --------------------------------------------------------------------------- #
# sigma_m fit
# --------------------------------------------------------------------------- #

def profile_points(profile, min_entries: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mass, mean sigma_m, error) of the usable profile bins."""
    n = profile.GetNbinsX()
    x = np.array([profile.GetBinCenter(b) for b in range(1, n + 1)])
    y = np.array([profile.GetBinContent(b) for b in range(1, n + 1)])
    yerr = np.array([profile.GetBinError(b) for b in range(1, n + 1)])
    w = np.array([profile.GetBinEffectiveEntries(b) for b in range(1, n + 1)])
    mask = (w >= min_entries) & (y > 0.1) & (y < x)
    return x[mask], y[mask], yerr[mask]


def fit_sigma_curve(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                    grid: np.ndarray, settings: RebinSettings) -> Tuple[np.ndarray, str]:
    """sigma_m on ``grid``: GPR as in WLCG, constant mean for few points."""
    if len(x) == 0:
        raise ValueError("no usable profile points")
    if len(x) < settings.min_gpr_points:
        return np.full_like(grid, float(np.mean(y))), "constant"

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    kernel = RBF(length_scale=settings.gpr_length_scale,
                 length_scale_bounds=settings.gpr_length_scale_bounds)
    alpha = np.maximum(yerr ** 2, settings.gpr_alpha_floor)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                   n_restarts_optimizer=1, random_state=42)
    gpr.fit(x.reshape(-1, 1), y)
    # Outside the fitted points the GP falls back to its mean; hold the values
    # at the outermost points instead, and never let the width go negative.
    pred = gpr.predict(np.clip(grid, x[0], x[-1]).reshape(-1, 1))
    return np.maximum(pred, 0.0), "gpr"


def resolution_edges(start: float, stop: float, sigma_at, grid_origin: float,
                     grid_width: float, rho: float = 1.0) -> List[float]:
    """Edges from ``start`` to ``stop``, width sigma_m(edge)/rho snapped to the grid.

    Each bin spans at least one grid bin. A last bin narrower than half the
    width it should have is merged into the previous one.
    """
    k = int(round((start - grid_origin) / grid_width))
    k_stop = int(round((stop - grid_origin) / grid_width))
    ks = [k]
    while ks[-1] < k_stop:
        edge = grid_origin + ks[-1] * grid_width
        step = max(1, int(round(sigma_at(edge) / rho / grid_width)))
        ks.append(min(ks[-1] + step, k_stop))
    if len(ks) >= 3:
        last_width = ks[-1] - ks[-2]
        wanted = max(1, int(round(sigma_at(grid_origin + ks[-2] * grid_width) / rho / grid_width)))
        if last_width < 0.5 * wanted:
            ks.pop(-2)
    return [grid_origin + kk * grid_width for kk in ks]


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #

def _rebinned_contents(contents_fine: np.ndarray, origin: float, width: float,
                       edges: List[float]) -> np.ndarray:
    """Sum fine-bin contents into ``edges`` (edges lie on the fine grid)."""
    idx = np.rint((np.asarray(edges) - origin) / width).astype(np.int64)
    idx = np.clip(idx, 0, len(contents_fine))
    cum = np.concatenate([[0.0], np.cumsum(contents_fine)])
    return cum[idx[1:]] - cum[idx[:-1]]


def roi_edges(full_edges: List[float], rebinned: np.ndarray, objects: Dict[str, int],
              n_total: float, n_top: float, n_ztautau: float,
              settings: RebinSettings, notes: List[str]) -> List[float]:
    """WLCG ROI logic on the rebinned contents (one value per full bin)."""
    edges = np.asarray(full_edges, dtype=np.float64)
    low = edges[:-1]
    if not np.any(rebinned != 0):
        return []
    max_edge = float(low[int(np.argmax(rebinned))])
    last_upper = float(edges[np.nonzero(rebinned)[0][-1] + 1])

    if is_same_flavour_dilepton(objects):
        notes.append(f"Z veto: edges > {settings.z_veto_edge_gev:.2f} and >= max {max_edge:.1f}")
        return [float(e) for e in edges if e > settings.z_veto_edge_gev and e >= max_edge
                and e <= last_upper]

    first = max_edge
    if is_1l_1jet(objects) and n_top > math.sqrt(max(n_total, 0.0)):
        before = first
        first = max(first, settings.top_roi_start_gev)
        notes.append(f"top override (N_top={n_top:.1f}, N_tot={n_total:.1f}): {before:.1f} -> {first:.1f}")
    if is_emu(objects) and n_total > 0 and n_ztautau > math.sqrt(n_total):
        before = first
        above = low >= settings.ztautau_cutoff_gev
        if np.any(above):
            first = float(low[above][int(np.argmax(rebinned[above]))])
        else:
            first = settings.ztautau_cutoff_gev
        notes.append(f"ztautau override (N_ztt={n_ztautau:.1f}, N_tot={n_total:.1f}): {before:.1f} -> {first:.1f}")
    return [float(e) for e in edges if first <= e <= last_upper]


# --------------------------------------------------------------------------- #
# one histogram
# --------------------------------------------------------------------------- #

def derive_binning(base: str, hraw, profile, process_hists: Dict[str, object],
                   settings: RebinSettings) -> Optional[RebinResult]:
    """Full and ROI binning of one histogram (``mode: fit``)."""
    filled = _filled_range(hraw)
    if filled is None or profile is None or profile.GetEntries() == 0:
        return None
    nbins = hraw.GetNbinsX()
    origin = hraw.GetXaxis().GetXmin()
    width = (hraw.GetXaxis().GetXmax() - origin) / nbins
    if not math.isclose(width, settings.base_bin_width_gev, rel_tol=1e-6):
        raise ValueError(f"{base}: input bin width {width} != base_bin_width_gev "
                         f"{settings.base_bin_width_gev}")

    stop = filled[1]
    if settings.min_tail_events > 0:
        end = tail_end(hraw, settings.min_tail_events)
        if end is None or end <= filled[0]:
            return None
        stop = min(stop, end)

    x, y, yerr = profile_points(profile, settings.min_profile_entries)
    if len(x) == 0:
        return None
    grid = np.arange(filled[0], stop + width, width)
    sigma, method = fit_sigma_curve(x, y, yerr, grid, settings)

    def sigma_at(m):
        return float(np.interp(m, grid, sigma))

    full = resolution_edges(filled[0], stop, sigma_at, origin, width, settings.rho)
    if len(full) < 2:
        return None
    notes = [f"sigma fit: {method} on {len(x)} profile points"]
    if stop < filled[1]:
        notes.append(f"end at {stop:.0f} GeV: < {settings.min_tail_events:g} expected events above "
                     f"(last filled bin {filled[1]:.0f} GeV)")
    rebinned = _rebinned_contents(_contents(hraw), origin, width, full)
    roi = roi_edges(full, rebinned, combination_objects(base), _integral(hraw),
                    _integral(process_hists.get(TOP_GROUP)),
                    _integral(process_hists.get(ZTAUTAU_GROUP)), settings, notes)
    return RebinResult(base, full, roi, notes, sigma_grid=(grid, sigma))


def _binning_path(settings: RebinSettings, base: str, which: str) -> Path:
    return Path(settings.binning_dir) / "binning_arrays" / f"{base.replace('/', '_')}_binning_{which}.npy"


def load_cached_binning(base: str, settings: RebinSettings) -> Optional[RebinResult]:
    full_p, roi_p = _binning_path(settings, base, "full"), _binning_path(settings, base, "roi")
    if not full_p.exists() or not roi_p.exists():
        return None
    return RebinResult(base, np.load(full_p).tolist(), np.load(roi_p).tolist(), ["cached"])


def save_binning(result: RebinResult, settings: RebinSettings) -> None:
    full_p = _binning_path(settings, result.base, "full")
    full_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(full_p, np.asarray(result.full_edges, dtype=np.float64))
    np.save(_binning_path(settings, result.base, "roi"), np.asarray(result.roi_edges, dtype=np.float64))
    if result.sigma_grid is not None:
        fits = Path(settings.binning_dir) / "fits_arrays"
        fits.mkdir(parents=True, exist_ok=True)
        np.save(fits / f"{result.base}_fit.npy", np.vstack(result.sigma_grid))


# --------------------------------------------------------------------------- #
# file level
# --------------------------------------------------------------------------- #

def rebin_file(input_path: str, settings: RebinSettings) -> Dict:
    """Rebin every base-width mass histogram of ``input_path``.

    Writes ``settings.output_filename`` next to the input and, in fit mode,
    the binning arrays and a summary to ``settings.binning_dir``.
    """
    import ROOT

    if settings.mode not in ("fit", "cached"):
        raise ValueError(f"rebin mode must be 'fit' or 'cached', got {settings.mode!r}")

    output_path = os.path.join(os.path.dirname(input_path), settings.output_filename)
    summary = {"input": input_path, "output": output_path, "mode": settings.mode,
               "histograms": {}, "skipped": []}

    fin = ROOT.TFile.Open(input_path, "READ")
    if not fin or fin.IsZombie():
        raise OSError(f"cannot open {input_path}")
    fout = ROOT.TFile(output_path, "RECREATE")
    try:
        names = [k.GetName() for k in fin.GetListOfKeys()]
        name_set = set(names)
        for name in sorted(names):
            m = _MASS_NAME.match(name)
            if not m or not math.isclose(float(m.group("width")), settings.base_bin_width_gev,
                                         rel_tol=1e-6):
                continue
            base, width_str = m.group("base"), m.group("width")
            hraw = fin.Get(name)
            if not hraw or hraw.InheritsFrom("TProfile"):
                continue

            if settings.mode == "fit":
                prof_name = f"{RESOLUTION_PROFILE_PREFIX}{base}"
                profile = fin.Get(prof_name) if prof_name in name_set else None
                procs = {}
                for group in (TOP_GROUP, ZTAUTAU_GROUP):
                    pname = f"{PROCESS_HIST_PREFIX}{group}_{base}_width_{width_str}"
                    if pname in name_set:
                        procs[group] = fin.Get(pname)
                result = derive_binning(base, hraw, profile, procs, settings)
            else:
                result = load_cached_binning(base, settings)

            if result is None or len(result.roi_edges) < 2:
                summary["skipped"].append(base)
                logger.warning(f"Skipping {base}: no usable binning")
                continue
            if settings.mode == "fit":
                save_binning(result, settings)

            full = np.asarray(result.full_edges, dtype=np.float64)
            roi = np.asarray(result.roi_edges, dtype=np.float64)
            for out_name, edges in ((f"rebinned_{base}", full), (f"ROI_{base}", roi)):
                hout = hraw.Rebin(len(edges) - 1, out_name, edges)
                hout.SetDirectory(0)
                ROOT.SetOwnership(hout, True)
                fout.WriteObject(hout, out_name)
            summary["histograms"][base] = {
                "n_bins_full": len(full) - 1, "n_bins_roi": len(roi) - 1,
                "roi": [float(roi[0]), float(roi[-1])], "notes": result.notes,
            }
    finally:
        fout.Close()
        fin.Close()

    if settings.mode == "fit":
        out_dir = Path(settings.binning_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "rebin_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        for tag, path in (("top override", "top_peak_removed_histograms.txt"),
                          ("ztautau override", "ztautau_peak_removed_histograms.txt")):
            with open(out_dir / path, "w") as f:
                f.write(f"# histograms with a {tag} of the ROI start\n")
                for base, info in summary["histograms"].items():
                    for note in info["notes"]:
                        if note.startswith(tag):
                            f.write(f"{base} | {note}\n")
    logger.info(f"Rebinned {len(summary['histograms'])} histograms into {output_path} "
                f"({len(summary['skipped'])} skipped)")
    return summary

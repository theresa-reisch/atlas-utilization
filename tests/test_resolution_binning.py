"""
Tests of the resolution-binning chain: per-event sigma_m, sibling arrays
through post-processing and histogramming, and the post-merge rebinning.

Needs awkward, vector, ROOT and scikit-learn (e.g. the LCG_107 view):
    python -m pytest tests/test_resolution_binning.py
"""
import logging
import math
import os
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.calculations.mass_resolution import (  # noqa: E402
    DEFAULT_TABLE_PATH, ObjectResolutionTable, mass_resolution,
)
from services.calculations.im_calculator import IMCalculator  # noqa: E402
from services.pipelines import rebin_pipeline as rb  # noqa: E402

ROOT = pytest.importorskip("ROOT")


def _toy_table(r_e=0.02, r_j=0.10, k_e=1.5):
    edges = {"eta_edges": [0.0, 1.0, 2.5], "pt_edges_gev": [25, 100, 1000]}
    return {
        "Electrons": {**edges, "rel_pt_resolution": [[r_e, r_e / 2], [2 * r_e, r_e]],
                      "data_mc_scale": [k_e, None]},
        "Jets": {**edges, "rel_pt_resolution": [[r_j, r_j], [r_j, r_j]]},
    }


def _events(pt_mev, eta, phi, kind="Electrons"):
    return ak.Array({kind: ak.zip({"pt": pt_mev, "eta": eta, "phi": phi,
                                    "mass": ak.zeros_like(ak.Array(pt_mev))})})


# --------------------------------------------------------------------------- #
# table and sigma_m
# --------------------------------------------------------------------------- #

def test_packaged_table_loads():
    table = ObjectResolutionTable.from_json(DEFAULT_TABLE_PATH)
    for obj in ("Electrons", "Muons", "Photons", "Jets", "BJets"):
        assert table.has(obj)
        rel = table.relative(obj, np.array([10.0, 50.0, 5000.0]), np.array([0.1, 1.2, 3.0]))
        assert np.all((rel > 0.001) & (rel < 0.5))


def test_table_lookup_clamps_and_scales():
    table = ObjectResolutionTable(_toy_table())
    rel = table.relative("Electrons", np.array([10.0, 50.0, 500.0, 5000.0, 50.0]),
                         np.array([0.5, -0.5, 0.5, 0.5, 2.0]))
    # rows 0 scaled by k = 1.5; row 1 has k = None -> unscaled; |eta| used
    assert np.allclose(rel, [0.03, 0.03, 0.015, 0.015, 0.04])
    unscaled = ObjectResolutionTable(_toy_table(), apply_data_mc_scale=False)
    assert unscaled.relative("Electrons", [50.0], [0.5])[0] == pytest.approx(0.02)


def test_sigma_back_to_back_pair():
    # Two massless objects, pT = p each, back to back at eta = 0: m = 2p and
    # sigma_m / m = r / sqrt(2).
    table = ObjectResolutionTable(_toy_table(k_e=1.0))
    ev = _events([[50e3, 50e3]], [[0.0, 0.0]], [[0.0, math.pi]])
    sigma = mass_resolution(ev, table)
    m = IMCalculator(ev, 0, 1, 4, 1, 4).calculate_invariant_mass(ev)
    assert float(m[0]) == pytest.approx(100e3, rel=1e-9)
    assert sigma[0] == pytest.approx(100e3 * 0.02 / math.sqrt(2), rel=1e-9)


def test_sigma_matches_toy_smearing():
    rng = np.random.default_rng(1)
    table = ObjectResolutionTable(_toy_table())
    # one electron pair + one jet, random kinematics
    pt_e = rng.uniform(30e3, 300e3, (2,))
    eta_e = rng.uniform(-2.4, 2.4, (2,))
    phi_e = rng.uniform(-math.pi, math.pi, (2,))
    pt_j, eta_j, phi_j = 80e3, 0.3, 1.0
    ev = ak.Array({
        "Electrons": ak.zip({"pt": [pt_e.tolist()], "eta": [eta_e.tolist()],
                             "phi": [phi_e.tolist()], "mass": [[0.511, 0.511]]}),
        "Jets": ak.zip({"pt": [[pt_j]], "eta": [[eta_j]], "phi": [[phi_j]], "mass": [[5e3]]}),
    })
    sigma = mass_resolution(ev, table)[0]

    n = 200_000
    r_e = table.relative("Electrons", pt_e * 1e-3, eta_e)
    r_j = table.relative("Jets", [pt_j * 1e-3], [eta_j])[0]
    s_e = 1 + rng.normal(0, 1, (n, 2)) * r_e
    s_j = 1 + rng.normal(0, 1, n) * r_j

    def four(pt, eta, phi, mass):
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        return np.sqrt(px**2 + py**2 + pz**2 + mass**2), px, py, pz

    tot = np.zeros((4, n))
    for i in range(2):
        tot += np.array(four(pt_e[i] * s_e[:, i], eta_e[i], phi_e[i], 0.511))
    tot += np.array(four(pt_j * s_j, eta_j, phi_j, 5e3))
    m = np.sqrt(tot[0]**2 - tot[1]**2 - tot[2]**2 - tot[3]**2)
    assert np.std(m) == pytest.approx(sigma, rel=0.03)


def test_sigma_unknown_object_raises():
    table = ObjectResolutionTable(_toy_table())
    with pytest.raises(KeyError):
        mass_resolution(_events([[50e3, 40e3]], [[0, 0]], [[0, 3]], kind="Muons"), table)


# --------------------------------------------------------------------------- #
# siblings: post-processing -> histograms
# --------------------------------------------------------------------------- #

def _write_shard(path, signature, masses, weights, sigmas, dsids):
    from services.storage.sqlite_shards import SqliteArrayShardWriter
    w = SqliteArrayShardWriter(path)
    w.append_many({signature: masses, signature + "_mcw": weights,
                   signature + "_sigm": sigmas, signature + "_dsid": dsids})
    w.commit()
    w.close()


def test_siblings_survive_postprocessing_and_fill(tmp_path):
    from services.pipelines.post_processing_pipeline import process_im_arrays
    from services.pipelines.histograms_pipeline import create_histograms
    from services.storage.sqlite_shards import iter_arrays_for_signature

    rng = np.random.default_rng(2)
    n = 5000
    masses = rng.uniform(20, 400, n)
    weights = rng.uniform(0.5, 2.0, n)
    sigmas = (0.02 * masses).astype(np.float32)
    dsids = np.where(rng.uniform(size=n) < 0.3, 410470, 700320).astype(np.int32)
    sig = "f_FS_2e_0m_0j_0g_IM_e0e1"
    im_dir, pp_dir, h_dir = tmp_path / "im", tmp_path / "pp", tmp_path / "h"
    im_dir.mkdir()
    _write_shard(str(im_dir / "im_batch_1.sqlite"), sig, masses, weights, sigmas, dsids)

    process_im_arrays({"input_dir": str(im_dir), "output_dir": str(pp_dir),
                       "peak_detection_bin_width_gev": 10.0, "z_peak_cutoff": 110.0,
                       "max_mass_cutoff": 10000.0, "enable_peak_removal": False})
    out = str(pp_dir / "processed_batch_1.sqlite")
    m_pp = np.concatenate(list(iter_arrays_for_signature(out, "_FS_2e_0m_0j_0g_IM_e0e1_main")))
    s_pp = np.concatenate(list(iter_arrays_for_signature(out, "_FS_2e_0m_0j_0g_IM_e0e1_main_sigm")))
    d_pp = np.concatenate(list(iter_arrays_for_signature(out, "_FS_2e_0m_0j_0g_IM_e0e1_main_dsid")))
    keep = masses >= 110.0
    assert len(m_pp) == keep.sum() == len(s_pp) == len(d_pp)
    assert np.allclose(s_pp, 0.02 * m_pp, rtol=1e-5)          # alignment kept

    create_histograms({"input_dir": str(pp_dir), "output_dir": str(h_dir), "bin_width_gev": [1.0],
                       "single_output_file": True, "output_filename": "h.root",
                       "use_bumpnet_naming": True, "exclude_outliers": False,
                       "resolution_profile_bin_width_gev": 10.0,
                       "process_groups": {"top": [410470], "ztautau": [700360]}})
    f = ROOT.TFile.Open(str(h_dir / "h.root"))
    base = "mass_e0e1_cat_2ex_0mx_0jx_0gx"
    h = f.Get(f"ROI_{base}_width_1.0")
    prof = f.Get(f"RESO_{base}")
    top = f.Get(f"PROC_top_{base}_width_1.0")
    assert h and prof and top
    assert not f.Get(f"PROC_ztautau_{base}_width_1.0")          # no Z->tautau events
    assert h.Integral() == pytest.approx(weights[keep].sum(), rel=1e-9)
    assert top.Integral() == pytest.approx(weights[keep & (dsids == 410470)].sum(), rel=1e-9)
    b = prof.FindBin(205.0)
    assert prof.GetBinContent(b) == pytest.approx(0.02 * 205.0, rel=0.03)
    f.Close()


# --------------------------------------------------------------------------- #
# rebinning
# --------------------------------------------------------------------------- #

def test_resolution_edges_constant_sigma():
    edges = rb.resolution_edges(100.0, 200.0, lambda m: 10.0, 0.0, 1.0)
    assert edges == [100.0 + 10 * i for i in range(11)]
    # sigma below one grid bin -> one grid bin
    assert rb.resolution_edges(0.0, 5.0, lambda m: 0.2, 0.0, 1.0) == [0, 1, 2, 3, 4, 5]
    # short last bin merged
    assert rb.resolution_edges(0.0, 21.0, lambda m: 10.0, 0.0, 1.0) == [0.0, 10.0, 21.0]


def test_combination_objects():
    assert rb.combination_objects("mass_e0e1_cat_2ex_0mx_0jx_0gx") == \
        {"e": 2, "m": 0, "j": 0, "b": 0, "g": 0, "t": 0}
    o = rb.combination_objects("x_FS_1e_0m_1j_0g_IM_e0j0_main")
    assert rb.is_1l_1jet(o) and not rb.is_emu(o)
    assert rb.is_same_flavour_dilepton(rb.combination_objects("mass_m0m1_cat_0ex_2mx_0jx_0gx"))
    assert not rb.is_same_flavour_dilepton(rb.combination_objects("mass_e0e1j0_cat_2ex_0mx_1jx_0gx"))


def test_roi_rules():
    s = rb.RebinSettings()
    edges = [float(e) for e in range(0, 401, 20)]
    contents = np.zeros(20)
    contents[4] = 100.0      # max at 80
    contents[5:19] = 10.0
    ee = rb.combination_objects("mass_e0e1_cat_2ex")
    assert rb.roi_edges(edges, contents, ee, 1, 0, 0, s, [])[0] == 120.0   # > 106.18
    plain = rb.combination_objects("mass_e0e1j0_cat_2ex_1jx")
    roi = rb.roi_edges(edges, contents, plain, 1, 0, 0, s, [])
    assert roi[0] == 80.0 and roi[-1] == 380.0                            # incl. last filled bin
    ej = rb.combination_objects("mass_e0j0_cat_1ex_1jx")
    assert rb.roi_edges(edges, contents, ej, 100, 20, 0, s, [])[0] == 180.0  # top: >= 175
    assert rb.roi_edges(edges, contents, ej, 1000, 20, 0, s, [])[0] == 80.0  # 20 < sqrt(1000)
    em = rb.combination_objects("mass_e0m0_cat_1ex_1mx")
    contents_em = contents.copy()
    contents_em[2] = 500.0                                                 # Z->tautau at 40
    contents_em[8] = 150.0                                                 # max above 80 at 160
    assert rb.roi_edges(edges, contents_em, em, 100, 0, 0, s, [])[0] == 40.0
    assert rb.roi_edges(edges, contents_em, em, 100, 0, 50, s, [])[0] == 160.0


def _fake_merged_file(path, base, rel_sigma=0.03, with_top=False):
    rng = np.random.default_rng(3)
    m = 60 + rng.exponential(120.0, 200_000)
    f = ROOT.TFile(path, "RECREATE")
    h = ROOT.TH1D(f"ROI_{base}_width_1.0", "", 10000, 0, 10000)
    h.Sumw2()
    p = ROOT.TProfile(f"RESO_{base}", "", 1000, 0, 10000)
    h.SetDirectory(0)
    p.SetDirectory(0)
    h.FillN(len(m), m, np.ones(len(m)))
    p.FillN(len(m), m, rel_sigma * m * rng.uniform(0.9, 1.1, len(m)), np.ones(len(m)))
    f.WriteObject(h, h.GetName())
    f.WriteObject(p, p.GetName())
    if with_top:
        t = h.Clone(f"PROC_top_{base}_width_1.0")
        t.SetDirectory(0)
        t.Scale(0.5)
        f.WriteObject(t, t.GetName())
    f.Close()


def test_rebin_file_fit_and_cached(tmp_path):
    base = "mass_e0j0_cat_1ex_0mx_1jx_0gx"
    merged = str(tmp_path / "merged.root")
    _fake_merged_file(merged, base, with_top=True)
    settings = rb.RebinSettings(binning_dir=str(tmp_path / "rebin"))
    summary = rb.rebin_file(merged, settings)
    info = summary["histograms"][base]
    assert info["roi"][0] >= 175.0                        # top override applied
    full = np.load(tmp_path / "rebin" / "binning_arrays" / f"{base}_binning_full.npy")
    widths, lows = np.diff(full), full[:-1]
    mid = (lows > 150) & (lows < 800)
    # bin width ~ sigma_m = 3 % of the mass (GPR smooths, grid rounds to 1 GeV)
    assert np.allclose(widths[mid], 0.03 * lows[mid], rtol=0.25, atol=1.0)

    f = ROOT.TFile.Open(str(tmp_path / "rebinned.root"))
    roi = f.Get(f"ROI_{base}")
    full_h = f.Get(f"rebinned_{base}")
    assert roi.GetNbinsX() == info["n_bins_roi"]
    assert full_h.Integral() == pytest.approx(200_000, rel=1e-9)
    f.Close()

    # cached mode on a "data" file without profile/process hists
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data = str(data_dir / "merged.root")
    _fake_merged_file(data, base)
    cached = rb.RebinSettings(mode="cached", binning_dir=str(tmp_path / "rebin"))
    s2 = rb.rebin_file(data, cached)
    assert s2["histograms"][base]["roi"] == info["roi"]
    assert s2["histograms"][base]["n_bins_full"] == info["n_bins_full"]


def test_trim_keeps_profiles():
    from services.pipelines.histograms_pipeline import slice_empty_bins
    p = ROOT.TProfile("p_trim", "", 10, 0, 10)
    p.SetDirectory(0)
    p.Fill(5.5, 2.0)
    assert slice_empty_bins(p) is p

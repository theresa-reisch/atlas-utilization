"""Per-object pT resolution from public ATLAS Open Data MC truth (release 2024r-pp).

Only public inputs are used: Open Data background MC (no signal samples) and
Open Data collision data for the data/MC scale factors.

Steps (each one a sub-command, run on compute nodes):

  collect  --kind {e,mu,gamma,jet} --dsid D --file-index I --out DIR
      Reads one file, selects reco objects like the resonex parser does (but
      with a looser pT cut so the lowest truth-pT bin is not threshold-biased),
      matches them to truth and writes DIR/<kind>_<dsid>_<i>.npz with
      (truth pT, reco pT, reco eta, is_btag).

  zpeak    --flav {e,mu} --source {mc,data} --file-index I --out DIR
      Leading-pair m_ll for events with exactly two selected same-flavour
      leptons. For MC also the dressed-truth m_ll of matched pairs, so the
      residual m_reco - m_truth can be rescaled. Writes DIR/z_<flav>_<src>_<i>.npz.

  combine  --in DIR --out table.json
      Builds the resolution table: core width (IQR/1.349) of pT_reco/pT_truth
      per (|eta| bin, truth-pT bin), per object type, plus the data/MC scale
      factors from the Z peaks.

Truth references: dressed truth leptons (TruthLink), truth photons
(TruthLink), AntiKt4TruthDressedWZJets for jets (dR < 0.3 to the closest
truth jet, truth jet isolated: no other truth jet pT > 7 GeV within dR < 0.6).
"""
import argparse
import glob
import json
import os
import re

import awkward as ak
import numpy as np

INVALID_INDEX = 4294967295

# Analysis selection (resonex config.yaml kinematic_cuts); pT cut loosened for derivation.
ETA_MAX = {"e": 2.47, "mu": 2.5, "gamma": 2.37, "jet": 2.5}
ANALYSIS_PT_MIN = {"e": 25.0, "mu": 25.0, "gamma": 25.0, "jet": 30.0}
DERIVE_PT_MIN = {"e": 15.0, "mu": 15.0, "gamma": 15.0, "jet": 20.0}
REL_ISO_MAX = 0.06
ISO = {
    "e": "ptvarcone30_Nonprompt_All_MaxWeightTTVALooseCone_pt1000",
    "mu": "ptvarcone30_Nonprompt_All_MaxWeightTTVA_pt1000",
}
RECO = {"e": "AnalysisElectrons", "mu": "AnalysisMuons", "gamma": "AnalysisPhotons", "jet": "AnalysisJets"}
TRUTH = {"e": "TruthElectrons", "mu": "TruthMuons", "gamma": "TruthPhotons"}
DL1D_THRESHOLD = 2.51  # resonex jet_btagging_thresholds.DL1d
DL1D_FC = 0.018

# Binning of the table. |eta| edges follow detector regions.
ETA_BINS = {
    "e": [0.0, 1.37, 1.52, 2.47],        # barrel, crack, endcap
    "gamma": [0.0, 1.37, 1.52, 2.37],
    "mu": [0.0, 1.05, 2.0, 2.5],          # barrel, endcap, forward endcap
    "jet": [0.0, 0.8, 1.6, 2.5],
}
PT_BINS = {
    "e": [25, 35, 45, 60, 80, 120, 200, 400, 1000, 3000],
    "gamma": [25, 35, 45, 60, 80, 120, 200, 400, 1000, 3000],
    "mu": [25, 35, 45, 60, 80, 120, 200, 400, 1000, 3000],
    "jet": [30, 40, 50, 65, 85, 110, 150, 200, 300, 450, 700, 1000, 1500, 2500, 4000],
}
MIN_ENTRIES = 50


def _urls(dsid):
    import atlasopenmagic as atom
    atom.set_release("2024r-pp")
    return atom.get_urls(str(dsid))


def _open(url):
    from od_io import local_tree
    return local_tree(url)


def _dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _gather(values, idx):
    """values[idx] per event with idx possibly out of range (pad with 0)."""
    return ak.fill_none(ak.pad_none(values, 1), 0.0)[idx]


# --------------------------------------------------------------------- collect
def _collect_lepton_photon(kind, tree):
    r, t = RECO[kind], TRUTH[kind]
    br = {
        "pt": f"{r}AuxDyn.pt", "eta": f"{r}AuxDyn.eta", "phi": f"{r}AuxDyn.phi",
        "link": f"{r}AuxDyn.TruthLink/{r}AuxDyn.TruthLink.m_persIndex",
    }
    if kind in ISO:
        br["iso"] = f"{r}AuxDyn.{ISO[kind]}"
    if kind == "gamma":
        br.update(tpx=f"{t}AuxDyn.px", tpy=f"{t}AuxDyn.py", tpz=f"{t}AuxDyn.pz")
    else:
        br.update(tpt=f"{t}AuxDyn.pt_dressed", teta=f"{t}AuxDyn.eta_dressed", tphi=f"{t}AuxDyn.phi_dressed")
    a = tree.arrays(list(br.values()), library="ak")
    a = {k: a[v] for k, v in br.items()}

    pt = a["pt"] / 1000.0
    sel = (pt > DERIVE_PT_MIN[kind]) & (abs(a["eta"]) < ETA_MAX[kind])
    if "iso" in a:
        sel = sel & (a["iso"] / a["pt"] < REL_ISO_MAX)
    ntruth_src = a["tpx"] if kind == "gamma" else a["tpt"]
    ntruth = ak.num(ntruth_src)
    valid = (a["link"] != INVALID_INDEX) & (a["link"] < ntruth)
    idx = ak.where(valid, a["link"], 0)
    if kind == "gamma":
        px, py, pz = (_gather(a[k], idx) for k in ("tpx", "tpy", "tpz"))
        tpt = np.hypot(px, py) / 1000.0
        teta = np.arcsinh(pz / np.maximum(np.hypot(px, py), 1e-9))
        tphi = np.arctan2(py, px)
    else:
        tpt = _gather(a["tpt"], idx) / 1000.0
        teta = _gather(a["teta"], idx)
        tphi = _gather(a["tphi"], idx)
    dr = np.sqrt((a["eta"] - teta) ** 2 + _dphi(a["phi"], tphi) ** 2)
    m = sel & valid & (ntruth > 0) & (dr < 0.1)
    n_sel = int(ak.sum(sel))
    out = dict(
        tpt=ak.to_numpy(ak.flatten(tpt[m])), pt=ak.to_numpy(ak.flatten(pt[m])),
        eta=ak.to_numpy(ak.flatten(a["eta"][m])),
    )
    out["btag"] = np.zeros(len(out["pt"]), dtype=bool)
    return out, n_sel


def _collect_jets(tree):
    r = "AnalysisJetsAuxDyn"
    tj = "AntiKt4TruthDressedWZJetsAux./AntiKt4TruthDressedWZJetsAux"
    br = {
        "pt": f"{r}.pt", "eta": f"{r}.eta", "phi": f"{r}.phi",
        "blink": f"{r}.btaggingLink/{r}.btaggingLink.m_persIndex",
        "pb": "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb",
        "pc": "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc",
        "pu": "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu",
        "tpt": f"{tj}.pt", "teta": f"{tj}.eta", "tphi": f"{tj}.phi",
    }
    a = tree.arrays(list(br.values()), library="ak")
    a = {k: a[v] for k, v in br.items()}

    # b-tagging exactly as services/parsing/file_parser.py
    idx = a["blink"]
    pb, pc, pu = a["pb"][idx], a["pc"][idx], a["pu"][idx]
    den = DL1D_FC * pc + (1 - DL1D_FC) * pu
    ok = (pb > 0) & (den > 0)
    dl1d = np.log(ak.where(ok, pb, 1.0) / ak.where(ok, den, 1.0))
    btag = ak.where(ok, dl1d > DL1D_THRESHOLD, False)

    pt = a["pt"] / 1000.0
    sel = (pt > DERIVE_PT_MIN["jet"]) & (abs(a["eta"]) < ETA_MAX["jet"])
    tpt_all = a["tpt"] / 1000.0
    tsel = tpt_all > 7.0
    tpt, teta, tphi = tpt_all[tsel], a["teta"][tsel], a["tphi"][tsel]

    pairs = ak.cartesian({"r": ak.zip({"eta": a["eta"], "phi": a["phi"]}),
                          "t": ak.zip({"eta": teta, "phi": tphi})}, nested=True)
    dr = np.sqrt((pairs.r.eta - pairs.t.eta) ** 2 + _dphi(pairs.r.phi, pairs.t.phi) ** 2)
    order = ak.argsort(dr, axis=2)
    dr_sorted = dr[order]
    best = ak.firsts(order, axis=2)
    dr_best = ak.fill_none(ak.firsts(dr_sorted, axis=2), 99.0)
    # Isolation of the matched truth jet from other truth jets.
    tt = ak.cartesian({"a": ak.zip({"eta": teta, "phi": tphi}),
                       "b": ak.zip({"eta": teta, "phi": tphi})}, nested=True)
    dtt = np.sqrt((tt.a.eta - tt.b.eta) ** 2 + _dphi(tt.a.phi, tt.b.phi) ** 2)
    dtt = ak.where(dtt < 1e-6, 99.0, dtt)  # self-pairs
    t_iso = ak.fill_none(ak.min(dtt, axis=2), 99.0) > 0.6

    best_safe = ak.fill_none(best, 0)
    has_t = ak.num(tpt) > 0
    m_tpt = _gather(tpt, best_safe)
    m_iso = ak.values_astype(ak.fill_none(ak.pad_none(t_iso, 1), False)[best_safe], bool)
    m = sel & (dr_best < 0.3) & m_iso & has_t
    n_sel = int(ak.sum(sel))
    out = dict(
        tpt=ak.to_numpy(ak.flatten(m_tpt[m])), pt=ak.to_numpy(ak.flatten(pt[m])),
        eta=ak.to_numpy(ak.flatten(a["eta"][m])), btag=ak.to_numpy(ak.flatten(btag[m])),
    )
    return out, n_sel


def cmd_collect(args):
    url = _urls(args.dsid)[args.file_index]
    with _open(url) as tree:
        if args.kind == "jet":
            out, n_sel = _collect_jets(tree)
        else:
            out, n_sel = _collect_lepton_photon(args.kind, tree)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.kind}_{args.dsid}_{args.file_index}.npz")
    np.savez_compressed(path, **out, n_sel=n_sel, url=url)
    print(f"{path}: {len(out['pt'])} matched of {n_sel} selected", flush=True)


# ----------------------------------------------------------------------- zpeak
Z_MC = {"e": "700322", "mu": "700325"}


def _data_urls(cache_dir):
    """All 2015-18 collision-data file URLs of the release (cached as JSON in cache_dir)."""
    cache = os.path.join(cache_dir, "data_urls.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    import atlasopenmagic as atom
    atom.set_release("2024r-pp")
    urls = set()
    for d in atom.available_datasets():
        try:
            urls.update(u for u in atom.get_urls(d) if re.search(r"data1[5-8]_13TeV", u))
        except Exception:  # noqa: BLE001
            continue
    urls = sorted(urls)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cache + f".{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(urls, f)
    os.replace(tmp, cache)
    return urls


def _select_pair(a, flav, pt_key="pt"):
    pt = a["pt"] / 1000.0
    sel = (pt > ANALYSIS_PT_MIN[flav]) & (abs(a["eta"]) < ETA_MAX[flav]) & (a["iso"] / a["pt"] < REL_ISO_MAX)
    return sel


def cmd_zpeak(args):
    """MC: one Z sample per flavour. Data: one download serves both flavours."""
    flavs = ["e", "mu"] if args.flav == "both" else [args.flav]
    if args.source == "mc":
        if len(flavs) != 1:
            raise SystemExit("--source mc needs --flav e or mu")
        url = _urls(Z_MC[flavs[0]])[args.file_index]
    else:
        urls = _data_urls(args.out)
        # Spread over the four years: stride through the sorted list.
        step = max(1, len(urls) // 400)
        url = urls[(args.file_index * step) % len(urls)]
    with _open(url) as tree:
        for flav in flavs:
            _zpeak_from_tree(args, flav, RECO[flav], url, tree)


def _zpeak_from_tree(args, flav, r, url, tree):
    br = {"pt": f"{r}AuxDyn.pt", "eta": f"{r}AuxDyn.eta", "phi": f"{r}AuxDyn.phi",
          "q": f"{r}AuxDyn.charge", "iso": f"{r}AuxDyn.{ISO[flav]}"}
    if args.source == "mc":
        t = TRUTH[flav]
        br.update(link=f"{r}AuxDyn.TruthLink/{r}AuxDyn.TruthLink.m_persIndex",
                  tpt=f"{t}AuxDyn.pt_dressed", teta=f"{t}AuxDyn.eta_dressed", tphi=f"{t}AuxDyn.phi_dressed")
    missing = [v for v in br.values() if v not in tree.keys()]
    if missing:
        print(f"skip {url}: missing {missing[:2]}", flush=True)
        return
    a = tree.arrays(list(br.values()), library="ak")
    a = {k: a[v] for k, v in br.items()}
    sel = _select_pair(a, flav)
    two = ak.sum(sel, axis=1) == 2
    s = {k: v[sel][two] for k, v in a.items() if k not in ("tpt", "teta", "tphi")}
    os_ = (s["q"][:, 0] * s["q"][:, 1]) < 0
    pt, eta, phi = s["pt"] / 1000.0, s["eta"], s["phi"]

    def mass(p1, e1, f1, p2, e2, f2):
        return np.sqrt(2 * p1 * p2 * (np.cosh(e1 - e2) - np.cos(f1 - f2)))

    m_reco = mass(pt[:, 0], eta[:, 0], phi[:, 0], pt[:, 1], eta[:, 1], phi[:, 1])
    out = dict(m=ak.to_numpy(m_reco[os_]), eta1=ak.to_numpy(eta[:, 0][os_]), eta2=ak.to_numpy(eta[:, 1][os_]))
    if args.source == "mc":
        ntruth = ak.num(a["tpt"])[two]
        link = s["link"]
        valid = ak.all((link != INVALID_INDEX) & (link < ntruth), axis=1) & (ntruth > 0)
        idx = ak.where((link != INVALID_INDEX) & (link < ntruth), link, 0)
        tpt = _gather(a["tpt"][two], idx) / 1000.0
        teta = _gather(a["teta"][two], idx)
        tphi = _gather(a["tphi"][two], idx)
        m_true = mass(tpt[:, 0], teta[:, 0], tphi[:, 0], tpt[:, 1], teta[:, 1], tphi[:, 1])
        out["m_true"] = ak.to_numpy(ak.where(valid, m_true, np.nan)[os_])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"z_{flav}_{args.source}_{args.file_index}.npz")
    np.savez_compressed(path, **out, url=url)
    print(f"{path}: {len(out['m'])} OS pairs", flush=True)


# --------------------------------------------------------------------- combine
def core_width(x):
    if len(x) < MIN_ENTRIES:
        return np.nan, np.nan
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    return (q75 - q25) / 1.349, q50


def _table(kind, d, btag=None):
    eta_bins, pt_bins = ETA_BINS[kind], PT_BINS[kind]
    sel = np.ones(len(d["pt"]), dtype=bool) if btag is None else (d["btag"] == btag)
    tpt, resp, aeta = d["tpt"][sel], (d["pt"] / d["tpt"])[sel], np.abs(d["eta"][sel])
    width = np.full((len(eta_bins) - 1, len(pt_bins) - 1), np.nan)
    scale = np.full_like(width, np.nan)
    count = np.zeros_like(width, dtype=int)
    for i in range(len(eta_bins) - 1):
        for j in range(len(pt_bins) - 1):
            b = (aeta >= eta_bins[i]) & (aeta < eta_bins[i + 1]) & (tpt >= pt_bins[j]) & (tpt < pt_bins[j + 1])
            count[i, j] = int(b.sum())
            width[i, j], scale[i, j] = core_width(resp[b])
    return dict(eta_edges=eta_bins, pt_edges_gev=pt_bins,
                rel_pt_resolution=_fill_gaps(width).tolist(), response_median=scale.tolist(),
                entries=count.tolist())


def _fill_gaps(w):
    """Bins below MIN_ENTRIES take the nearest filled pT bin of the same eta row."""
    w = w.copy()
    for i in range(w.shape[0]):
        filled = np.where(np.isfinite(w[i]))[0]
        if len(filled) == 0:
            continue
        for j in np.where(~np.isfinite(w[i]))[0]:
            w[i, j] = w[i, filled[np.argmin(np.abs(filled - j))]]
    return w


# Pythia8 gamma+jet slices (lower truth photon pT edge of each slice, GeV). Below its edge a slice
# has almost no prompt photons, and the truth link then points to a soft (non-prompt) photon, which
# gives pT_reco/pT_truth >> 1. Those entries are dropped.
GAMMA_SLICE_MIN_GEV = {"423104": 140.0, "423105": 280.0, "423106": 500.0, "423107": 800.0,
                       "423108": 1000.0}


def _truth_pt_floor(path):
    m = re.match(r"gamma_(\d+)_\d+\.npz$", os.path.basename(path))
    return GAMMA_SLICE_MIN_GEV.get(m.group(1), 0.0) if m else 0.0


def _load(pattern):
    files = sorted(glob.glob(pattern))
    parts = [np.load(f) for f in files]
    if not parts:
        return None, 0
    keys = [k for k in parts[0].files if k not in ("url", "n_sel")]
    masks = [p["tpt"] >= _truth_pt_floor(f) if "tpt" in p.files else slice(None)
             for f, p in zip(files, parts)]
    return {k: np.concatenate([p[k][m] for p, m in zip(parts, masks)]) for k in keys}, len(files)


def _z_core(m):
    w = m[(m > 80) & (m < 100)]
    q = np.quantile(w, [0.25, 0.75])
    return (q[1] - q[0]) / 1.349


def _region(eta, edges):
    return np.clip(np.searchsorted(edges, np.abs(eta), side="right") - 1, 0, len(edges) - 2)


def _z_scale_factor(mc, data, edges, region=None):
    """k such that m_true + k (m_reco - m_true) in MC has the data Z core width.

    Restricted to pairs with both leptons in |eta| region ``region`` when given.
    """
    def pick(d):
        if region is None:
            return np.ones(len(d["m"]), dtype=bool)
        return (_region(d["eta1"], edges) == region) & (_region(d["eta2"], edges) == region)

    md, mm = data["m"][pick(data)], mc["m"][pick(mc)]
    mt = mc["m_true"][pick(mc)]
    ok = np.isfinite(mt)
    mm, mt = mm[ok], mt[ok]
    if len(md) < 200 or len(mm) < 2000:
        return None
    target = _z_core(md)
    ks = np.linspace(0.5, 1.6, 111)
    widths = np.array([_z_core(mt + k * (mm - mt)) for k in ks])
    k = float(np.interp(target, widths, ks)) if np.all(np.diff(widths) > 0) else float(ks[np.argmin(np.abs(widths - target))])
    # Statistical uncertainty of the data core width by bootstrap.
    rng = np.random.default_rng(1)
    boots = [_z_core(rng.choice(md, len(md))) for _ in range(100)]
    dk = float(np.std([np.interp(b, widths, ks) for b in boots]))
    return dict(k=k, k_stat=dk, data_core_gev=float(target), mc_core_gev=float(_z_core(mm)),
                n_data=int(len(md)), n_mc=int(len(mm)))


def cmd_combine(args):
    table = {"description": (
        "Relative pT resolution sigma(pT)/pT = core width (IQR/1.349) of pT_reco/pT_truth "
        "in bins of |eta| (rows) and truth pT (columns), from ATLAS Open Data 2024r-pp MC. "
        "data_mc_scale multiplies rel_pt_resolution to match the data Z peak."),
        "release": "2024r-pp", "objects": {}}
    for kind, name in (("e", "Electrons"), ("mu", "Muons"), ("gamma", "Photons")):
        d, n = _load(os.path.join(args.inp, f"{kind}_*.npz"))
        if d is None:
            continue
        table["objects"][name] = dict(_table(kind, d), n_files=n)
    d, n = _load(os.path.join(args.inp, "jet_*.npz"))
    if d is not None:
        table["objects"]["Jets"] = dict(_table("jet", d, btag=False), n_files=n)
        table["objects"]["BJets"] = dict(_table("jet", d, btag=True), n_files=n)

    for flav, name in (("e", "Electrons"), ("mu", "Muons")):
        mc, _ = _load(os.path.join(args.inp, f"z_{flav}_mc_*.npz"))
        data, _ = _load(os.path.join(args.inp, f"z_{flav}_data_*.npz"))
        if mc is None or data is None or name not in table["objects"]:
            continue
        edges = ETA_BINS[flav]
        inclusive = _z_scale_factor(mc, data, edges)
        per_region = [_z_scale_factor(mc, data, edges, region=i) for i in range(len(edges) - 1)]
        scale = [r["k"] if r else inclusive["k"] for r in per_region]
        table["objects"][name]["data_mc_scale"] = scale
        table["objects"][name]["data_mc_scale_detail"] = dict(inclusive=inclusive, per_eta_region=per_region)
    if "Electrons" in table["objects"] and "Photons" in table["objects"]:
        # No clean public photon peak: photons take the electron calorimeter scale factors.
        table["objects"]["Photons"]["data_mc_scale"] = table["objects"]["Electrons"].get("data_mc_scale")
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("rel_pt_resolution", "data_mc_scale")}
                      for k, v in table["objects"].items()}, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--kind", choices=["e", "mu", "gamma", "jet"], required=True)
    c.add_argument("--dsid", required=True)
    c.add_argument("--file-index", type=int, default=0)
    c.add_argument("--out", required=True)
    z = sub.add_parser("zpeak")
    z.add_argument("--flav", choices=["e", "mu", "both"], required=True)
    z.add_argument("--source", choices=["mc", "data"], required=True)
    z.add_argument("--file-index", type=int, default=0)
    z.add_argument("--out", required=True)
    m = sub.add_parser("combine")
    m.add_argument("--in", dest="inp", required=True)
    m.add_argument("--out", required=True)
    args = p.parse_args()
    {"collect": cmd_collect, "zpeak": cmd_zpeak, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    main()

# Object pT resolution table (public inputs only)

`object_resolution_2024r-pp_v1.json` gives sigma(pT)/pT per object type
(Electrons, Muons, Photons, Jets, BJets) in bins of |eta| (rows) and pT
(columns, GeV). The mass step uses it to compute the per-event mass resolution
(`services/calculations/mass_resolution.py`), and the rebin step uses that in
turn (`services/pipelines/rebin_pipeline.py`).

## How it was made

Script: `tools/resolution/derive_object_resolution.py`. Job list:
`tools/resolution/jobs_v1.txt` (108 jobs; 104 ran, and the 4 that failed
asked for a second file of samples that have only one).

- **Inputs:** ATLAS Open Data release 2024r-pp only: background MC and 2015+16
  collision data. No internal ATLAS numbers and no BSM signal samples.
- **Object selection:** as in the resonex parser, except that the pT cut is
  loosened to 15 GeV (20 GeV for jets), so the lowest bin is not biased by
  the threshold.
- **Truth matching:**
  - leptons and photons through their reco TruthLink to the dressed truth
    particle;
  - jets to `AntiKt4TruthDressedWZJets` with dR < 0.3, where the truth jet
    must be isolated (no other truth jet with pT > 7 GeV within dR < 0.6).
- **Resolution value:** the core width of pT_reco / pT_truth, i.e. IQR/1.349,
  per (|eta|, truth pT) bin.
- **Gap filling:** bins with fewer than 50 entries take the nearest filled pT
  bin of the same |eta| row. In v1 this affects electrons and muons above
  1 TeV, and jets and b-jets at 2.5–4 TeV.
- **Samples:**
  - electrons: Z -> ee, 700322;
  - muons: Z -> mumu, 700325;
  - photons: Pythia8 gamma+jet slices 423104–423108 and Sherpa 700398/700399;
  - jets: dijet slices 364703–364707 and ttbar 410470 (b-jets are the
    b-tagged jets).
- **Photon slice floor:** in each Pythia8 gamma+jet slice, only photons above
  that slice's lower truth-pT edge are used. Below the edge the truth link
  points to soft, non-prompt photons.
- **`data_mc_scale` (one factor per |eta| row):** the factor k for which
  m_true + k (m_reco - m_true) in MC has the same Z-peak core width as data.
  It multiplies `rel_pt_resolution` unless `apply_data_mc_scale: false`.
  - Electrons: k = 1.12–1.18. The crack row uses the inclusive value.
  - Muons: k = 0.61–0.71. The Open Data MC Z -> mumu peak is wider than in
    data (core width 3.5 vs 2.85 GeV). After scaling, the central muon
    resolution is about 1.75 % at the Z, close to the published ATLAS value.
  - Photons: take the electron factors.
  - Jets: no factor yet (planned: W and top peaks in ttbar).

`data_mc_scale_detail` holds the inclusive and per-region fits with their
statistical uncertainties.

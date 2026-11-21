"""
Centralized constants for particle physics calculations.
"""
# Fallback masses, used only when a collection has no "mass" field.
# Units are MeV: four-vectors are built from PHYSLITE pt (MeV) and the
# invariant mass is converted to GeV only at the end (im_pipeline).
KNOWN_MASSES = {
    "Muons": 105.658,
    "Photons": 0.0,
    "Electrons": 0.511,
    "Jets": 0.0,
    "BJets": 0.0,
    "Taus": 1776.86,
}

# ATLAS PHYSLITE / AnalysisElectronsAuxDyn — relative isolation uses cone energy / pT
ELECTRON_REL_ISOLATION_FIELD = "ptvarcone30_Nonprompt_All_MaxWeightTTVALooseCone_pt1000"

LETTER_PARTICLE_MAPPING = {
    "e": "Electrons",
    "j": "Jets",
    "g": "Photons",
    "m": "Muons",
    "t": "Taus",
    "b": "BJets",
}

"""
Per-event invariant-mass resolution from per-object pT resolutions.

The object resolutions come from a public table (see
``resources/resolution/README.md``): sigma(pT)/pT per object type in bins of
|eta| and pT, plus a data/MC scale factor per |eta| row. The table is derived
from ATLAS Open Data MC truth only and its scale factors from the Z peak in
Open Data collision data.

Mass resolution of a combination of objects i with four-momenta p_i and total
P = sum_i p_i, treating the pT measurements as independent and the directions
as exact (each p_i scales with its pT):

    dm / d(ln pT_i) = (p_i . P) / m
    sigma_m = sqrt( sum_i (r_i * p_i . P)^2 ) / m,     r_i = sigma(pT_i) / pT_i

MET is never part of a combination, so it never needs a resolution.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import awkward as ak
import numpy as np

from domain.events import particle_fields
from services.calculations import physics_calcs

DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "resolution", "object_resolution_2024r-pp_v1.json",
)


class ObjectResolutionTable:
    """sigma(pT)/pT lookup per object type, with the data/MC scale applied."""

    def __init__(self, objects: Dict[str, dict], apply_data_mc_scale: bool = True):
        self._tables = {}
        for name, obj in objects.items():
            eta_edges = np.asarray(obj["eta_edges"], dtype=np.float64)
            pt_edges = np.asarray(obj["pt_edges_gev"], dtype=np.float64)
            rel = np.asarray(obj["rel_pt_resolution"], dtype=np.float64)
            if rel.shape != (len(eta_edges) - 1, len(pt_edges) - 1):
                raise ValueError(
                    f"{name}: rel_pt_resolution has shape {rel.shape}, expected "
                    f"{(len(eta_edges) - 1, len(pt_edges) - 1)}"
                )
            if not np.all(np.isfinite(rel)) or np.any(rel <= 0):
                raise ValueError(f"{name}: rel_pt_resolution must be finite and positive")
            scale = obj.get("data_mc_scale")
            if apply_data_mc_scale and scale is not None:
                rel = rel * np.asarray(
                    [1.0 if k is None else float(k) for k in scale], dtype=np.float64
                )[:, None]
            self._tables[name] = (eta_edges, pt_edges, rel)

    @classmethod
    def from_json(cls, path: Optional[str] = None, apply_data_mc_scale: bool = True):
        with open(path or DEFAULT_TABLE_PATH) as f:
            payload = json.load(f)
        return cls(payload["objects"], apply_data_mc_scale=apply_data_mc_scale)

    def has(self, object_type: str) -> bool:
        return object_type in self._tables

    def relative(self, object_type: str, pt_gev, eta) -> np.ndarray:
        """sigma(pT)/pT for flat arrays of pT [GeV] and eta.

        Values outside the table use the nearest bin: pT below the first edge
        the first column, pT above the last edge the last column, |eta| beyond
        the last edge the outermost row.
        """
        eta_edges, pt_edges, rel = self._tables[object_type]
        pt_gev = np.asarray(pt_gev, dtype=np.float64)
        abs_eta = np.abs(np.asarray(eta, dtype=np.float64))
        i_eta = np.clip(np.searchsorted(eta_edges, abs_eta, side="right") - 1, 0, rel.shape[0] - 1)
        i_pt = np.clip(np.searchsorted(pt_edges, pt_gev, side="right") - 1, 0, rel.shape[1] - 1)
        return rel[i_eta, i_pt]


def mass_resolution(
    particle_events: ak.Array,
    table: ObjectResolutionTable,
    pt_to_gev: float = 1e-3,
) -> np.ndarray:
    """
    Per-event sigma_m for the objects in ``particle_events``.

    ``particle_events`` must hold exactly the objects of the combination (after
    count filtering and rank slicing), as passed to
    ``IMCalculator.calculate_invariant_mass``. Returns sigma_m in the units of
    pT (MeV for parsed files); convert like the mass.

    Raises:
        KeyError: an object type of the combination is not in the table.
    """
    n_events = len(particle_events)
    if n_events == 0:
        return np.zeros(0, dtype=np.float64)

    parts = []
    for particle_type in particle_fields(particle_events):
        arr = particle_events[particle_type]
        if not table.has(particle_type):
            raise KeyError(f"no resolution for object type {particle_type!r}")
        mass = physics_calcs.get_particle_known_mass(particle_type, arr)
        pt, eta, phi = arr.pt, arr.eta, arr.phi
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        counts = ak.num(pt, axis=1)
        rel = table.relative(
            particle_type,
            ak.to_numpy(ak.flatten(pt, axis=None)) * pt_to_gev,
            ak.to_numpy(ak.flatten(eta, axis=None)),
        )
        parts.append((energy, px, py, pz, ak.unflatten(rel, counts)))

    tot = [sum(ak.sum(p[c], axis=1) for p in parts) for c in range(4)]
    m2 = tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2 - tot[3] ** 2

    var = 0.0
    for energy, px, py, pz, rel in parts:
        dot = energy * tot[0] - px * tot[1] - py * tot[2] - pz * tot[3]
        var = var + ak.sum((rel * dot) ** 2, axis=1)

    m2 = np.asarray(ak.to_numpy(m2), dtype=np.float64)
    var = np.asarray(ak.to_numpy(var), dtype=np.float64)
    sigma = np.zeros(n_events, dtype=np.float64)
    positive = m2 > 0
    sigma[positive] = np.sqrt(var[positive] / m2[positive])
    return sigma

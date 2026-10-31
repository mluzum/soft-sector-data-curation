#!/usr/bin/env python3
"""Curate the isobar identified-particle spectra from ins2822399.

Paper: STAR, "Tracking the baryon number with nuclear collisions",
Science (2026), arXiv:2408.15441.

Figure S2 contains 16 tables (8 species x Ru+Ru/Zr+Zr), each with five
centrality dependent variables (0-10, 10-20, 20-40, 40-60, 60-80%).  We
write one curated record per (table, centrality) pair, using ``dv_index``
to select the dependent variable.  The x-axis is pT/A; for pions, kaons
and protons A=1, so the stored pT is the physical pT.  Deuterons (A=2)
are skipped for now (the 2D campaign has no deuterons).
"""

from __future__ import annotations

import glob
import os
import re

import yaml

BASE = "hepdata_raw/records/ins2822399/HEPData-ins2822399-v1-yaml"
OUT = "data/particle_spectra"

# table filename fragment -> (system, species, pdg_id)
SPECIES = {
    "ru_$\\pi^+$": ("Ru+Ru", "pi+", 211),
    "ru_$\\pi^-$": ("Ru+Ru", "pi-", -211),
    "ru_$k^+$": ("Ru+Ru", "K+", 321),
    "ru_$k^-$": ("Ru+Ru", "K-", -321),
    "ru_proton": ("Ru+Ru", "p", 2212),
    "ru_anti-proton": ("Ru+Ru", "pbar", -2212),
    "zr_$\\pi^+$": ("Zr+Zr", "pi+", 211),
    "zr_$\\pi^-$": ("Zr+Zr", "pi-", -211),
    "zr_$k^+$": ("Zr+Zr", "K+", 321),
    "zr_$k^-$": ("Zr+Zr", "K-", -321),
    "zr_proton": ("Zr+Zr", "p", 2212),
    "zr_anti-proton": ("Zr+Zr", "pbar", -2212),
}

SYSTEM_TAG = {"Ru+Ru": "RuRu", "Zr+Zr": "ZrZr"}


def _table_label_map():
    """Map HEPData data_file -> (table name, description)."""
    out = {}
    for doc in yaml.safe_load_all(open(os.path.join(BASE, "submission.yaml"))):
        if isinstance(doc, dict) and doc.get("data_file"):
            out[doc["data_file"]] = (
                doc.get("name", ""), doc.get("description", ""))
    return out


def _pt_edges(indep):
    vals = indep["values"]
    edges = [float(vals[0]["low"])]
    for v in vals:
        edges.append(float(v["high"]))
    # Guard against float noise / non-monotonic input.
    for a, b in zip(edges, edges[1:]):
        assert b > a, f"non-monotonic pT edges: {edges}"
    return edges


def _centrality(dv):
    q = {x.get("name", "").lower(): x.get("value") for x in dv.get("qualifiers", [])}
    c = q.get("centrality")
    if c is None:
        raise ValueError(f"no centrality qualifier in {dv.get('header')}")
    return str(c).strip()


def main():
    labels = _table_label_map()
    os.makedirs(OUT, exist_ok=True)
    written = 0
    for path in sorted(glob.glob(os.path.join(BASE, "figure_s2_*.yaml"))):
        fn = os.path.basename(path)
        key = fn[len("figure_s2_"):-len(".yaml")]
        if key not in SPECIES:
            continue
        system, species, pdg = SPECIES[key]
        data = yaml.safe_load(open(path))
        pt = data["independent_variables"][0]
        edges = _pt_edges(pt)
        table_name, table_desc = labels.get(fn, ("", ""))
        docs = []
        for dv_index, dv in enumerate(data["dependent_variables"]):
            cent = _centrality(dv)
            docs.append({
                "observable_type": "particle_spectra",
                "inspire_id": "2822399",
                "version": 1,
                "hepdata_file": fn,
                "hepdata_table_label": table_desc or table_name,
                "qualifiers": {
                    "reaction": system,
                    "centrality": cent,
                    "particle": species,
                },
                "collision_system": {
                    "energy_gev": 200.0,
                    "reaction": system,
                },
                "centrality": {
                    "classes": [cent],
                    "estimator": {
                        "name": "RefMult",
                        "observable": "Nch",
                        "detector": "TPC",
                    },
                    "eta": [{"min": -0.5, "max": 0.5}],
                    # STAR Ntrk offline: pT > 0.2 GeV, no upper cut.
                    "pT": {"min": 0.2, "max": 100.0},
                },
                "particle_id": {
                    "species": species,
                    "pdg_id": pdg,
                },
                "kinematic_range": {
                    "pT": {"min": edges[0], "max": edges[-1], "bins": edges},
                    "y": {"min": -0.5, "max": 0.5},
                },
                "observable": "dN_dpT_dy",
                "value_type": "invariant_yield",
                "dv_index": dv_index,
            })
        out_name = (
            f"star_particle_spectra_identified_particle_spectra_"
            f"{SYSTEM_TAG[system]}_{species.replace('+', 'p').replace('-', 'm')}_200gev.yaml"
        )
        out_path = os.path.join(OUT, out_name)
        with open(out_path, "w") as f:
            yaml.safe_dump_all(docs, f, default_flow_style=False,
                               sort_keys=False, width=120, allow_unicode=True)
        written += len(docs)
        print(f"wrote {out_path} ({len(docs)} docs)")
    print(f"total spectra docs: {written}")


if __name__ == "__main__":
    main()

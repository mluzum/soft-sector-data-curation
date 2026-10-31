#!/usr/bin/env python3
"""Curate the ratio measurements from ins2822399.

  Figure S3: Ru+Ru/Zr+Zr double ratios of pi, K, p (5 centrality bins each)
  Figure 4:  <B>/<Delta Q> * Delta Z/A vs centrality

The S3 records point at the per-system, per-species spectra curated by
`curate_isobar_2822399.py` where possible.
"""

from __future__ import annotations

import os

import yaml

BASE = "hepdata_raw/records/ins2822399/HEPData-ins2822399-v1-yaml"
OUT = "data/observable_ratios"

S3 = {
    "figure_s3_ru_zr_pion.yaml": ("pi", "pi+/pi-", 40),
    "figure_s3_ru_zr_kaon.yaml": ("K", "K+/K-", 40),
    "figure_s3_ru_zr_proton.yaml": ("p", "p/pbar", 35),
}


def _pt_edges(indep):
    vals = indep["values"]
    edges = [float(vals[0]["low"])]
    for v in vals:
        edges.append(float(v["high"]))
    return edges


def _cent(dv):
    q = {x.get("name", "").lower(): x.get("value")
         for x in dv.get("qualifiers", [])}
    return str(q.get("centrality", "")).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for fn, (species, ratio_label, _nbins) in S3.items():
        data = yaml.safe_load(open(os.path.join(BASE, fn)))
        edges = _pt_edges(data["independent_variables"][0])
        for dv_index, dv in enumerate(data["dependent_variables"]):
            cent = _cent(dv)
            doc = {
                "observable_type": "observable_ratios",
                "inspire_id": "2822399",
                "version": 1,
                "hepdata_file": fn,
                "hepdata_table_label": "",
                "collision_system": {
                    "energy_gev": 200.0,
                    "reaction": "Ru+Ru and Zr+Zr",
                },
                "centrality": {
                    "classes": [cent],
                    "estimator": {"name": "RefMult", "observable": "Nch",
                                  "detector": "TPC"},
                    "eta": [{"min": -0.5, "max": 0.5}],
            "pT": {"min": 0.2, "max": 100.0},
                    "pT": {"min": 0.2, "max": 100.0},
                },
                "particle_id": {"species": species},
                "kinematic_range": {
                    "pT": {"min": edges[0], "max": edges[-1], "bins": edges},
                    "y": {"min": -0.5, "max": 0.5},
                },
                "observable": "particle_yield",
                "value_type": f"double_ratio_{species}",
                "numerator": "Ru+Ru / Zr+Zr",
                "denominator": ratio_label,
                "ratio_type": "double_ratio",
                "numerator_system": "Ru+Ru",
                "denominator_system": "Zr+Zr",
                "numerator_particle": ratio_label.split("/")[0],
                "denominator_particle": ratio_label.split("/")[1],
                "dv_index": dv_index,
            }
            out = (f"{OUT}/star_observable_ratios_isobar_double_ratio_"
                   f"{species}_{cent.replace('%','pct').replace('-','_')}"
                   f"_200gev.yaml")
            with open(out, "w") as f:
                yaml.safe_dump(doc, f, default_flow_style=False,
                               sort_keys=False, width=120,
                               allow_unicode=True)
            n += 1

    # Figure 4: B/DeltaQ * DeltaZ/A vs centrality (one DV, 5 points).
    data = yaml.safe_load(open(os.path.join(BASE, "figure_4.yaml")))
    doc = {
        "observable_type": "observable_ratios",
        "inspire_id": "2822399",
        "version": 1,
        "hepdata_file": "figure_4.yaml",
        "hepdata_table_label": "",
        "collision_system": {
            "energy_gev": 200.0,
            "reaction": "Ru+Ru and Zr+Zr",
        },
        "centrality": {
            # figure_4 lists centrality peripheral-first.
            "classes": ["60-80%", "40-60%", "20-40%", "10-20%", "0-10%"],
            "estimator": {"name": "RefMult", "observable": "Nch",
                          "detector": "TPC"},
            "eta": [{"min": -0.5, "max": 0.5}],
            "pT": {"min": 0.2, "max": 100.0},
        },
        "kinematic_range": {"y": {"min": -0.5, "max": 0.5}},
        "observable": "B_over_DeltaQ",
        "value_type": "B_over_DeltaQ_times_DeltaZ_over_A",
        "numerator": "B",
        "denominator": "DeltaQ",
        "ratio_type": "quantity_ratio",
        "dv_index": 0,
        "notes": "Net-baryon number over net-charge difference at midrapidity, "
                 "scaled by Delta Z / A = 4/96.",
    }
    out = f"{OUT}/star_observable_ratios_isobar_B_over_DeltaQ_200gev.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False,
                       width=120, allow_unicode=True)
    n += 1
    print(f"wrote {n} ratio records")


if __name__ == "__main__":
    main()

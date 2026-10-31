#!/usr/bin/env python3
"""Curate the Ru+Ru/Zr+Zr flow ratios from ins1914564.

These are the lower panels of Figs. 4 and 10.  They are stored as
`observable_ratios` records for the Bayesian analysis; the pipeline does
not dispatch them.  Numerator/denominator record paths point at the
per-system flow records curated by `curate_isobar_1914564_bulk.py`.
"""

from __future__ import annotations

import os

import yaml

BASE = "hepdata_raw/records/ins1914564/HEPData-ins1914564-v1-yaml"
OUT = "data/observable_ratios"

CENT_CLASSES = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%",
                "40-50%", "50-60%", "60-70%", "70-80%"]
CENT_MAP = {
    2.5: "0-5%", 7.5: "5-10%", 15.0: "10-20%", 25.0: "20-30%",
    35.0: "30-40%", 45.0: "40-50%", 55.0: "50-60%", 65.0: "60-70%",
    75.0: "70-80%",
}


def _classes_from_indep(indep):
    return [CENT_MAP[float(v["value"])] for v in indep.get("values", [])]

# table filename -> (harmonic, value_type, numerator record tag)
RATIOS = {
    "fig4_left_low_isobarpaper_star_blue_v2_tpc_ratio.yaml":
        (2, "v2{2}", "v22"),
    "fig4_left_low_isobarpaper_star_green_v2_tpc_eta_gt1_ratio.yaml":
        (2, "v2{2}(dEta>1)", "v22_dEtagt1"),
    "fig4_left_low_isobarpaper_star_purple_v2_subEv_ratio.yaml":
        (2, "v2{SP}", "v2SP"),
    "fig4_left_low_isobarpaper_star_red_v2_epd_ratio.yaml":
        (2, "v2{EP,TPC-EPD}", "v2EP_TPC-EPD"),
    "fig4_left_low_isobarpaper_star_yellow_v2_EP_ratio.yaml":
        (2, "v2{EP}", "v2EP"),
    "fig4_right_low_isobarpaper_star_green_v2_4_ratio.yaml":
        (2, "v2{4}", "v24"),
    "fig4_right_low_isobarpaper_star_green_v2_zdc_ratio.yaml":
        (2, "v2{ZDC}", "v2ZDC"),
    "fig10_left_low_isobarpaper_star_green_v3_tpc_ratio.yaml":
        (3, "v3{2}", "v32"),
    "fig10_left_low_isobarpaper_star_yellow_v3_epd_ratio.yaml":
        (3, "v3{EP,EPD}", "v3EP_EPD"),
    "fig10_right_low_isobarpaper_star_blue_v3_subEv_ratio.yaml":
        (3, "v3{SP}", "v3SP"),
    "fig10_right_low_isobarpaper_star_purple_v3_tpc_eta_gt1_ratio.yaml":
        (3, "v3{2}(dEta>1)", "v32_dEtagt1"),
}

TAG = {"v22": "v22", "v22_dEtagt1": "v22_dEtagt1", "v2SP": "v2SP",
       "v2EP_TPC-EPD": "v2EP_TPC-EPD", "v2EP": "v2EP", "v24": "v24",
       "v2ZDC": "v2ZDC", "v32": "v32", "v3EP_EPD": "v3EP_EPD",
       "v3SP": "v3SP", "v32_dEtagt1": "v32_dEtagt1"}


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for fn, (harmonic, vt, tag) in RATIOS.items():
        path = os.path.join(BASE, fn)
        data = yaml.safe_load(open(path))
        classes = _classes_from_indep(data["independent_variables"][0])
        doc = {
            "observable_type": "observable_ratios",
            "inspire_id": "1914564",
            "version": 1,
            "hepdata_file": fn,
            "hepdata_table_label": "",
            "collision_system": {
                "energy_gev": 200.0,
                "reaction": "Ru+Ru and Zr+Zr",
            },
            "centrality": {
                "classes": classes,
                "estimator": {"name": "RefMult", "observable": "Nch",
                              "detector": "TPC"},
                "eta": [{"min": -0.5, "max": 0.5}],
                "pT": {"min": 0.2, "max": 100.0},
            },
            "particle_id": {"species": "charged"},
            "kinematic_range": {
                "eta": {"min": -1.0, "max": 1.0, "bins": [-1.0, 1.0]},
                "pT": {"min": 0.2, "max": 100.0, "bins": [0.2, 100.0]},
            },
            "observable": f"v{harmonic}",
            "value_type": f"{vt}_Ru_over_Zr",
            "numerator": "Ru+Ru",
            "denominator": "Zr+Zr",
            "ratio_type": "system_ratio",
            "numerator_system": "Ru+Ru",
            "denominator_system": "Zr+Zr",
            "numerator_record": (
                f"data/flow_vn/star_flow_vn_isobar_v{harmonic}_{tag}_"
                f"RuRu_200gev.yaml"),
            "denominator_record": (
                f"data/flow_vn/star_flow_vn_isobar_v{harmonic}_{tag}_"
                f"ZrZr_200gev.yaml"),
            "dv_index": 0,
        }
        # Keep the filename simple and deterministic.
        out = (f"{OUT}/star_observable_ratios_isobar_v{harmonic}_"
               f"{tag}_Ru_over_Zr_200gev.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(doc, f, default_flow_style=False,
                           sort_keys=False, width=120, allow_unicode=True)
        n += 1
        print("wrote", out)
    print(f"total ratio records: {n}")


if __name__ == "__main__":
    main()

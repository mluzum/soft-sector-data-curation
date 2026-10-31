#!/usr/bin/env python3
"""Curate the bulk (multiplicity + v2/v3) tables from ins1914564.

Paper: STAR, "Search for the Chiral Magnetic Effect with Isobar Collisions
at sqrt(s_NN) = 200 GeV", PRC 105, 014901 (2022), arXiv:2109.00131.

The paper contains many CME-specific tables; this script extracts the
bulk-flow and mean-multiplicity measurements that a hydrodynamic model
should reproduce:

  fig3  <N_trk^offline> vs centrality, Ru+Ru and Zr+Zr
  fig4  v2 with several methods (v2{2}, v2{2}(dEta>1), v2{SP},
        v2{2}(TPC-EPD), v2{EP}, v2{4}, v2{ZDC})
  fig10 v3 with several methods (v3{2}, v3{2}(dEta>1), v3{SP}, v3{EP})

Only the canonical (non-duplicated) tables are written; fig6 and fig23
repeat values already present in fig4.  The ZDC and EPD reference
measurements are kept in the database but the 2D campaign (|eta| < 5) cannot
compute the ZDC one.
"""

from __future__ import annotations

import glob
import os

import yaml

BASE = "hepdata_raw/records/ins1914564/HEPData-ins1914564-v1-yaml"
OUT_FLOW = "data/flow_vn"
OUT_YIELD = "data/particle_yields"

CENT_CLASSES = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%",
                "40-50%", "50-60%", "60-70%", "70-80%"]
# HEPData tables are not consistent about centrality ordering: the v2{2}
# and v3{2} tables are central-first (2.5, 7.5, ..., 75), while v2{EP},
# v2{4}, v2{ZDC}, and the fig3 mean-multiplicity tables are
# peripheral-first (75, 65, ..., 2.5).  Some v3 EPD tables have only 8
# bins.  Always derive the class list from the table's own independent
# variable values instead of assuming a fixed order.
CENT_MAP = {
    2.5: "0-5%", 7.5: "5-10%", 15.0: "10-20%", 25.0: "20-30%",
    35.0: "30-40%", 45.0: "40-50%", 55.0: "50-60%", 65.0: "60-70%",
    75.0: "70-80%",
}


def _classes_from_indep(indep):
    """Map a table's centrality bin centres to class labels, in table order."""
    out = []
    for v in indep.get("values", []):
        c = float(v["value"])
        if c not in CENT_MAP:
            raise ValueError(f"unmapped centrality centre {c}")
        out.append(CENT_MAP[c])
    return out

# table filename -> (system, harmonic, method, cumulant_order, value_type,
#                   eta_gap, flow_reference)
V2_TABLES = {
    "fig4_left_top_isobarpaper_star_blue_solid_v2_2_ruru.yaml":
        ("Ru+Ru", 2, "cumulant", 2, "v2{2}", 0.05, None),
    "fig4_left_top_isobarpaper_star_blue_open_v2_2_zrzr.yaml":
        ("Zr+Zr", 2, "cumulant", 2, "v2{2}", 0.05, None),
    "fig4_left_top_isobarpaper_star_green_solid_v2_tpc_eta_gt1_ruru.yaml":
        ("Ru+Ru", 2, "cumulant", 2, "v2{2}(dEta>1)", 1.0, None),
    "fig4_left_top_isobarpaper_star_green_open_v2_tpc_eta_gt1_zrzr.yaml":
        ("Zr+Zr", 2, "cumulant", 2, "v2{2}(dEta>1)", 1.0, None),
    "fig4_left_top_isobarpaper_star_purple_solid_v2_subEv_ruru.yaml":
        ("Ru+Ru", 2, "scalar_product", None, "v2{SP}", 0.2,
         {"detector": "TPC", "same_sample_as_poi": False,
          "eta": [{"min": 0.1, "max": 1.0}, {"min": -1.0, "max": -0.1}]}),
    "fig4_left_top_isobarpaper_star_purple_open_v2_subEv_zrzr.yaml":
        ("Zr+Zr", 2, "scalar_product", None, "v2{SP}", 0.2,
         {"detector": "TPC", "same_sample_as_poi": False,
          "eta": [{"min": 0.1, "max": 1.0}, {"min": -1.0, "max": -0.1}]}),
    "fig4_left_top_isobarpaper_star_red_solid_v2_tpcepd_ruru.yaml":
        ("Ru+Ru", 2, "event_plane", None, "v2{EP,TPC-EPD}", None,
         {"detector": "EPD", "same_sample_as_poi": False,
          "eta": [{"min": 2.1, "max": 5.1}, {"min": -5.1, "max": -2.1}]}),
    "fig4_left_top_isobarpaper_star_red_open_v2_tpcepd_zrzr.yaml":
        ("Zr+Zr", 2, "event_plane", None, "v2{EP,TPC-EPD}", None,
         {"detector": "EPD", "same_sample_as_poi": False,
          "eta": [{"min": 2.1, "max": 5.1}, {"min": -5.1, "max": -2.1}]}),
    "fig4_left_top_isobarpaper_star_yellow_solid_v2_EP_ruru.yaml":
        ("Ru+Ru", 2, "event_plane", None, "v2{EP}", None,
         {"detector": "TPC", "same_sample_as_poi": True}),
    "fig4_left_top_isobarpaper_star_yellow_open_v2_EP_zrzr.yaml":
        ("Zr+Zr", 2, "event_plane", None, "v2{EP}", None,
         {"detector": "TPC", "same_sample_as_poi": True}),
    "fig4_right_top_isobarpaper_star_green_solid_v2_4_ruru.yaml":
        ("Ru+Ru", 2, "cumulant", 4, "v2{4}", None, None),
    "fig4_right_top_isobarpaper_star_green_open_v2_4_zrzr.yaml":
        ("Zr+Zr", 2, "cumulant", 4, "v2{4}", None, None),
    "fig4_right_top_isobarpaper_star_grey_solid_v2_zdc_ruru.yaml":
        ("Ru+Ru", 2, "event_plane", None, "v2{ZDC}", None,
         {"detector": "ZDC", "same_sample_as_poi": False,
          "eta": [{"min": 6.3, "max": 99.0}, {"min": -99.0, "max": -6.3}]}),
    "fig4_right_top_isobarpaper_star_grey_open_v2_zdc_zrzr.yaml":
        ("Zr+Zr", 2, "event_plane", None, "v2{ZDC}", None,
         {"detector": "ZDC", "same_sample_as_poi": False,
          "eta": [{"min": 6.3, "max": 99.0}, {"min": -99.0, "max": -6.3}]}),
}

V3_TABLES = {
    "fig10_left_mid_isobarpaper_star_green_solid_v3_tpc_ruru.yaml":
        ("Ru+Ru", 3, "cumulant", 2, "v3{2}", 0.05, None),
    "fig10_left_mid_isobarpaper_star_green_open_v3_tpc_zrzr.yaml":
        ("Zr+Zr", 3, "cumulant", 2, "v3{2}", 0.05, None),
    "fig10_right_top_isobarpaper_star_purple_solid_v3_tpc_eta_gt1_ruru.yaml":
        ("Ru+Ru", 3, "cumulant", 2, "v3{2}(dEta>1)", 1.0, None),
    "fig10_right_top_isobarpaper_star_purple_open_v3_tpc_eta_gt1_zrzr.yaml":
        ("Zr+Zr", 3, "cumulant", 2, "v3{2}(dEta>1)", 1.0, None),
    "fig10_right_top_isobarpaper_star_blue_solid_v3_subEv_ruru.yaml":
        ("Ru+Ru", 3, "scalar_product", None, "v3{SP}", 0.2,
         {"detector": "TPC", "same_sample_as_poi": False,
          "eta": [{"min": 0.1, "max": 1.0}, {"min": -1.0, "max": -0.1}]}),
    "fig10_right_top_isobarpaper_star_blue_open_v3_subEv_zrzr.yaml":
        ("Zr+Zr", 3, "scalar_product", None, "v3{SP}", 0.2,
         {"detector": "TPC", "same_sample_as_poi": False,
          "eta": [{"min": 0.1, "max": 1.0}, {"min": -1.0, "max": -0.1}]}),
    "fig10_right_top_isobarpaper_star_yellow_solid_v3_epd_ruru.yaml":
        ("Ru+Ru", 3, "event_plane", None, "v3{EP,EPD}", None,
         {"detector": "EPD", "same_sample_as_poi": False,
          "eta": [{"min": 2.1, "max": 5.1}, {"min": -5.1, "max": -2.1}]}),
    "fig10_right_top_isobarpaper_star_yellow_open_v3_epd_zrzr.yaml":
        ("Zr+Zr", 3, "event_plane", None, "v3{EP,EPD}", None,
         {"detector": "EPD", "same_sample_as_poi": False,
          "eta": [{"min": 2.1, "max": 5.1}, {"min": -5.1, "max": -2.1}]}),
}

MULT_TABLES = {
    "fig3_otop_isobarpaper_star_blue_solid_mean_multiplicity_ruru.yaml": "Ru+Ru",
    "fig3_otop_isobarpaper_star_blue_open_mean_multiplicity_zrzr.yaml": "Zr+Zr",
}

SYSTEM_TAG = {"Ru+Ru": "RuRu", "Zr+Zr": "ZrZr"}


def _centrality_block(classes=None):
    return {
        "classes": list(classes if classes is not None else CENT_CLASSES),
        "estimator": {"name": "RefMult", "observable": "Nch", "detector": "TPC"},
        "eta": [{"min": -0.5, "max": 0.5}],
        # STAR Ntrk offline: pT > 0.2 GeV, no upper cut.
        "pT": {"min": 0.2, "max": 100.0},
    }


def _kinematic_block():
    # TPC acceptance |eta| < 1 with 0.2 < pT < 2 GeV/c.  The paper prints
    # "0.2 < pT < 2 GeV/c" on every figure that carries these flow
    # measurements: Fig. 5 (the v2 entering Delta-gamma/v2 for all groups,
    # full-TPC and sub-event), Fig. 6 (Group-1 v2{EP}, = fig4 v2{EP}),
    # Fig. 10 right (Group-2 v3{2}, v3{2}(dEta>1), v3{SP}, v3 TPC-EPD;
    # Eq. (45) defines v_{n=2,3}{2} in one analysis, and the fig10 v2{2}
    # table equals fig4 v2{2}) and Fig. 23 (v2{2}, v2{4}, v2{ZDC}).
    return {
        "eta": {"min": -1.0, "max": 1.0, "bins": [-1.0, 1.0]},
        "pT": {"min": 0.2, "max": 2.0, "bins": [0.2, 2.0]},
    }


def _flow_doc(fn, spec, classes):
    system, harmonic, method, order, vt, gap, ref = spec
    doc = {
        "observable_type": "flow_vn",
        "inspire_id": "1914564",
        "version": 1,
        "hepdata_file": fn,
        "hepdata_table_label": "",
        "collision_system": {"energy_gev": 200.0, "reaction": system},
        "centrality": _centrality_block(classes),
        "particle_id": {"species": "charged"},
        "kinematic_range": _kinematic_block(),
        "harmonic": harmonic,
        "method": method,
        "value_type": vt,
        "dimensionality": "integrated",
        "notes": {"value": "TPC |eta|<1, 0.2<pT<2 GeV/c: the paper labels "
                           "the isobar flow figures (Figs. 5, 6, 10, 23) "
                           "'0.2< pT < 2 GeV/c'."},
    }
    if order is not None:
        doc["cumulant_order"] = order
    if gap is not None:
        doc["eta_gap"] = gap
    if ref is not None:
        doc["flow_reference"] = ref
    return doc


def _mult_doc(fn, system, classes):
    return {
        "observable_type": "particle_yields",
        "inspire_id": "1914564",
        "version": 1,
        "hepdata_file": fn,
        "hepdata_table_label": "",
        "collision_system": {"energy_gev": 200.0, "reaction": system},
        "centrality": _centrality_block(classes),
        "particle_id": {"species": "charged"},
        "kinematic_range": {
            "eta": {"min": -0.5, "max": 0.5, "bins": [-0.5, 0.5]},
            # STAR Ntrk offline has pT > 0.2 with no upper cut, unlike the
            # 0.2-2 GeV flow analysis window.
            "pT": {"min": 0.2, "max": 100.0, "bins": [0.2, 100.0]},
        },
        "observable": "Nch",
        "value_type": "Nch",
        "hepdata_table_label": (
            "Efficiency-uncorrected mean Ntrk offline in |eta|<0.5 vs "
            "centrality (table listed peripheral-first)."),
        "qualifiers": {
            "note": "Efficiency-uncorrected Ntrk offline, pT > 0.2 GeV "
                    "with no upper pT cut; the simulation value is the true "
                    "charged multiplicity, so compare the shape or apply "
                    "the tracking efficiency.",
        },
    }


def _write(path, doc):
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False,
                       width=120, allow_unicode=True)


def main():
    os.makedirs(OUT_FLOW, exist_ok=True)
    os.makedirs(OUT_YIELD, exist_ok=True)
    n = 0
    for tables, prefix in ((V2_TABLES, "v2"), (V3_TABLES, "v3")):
        for fn, spec in tables.items():
            system, harmonic, method, order, vt, gap, ref = spec
            data = yaml.safe_load(open(os.path.join(BASE, fn)))
            classes = _classes_from_indep(data["independent_variables"][0])
            tag = vt.replace("{", "").replace("}", "").replace(",", "_")
            tag = tag.replace("(", "_").replace(")", "").replace(">", "gt")
            out = (f"{OUT_FLOW}/star_flow_vn_isobar_{prefix}_{tag}_"
                   f"{SYSTEM_TAG[system]}_200gev.yaml")
            _write(out, _flow_doc(fn, spec, classes))
            n += 1
    for fn, system in MULT_TABLES.items():
        data = yaml.safe_load(open(os.path.join(BASE, fn)))
        classes = _classes_from_indep(data["independent_variables"][0])
        out = (f"{OUT_YIELD}/star_particle_yields_isobar_mean_Nch_"
               f"{SYSTEM_TAG[system]}_200gev.yaml")
        _write(out, _mult_doc(fn, system, classes))
        n += 1
    print(f"wrote {n} bulk records")


if __name__ == "__main__":
    main()

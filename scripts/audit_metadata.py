#!/usr/bin/env python3
"""Audit curated-record metadata against the raw HEPData tables.

For each curated document, compare:
  - collision_system.reaction / energy_gev
  - centrality.classes / estimator / eta / pT
  - particle_id.species
  - kinematic_range.pT / eta / y

against the HEPData table's:
  - dependent-variable qualifiers
  - independent variables and their bin order
  - submission.yaml table description

The audit is read-only.  It writes a YAML report with one entry per
curated document and a `flags` list for anything that cannot be
confirmed from the HEPData record.

Usage:
    python3 scripts/audit_metadata.py --inspire-ids 1914564,2822399
    python3 scripts/audit_metadata.py --type flow_vn --output audit.yaml
    python3 scripts/audit_metadata.py --all
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
RAW = os.path.join(REPO, "hepdata_raw", "records")

CENT_MAP = {
    2.5: "0-5%", 7.5: "5-10%", 15.0: "10-20%", 25.0: "20-30%",
    35.0: "30-40%", 45.0: "40-50%", 55.0: "50-60%", 65.0: "60-70%",
    75.0: "70-80%",
}

# Qualifier names that carry a centrality label/value.
CENT_KEYS = {"centrality", "centrality [%]", "centrality (%)", "cent", "multiplicity class"}
# Qualifier names that carry a pT selection.
PT_KEYS = {"pt", "p_t", "p t", "$p_t$", "$p_{t}$", "transverse momentum"}
# Qualifier names that carry an eta/y selection.
ETA_KEYS = {"eta", "rapidity", "y", "$\\eta$", "$y$", "pseudorapidity"}


_RAW_CACHE = {}
_LABEL_CACHE = {}


def _raw_dir(iid, version):
    return os.path.join(RAW, f"ins{iid}", f"HEPData-ins{iid}-v{version}-yaml")


def _table_label(iid, version, hepdata_file):
    key = (iid, version)
    if key not in _LABEL_CACHE:
        path = os.path.join(_raw_dir(iid, version), "submission.yaml")
        labels = {}
        if os.path.exists(path):
            for doc in yaml.safe_load_all(open(path)):
                if isinstance(doc, dict) and doc.get("data_file"):
                    labels[doc["data_file"]] = str(
                        doc.get("description", "") or "")
        _LABEL_CACHE[key] = labels
    return _LABEL_CACHE[key].get(hepdata_file, "")


def _load_raw(iid, version, hepdata_file):
    key = (iid, version, hepdata_file)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    path = os.path.join(_raw_dir(iid, version), hepdata_file)
    if not os.path.exists(path):
        _RAW_CACHE[key] = None
        return None
    _RAW_CACHE[key] = yaml.safe_load(open(path))
    return _RAW_CACHE[key]


def _quals(dv):
    return {str(q.get("name", "")).strip().lower(): q.get("value")
            for q in dv.get("qualifiers", [])}


def _indep_name(iv):
    return str(iv.get("header", {}).get("name", "")).strip()


def _indep_values(iv):
    return [v.get("value") if "value" in v else
            (v.get("low"), v.get("high"))
            for v in iv.get("values", [])]


def _cent_classes_from_indep(iv):
    vals = _indep_values(iv)
    out = []
    for v in vals:
        if isinstance(v, (int, float)) and float(v) in CENT_MAP:
            out.append(CENT_MAP[float(v)])
        else:
            return None
    return out


def _num_range(text):
    """Extract a numeric range from a qualifier string."""
    if text is None:
        return None
    s = str(text)
    # "0.2 < pT < 2.0", "0.2 < pT < 5", "pT > 0.2", "0.2-2.0", "0.2 TO 5.0"
    m = re.search(r"([\d.]+)\s*<\s*[^<]*<\s*([\d.]+)", s) or re.search(r"([\d.]+)\s+TO\s+([\d.]+)", s, re.I)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"([\d.]+)\s*-\s*([\d.]+)", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r">\s*([\d.]+)", s)
    if m:
        return (float(m.group(1)), None)
    return None


def _fmt_range(r):
    if r is None:
        return None
    lo, hi = r
    return {"min": lo, "max": hi}


def _norm_cent(s):
    s = str(s).strip().replace("TO", "-").replace("to", "-")
    s = re.sub(r"\s+", "", s.replace("%", ""))      # tabs too ('0\t-5', STAR 660793)
    m = re.match(r"([\d.]+)-([\d.]+)", s)
    if m:
        return f"{float(m.group(1)):g}-{float(m.group(2)):g}%"
    return s


def _cent_bounds(s):
    m = re.match(r"([\d.]+)-([\d.]+)", _norm_cent(s))
    return (float(m.group(1)), float(m.group(2))) if m else None


def _is_cent_class(s):
    return _cent_bounds(s) is not None


def _covers_all(cur, raw_classes):
    """True if one curated class spans the union of the raw bins."""
    if len(cur) != 1:
        return False
    b = _cent_bounds(cur[0])
    if b is None:
        return False
    bounds = [_cent_bounds(r) for r in raw_classes]
    if any(x is None for x in bounds):
        return False
    return b[0] <= min(x[0] for x in bounds) and \
        b[1] >= max(x[1] for x in bounds)


# ---- checks added November 2026 (each caught real public records) ---------
# A flow_vn record that is not a flow coefficient: slopes (dv1/dy, charge-
# asymmetry slope r), counts, covariances/variances, R_AA ...
_NOT_VN = re.compile(r"slope|counts|events|cov_|cov\(|var\(|\bc_\{?k\b|"
                     r"R_\{?(AA|CP|AB|dAu|pPb)|d\s*\$?v_?\{?1\}?\$?\s*/\s*d\s*y", re.I)
_GAP_TEXT = re.compile(r"Delta\s*\\?eta|\|\s*\\?Delta|eta.gap|sub-?event", re.I)
_PID_TEXT = re.compile(r"\\pi\b|\bpions?\b|\bkaons?\b|\bprotons?\b|K\^\{?0|K_\{?S|\\Lambda|\bLambda|\\phi\b|"
                       r"\\Xi|\bXi\b|\\Omega", re.I)
_ENERGY_KEYS = {"sqrt(s)/nucleon", "sqrt(s)", "sqrt(s_nn)", "$\\sqrt{s_{nn}}$", "sqrt(s) per nucleon"}


def _energy_gev(text):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(text))
    if not m:
        return None
    v = float(m.group(1))
    return v * 1000.0 if "tev" in str(text).lower() else v


def _beams(reaction):
    head = str(reaction).split("-->")[0].upper().replace("+", " ").replace("-", " ")
    toks = [t for t in head.split() if t not in ("AND",)]
    return tuple(toks[:2]) if len(toks) >= 2 else None


def _same_centres(classes, bins):
    """Do the curated classes share the centres of the raw (lo, hi) bins?"""
    if len(classes) != len(bins):
        return False
    for c, (lo, hi) in zip(classes, bins):
        b = _cent_bounds(c)
        if b is None or abs(0.5 * (b[0] + b[1]) - 0.5 * (float(lo) + float(hi))) > 1e-6:
            return False
    return True


def _is_pt_cut_axis(name):
    """A table axis of lower pT cuts ('p_T,min'), not pT bins."""
    return bool(re.match(r"p_?t,?min|ptmin|p_t\^?min|minimump_?t", name))


def _check_pt_cut_point(doc, iv, flag):
    """A pT-cut axis: one integrated record per point (table_rows [j]),
    with kinematic_range.pT.min at that cut."""
    rows = doc.get("table_rows")
    if doc.get("dimensionality", "integrated") != "integrated" or not rows or len(rows) != 1:
        flag("error", "table is a pT_min dependence: one integrated record per point (table_rows [j])")
        return
    v = iv["values"][rows[0]]
    cut = float(v["value"] if "value" in v else v["low"])
    lo = ((doc.get("kinematic_range") or {}).get("pT") or {}).get("min")
    if lo is None or abs(float(lo) - cut) > 1e-9:
        flag("error", f"row {rows[0]} is pT_min = {cut:g}, record pT.min = {lo}")


def _extra_checks(doc, raw, sel_dv, sel_q, desc, flag):
    otype = doc.get("observable_type")
    header = str(((sel_dv or {}).get("header") or {}).get("name", ""))
    text = " ".join([header, str(doc.get("value_type", "")), desc])
    indeps = raw.get("independent_variables", []) if raw else []
    iv0 = indeps[0] if indeps else None
    n0 = len(iv0.get("values", [])) if iv0 else 0
    # '$p_{\rm T}$', 'p_{T} [GeV/c]', 'PT' -> 'p_t...'
    name0 = re.sub(r"\\(rm|mathrm|text|it)|[${}\\ ]", "", _indep_name(iv0).lower()) if iv0 else ""
    if n0 > 1 and _is_pt_cut_axis(name0):
        _check_pt_cut_point(doc, iv0, flag)
    # A <pT> column next to v_n in a pT-binned table is the bin's mean pT,
    # not a measurement (PHENIX tables; 46 such records were public until
    # 2026-11-02).
    if otype == "mean_pT" and n0 > 1 and (name0.startswith(("p_t", "pt")) or "transversemomentum" in name0):
        flag("error", "mean_pT of a pT-binned table: the bin centre, not a measurement")
    labels = {str(e.get("label")) for v in (sel_dv or {}).get("values", []) for e in v.get("errors") or []}
    for lab in doc.get("errors_excluded") or []:
        if lab not in labels:
            flag("error", f"errors_excluded {lab!r} is not an error label of the table ({sorted(labels)})")
    # Spectrum form: the invariant yield carries 1/(2 pi pT) (or E d3N/dp3);
    # d2N/(dpT dy) does not.  The pipeline computes the form value_type names.
    if otype == "particle_spectra":
        vt = doc.get("value_type", "invariant_yield")
        if doc.get("observable") and doc.get("observable") != vt:
            flag("error", f"observable {doc.get('observable')!r} differs from value_type {vt!r}")
        h = header.upper().replace(" ", "")
        inv = bool(re.search(r"2\*?\\?PI|E\*?D3|INVARIANT", h))
        if h and vt == "invariant_yield" and not inv and re.search(r"D2?\(?N", h):
            flag("error", f"header {header!r} has no 1/(2 pi pT): d2N/(dpT dy), value_type dN_dpT_dy")
        if vt in ("dN_dpT_dy", "d2N_dpT_dy") and inv:
            flag("error", f"header {header!r} is an invariant yield; value_type {vt!r}")
    if otype == "flow_vn":
        if _NOT_VN.search(header) or _NOT_VN.search(str(doc.get("value_type", ""))):
            flag("error", f"header {header!r} is not a flow coefficient")
        elif "/" in header.replace("GeV/c", ""):
            flag("warning", f"header {header!r} looks like a ratio")
        dim = doc.get("dimensionality", "integrated")
        if n0 > 1 and not _is_pt_cut_axis(name0) and (name0.startswith(("p_t", "pt")) or "transversemomentum" in name0) and dim != "differential_pT":
            flag("error", f"table is pT-differential ({n0} points), record is {dim}")
        if n0 > 1 and (name0.startswith(("eta", "\\eta", "y", "pseudorapidity", "rapidity"))) and dim != "differential_eta":
            flag("error", f"table is eta-differential ({n0} points), record is {dim}")
        # Percent tables whose header carries no unit (STAR PRC 72, 014904:
        # v2 = 1.89 for 1.89%): a flow coefficient's median |value| stays
        # far below 0.5; single noisy high-pT points do not move the median.
        units = str(((sel_dv or {}).get("header") or {}).get("units", ""))
        vals = []
        for v in (sel_dv or {}).get("values", []):
            try:
                vals.append(abs(float(v.get("value"))))
            except (TypeError, ValueError):
                pass
        scale = float(doc.get("value_scale") or 1.0)
        if vals and "%" not in units and "*100" not in header.replace(" ", "") \
                and not re.search(r"\(\s*%\s*\)", header) \
                and sorted(vals)[len(vals) // 2] / scale > 0.5:
            flag("error", f"median |value| {sorted(vals)[len(vals) // 2]:.3g}: percent without a unit "
                          f"(value_scale) or not a flow coefficient")
        has_gap = (doc.get("particles") or doc.get("eta_gap") is not None or doc.get("flow_reference")
                   or doc.get("method") in ("scalar_product", "event_plane"))
        if _GAP_TEXT.search(text) and not has_gap:
            flag("warning", "table mentions an eta gap or subevents; the record has none")
    species = (doc.get("particle_id") or {}).get("species")
    if species == "charged" and _PID_TEXT.search(header + " " + desc) and "charged" not in (header + desc).lower():
        flag("warning", "table names an identified species; the record says charged")
    rec_e = (doc.get("collision_system") or {}).get("energy_gev")
    rec_r = (doc.get("collision_system") or {}).get("reaction")
    for k, v in sel_q.items():
        if k in _ENERGY_KEYS or k.startswith("sqrt(s"):
            e = _energy_gev(v)
            if e and e < 30 and rec_e and float(rec_e) > 1000:
                e *= 1000.0                    # '2.76', '5.02': TeV without a unit
            if e and rec_e and abs(e - float(rec_e)) > 0.01 * e:
                flag("error", f"energy qualifier {v!r} -> {e:g} GeV, record {rec_e} GeV")
        if k in ("re", "reaction"):
            b = _beams(v)
            if b and _beams(rec_r) and set(b) != set(_beams(rec_r)):
                flag("error", f"reaction qualifier {v!r}, record {rec_r!r}")
    if str(rec_r).strip().lower() == "unknown":
        flag("warning", "reaction unknown")


def audit_doc(path, doc):
    iid = str(doc.get("inspire_id", ""))
    version = doc.get("version", 1)
    hf = doc.get("hepdata_file", "")
    # PHOBOS records are ASCII data, not HEPData YAML.
    if iid.startswith("PHOBOS"):
        return None
    raw = _load_raw(iid, version, hf)
    if doc.get("value_table"):        # curated from another source (value_tables.py)
        vt = os.path.join(os.path.dirname(path), doc["value_table"])
        raw = yaml.safe_load(open(vt)) if os.path.exists(vt) else None
    entry = {
        "file": os.path.relpath(path, REPO),
        "inspire_id": iid,
        "hepdata_file": hf,
        "observable_type": doc.get("observable_type"),
        "value_type": doc.get("value_type"),
        "curated": {
            "reaction": (doc.get("collision_system") or {}).get("reaction"),
            "energy_gev": (doc.get("collision_system") or {}).get("energy_gev"),
            "centrality_classes": (doc.get("centrality") or {}).get("classes"),
            "centrality_estimator": (doc.get("centrality") or {}).get("estimator"),
            "centrality_eta": (doc.get("centrality") or {}).get("eta"),
            "centrality_pT": (doc.get("centrality") or {}).get("pT"),
            "species": (doc.get("particle_id") or {}).get("species"),
            "pT": (doc.get("kinematic_range") or {}).get("pT"),
            "eta": (doc.get("kinematic_range") or {}).get("eta"),
            "y": (doc.get("kinematic_range") or {}).get("y"),
        },
        "raw": {"table_description": _table_label(iid, version, hf),
                "qualifiers": [], "independent_variables": []},
        "flags": [],
    }

    def flag(sev, msg):
        entry["flags"].append({"severity": sev, "message": msg})

    if raw is None:
        flag("error", "raw HEPData table not found")
        return entry

    for iv in raw.get("independent_variables", []):
        entry["raw"]["independent_variables"].append({
            "name": _indep_name(iv),
            "values": _indep_values(iv)[:12],
            "n": len(iv.get("values", [])),
        })
    for dv in raw.get("dependent_variables", []):
        entry["raw"]["qualifiers"].append(_quals(dv))

    dv_index = int(doc.get("dv_index", 0) or 0)
    dvs = raw.get("dependent_variables", [])
    sel_dv = dvs[dv_index] if dvs and 0 <= dv_index < len(dvs) else (
        dvs[0] if dvs else None)
    sel_q = _quals(sel_dv) if sel_dv is not None else {}
    indeps = raw.get("independent_variables", [])

    # ---- centrality ---------------------------------------------------
    cent_iv = None
    for iv in indeps:
        if "cent" in _indep_name(iv).lower():
            cent_iv = iv
            break
    cur_classes = (doc.get("centrality") or {}).get("classes") or []

    # A centrality qualifier on the selected DV is the most specific
    # statement (e.g. a fixed 20-50% analysis bin).  Only fall back to a
    # centrality independent variable when the DV has no centrality.
    dv_cent = None
    for k, v in sel_q.items():
        if k in CENT_KEYS and v is not None:
            if _is_cent_class(v):
                dv_cent = _norm_cent(v)
            else:
                flag("warning", f"centrality-named qualifier has non-class "
                                f"value {v!r}")
    if dv_cent is not None:
        corrected = (doc.get("hepdata_corrections") or {}).get("centrality")
        if cur_classes and dv_cent not in [_norm_cent(c)
                                           for c in cur_classes]:
            if corrected:       # a documented HEPData labelling error
                flag("warning", f"selected DV centrality {dv_cent!r} corrected to "
                                f"{cur_classes} (hepdata_corrections: {corrected[:60]})")
            else:
                flag("error", f"selected DV centrality {dv_cent!r} not in "
                              f"curated classes {cur_classes}")
        elif not cur_classes:
            flag("error", f"selected DV centrality {dv_cent!r} but curated "
                          f"centrality empty")
    elif cent_iv is not None:
        cent_vals = _indep_values(cent_iv)
        if doc.get("table_rows") is not None:          # the record holds these rows only
            cent_vals = [cent_vals[r] for r in doc["table_rows"] if r < len(cent_vals)]
        if cent_vals and all(isinstance(v, tuple) for v in cent_vals):
            raw_classes = [f"{float(lo):g}-{float(hi):g}%"
                           for lo, hi in cent_vals]
            cur_norm = [_norm_cent(c) for c in cur_classes]
            if not cur_classes:
                flag("error", f"raw centrality present ({raw_classes[:3]}...) "
                              f"but curated centrality empty")
            elif _covers_all(cur_classes, raw_classes):
                flag("warning", f"coarse curated centrality {cur_norm} "
                                f"spans raw bins {raw_classes[:3]}...")
            elif cur_norm != raw_classes and _same_centres(cur_norm, cent_vals):
                # plotting positions (points displaced for visibility,
                # ATLAS 2967110 Fig. 7a): classes centred on the raw bins
                flag("warning", f"raw centrality bins are plotting positions centred on the curated "
                                f"classes ({raw_classes[:2]}... -> {cur_norm[:2]}...)")
            elif cur_norm != raw_classes:
                flag("error", f"centrality classes/order mismatch: "
                              f"curated={cur_norm} raw={raw_classes}")
        elif cent_vals and all(isinstance(v, str) for v in cent_vals):
            raw_classes = [_norm_cent(v) for v in cent_vals]
            cur_norm = [_norm_cent(c) for c in cur_classes]
            if not cur_classes:
                flag("error", f"raw centrality present ({raw_classes[:3]}...) "
                              f"but curated centrality empty")
            elif _covers_all(cur_classes, raw_classes):
                flag("warning", f"coarse curated centrality {cur_norm} "
                                f"spans raw bins {raw_classes[:3]}...")
            elif cur_norm != raw_classes:
                flag("error", f"centrality classes/order mismatch: "
                              f"curated={cur_norm} raw={raw_classes}")
        else:
            # rows with a blank value in the selected column are dropped by
            # load_experimental, and the record's classes skip them too
            vals = (sel_dv or {}).get("values", [])
            if doc.get("table_rows") is not None:
                vals = [vals[r] for r in doc["table_rows"] if r < len(vals)]
            blank = {j for j, v in enumerate(vals) if str(v.get("value")).strip() in ("", "-")}
            centers = [float(v) for j, v in enumerate(cent_vals)
                       if isinstance(v, (int, float)) and j not in blank]
            if not cur_classes:
                flag("error", f"raw centrality present ({centers[:3]}...) but "
                              f"curated centrality empty")
            elif len(cur_classes) != len(centers):
                flag("error", f"centrality bin count mismatch: curated "
                              f"{len(cur_classes)} vs raw {len(centers)}")
            else:
                for c, center in zip(cur_classes, centers):
                    b = _cent_bounds(c)
                    if b is None:
                        flag("warning", f"unparsed curated centrality {c!r}")
                    elif not (b[0] - 1e-6 <= center < b[1] + 1e-6):
                        flag("error", f"curated class {c!r} does not contain "
                                      f"raw centre {center}")
    elif cur_classes:
        flag("warning", "curated centrality present but no raw centrality "
                        "qualifier or independent variable")

    # ---- pT qualifier -------------------------------------------------
    # The selected DV's qualifiers (a table's DVs may differ in pT window,
    # PHENIX PPG124 Fig. 11); hepdata_corrections.pT: a stated departure.
    pt_fixed = (doc.get("hepdata_corrections") or {}).get("pT")
    for q in [sel_q]:
        for k, v in q.items():
            if k in PT_KEYS:
                r = _num_range(v)
                cur = (doc.get("kinematic_range") or {}).get("pT") or {}
                cur_r = (cur.get("min"), cur.get("max")) if cur else None
                if r is None:
                    flag("warning", f"unparsed pT qualifier {v!r}")
                elif r[1] is None:
                    flag("warning", f"open-ended pT qualifier {v!r}; curated "
                                    f"max {cur_r[1] if cur_r else None} assumed")
                elif cur_r != r:
                    flag("warning" if pt_fixed else "error",
                         f"pT qualifier {v!r} -> {r}, curated={cur_r}"
                         + (f" (corrected: {pt_fixed})" if pt_fixed else ""))
            if k in ETA_KEYS:
                r = _num_range(v)
                cur = (doc.get("kinematic_range") or {}).get("eta") or {}
                cur_r = (cur.get("min"), cur.get("max")) if cur else None
                if r is None:
                    flag("warning", f"unparsed eta/y qualifier {v!r}")
                elif r[1] is None:
                    flag("warning", f"open-ended eta/y qualifier {v!r}")
                elif cur_r != r:
                    flag("warning", f"eta/y qualifier {v!r} -> {r}, "
                                    f"curated={cur_r}")

    _extra_checks(doc, raw, sel_dv, sel_q, entry["raw"]["table_description"], flag)

    # ---- pT bins ------------------------------------------------------
    pt_iv = None
    for iv in indeps:
        n = _indep_name(iv).lower()
        if "pt" in n or "p_t" in n or "transverse momentum" in n:
            pt_iv = iv
            break
    if pt_iv is not None:
        vals = pt_iv.get("values", [])
        if vals and all("low" in v and "high" in v for v in vals):
            edges = [float(vals[0]["low"])] + [float(v["high"])
                                                for v in vals]
            cur = (doc.get("kinematic_range") or {}).get("pT") or {}
            cur_bins = cur.get("bins")
            if cur_bins:
                cur_f = [float(x) for x in cur_bins]
                # every curated edge must be a raw edge (a record may hold
                # some of the table's rows; until November 2026 this tested
                # the reverse and flagged such records)
                subset = all(any(abs(e - c) < 1e-6 for e in edges)
                             for c in cur_f)
                if subset and len(cur_f) != len(edges):
                    flag("warning", f"pT bins are a subset/rebin: curated "
                                    f"{len(cur_f)} vs raw {len(edges)}")
                elif not subset:
                    flag("error", f"pT bin edges not a subset of raw: "
                                  f"curated {cur_f[:3]}... raw {edges[:3]}...")

    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default=None,
                    help="Observable type subdirectory (default: all).")
    ap.add_argument("--inspire-ids", default=None,
                    help="Comma-separated INSPIRE IDs to include.")
    ap.add_argument("--output", default=None,
                    help="YAML report path (default: stdout summary).")
    ap.add_argument("--all", action="store_true",
                    help="Audit all curated records.")
    ap.add_argument("--by-paper", action="store_true",
                    help="Summarize errors per publication (most affected first).")
    args = ap.parse_args()

    ids = set(x.strip() for x in args.inspire_ids.split(",")) if args.inspire_ids else None
    if args.type:
        paths = sorted(glob.glob(os.path.join(DATA, args.type, "*.yaml")))
    else:
        paths = sorted(glob.glob(os.path.join(DATA, "**", "*.yaml"),
                                 recursive=True))

    entries = []
    for path in paths:
        for doc in yaml.safe_load_all(open(path)):
            if not isinstance(doc, dict):
                continue
            if ids is not None and str(doc.get("inspire_id")) not in ids:
                continue
            res = audit_doc(path, doc)
            if res is not None:
                entries.append(res)

    flagged = [e for e in entries if e["flags"]]
    n_err = sum(1 for e in entries for f in e["flags"]
                if f["severity"] == "error")
    n_warn = sum(1 for e in entries for f in e["flags"]
                 if f["severity"] == "warning")
    print(f"audited {len(entries)} documents; {len(flagged)} with flags "
          f"({n_err} errors, {n_warn} warnings)")
    if args.output:
        with open(args.output, "w") as f:
            yaml.safe_dump(entries, f, default_flow_style=False,
                           sort_keys=False, width=140, allow_unicode=True)
        print(f"wrote {args.output}")
    if args.by_paper:
        import collections
        papers = collections.defaultdict(lambda: [0, 0, collections.Counter(), set()])
        for e in entries:
            rec = papers[e["inspire_id"]]
            rec[0] += 1
            rec[3].add(os.path.basename(e["file"]))
            errs = [f["message"] for f in e["flags"] if f["severity"] == "error"]
            if errs:
                rec[1] += 1
            for m in errs:
                rec[2][re.sub(r"[\[(\'\"].*", "", m).strip()[:50]] += 1
        rows = sorted(papers.items(), key=lambda kv: (-kv[1][1], -kv[1][0]))
        print(f"\n{'inspire':>9} {'docs':>5} {'w/err':>5}  kinds / files")
        for iid, (n, nerr, kinds, files) in rows:
            if nerr:
                print(f"{iid:>9} {n:5d} {nerr:5d}  {dict(kinds.most_common(3))}  {sorted(files)[0]}")
        return
    for e in flagged[:80]:
        print(f"\n{e['file']} :: {e['hepdata_file']}")
        for f in e["flags"]:
            print(f"  [{f['severity']}] {f['message']}")



if __name__ == "__main__":
    main()

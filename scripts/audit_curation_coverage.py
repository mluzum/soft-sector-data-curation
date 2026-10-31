#!/usr/bin/env python3
"""Audit curation coverage of hepdata_raw/ for the IMPLEMENTED (routable) observables.

Answers the question: "Have we curated every raw measurement in hepdata_raw/
that corresponds to an observable our pipeline can actually compute?"

For each implemented observable type it cross-references the normalized
(inspire_id, hepdata_file) keys between the auto-classified candidates in
data_candidates/<type>/ and the curated records in data/<type>/.  A table that
is classified but not curated is a genuine curation gap (event_plane flow is
excluded by design -- it is deferred).

It also reconstructs the classifier's coverage of hepdata_raw/:
  Set A : downloaded records that produced NO candidates at all
  Set B : records with candidates but no curated record of ANY type
and runs a tight per-table signature scan over Set A to confirm that no
genuine implemented-type heavy-ion table was silently dropped by the
classifier.

Usage:
  python scripts/audit_curation_coverage.py
  python scripts/audit_curation_coverage.py --type flow_vn
"""
import argparse
import os
import re
import sys
import yaml
import warnings
from collections import Counter, defaultdict

warnings.simplefilter("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEPDATA_RECORDS = os.path.join(ROOT, "hepdata_raw", "records")
RECORD_INDEX = os.path.join(ROOT, "hepdata_raw", "record_index.yaml")
CANDIDATES = os.path.join(ROOT, "data_candidates")
DATA = os.path.join(ROOT, "data")

# Observable types the pipeline can actually compute (routable).
IMPLEMENTED_TYPES = [
    "flow_vn",
    "particle_yields",
    "particle_spectra",
    "mean_pT",
    "fluctuations",
    "transverse_energy",
]

# Tight signatures applied to dependent-variable NAMES and table name/description
# only (not buried prose) so we do not false-positive on e.g. "SIGMA" baryon names.
IMPL_SIG = {
    "flow_vn": re.compile(r"v_\{?n?\}?|v\d\{|\b(cumulant|anisotropic|directed|elliptic|triangular) flow\b|v_\d", re.I),
    "yields": re.compile(r"dN/d(|y|η|eta)|dN/(dη|deta)|(yield|multiplicity|N_\{?ch\}?|dN/dy)", re.I),
    "spectra": re.compile(r"(invariant yield|1/(2π|2 ?π)|spectr(a|um)|d\^?2N)", re.I),
    "mean_pT": re.compile(r"(mean p|〈?p[_]?T〉?|<p_T>|average p_T)", re.I),
    "fluct": re.compile(r"(p_T fluctuation|〈Δp)", re.I),
    "dET": re.compile(r"(dE_T/d(η|eta)|transverse energ)", re.I),
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def scan_file(path):
    """Fast line-scan returning one key-dict PER YAML document (handles multi-doc
    files).  Captures inspire_id, hepdata_file, method, observable_type."""
    docs = []
    cur = {}
    try:
        with open(path) as fh:
            for line in fh:
                if line.strip() in ("---", "..."):
                    if cur.get("inspire_id"):
                        docs.append(cur)
                    cur = {}
                    continue
                if re.match(r"\s*inspire_id\s*:", line) and "inspire_id" not in cur:
                    m = re.search(r"(\d+)", line)
                    if m:
                        cur["inspire_id"] = m.group(1)
                elif re.match(r"\s*hepdata_file\s*:", line) and "hepdata_file" not in cur:
                    m = re.search(r":\s*'?\"?([^'\"#\n]+)", line)
                    if m:
                        cur["hepdata_file"] = m.group(1).strip().strip("'\"").rstrip()
                elif re.match(r"\s*method\s*:", line) and "method" not in cur:
                    m = re.search(r":\s*'?\"?([^'\"#\n]+)", line)
                    if m:
                        cur["method"] = m.group(1).strip().strip("'\"").rstrip()
                elif re.match(r"\s*observable_type\s*:", line) and "observable_type" not in cur:
                    m = re.search(r":\s*'?\"?([^'\"#\n]+)", line)
                    if m:
                        cur["observable_type"] = m.group(1).strip().strip("'\"").rstrip()
    except Exception:
        pass
    if cur.get("inspire_id"):
        docs.append(cur)
    return docs


def collect(root):
    """Map (inspire_id, normalized hepdata_file) -> dict with method/type."""
    out = {}
    if not os.path.isdir(root):
        return out
    for dp, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".yaml"):
                continue
            for k in scan_file(os.path.join(dp, fn)):
                key = (str(k["inspire_id"]), norm(k.get("hepdata_file")))
                out.setdefault(key, k)
    return out


def all_keys(root):
    """Yield every (inspire_id, observable_type, method) under root via line-scan."""
    if not os.path.isdir(root):
        return
    for dp, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".yaml"):
                continue
            for k in scan_file(os.path.join(dp, fn)):
                yield (str(k["inspire_id"]), k.get("observable_type"), k.get("method"))


def dep_names(p):
    """Return (dependent_var_names, [table_name+description]) for a HEPData yaml."""
    try:
        doc = yaml.safe_load(open(p))
    except Exception:
        return [], []
    if isinstance(doc, dict):
        doc = [doc]
    dnames, meta = [], []
    for t in doc:
        if not isinstance(t, dict):
            continue
        meta.append(str(t.get("name", "")) + " " + str(t.get("description", "")))
        for dv in t.get("dependent_variables", []) or []:
            dnames.append(str(dv.get("name", "")))
    return dnames, meta


def downloaded_ids():
    if not os.path.isfile(RECORD_INDEX):
        return set()
    idx = yaml.safe_load(open(RECORD_INDEX))
    if isinstance(idx, dict):
        return set(str(k) for k in idx.keys())
    return set(str(r.get("inspire_id")) for r in idx if isinstance(r, dict))


def tight_scan_setA(set_a_ids):
    """For each Set A record, scan raw tables with tight signatures.

    Returns dict type -> set of inspire_ids that genuinely match.
    """
    hits = defaultdict(set)
    for i in set_a_ids:
        base = os.path.join(HEPDATA_RECORDS, f"ins{i}")
        if not os.path.isdir(base):
            continue
        for dp, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".yaml"):
                    continue
                dnames, meta = dep_names(os.path.join(dp, fn))
                text = " ".join(meta)
                for t, rx in IMPL_SIG.items():
                    if any(rx.search(x) for x in dnames) or rx.search(text):
                        hits[t].add(i)
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", help="limit to one implemented observable type")
    args = ap.parse_args()

    types = [args.type] if args.type else IMPLEMENTED_TYPES

    print("=" * 70)
    print("CURATION COVERAGE AUDIT -- implemented (routable) observables")
    print("=" * 70)

    # ---- Step 1: per-type candidate-vs-curated cross reference ----
    print("\n[1] Candidate -> Curated cross-reference (normalized key)")
    print(f"{'type':20s} {'cand':>7s} {'curated':>8s} {'genuine_gap':>12s}")
    total_gap = 0
    for t in types:
        cand = collect(os.path.join(CANDIDATES, t))
        cur = collect(os.path.join(DATA, t))
        # genuine gap = candidate key not present in curated
        if t == "flow_vn":
            # exclude event_plane (deferred by design)
            new = [
                k for k, d in cand.items()
                if k not in cur and str(d.get("method")) != "event_plane"
            ]
        else:
            new = [k for k in cand if k not in cur]
        print(f"{t:20s} {len(cand):7d} {len(cur):8d} {len(new):12d}")
        total_gap += len(new)
        for k in new[:10]:
            d = cand[k]
            print(f"      gap {k} vt={d.get('value_type')} method={d.get('method')} dim={d.get('dimensionality')}")
    print(f"\n  Total genuine gaps across implemented types: {total_gap}")

    # ---- Step 2: classifier coverage of hepdata_raw ----
    print("\n[2] Classifier coverage of hepdata_raw/")
    dl = downloaded_ids()
    cand_ids = set()
    for i, _, _ in all_keys(CANDIDATES):
        cand_ids.add(str(i))
    cur_ids = set()
    for i, _, _ in all_keys(DATA):
        cur_ids.add(str(i))

    set_a = dl - cand_ids          # downloaded, no candidates
    set_b = cand_ids - cur_ids     # has candidates, not curated anywhere
    print(f"  downloaded records : {len(dl)}")
    print(f"  with candidates    : {len(cand_ids)}")
    print(f"  curated (any type) : {len(cur_ids)}")
    print(f"  Set A (no candidates at all)        : {len(set_a)}")
    print(f"  Set B (candidates but uncurated)    : {len(set_b)}")

    # Set B type composition (should be NON-implemented types)
    b_types = Counter()
    b_impl = []
    for i, ot, meth in all_keys(CANDIDATES):
        if i in set_b:
            b_types[ot] += 1
            if ot in IMPLEMENTED_TYPES:
                # event_plane flow is deferred by design, not a genuine gap
                if ot == "flow_vn" and meth == "event_plane":
                    continue
                b_impl.append((i, ot, meth))
    print(f"  Set B type composition: {dict(b_types)}")
    if b_impl:
        print(f"  WARNING: {len(b_impl)} Set B docs are implemented types (genuine gap!):")
        for x in b_impl[:20]:
            print("     ", x)
    else:
        print("  Set B contains NO implemented-type records (good).")

    # Tight per-table scan of Set A for genuine implemented-type tables
    print("\n  Tight per-table signature scan of Set A (dependent-var/table-name only):")
    hits = tight_scan_setA(set_a)
    genuine = 0
    for t in IMPLEMENTED_TYPES:
        n = len(hits.get(t, set()))
        genuine += n
        print(f"    {t:20s} records with genuine signature: {n}")
    if genuine == 0:
        print("    => No genuine implemented-type heavy-ion tables dropped by classifier.")
    else:
        print(f"    => {genuine} record(s) may contain dropped tables; inspect manually.")


if __name__ == "__main__":
    main()

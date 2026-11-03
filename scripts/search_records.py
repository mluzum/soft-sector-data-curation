#!/usr/bin/env python3
"""Search curated records and generate observable manifests.

Usage:
  # Preview matching records in a table
  python scripts/search_records.py \\
    --type flow_vn \\
    --reaction "AU AU" \\
    --energy 200 \\
    --dimensionality integrated \\
    --method cumulant

  # Generate a manifest YAML for the pipeline
  python scripts/search_records.py \\
    --type flow_vn \\
    --reaction "AU AU" --energy 200 \\
    --value-type-regex 'v_2\\\\{2\\\\}' \\
    --output manifest.yaml

  # Count only
  python scripts/search_records.py --type flow_vn --count-only

  # Include non-dispatchable records
  python scripts/search_records.py --type flow_vn --include-undispatchable
"""

import sys
import os
import re
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.dispatcher import resolve, extract_measurement_params
from analysis.record_id import record_short_id, record_label, record_key


def _safe_str(v):
    if v is None:
        return ""
    return str(v)


def load_records(data_dir):
    """Yield (file_path, doc_index, doc) for every record in data_dir."""
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".yaml"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            docs = list(yaml.safe_load_all(f))
        for idx, doc in enumerate(docs):
            if doc is None:
                continue
            yield fpath, idx, doc


def matches(doc, args):
    """Check if a document matches all search filters."""
    cs = doc.get("collision_system", {})

    if args.reaction:
        # a regex as given (until November 2026 every '*' became '.*', so a
        # quantifier such as '^\s*O' turned into '^\s.*O' and matched nothing)
        if not re.search(args.reaction, _safe_str(cs.get("reaction", "")), re.IGNORECASE):
            return False

    if args.energy is not None:
        try:
            doc_energy = float(cs.get("energy_gev", 0))
            if abs(doc_energy - args.energy) > 0.1:
                return False
        except (TypeError, ValueError):
            pass

    if args.value_type_regex:
        vt = _safe_str(doc.get("value_type", ""))
        try:
            if not re.search(args.value_type_regex, vt):
                return False
        except re.error:
            if args.value_type_regex not in vt:
                return False

    if args.dimensionality:
        if doc.get("dimensionality") != args.dimensionality:
            return False

    if args.method:
        if doc.get("method") != args.method:
            return False

    if args.harmonic is not None:
        try:
            if int(doc.get("harmonic", 0)) != args.harmonic:
                return False
        except (TypeError, ValueError):
            pass

    if args.cumulant_order is not None:
        try:
            if int(doc.get("cumulant_order", 0)) != args.cumulant_order:
                return False
        except (TypeError, ValueError):
            pass

    if not args.include_undispatchable:
        try:
            resolve(doc)
        except ValueError:
            return False

    return True


def format_cent_classes(doc):
    """Return a short centrality class string."""
    cent = doc.get("centrality", {})
    classes = cent.get("classes", [])
    if not classes:
        return "MB"
    n = len(classes)
    if n <= 3:
        return ", ".join(classes)
    return f"{classes[0]}..{classes[-1]} ({n} bins)"


def format_kinematic(doc):
    """Return a short kinematic range string."""
    kr = doc.get("kinematic_range", {})
    eta = kr.get("eta", {})
    pT = kr.get("pT", {})
    eta_str = f"η∈[{eta.get('min','?')},{eta.get('max','?')}]" if eta else ""
    pt_str = f"pT∈[{pT.get('min','?')},{pT.get('max','?')}]" if pT else ""
    sep = " " if eta_str and pt_str else ""
    return f"{eta_str}{sep}{pt_str}"


def print_table(matches_list, args):
    """Print matching records as a formatted table."""
    if not matches_list:
        print("No matching records found.")
        return

    print(f"\n{'ID':<14} {'Obs. type':<10} {'Value type':<40} {'Dim':<18} {'Method':<16} "
          f"{'n':<4} {'Harm':<5} {'Reaction':<25} {'Energy':<8} {'Cent':<20} "
          f"{'Kinematic':<30} {'HEPData file'}")
    print("-" * 210)

    for seq_id, (fpath, doc_idx, doc) in enumerate(matches_list, 1):
        rid = record_short_id(doc)
        dim = doc.get("dimensionality", "integrated")
        method = doc.get("method", "?")
        harm = doc.get("harmonic", "?")
        cs = doc.get("collision_system", {})
        reaction = cs.get("reaction", "?")
        energy = cs.get("energy_gev", "?")
        vt = doc.get("value_type", "?")
        hf = doc.get("hepdata_file", "?")
        cent_str = format_cent_classes(doc)
        kin_str = format_kinematic(doc)

        # Trim long values
        vt_display = (vt[:38] + "..") if len(vt) > 40 else vt
        hf_display = (hf[:28] + "..") if len(hf) > 30 else hf

        print(f"{rid:<14} {dim:<10} {vt_display:<40} {dim:<18} {method:<16} "
              f"{'':<4} {str(harm):<5} {reaction[:24]:<25} {str(energy):<8} {cent_str:<20} "
              f"{kin_str:<30} {hf_display}")

    print(f"\nTotal: {len(matches_list)} matching records")


def generate_manifest(matches_list, output_path):
    """Write a manifest YAML file from matching records."""
    records = []
    for seq_id, (fpath, doc_idx, doc) in enumerate(matches_list, 1):
        rid = record_short_id(doc)
        entry = {
            "id": rid,
            # fpath + doc_index pin the EXACT document.  Without them
            # run_pipeline falls back to id/content-key lookup, which cannot
            # distinguish per-centrality docs of one HEPData table (they
            # share a content key AND a short id) -- it then silently picks
            # the wrong doc, e.g. reading the MB v2(eta) series when the
            # manifest asked for the 3-15% one.
            "fpath": fpath,
            "doc_index": doc_idx,
            "inspire_id": doc.get("inspire_id"),
            "version": doc.get("version"),
            "hepdata_file": doc.get("hepdata_file"),
            "value_type": doc.get("value_type"),
            "dimensionality": doc.get("dimensionality"),
            "method": doc.get("method"),
            "harmonic": doc.get("harmonic"),
            "cumulant_order": doc.get("cumulant_order"),
            "particles": doc.get("particles"),
            # Tells apart the per-centrality documents of one table, which
            # share the id; run_pipeline checks the pin against it.
            "centrality_classes": (doc.get("centrality") or {}).get("classes"),
            "select": True,
        }
        # Remove None fields for cleanliness
        entry = {k: v for k, v in entry.items() if v is not None}
        records.append(entry)

    manifest = {
        "manifest_version": 1,
        "description": f"Auto-generated manifest with {len(records)} records",
        "records": records,
    }

    with open(output_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(records)} records to {output_path}")
    print("Edit 'select: true/false' for each record to include/exclude.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Search curated records and generate observable manifests")
    parser.add_argument("--type", default="flow_vn",
                        help="Observable type directory under data/ (default: flow_vn)")
    parser.add_argument("--reaction", default=None,
                        help="Reaction pattern (regex, case-insensitive, e.g. 'AU AU')")
    parser.add_argument("--energy", type=float, default=None,
                        help="Collision energy in GeV (e.g. 200, 2760)")
    parser.add_argument("--value-type-regex", default=None,
                        help="Regex pattern for value_type field")
    parser.add_argument("--dimensionality", default=None,
                        help="Filter by dimensionality (integrated, differential_pT, differential_eta)")
    parser.add_argument("--method", default=None,
                        help="Filter by method (cumulant, event_plane, scalar_product)")
    parser.add_argument("--harmonic", type=int, default=None,
                        help="Filter by harmonic n (e.g. 2, 3, 4)")
    parser.add_argument("--cumulant-order", type=int, default=None,
                        help="Filter by cumulant order (e.g. 2, 4)")
    parser.add_argument("--include-undispatchable", action="store_true",
                        help="Include records that can't be dispatched (no compute function)")
    parser.add_argument("--extra-records", action="append", default=None, metavar="DIR",
                        help="Additional record directory laid out like data/ (repeatable)")
    parser.add_argument("--count-only", action="store_true",
                        help="Just print the count, no details")
    parser.add_argument("--output", "-o", default=None,
                        help="Write manifest to this YAML file (otherwise print table)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from analysis.dispatcher import record_roots
    dirs = [os.path.join(root, args.type) for root in record_roots(args.extra_records)]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print(f"Data directory not found: {os.path.join('data', args.type)}")
        sys.exit(1)

    # Load and filter
    all_records = [r for d in dirs for r in load_records(d)]
    filtered = []
    for fpath, idx, doc in all_records:
        if matches(doc, args):
            filtered.append((fpath, idx, doc))

    if args.count_only:
        print(f"Total: {len(filtered)} matching records (out of {len(all_records)} total)")
        return

    if args.output:
        generate_manifest(filtered, args.output)
    else:
        print_table(filtered, args)


if __name__ == "__main__":
    main()

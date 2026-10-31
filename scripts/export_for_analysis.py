#!/usr/bin/env python3
"""
Export curated records + raw HEPData into a single JSON manifest for
downstream analysis (jetscape-analysis, etc.).

Usage:
    python scripts/export_for_analysis.py                     # export from data/
    python scripts/export_for_analysis.py --candidates        # include data_candidates/
    python scripts/export_for_analysis.py --type flow_vn      # filter by type
    python scripts/export_for_analysis.py --output manifest.json
    python scripts/export_for_analysis.py --validate          # validate schemas too
"""

import json
import re
import sys
import yaml
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
CANDIDATES_DIR = REPO_DIR / "data_candidates"
HEPDATA_DIR = REPO_DIR / "hepdata_raw" / "records"
SCHEMA_DIR = REPO_DIR / "observables"
DEFAULT_OUTPUT = REPO_DIR / "analysis_manifest.json"

_HEPDATA_CACHE: dict[str, dict | None] = {}
_EXPERIMENT_CACHE: dict[str, str | None] = {}


def resolve_experiment(inspire_id):
    """Resolve experiment/collaboration from HEPData metadata or record filename."""
    iid = str(inspire_id)

    # PHOBOS records have non-numeric inspire_ids
    if "PHOBOS" in iid.upper():
        return "PHOBOS"

    # Canonicalize common variants
    _CANON = {
        "A LARGE ION COLLIDER EXPERIMENT": "ALICE",
        "AACHEN-BERLIN-BONN-CERN-CRACOW-HEIDELBERG-WARSAW": "NA49",
        "AACHEN-BERLIN-BONN-CERN-CRACOW-HEIDELBERG": "NA49",
        "AACHEN-BERLIN-BONN-CERN-HEIDELBERG-LONDON-VIENNA": "NA49",
        "AACHEN-HAMBURG-HEIDELBERG-MUNICH": "NA49",
        "AACHEN-BONN-CERN-MUNICH-OXFORD": "NA49",
        "AMSTERDAM-CERN-NIJMEGEN-OXFORD": "NA49",
        "ATHENS-DEMOCRITOS-LIVERPOOL-VIENNA": "NA49",
        "ACCMOR": "NA22",
        "BERLIN-BUDAPEST-DUBNA-PRAGUE-SERPUKHOV-SOFIA": "NA61/SHINE",
        "CERN-MUNICH": "NA49",
    }

    if iid in _EXPERIMENT_CACHE:
        exp = _EXPERIMENT_CACHE[iid]
        return _CANON.get(exp.upper(), exp) if exp else None

    meta_path = HEPDATA_DIR / f"ins{iid}" / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            collabs = meta.get("collaborations", [])
            if collabs:
                raw = collabs[0]
                _EXPERIMENT_CACHE[iid] = raw
                return _CANON.get(raw.upper(), raw)
        except Exception:
            pass

    _EXPERIMENT_CACHE[iid] = None
    return None


def resolve_hepdata_path(inspire_id, version, filename):
    """Resolve the full path to a raw HEPData YAML file."""
    base = HEPDATA_DIR / f"ins{inspire_id}"
    if not base.exists():
        return None
    yaml_dirs = sorted(d for d in base.iterdir()
                       if d.is_dir() and d.name.endswith("-yaml"))
    for d in yaml_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


def load_hepdata_table(inspire_id, version, filename):
    """Load raw HEPData YAML table data."""
    cache_key = f"{inspire_id}/{version}/{filename}"
    if cache_key in _HEPDATA_CACHE:
        return _HEPDATA_CACHE[cache_key]
    path = resolve_hepdata_path(inspire_id, version, filename)
    if path is None:
        _HEPDATA_CACHE[cache_key] = None
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        _HEPDATA_CACHE[cache_key] = data
        return data
    except Exception:
        _HEPDATA_CACHE[cache_key] = None
        return None


def _apply_value_scale(raw_data, factor):
    """Multiply every dependent-variable value (and its errors) by factor.

    Only the DEPENDENT variables are scaled.  Independent variables are
    bin coordinates (centrality, pT, eta) and must not be touched.
    """
    def _scale(x):
        # HEPData tables occasionally carry placeholder strings ("", "none",
        # "0,494") in numeric columns; leave those untouched instead of
        # crashing the whole export.
        return x * factor if isinstance(x, (int, float)) else x

    for dv in raw_data.get("dependent_variables", []):
        for v in dv.get("values", []):
            if v.get("value") is not None:
                v["value"] = _scale(v["value"])
            for e in v.get("errors", []) or []:
                if e.get("value") is not None:
                    e["value"] = _scale(e["value"])
                for k in ("symerror", "asymmerror"):
                    if e.get(k) is not None:
                        e[k] = _scale(e[k])


def extract_raw_values(raw_data):
    """Extract independent/dependent variable arrays from raw HEPData YAML."""
    if raw_data is None:
        return {"independent_variables": [], "dependent_variables": []}

    result = {"independent_variables": [], "dependent_variables": []}

    for iv in raw_data.get("independent_variables", []):
        h = iv.get("header", {})
        entry = {
            "name": h.get("name", ""),
            "unit": h.get("units", ""),
            "values": [],
        }
        for v in iv.get("values", []):
            if "low" in v and "high" in v:
                entry["values"].append({
                    "low": v["low"],
                    "high": v["high"],
                })
            elif "value" in v:
                entry["values"].append({"value": v["value"]})
        result["independent_variables"].append(entry)

    for dv in raw_data.get("dependent_variables", []):
        h = dv.get("header", {})
        entry = {
            "name": h.get("name", ""),
            "unit": h.get("units", ""),
            "qualifiers": [],
            "values": [],
        }
        for q in dv.get("qualifiers", []):
            entry["qualifiers"].append({
                "name": q.get("name", ""),
                "value": q.get("value", ""),
                "units": q.get("units", ""),
            })
        for v in dv.get("values", []):
            val_entry = {"value": v.get("value")}
            if "errors" in v:
                val_entry["errors"] = [
                    {"label": e.get("label", ""),
                     "value": e.get("value"),
                     "symerror": e.get("symerror"),
                     "asymmerror": e.get("asymmerror")}
                    for e in v["errors"]
                ]
            entry["values"].append(val_entry)
        result["dependent_variables"].append(entry)

    return result


def find_curated_records(search_dir, include_candidates=False):
    """Find all curated record YAML files in search_dir and optionally candidates."""
    records = []

    # Flat YAML files directly in observable-type subdirs
    for subdir in sorted(search_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        for f in sorted(subdir.iterdir()):
            if f.suffix in (".yaml", ".yml"):
                records.append(f)

    # Also scan directories that contain a tables/ subdir
    for subdir in sorted(search_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        for f in sorted(subdir.iterdir()):
            if f.is_dir() and (f / "tables").is_dir():
                curated_yaml = f / f"{f.name}.yaml"
                if not curated_yaml.exists():
                    curated_yaml = f / "curated.yaml"
                if curated_yaml.exists():
                    records.append(curated_yaml)

    if include_candidates:
        candidates_dir = REPO_DIR / "data_candidates"
        if candidates_dir.exists():
            for subdir in sorted(candidates_dir.iterdir()):
                if not subdir.is_dir() or subdir.name.startswith("."):
                    continue
                for f in sorted(subdir.iterdir()):
                    if f.suffix in (".yaml", ".yml"):
                        records.append(f)

    return sorted(set(records))


def load_schema(obs_type):
    """Load schema definition for an observable type."""
    path = SCHEMA_DIR / f"{obs_type}.yaml"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def record_matches_filter(record, obs_type_filter):
    """Check if a curated record file matches the observable type filter.
    
    For multi-doc files, checks all docs (any match = keep the file).
    """
    if obs_type_filter is None:
        return True
    try:
        with open(record) as f:
            docs = list(yaml.safe_load_all(f))
        return any(
            isinstance(d, dict) and d.get("observable_type") == obs_type_filter
            for d in docs
        )
    except Exception:
        return False


def export_record(record_path):
    """Export records from a curated YAML file (may be multi-doc).

    Returns (list_of_entries, list_of_errors).
    """
    try:
        with open(record_path) as f:
            docs = list(yaml.safe_load_all(f))
    except Exception as e:
        return [], [f"cannot parse {record_path}: {e}"]

    entries = []
    errors = []

    for i, curated in enumerate(docs):
        if not isinstance(curated, dict):
            errors.append(f"doc{i} in {record_path}: empty or invalid")
            continue

        inspire_id = str(curated.get("inspire_id", ""))
        version = str(curated.get("version", "1"))
        hepdata_file = curated.get("hepdata_file")
        obs_type = curated.get("observable_type", "unknown")

        entry = {
            "source": f"{record_path.relative_to(REPO_DIR)}:doc{i}",
            "observable_type": obs_type,
            "inspire_id": inspire_id,
            "version": version,
        }

        # A publication may tabulate the quantity pre-multiplied for
        # legibility -- "v_3(TPC) (%)", "v_3*100", "v_3^2*10^4".  HEPData
        # reports those verbatim with units:none, so nothing downstream can
        # tell a percent table from a fraction and the comparison is then
        # wrong by 100x while still looking plausible.  `value_scale` on the
        # record is the factor to the canonical unit; apply it HERE, once,
        # so every consumer of the manifest gets physical values.
        value_scale = curated.get("value_scale", 1.0) or 1.0
        if value_scale != 1.0:
            entry["value_scale_applied"] = value_scale

        if hepdata_file:
            entry["hepdata_file"] = hepdata_file
            raw = load_hepdata_table(inspire_id, version, hepdata_file)
            entry["raw_data"] = extract_raw_values(raw)
            entry["raw_data_available"] = raw is not None
            if raw is not None and value_scale != 1.0:
                _apply_value_scale(entry["raw_data"], value_scale)
        else:
            entry["raw_data"] = {"independent_variables": [], "dependent_variables": []}
            entry["raw_data_available"] = False

        entry["experiment"] = resolve_experiment(inspire_id)

        for k, v in curated.items():
            if k not in ("inspire_id", "version", "hepdata_file", "observable_type"):
                entry[k] = v

        cs = curated.get("collision_system", {})
        if isinstance(cs, dict):
            entry["energy_gev"] = cs.get("energy_gev")

        entries.append(entry)

    return entries, errors if errors else None


def validate_entry(entry, strict=False):
    """Validate an export entry against its schema (basic checks)."""
    obs_type = entry.get("observable_type", "")
    schema = load_schema(obs_type)
    if schema is None:
        return ["unknown observable type"] if strict else []

    warnings = []
    required = schema.get("required_fields", {})
    for field_name in required:
        if field_name not in entry or entry[field_name] is None:
            warnings.append(f"missing required field: {field_name}")

    if "hepdata_file" in entry and not entry.get("raw_data_available", False):
        warnings.append(f"raw data not found for hepdata_file={entry.get('hepdata_file')}")

    return warnings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export curated records for analysis")
    parser.add_argument("--candidates", action="store_true",
                        help="Include candidate records from data_candidates/")
    parser.add_argument("--type", type=str, default=None,
                        help="Filter by observable type (e.g. flow_vn)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--validate", action="store_true",
                        help="Run schema validation on each record")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on schema warnings")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print JSON output")
    args = parser.parse_args()

    records = find_curated_records(DATA_DIR, include_candidates=args.candidates)
    if not records:
        print(f"No curated records found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.type:
        records = [r for r in records if record_matches_filter(r, args.type)]

    exported = []
    errors = []
    total_warnings = 0

    for rec_path in records:
        if not record_matches_filter(rec_path, args.type):
            continue

        entries, err = export_record(rec_path)
        if err:
            for e in err:
                errors.append(e)
            continue

        for entry in entries:
            if args.validate:
                warnings = validate_entry(entry, strict=args.strict)
                if warnings:
                    total_warnings += len(warnings)
                    for w in warnings:
                        errors.append(f"{rec_path.relative_to(REPO_DIR)}: {w}")
                    if args.strict:
                        continue

            exported.append(entry)

    output = {
        "manifest_version": "1.0",
        "generator": "export_for_analysis.py",
        "total_records": len(records),
        "exported": len(exported),
        "errors": len(errors),
        "records": exported,
    }

    if errors:
        output["error_details"] = errors

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2 if args.pretty else None,
                  default=str)

    print(f"Exported {len(exported)}/{len(records)} records to {args.output}")
    if errors:
        print(f"  {len(errors)} errors/warnings", file=sys.stderr)
        for e in errors[:10]:
            print(f"  - {e}", file=sys.stderr)
    if total_warnings:
        print(f"  {total_warnings} schema warnings")


if __name__ == "__main__":
    main()

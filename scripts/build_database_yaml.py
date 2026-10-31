#!/usr/bin/env python3
"""
Build hepdata_database.yaml — the catalog of all curated records in data/.

Scans data/ for curated record YAML files, reads their metadata,
and writes a registry that downstream tools can use to discover records.

Usage:
    python scripts/build_database_yaml.py
    python scripts/build_database_yaml.py --check     # check for issues without writing
    python scripts/build_database_yaml.py --schema    # validate against schemas too
"""

import sys
import yaml
from pathlib import Path
from collections import Counter

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
SCHEMA_DIR = REPO_DIR / "observables"
OUTPUT_PATH = REPO_DIR / "hepdata_database.yaml"

SCHEMA_CACHE = {}


def load_schema(obs_type):
    if obs_type in SCHEMA_CACHE:
        return SCHEMA_CACHE[obs_type]
    path = SCHEMA_DIR / f"{obs_type}.yaml"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            schema = yaml.safe_load(f)
        SCHEMA_CACHE[obs_type] = schema
        return schema
    except Exception:
        return None


def find_curated_records():
    """Find all curated record YAML files under data/."""
    records = []
    for subdir in sorted(DATA_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        for f in sorted(subdir.iterdir()):
            if f.suffix in (".yaml", ".yml"):
                records.append(f)
    return records


def read_curated_records(path):
    """Read a curated record YAML file (may be multi-doc) and return list of doc dicts."""
    try:
        with open(path) as f:
            docs = list(yaml.safe_load_all(f))
        return [d for d in docs if isinstance(d, dict)]
    except Exception as e:
        return [{"_error": str(e)}]


def schema_has_kinematic_range(obs_type):
    """Check if a schema composes kinematic_range."""
    schema = load_schema(obs_type)
    if not schema:
        return False
    composes = schema.get("composes", [])
    return any("kinematic_range" in c for c in composes)


def check_record(record, path):
    """Check a curated record for common issues. Returns list of warnings."""
    warnings = []
    if not isinstance(record, dict):
        warnings.append("file is not valid YAML dict")
        return warnings

    obs_type = record.get("observable_type", "")
    if not obs_type:
        warnings.append("missing observable_type")
        return warnings

    schema = load_schema(obs_type)
    if schema is None:
        warnings.append(f"unknown observable_type: {obs_type}")
    else:
        required = schema.get("required_fields", {})
        for name, defn in required.items():
            req = defn.get("required", True)
            if req and name not in record:
                warnings.append(f"missing required field: {name}")

    if not record.get("inspire_id"):
        warnings.append("missing inspire_id")
    if not record.get("version"):
        warnings.append("missing version")

    hepdata_file = record.get("hepdata_file")
    if hepdata_file:
        inspire_id = str(record.get("inspire_id", ""))
        version = str(record.get("version", "1"))
        hepdata_dir = REPO_DIR / "hepdata_raw" / "records" / f"ins{inspire_id}"
        if hepdata_dir.exists():
            yaml_dirs = sorted(
                d for d in hepdata_dir.iterdir()
                if d.is_dir() and d.name.endswith("-yaml")
            )
            found = any((d / hepdata_file).exists() for d in yaml_dirs)
            if not found:
                warnings.append(f"hepdata_file '{hepdata_file}' not found on disk")

    kr = record.get("kinematic_range", {})
    if isinstance(kr, dict) and not kr:
        if schema_has_kinematic_range(obs_type):
            warnings.append("empty kinematic_range (schema requires it)")

    return warnings


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build hepdata_database.yaml catalog")
    parser.add_argument("--check", action="store_true",
                        help="Check records for issues without writing")
    parser.add_argument("--schema", action="store_true",
                        help="Also validate against observable schemas")
    args = parser.parse_args()

    files = find_curated_records()
    if not files:
        print(f"No curated records found in {DATA_DIR}", file=sys.stderr)
        sys.exit(0 if args.check else 1)

    database = []
    total_warnings = 0
    issues_found = False

    for rec_path in files:
        docs = read_curated_records(rec_path)
        rel_path = rec_path.relative_to(REPO_DIR)
        file_dir = str(rec_path.parent.relative_to(REPO_DIR))
        file_name = rec_path.name

        # Build one entry per multi-doc file, aggregating tables_to_filenames
        entry = None
        tables = {}

        for doc in docs:
            if not isinstance(doc, dict):
                continue

            warnings = check_record(doc, rec_path) if isinstance(doc, dict) else ["invalid YAML"]
            if args.schema and isinstance(doc, dict):
                from import_manual import validate_against_schema
                schema_ok = validate_against_schema(
                    doc, doc.get("observable_type", ""), verbose=False)
                if not schema_ok:
                    warnings.append("schema validation failed")

            if warnings:
                issues_found = True
                total_warnings += len(warnings)
                print(f"  ⚠ {rel_path}", file=sys.stderr)
                for w in warnings:
                    print(f"    - {w}", file=sys.stderr)

            if not isinstance(doc, dict):
                continue

            hf = doc.get("hepdata_file", "")
            if hf:
                table_name = hf.replace(".yaml", "").replace(".yml", "")
                tables[table_name] = hf

            if entry is None:
                entry = {
                    "directory": f"{file_dir}/{file_name}",
                    "inspire_hep_id": str(doc.get("inspire_id", "")),
                    "version": str(doc.get("version", "1")),
                }

        if entry and tables:
            entry["tables_to_filenames"] = tables
            database.append(entry)

    if args.check:
        if issues_found:
            print(f"\nChecked {len(files)} files, {total_warnings} issue(s) found.",
                  file=sys.stderr)
        else:
            print(f"Checked {len(files)} files — all clean.")
        sys.exit(1 if issues_found else 0)

    output = {
        "hepdata_database": database,
        "_meta": {
            "generator": "build_database_yaml.py",
            "total_files": len(database),
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(output, f, default_flow_style=False,
                  sort_keys=False, width=120, allow_unicode=True)

    print(f"Wrote {len(database)} entries to {OUTPUT_PATH}")
    if total_warnings:
        print(f"  ({total_warnings} issues found during scan — see above)")


if __name__ == "__main__":
    main()

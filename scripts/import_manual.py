#!/usr/bin/env python3
"""
Create curated records for measurements not available on HEPData.

Reads a simplified YAML template, validates against the schema, and writes
a curated record into the data/ directory tree.

Usage:
    # Interactive prompt
    python scripts/import_manual.py

    # From YAML template
    python scripts/import_manual.py my_measurement.yaml

    # Validate an existing manual record
    python scripts/import_manual.py --validate data/flow_vn/my_record.yaml

Template format:
    observable_type: flow_vn
    inspire_id: "1234567"          # or a placeholder like "manual-001"
    version: 1
    hepdata_file: "table.yaml"     # optional, for non-HEPData omit this

    collision_system:
      energy_gev: 200.0
      reaction: "AU AU --> CHARGED X"

    centrality:
      classes: ["0-5%", "5-10%"]

    particle_id:
      species: "pion"

    kinematic_range:
      pT:
        min: 0.2
        max: 2.0

    # Observable-specific fields as a flat mapping
    harmonic: 2
    method: "event_plane"
    value_type: "v2"
    dimensionality: "integrated"
"""

import sys
import os
import yaml
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
SCHEMA_DIR = REPO_DIR / "observables"

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


def get_known_fields(obs_type):
    """Return the set of known field names for an observable type."""
    schema = load_schema(obs_type)
    if schema is None:
        return set()

    fields = set()

    # Top-level required/optional fields
    for section in ("required_fields", "optional_fields"):
        for name in schema.get(section, {}):
            fields.add(name)

    # Composed blocks from _common
    composed = schema.get("composes", [])
    common = load_schema("_common")
    if common:
        for section in ("required_fields", "optional_fields"):
            for name in common.get(section, {}):
                # Composed block names include the hash like _common#collision_system
                fields.add(name)

    # Observables schema has no direct field list for composes at top level,
    # but we can get composed section names
    for c in composed:
        parts = c.split("#")
        if len(parts) == 2:
            fields.add(parts[1])

    # Conditional fields
    for cf in schema.get("conditional_fields", []):
        fields.add(cf.get("field", ""))

    return fields


def validate_against_schema(record, obs_type, verbose=True):
    """Basic validation of a record against its schema."""
    schema = load_schema(obs_type)
    if schema is None:
        print(f"  ⚠ unknown schema: {obs_type}", file=sys.stderr)
        return True

    errors = []

    # Check required fields
    for name, defn in schema.get("required_fields", {}).items():
        required = defn.get("required", True)
        if required and name not in record:
            errors.append(f"missing required field: {name}")

    # Check composed sections
    composed = schema.get("composes", [])
    for c in composed:
        parts = c.split("#")
        if len(parts) == 2:
            section_name = parts[1]
            common = load_schema("_common")
            if common and section_name in common.get("required_fields", {}):
                if section_name not in record:
                    errors.append(f"missing required section: {section_name}")

    # Check conditional fields
    cf_defs = schema.get("conditional_fields", {})
    if isinstance(cf_defs, dict):
        for field_name, field_defn in cf_defs.items():
            required_if = field_defn.get("required_if", {})
            cond_field = required_if.get("field")
            cond_value = required_if.get("value")
            if cond_field and cond_value is not None:
                actual = record.get(cond_field)
                if str(actual) == str(cond_value):
                    if field_name not in record:
                        errors.append(f"missing field '{field_name}' required when "
                                      f"'{cond_field}' = '{cond_value}'")
    elif isinstance(cf_defs, list):
        for entry in cf_defs:
            field_name = entry.get("field")
            required_if = entry.get("required_if", {})
            cond_field = required_if.get("field")
            cond_value = required_if.get("value")
            if cond_field and cond_value is not None:
                actual = record.get(cond_field)
                if str(actual) == str(cond_value):
                    if field_name not in record:
                        errors.append(f"missing field '{field_name}' required when "
                                      f"'{cond_field}' = '{cond_value}'")

    if errors:
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return False

    if verbose:
        print(f"  ✓ schema validation passed", file=sys.stderr)
    return True


def prompt_user(record):
    """Interactive prompting to fill in missing fields."""
    obs_type = record.get("observable_type", "")
    if not obs_type:
        print("Error: observable_type is required", file=sys.stderr)
        return False

    schema = load_schema(obs_type)
    if schema is None:
        print(f"Unknown observable type: {obs_type}", file=sys.stderr)
        return False

    known = get_known_fields(obs_type)

    # These are always allowed
    always_known = {"inspire_id", "version", "hepdata_file", "observable_type"}

    # Check for unknown fields
    for key in record:
        if key not in known and key not in ["inspire_id", "version", "hepdata_file", "observable_type"]:
            print(f"  ⚠ unknown field: {key}", file=sys.stderr)

    return True


def write_record(record, output_path):
    """Write a curated record to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(record, f, default_flow_style=False,
                  sort_keys=False, width=120, allow_unicode=True)
    print(f"  ✓ wrote {output_path}", file=sys.stderr)


def auto_filename(record):
    """Generate a sensible filename from the record."""
    obs_type = record.get("observable_type", "unknown")
    inspire_id = str(record.get("inspire_id", ""))
    hepdata_file = record.get("hepdata_file", "")
    base = os.path.splitext(hepdata_file)[0] if hepdata_file else "manual"

    # Try to make a descriptive name
    species = ""
    cs = record.get("collision_system", {})
    if isinstance(cs, dict):
        energy = cs.get("energy_gev", "")
        if energy:
            species += f"_{energy}gev"
    pid = record.get("particle_id", {})
    if isinstance(pid, dict) and pid.get("species", "") not in ("", "charged"):
        species += f"_{pid['species']}"

    return f"{inspire_id}_{base}{species}.yaml"


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Import manual records not on HEPData")
    parser.add_argument("input", nargs="?", default=None,
                        help="YAML template file (omit for interactive)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate a record file and exit")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory under data/ "
                             "(default: auto-detect from observable_type)")
    parser.add_argument("--no-write", action="store_true",
                        help="Parse and validate but do not write")
    args = parser.parse_args()

    record = None

    if args.input:
        try:
            with open(args.input) as f:
                record = yaml.safe_load(f)
        except Exception as e:
            print(f"Error reading {args.input}: {e}", file=sys.stderr)
            sys.exit(1)

        if args.validate:
            obs_type = record.get("observable_type", "")
            ok = validate_against_schema(record, obs_type, verbose=True)
            sys.exit(0 if ok else 1)

        prompt_user(record)
    else:
        # Interactive mode
        print("Interactive manual record entry")
        print("-------------------------------")
        obs_type = input("Observable type: ").strip()
        if not obs_type:
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

        schema = load_schema(obs_type)
        if schema is None:
            print(f"Unknown type. Available:", file=sys.stderr)
            for p in sorted(SCHEMA_DIR.glob("*.yaml")):
                print(f"  {p.stem}", file=sys.stderr)
            sys.exit(1)

        record = {"observable_type": obs_type}

        inspire_id = input("INSPIRE ID (or manual placeholder): ").strip()
        record["inspire_id"] = inspire_id or "manual-unknown"

        version = input("Version [1]: ").strip()
        record["version"] = version or "1"

        hepdata_file = input("HEPData filename (or blank): ").strip()
        if hepdata_file:
            record["hepdata_file"] = hepdata_file

        # Collision system
        print("\nCollision system:")
        energy = input("  Energy (GeV): ").strip()
        reaction = input("  Reaction (e.g. PB PB --> CHARGED X): ").strip()
        if energy or reaction:
            record["collision_system"] = {}
            if energy:
                record["collision_system"]["energy_gev"] = float(energy)
            if reaction:
                record["collision_system"]["reaction"] = reaction

        # Centrality
        cent_str = input("\nCentrality classes (comma-separated, e.g. '0-5%,5-10%'): ").strip()
        if cent_str:
            record["centrality"] = {"classes": [c.strip() for c in cent_str.split(",")]}

        print("\nRecord preview:")
        print(yaml.dump(record, default_flow_style=False, sort_keys=False))

        ok = input("\nWrite this record? [y/N]: ").strip().lower()
        if ok != "y":
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

    # Determine output path
    obs_type = record["observable_type"]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DATA_DIR / obs_type

    fname = auto_filename(record)
    output_path = output_dir / fname

    if not args.no_write:
        # Validate before writing
        ok = validate_against_schema(record, obs_type, verbose=True)
        if ok:
            write_record(record, output_path)
    else:
        print(yaml.dump(record, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()

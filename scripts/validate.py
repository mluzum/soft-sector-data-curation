#!/usr/bin/env python3
"""
Validate curated measurement records against their schema definitions.

Usage:
    python scripts/validate.py data/particle_spectra/my_record.yaml
    python scripts/validate.py data/                        # validate all
    python scripts/validate.py data/ --verbose               # detailed output
    python scripts/validate.py data/ --summary-only           # just pass/fail counts
    python scripts/validate.py data/ --schema flow_vn         # only this schema
"""

import yaml
import sys
import os
import glob
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "observables"
COMMON_FILE = "_common.yaml"


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_ref(ref):
    """Parse '_common#collision_system' → ('_common', 'collision_system')."""
    parts = ref.split("#", 1)
    return parts[0], parts[1] if len(parts) > 1 else None


def _ensure_list(val):
    """Wrap scalar in a list so we can iterate uniformly."""
    if isinstance(val, list):
        return val
    return [val]


def _schema_path(observable_type):
    return SCHEMA_DIR / f"{observable_type}.yaml"


def _fmt(path, msg):
    """Return coloured (if terminal) error string."""
    return f"  ✗ {path}: {msg}"


def _fmt_ok(msg):
    return f"  ✓ {msg}"


# ── Schema loading & resolution ──────────────────────────────────────────

_COMMON_CACHE = None

def _load_common():
    global _COMMON_CACHE
    if _COMMON_CACHE is None:
        _COMMON_CACHE = _load_yaml(SCHEMA_DIR / COMMON_FILE)
    return _COMMON_CACHE


def resolve_schema(schema):
    """Flatten a schema into validated field definitions.

    Returns
    -------
    dict with keys:
        sections         : dict[section_name → dict[field_name → definition]]
        top_level_fields : dict[field_name → definition]
        conditional      : dict[field_name → definition]
        dimensionality   : dict[dim_name → rule_dict]
        observable_type  : str
        validation_rules : list[str]
    """
    common = _load_common()
    sections = {}
    top_level = {}
    field_order = []  # preserve insertion order for readable output

    # ---- required_fields ----
    for name, defn in schema.get("required_fields", {}).items():
        defn["_required"] = defn.get("required", True)
        top_level[name] = defn
        field_order.append(name)

    # ---- optional_fields ----
    for name, defn in schema.get("optional_fields", {}).items():
        defn["_required"] = False
        top_level[name] = defn
        field_order.append(name)

    # ---- composes (required) ----
    for compose_ref in schema.get("composes", []):
        file_name, section_name = _parse_ref(compose_ref)
        source = common if file_name == "_common" else _load_yaml(SCHEMA_DIR / f"{file_name}.yaml")
        if section_name and section_name in source:
            section_def = source[section_name]
            if "fields" in section_def:
                # Mark as required (default) — any required sub-field makes this needed
                sections[section_name] = section_def["fields"]
                sections.setdefault("_section_flags", {})[section_name] = {"optional": False}

    # ---- optional_composes ----
    for compose_ref in schema.get("optional_composes", []):
        file_name, section_name = _parse_ref(compose_ref)
        source = common if file_name == "_common" else _load_yaml(SCHEMA_DIR / f"{file_name}.yaml")
        if section_name and section_name in source:
            section_def = source[section_name]
            if "fields" in section_def:
                sections[section_name] = section_def["fields"]
                sections.setdefault("_section_flags", {})[section_name] = {"optional": True}

    return {
        "sections": sections,
        "top_level_fields": top_level,
        "field_order": field_order,
        "conditional_fields": schema.get("conditional_fields", {}),
        "dimensionality_rules": schema.get("dimensionality_rules", {}),
        "observable_type": schema.get("observable_type"),
        "validation_rules": schema.get("validation", {}).get("rules", []),
    }


# ── Type validation ──────────────────────────────────────────────────────

TYPE_VALIDATORS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int),
    "float": lambda v: isinstance(v, (int, float)),
    "boolean": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}

# ── Validation rule checkers ──────────────────────────────────────────────

def _check_harmonic_must_be_positive(record, record_name):
    errors = []
    harmonic = record.get("harmonic")
    if harmonic is not None and (not isinstance(harmonic, int) or harmonic < 1):
        errors.append(f"'{record_name}': 'harmonic' must be a positive integer >= 1 (got {harmonic})")
    return errors

def _check_harmonic_must_be_at_most_10(record, record_name):
    errors = []
    harmonic = record.get("harmonic")
    if harmonic is not None and harmonic > 10:
        errors.append(
            f"'{record_name}': 'harmonic' > 10 (got {harmonic}) — likely misclassified correlator"
        )
    return errors

def _check_cumulant_order_required_if_method_cumulant(record, record_name):
    errors = []
    method = record.get("method")
    if method == "cumulant":
        co = record.get("cumulant_order")
        if co is None:
            errors.append(f"'{record_name}': 'cumulant_order' required when method='cumulant'")
    return errors

def _check_vn_between_minus_one_and_one(record, record_name):
    errors = []
    val = record.get("value")
    if isinstance(val, (int, float)) and (val < -1 or val > 1):
        errors.append(f"'{record_name}': value {val} outside [-1, 1]")
    return errors

def _check_harmonics_must_be_positive(record, record_name):
    errors = []
    harmonics = record.get("harmonics")
    if isinstance(harmonics, list):
        for h in harmonics:
            if not isinstance(h, int) or h < 1:
                errors.append(f"'{record_name}': harmonics must be positive integers (got {h})")
    return errors

def _check_symmetric_cumulant_range(record, record_name):
    errors = []
    corr_type = record.get("correlation_type")
    harmonics = record.get("harmonics")
    if corr_type == "symmetric_cumulant":
        if not isinstance(harmonics, list) or len(harmonics) < 2:
            errors.append(f"'{record_name}': symmetric_cumulant requires at least 2 harmonics (got {harmonics})")
    if corr_type == "event_plane_correlation" and isinstance(harmonics, list):
        if len(harmonics) > 4:
            errors.append(f"'{record_name}': event_plane_correlation typically uses ≤4 harmonics (got {len(harmonics)})")
    return errors

def _check_subevent_consistency(record, record_name):
    errors = []
    particles = record.get("particles")
    subevents = record.get("subevents")
    cumulant_order = record.get("cumulant_order")

    if particles:
        if not isinstance(particles, str) or len(particles) < 2:
            errors.append(f"'{record_name}': 'particles' must be a string of length ≥ 2")
            return errors
        if cumulant_order and len(particles) != cumulant_order:
            errors.append(f"'{record_name}': 'particles' length ({len(particles)}) must equal cumulant_order ({cumulant_order})")
        unique_letters = set(particles)
        invalid = [c for c in unique_letters if not c.isupper() or not c.isalpha()]
        if invalid:
            errors.append(f"'{record_name}': 'particles' letters must be uppercase A-Z (got {invalid})")
        if len(unique_letters) > 1 and not subevents:
            errors.append(f"'{record_name}': 'subevents' required when 'particles' has >1 unique letter ({particles})")
        if subevents:
            if not isinstance(subevents, dict):
                errors.append(f"'{record_name}': 'subevents' must be an object (dict)")
            else:
                for c in unique_letters:
                    if c not in subevents:
                        errors.append(f"'{record_name}': 'subevents.{c}': missing subevent for letter '{c}' in particles='{particles}'")
                for key in subevents:
                    if key not in unique_letters:
                        errors.append(f"'{record_name}': 'subevents.{key}': unused subevent (not in particles='{particles}')")
                    se = subevents[key]
                    if not isinstance(se, dict):
                        errors.append(f"'{record_name}': 'subevents.{key}': must be an object")
                        continue
                    kr = se.get("kinematic_range", {})
                    if kr and isinstance(kr, dict):
                        eta = kr.get("eta", {})
                        if not isinstance(eta, dict) or "min" not in eta or "max" not in eta:
                            errors.append(f"'{record_name}': 'subevents.{key}.kinematic_range.eta': must have min and max")
                        pt = kr.get("pT", {})
                        if pt and isinstance(pt, dict):
                            if "min" not in pt or "max" not in pt:
                                errors.append(f"'{record_name}': 'subevents.{key}.kinematic_range.pT': if present, must have min and max")
        # eta_gap: recommend setting it when multiple subevents have non-overlapping η
        if len(unique_letters) > 1 and "eta_gap" not in record:
            pass  # optional — can be derived
    elif subevents:
        errors.append(f"'{record_name}': 'particles' required when 'subevents' is present")

    return errors


RULE_REGISTRY = {
    "harmonic_must_be_positive": _check_harmonic_must_be_positive,
    "harmonic_must_be_at_most_10": _check_harmonic_must_be_at_most_10,
    "cumulant_order_required_if_method_cumulant": _check_cumulant_order_required_if_method_cumulant,
    "vn_between_minus_one_and_one": _check_vn_between_minus_one_and_one,
    "harmonics_must_be_positive": _check_harmonics_must_be_positive,
    "subevent_consistency": _check_subevent_consistency,
    "symmetric_cumulant_range_validation": _check_symmetric_cumulant_range,
}


def validate_type_value(field_path, value, defn):
    """Return a list of error strings; empty = valid."""
    errors = []
    expected_type = defn.get("type")
    if expected_type and expected_type in TYPE_VALIDATORS:
        if not TYPE_VALIDATORS[expected_type](value):
            errors.append(f"{field_path}: expected type '{expected_type}', got {type(value).__name__}")
    if "allowed" in defn:
        vals = _ensure_list(value)
        for v in vals:
            if v not in defn["allowed"]:
                allowed_str = ", ".join(str(a) for a in defn["allowed"])
                errors.append(f"{field_path}: value '{v}' not in allowed [{allowed_str}]")
    if expected_type == "object" and "fields" in defn and isinstance(value, dict):
        for sub_name, sub_defn in defn["fields"].items():
            sub_path = f"{field_path}.{sub_name}"
            if sub_name in value:
                errors.extend(validate_type_value(sub_path, value[sub_name], sub_defn))
            elif sub_defn.get("required", False):
                errors.append(f"{sub_path}: missing required field")
    return errors


# ── Core validation ──────────────────────────────────────────────────────

def validate_doc(record, record_path, doc_index=None, verbose=False):
    """Validate a single record dict.

    Parameters
    ----------
    record : dict
        Parsed record data.
    record_path : str or Path
        Display name for error messages (file path).
    doc_index : int or None
        Document index in multi-doc file.

    Returns list of error strings (empty = valid).
    """
    record_name = str(record_path)
    if doc_index is not None:
        record_name = f"{record_path}:doc{doc_index}"

    obs_type = record.get("observable_type")
    if not obs_type:
        return [f"{record_name}: missing 'observable_type'"]

    schema_path = _schema_path(obs_type)
    if not schema_path.exists():
        return [f"{record_name}: no schema file for observable_type '{obs_type}'"]

    schema = _load_yaml(schema_path)
    resolved = resolve_schema(schema)
    errors = []

    # ── 1. Top-level required fields ──
    for name, defn in resolved["top_level_fields"].items():
        if defn.get("_required"):
            if name not in record:
                errors.append(f"'{record_name}': '{name}': missing required field")
            else:
                errors.extend(validate_type_value(f"'{record_name}': '{name}'", record[name], defn))

    # ── 2. Composed sections ──
    flags = resolved["sections"].pop("_section_flags", {})
    for section_name, fields in resolved["sections"].items():
        section_data = record.get(section_name)
        if section_data is None:
            if flags.get(section_name, {}).get("optional"):
                continue
            section_required = any(f.get("required", False) for f in fields.values())
            if section_required:
                errors.append(f"'{record_name}': '{section_name}': missing required section")
            continue
        if not isinstance(section_data, dict):
            errors.append(f"'{record_name}': '{section_name}': expected object (dict)")
            continue
        for field_name, defn in fields.items():
            sub_path = f"'{record_name}': '{section_name}.{field_name}'"
            if defn.get("required"):
                if field_name not in section_data:
                    errors.append(f"{sub_path}: missing required field")
                else:
                    errors.extend(validate_type_value(sub_path, section_data[field_name], defn))
            elif field_name in section_data:
                errors.extend(validate_type_value(sub_path, section_data[field_name], defn))

    # ── 3. Conditional fields ──
    for field_name, defn in resolved["conditional_fields"].items():
        required_if = defn.get("required_if")
        if required_if:
            cond_field = required_if.get("field")
            cond_value = required_if.get("value")
            actual_value = record.get(cond_field)
            if actual_value == cond_value:
                if field_name not in record:
                    errors.append(
                        f"'{record_name}': '{field_name}': required when '{cond_field}' = '{cond_value}'"
                    )
                else:
                    errors.extend(validate_type_value(f"'{record_name}': '{field_name}'", record[field_name], defn))

    # ── 4. Dimensionality rules ──
    dimensionality = record.get("dimensionality")
    if dimensionality and dimensionality in resolved["dimensionality_rules"]:
        rules = resolved["dimensionality_rules"][dimensionality]
        for qual_name, qual_desc in rules.get("qualifiers", {}).items():
            if "required" in qual_desc.lower():
                parts = qual_name.split(".")
                found = True
                cursor = record
                for p in parts:
                    if isinstance(cursor, dict) and p in cursor:
                        cursor = cursor[p]
                    else:
                        found = False
                        break
                if not found:
                    errors.append(
                        f"'{record_name}': dimensionality='{dimensionality}': missing required qualifier "
                        f"'{qual_name}' ({qual_desc})"
                    )

    # ── 4b. Validation rules from schema ──
    for rule_name in resolved.get("validation_rules", []):
        checker = RULE_REGISTRY.get(rule_name)
        if checker:
            errors.extend(checker(record, record_name))
        elif verbose:
            errors.append(
                f"'{record_name}': unknown validation rule '{rule_name}' "
                f"(no checker registered)"
            )

    # ── 5. Unknown fields (warnings, not errors) ──
    known_top = set(resolved["top_level_fields"].keys())
    known_sections = set(resolved["sections"].keys())
    known_conds = set(resolved["conditional_fields"].keys())
    _implicit = {"observable_type", "inspire_id", "hepdata_id", "version",
                 "hepdata_file", "hepdata_table_label", "table_doi",
                 "status", "value_table", "source", "staging_reason"}
    if record.get("status", "published") not in ("published", "preliminary"):
        errors.append(f"'{record_name}': 'status': must be published or preliminary, "
                      f"got {record.get('status')!r}")
    if record.get("status") == "preliminary" and not record.get("source"):
        errors.append(f"'{record_name}': preliminary records need 'source' (where the "
                      f"values come from: proceedings, figure, how they were obtained)")
    for ax, spec in ((record.get("kinematic_range") or {}).items()):
        if isinstance(spec, dict) and spec.get("bin_ranges") is not None:
            r = spec["bin_ranges"]
            if spec.get("bins") or spec.get("points"):
                errors.append(f"'{record_name}': 'kinematic_range.{ax}': bin_ranges excludes bins and points")
            if not (isinstance(r, list) and r and all(isinstance(b, list) and len(b) == 2 and b[0] < b[1] for b in r)):
                errors.append(f"'{record_name}': 'kinematic_range.{ax}.bin_ranges' must be [lo, hi] pairs, lo < hi")
        if isinstance(spec, dict) and spec.get("points") is not None:
            w = spec.get("point_window")
            if spec.get("bins"):
                errors.append(f"'{record_name}': 'kinematic_range.{ax}': points and bins are exclusive")
            if not isinstance(w, (int, float)) or w <= 0:
                errors.append(f"'{record_name}': 'kinematic_range.{ax}.points' need a positive point_window")
    known = known_top | known_sections | known_conds | _implicit
    for key in record:
        if key not in known:
            if verbose:
                errors.append(f"'{record_name}': '{key}': unknown field (ignored)")

    return errors


def validate_record(record_path, verbose=False):
    """Validate one curated record YAML file (may contain multiple docs).

    Returns list of error strings (empty = valid).
    """
    try:
        with open(record_path) as f:
            docs = list(yaml.safe_load_all(f))
    except Exception as e:
        return [f"{Path(record_path).name}: cannot parse: {e}"]

    all_errors = []
    for i, record in enumerate(docs):
        if record is None:
            continue
        all_errors.extend(validate_doc(record, record_path, doc_index=i, verbose=verbose))

    return all_errors


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate curated records against observable schemas."
    )
    parser.add_argument("path", nargs="?", default="data/",
                        help="Record YAML file or directory to validate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show warnings for unknown fields")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print pass/fail summary per file")
    parser.add_argument("--schema", "-s", type=str, default=None,
                        help="Only validate records of this observable_type")

    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        targets = [path]
    elif path.is_dir():
        targets = sorted(path.rglob("*.yaml"))
        # Also check .yml
        targets.extend(sorted(path.rglob("*.yml")))
    else:
        print(f"error: '{path}' is not a file or directory")
        sys.exit(1)

    if not targets:
        print(f"No YAML files found under '{path}'")
        sys.exit(1)

    passed = 0
    failed = 0
    fail_details = []

    for fp in targets:
        # Skip hidden files
        if fp.name.startswith("."):
            continue
        # Skip files in hepdata_raw
        if "hepdata_raw" in fp.parts:
            continue
        # Skip schema files themselves
        if fp.parent == SCHEMA_DIR:
            continue

        rel = fp.relative_to(Path.cwd()) if fp.is_relative_to(Path.cwd()) else fp

        # Quick pre-filter: if --schema specified, skip records of other types
        if args.schema:
            try:
                with open(fp) as fh:
                    docs = list(yaml.safe_load_all(fh))
                has_type = any(
                    isinstance(d, dict) and d.get("observable_type") == args.schema
                    for d in docs
                )
                if not has_type:
                    continue
            except Exception:
                pass

        errs = validate_record(fp, verbose=args.verbose)

        if errs:
            failed += 1
            if not args.summary_only:
                print(f"✗ {rel}")
                for e in errs:
                    print(f"    {e}")
                print()
            fail_details.append((rel, errs))
        else:
            passed += 1
            if not args.summary_only:
                print(f"✓ {rel}")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  {total} record{'s' if total != 1 else ''} checked")
    print(f"  ✓ passed: {passed}")
    print(f"  ✗ failed: {failed}")
    if failed > 0 and args.summary_only:
        for rel, errs in fail_details:
            print(f"  ✗ {rel}: {errs[0]}{' …' if len(errs) > 1 else ''}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Group schema-valid candidate records from data_candidates/ into multi-doc
YAML files organized by (inspire_id, observable_type), written to data/.

Usage:
    python scripts/copy_valid_candidates.py                # grouped multi-doc
    python scripts/copy_valid_candidates.py --dry-run       # preview only
    python scripts/copy_valid_candidates.py --regenerate    # wipe data/ first
"""

import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict

REPO_DIR = Path(__file__).resolve().parent.parent
CANDIDATE_DIR = REPO_DIR / "data_candidates"
DATA_DIR = REPO_DIR / "data"
INDEX_PATH = REPO_DIR / "hepdata_raw" / "record_index.yaml"

sys.path.insert(0, str(REPO_DIR / "scripts"))
from validate import _load_yaml, resolve_schema


# ── Record index ─────────────────────────────────────────────────────────

def load_record_index():
    try:
        with open(INDEX_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ── Slug generation ──────────────────────────────────────────────────────

_PREFIX_RE = re.compile(
    r"^(?:"
    r"Measurement of (?:the )?(?:a )?"
    r"|Evidence for (?:the )?(?:a )?"
    r"|Systematic (?:study|analysis) of (?:the )?"
    r"|Two[- ]particle correlations "
    r"|Elliptic flow of "
    r"|Higher harmonic flow "
    r"|Charged hadron "
    r"|Energy dependence (?:and|of) "
    r"|Mixed (?:higher-order|harmonics) "
    r"|Event shape engineering "
    r"|Nuclear[- ]Modification Factor "
    r"|Directed flow of "
    r"|Rapidity (?:distributions|density) of "
    r")",
    re.IGNORECASE,
)

_SUFFIX_RE = re.compile(
    r"(?:"
    r" (?:in|for|at) .*?(?:Pb|Au|p|pp|d|Xe|Cu|U).*?(?:collisions|reactions).*"
    r"| with (?:the )?(?:ATLAS|CMS|ALICE|STAR|PHENIX|LHCb|PHOBOS|BRAHMS) .*"
    r"| at (?:RHIC|STAR|LHC|CERN|GSI|SPS|BNL).*"
    r"| from \\(?:sqrt|√).*"
    r"| up to .*"
    r"| at \\(?:sqrt|√).*"
    r"| in (?:pp|p|Pb|Au|Xe|Cu).*"
    r")",
    re.IGNORECASE,
)

_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "for", "at", "to", "and", "by", "with"})


def generate_slug(title):
    """Extract a short descriptive slug from a paper title."""
    if not title:
        return "unknown"

    t = title
    t = re.sub(r'\$[^$]*\$', '', t)
    t = re.sub(r'[{}^_]', ' ', t)
    t = t.replace('\\', ' ')

    t = _PREFIX_RE.sub('', t)
    t = _SUFFIX_RE.sub('', t)

    t = t.strip().strip(',').strip('.')

    words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+(?:[-+][a-zA-Z0-9]+)*', t)]
    while words and words[0] in _STOPWORDS:
        words = words[1:]
    words = words[:5]

    return '_'.join(words) if words else "unknown"


# ── File naming ──────────────────────────────────────────────────────────

_COLLISION_SHORTHAND = {
    frozenset({"PB", "PB"}): "PbPb",
    frozenset({"AU", "AU"}): "AuAu",
    frozenset({"CU", "CU"}): "CuCu",
    frozenset({"XE", "XE"}): "XeXe",
    frozenset({"U", "U"}): "UU",
    frozenset({"P", "P"}): "pp",
    frozenset({"P", "PB"}): "pPb",
    frozenset({"D", "AU"}): "dAu",
    frozenset({"NE", "NE"}): "NeNe",
    frozenset({"SI", "SI"}): "SiSi",
    frozenset({"CA", "CA"}): "CaCa",
    frozenset({"NB", "NB"}): "NbNb",
    frozenset({"CR", "CR"}): "CrCr",
}


def collision_shorthand(reaction):
    """Extract collision system shorthand from a reaction string.

    'PB PB --> CHARGED X' → 'PbPb'
    'AU AU --> CHARGED X' → 'AuAu'
    'P PB --> CHARGED X'  → 'pPb'
    Returns None if unknown.
    """
    if not reaction:
        return None
    words = set(reaction.upper().replace("-", " ").replace(">", " ").replace("→", " ").split())
    for key_set, shorthand in _COLLISION_SHORTHAND.items():
        if key_set.issubset(words):
            return shorthand
    return None


def format_energy(e):
    """Format an energy value for use in filenames."""
    try:
        e = float(e)
        if e == int(e):
            return f"{int(e)}gev"
        return f"{e}gev"
    except (ValueError, TypeError):
        return "unknown"


def build_filename(collab, obs_type, slug, energies, collisions):
    """Build a multi-doc YAML filename for a group of records.

    energies  : set of energy_gev values
    collisions: set of collision shorthand strings
    """
    coll_part = ""
    if collisions:
        sorted_coll = "_".join(sorted(collisions))
        coll_part = f"_{sorted_coll}"

    if len(energies) == 1:
        energy = list(energies)[0]
        energy_suffix = f"_{format_energy(energy)}"
    else:
        energy_suffix = "_multigev"
    return f"{collab}_{obs_type}_{slug}{coll_part}{energy_suffix}.yaml"


def slug_from_path(path):
    """Extract a descriptive slug from a candidate file path."""
    stem = path.stem
    parts = stem.split("_")
    return "_".join(parts[:5]) if parts else "unknown"


# ── Validation ───────────────────────────────────────────────────────────

def validate_record(record, resolved):
    """Validate a record dict against a resolved schema.

    Returns list of error strings (empty = valid).
    """
    errors = []
    for name, defn in resolved["top_level_fields"].items():
        if defn.get("_required"):
            if name not in record:
                errors.append(f"'{name}': missing required field")
    flags = resolved["sections"].pop("_section_flags", {})
    for section_name, fields in resolved["sections"].items():
        section_data = record.get(section_name)
        if section_data is None:
            if flags.get(section_name, {}).get("optional"):
                continue
            section_required = any(f.get("required", False) for f in fields.values())
            if section_required:
                errors.append(f"'{section_name}': missing required section")
            continue
        if not isinstance(section_data, dict):
            errors.append(f"'{section_name}': expected object (dict)")
            continue
        for field_name, defn in fields.items():
            sub_path = f"{section_name}.{field_name}"
            if defn.get("required"):
                if field_name not in section_data:
                    errors.append(f"'{sub_path}': missing required field")
    for field_name, defn in resolved["conditional_fields"].items():
        required_if = defn.get("required_if")
        if required_if:
            cond_field = required_if.get("field")
            cond_value = required_if.get("value")
            actual_value = record.get(cond_field)
            if actual_value == cond_value and field_name not in record:
                errors.append(f"'{field_name}': required when '{cond_field}' = '{cond_value}'")
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
                        f"dimensionality='{dimensionality}': missing required qualifier "
                        f"'{qual_name}' ({qual_desc})"
                    )
    return errors


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Group valid candidates into multi-doc YAML files in data/"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without writing")
    parser.add_argument("--regenerate", action="store_true",
                        help="Clear data/ before writing grouped files")
    args = parser.parse_args()

    index = load_record_index()
    print(f"Loaded record index: {len(index)} papers")

    if args.regenerate:
        existing = list(DATA_DIR.rglob("*.yaml"))
        for f in existing:
            f.unlink()
        print(f"Removed {len(existing)} existing files from data/")

    # Gather all candidate YAML files
    candidates = sorted(CANDIDATE_DIR.rglob("*.yaml"))
    print(f"Found {len(candidates)} candidate files")

    # Group valid records by (inspire_id, observable_type)
    groups = defaultdict(list)  # (inspire_id, obs_type) -> [record_dict, ...]
    skipped_invalid = 0
    skipped_missing_schema = 0
    error_samples = []

    for src in candidates:
        record = _load_yaml(src)
        if not isinstance(record, dict):
            skipped_invalid += 1
            continue

        obs_type = record.get("observable_type")
        inspire_id = record.get("inspire_id")
        if not obs_type or not inspire_id:
            skipped_invalid += 1
            continue

        schema_path = REPO_DIR / "observables" / f"{obs_type}.yaml"
        if not schema_path.exists():
            skipped_missing_schema += 1
            continue

        schema = _load_yaml(schema_path)
        resolved = resolve_schema(schema)
        errs = validate_record(record, resolved)
        if errs:
            skipped_invalid += 1
            if len(error_samples) < 5:
                error_samples.append(f"  {src.relative_to(REPO_DIR)}: {errs[0]}")
            continue

        key = (str(inspire_id), obs_type)
        groups[key].append(record)

    print(f"  Valid records grouped: {sum(len(v) for v in groups.values())}")
    print(f"  Groups: {len(groups)}")
    print(f"  Skipped (invalid)       : {skipped_invalid}")
    print(f"  Skipped (no schema)     : {skipped_missing_schema}")
    if error_samples:
        for e in error_samples:
            print(e)

    # Build (filename, key, records) list for all groups, resolve collisions
    file_entries = []  # list of (filename, inspire_id, obs_type, records)
    for (inspire_id, obs_type), records in sorted(groups.items()):
        records.sort(key=lambda r: r.get("hepdata_file", ""))
        meta = index.get(inspire_id, {})
        title = meta.get("title", "")
        collab_list = meta.get("collaborations", ["na"])
        collab = collab_list[0].lower().replace(" ", "_") if collab_list else "na"

        slug = generate_slug(title)
        if slug == "unknown":
            slug = slug_from_path(Path(records[0].get("hepdata_file", "unknown")))

        energies = set()
        collisions = set()
        for r in records:
            cs = r.get("collision_system", {})
            if isinstance(cs, dict):
                energies.add(cs.get("energy_gev", 0))
                coll = collision_shorthand(cs.get("reaction", ""))
                if coll:
                    collisions.add(coll)

        filename = build_filename(collab, obs_type, slug, energies, collisions)
        file_entries.append((filename, inspire_id, obs_type, records))

    # Resolve filename collisions by appending inspire_id
    seen = defaultdict(list)
    for i, (fname, iid, obs, _) in enumerate(file_entries):
        seen[fname].append(i)

    resolved_count = 0
    for fname, indices in seen.items():
        if len(indices) > 1:
            for idx in indices:
                orig_fname, iid, obs, _ = file_entries[idx]
                stem = orig_fname.replace(".yaml", "")
                new_fname = f"{stem}_{iid}.yaml"
                file_entries[idx] = (new_fname, iid, obs, file_entries[idx][3])
                resolved_count += 1

    if resolved_count:
        print(f"  Resolved {resolved_count} filename collision(s) by appending inspire_id")

    # Write grouped multi-doc files
    written = 0
    written_files = 0

    for filename, inspire_id, obs_type, records in file_entries:
        dest_dir = DATA_DIR / obs_type
        dest_path = dest_dir / filename

        if args.dry_run:
            print(f"  would write: {dest_path.relative_to(REPO_DIR)} ({len(records)} docs)")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Write multi-doc YAML (explicit --- before every doc for consistency)
        with open(dest_path, "w") as f:
            for record in records:
                f.write("---\n")
                yaml.dump(record, f, default_flow_style=False,
                          sort_keys=False, width=120, allow_unicode=True)

        written += len(records)
        written_files += 1

    if not args.dry_run:
        print(f"\nWrote {written} records across {written_files} files to data/")
    else:
        print(f"\n(dry-run — no files written)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

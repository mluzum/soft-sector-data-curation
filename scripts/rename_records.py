#!/usr/bin/env python3
"""
Rename all records in data/ to informative filenames.

New format:  {collab}_{obs_code}_{species}_{energy}_{cent}.yaml

When collisions occur, appends _{hepdata_stem} to disambiguate.
If still colliding, appends a numeric suffix _N.

Existing hand-curated names are preserved.

Collaboration is looked up from hepdata_raw/record_index.yaml by inspire_id.

Usage:
    python scripts/rename_records.py [--dry-run]
"""

import sys
from pathlib import Path
from collections import defaultdict

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
RECORD_INDEX_PATH = REPO_DIR / "hepdata_raw" / "record_index.yaml"

# Hand-curated filenames to preserve as-is
PRESERVE = {
    "data/flow_vn/alice_v2_4pc_integrated.yaml",
    "data/flow_vn/atlas_v2_2pc_dpt_0_2pct.yaml",
    "data/particle_spectra/phenix_pion_200gev_0_10.yaml",
    "data/two_particle_correlations/cms_longrange_pPb_5020.yaml",
    "data/flow_harmonic_correlations/alice_sc_5_2_pbpb_5020.yaml",
    "data/mean_pT_fluctuations/alice_sigma_pt_5020.yaml",
}

# Load record index for collaboration lookup
_RECORD_INDEX = {}
if RECORD_INDEX_PATH.exists():
    import yaml
    with open(RECORD_INDEX_PATH) as f:
        _RECORD_INDEX = yaml.safe_load(f) or {}


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _collab_code(rec):
    iid = rec.get("inspire_id", "")
    if iid in _RECORD_INDEX:
        collabs = _RECORD_INDEX[iid].get("collaborations", [])
        if collabs:
            return collabs[0].lower().replace(" ", "_")
    return f"ins{iid}"


def _slugify(vt, max_len=20):
    slug = vt.lower()
    # Strip shell-unfriendly and LaTeX characters
    for ch in "{}()$\\^~`'\"!@#%&*+=|<>?[]":
        slug = slug.replace(ch, "")
    slug = slug.replace(" ", "_").replace("-", "_").replace("/", "_")
    slug = slug.replace(":", "_").replace(".", "_")
    slug = slug.strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug[:max_len].rstrip("_")
    return slug


def _short_observable(rec):
    ot = rec.get("observable_type", "")
    vt = rec.get("value_type", "")
    method = rec.get("method", "")
    harmonic = rec.get("harmonic", "")
    dim = rec.get("dimensionality", "")
    dim_code = {"differential_pT": "_dpt", "differential_eta": "_deta", "integrated": ""}.get(dim, "")

    if method and harmonic:
        vt_slug = _slugify(vt, max_len=10)
        base = f"v{harmonic}_{method}{dim_code}"
        if vt_slug:
            return f"{base}_{vt_slug}"
        return base

    vt_slug = _slugify(vt, max_len=20)
    if ot and vt_slug:
        return f"{ot}_{vt_slug}"
    if ot:
        return ot
    return "unknown"


def _species_code(rec):
    pid = rec.get("particle_id", {})
    species = pid.get("species", "charged")
    mapping = {
        "charged": "h",
        "p": "p",
        "pbar": "pbar",
        "pi+": "pip", "pi-": "pim", "pi0": "pi0",
        "K+": "kp", "K-": "km", "K0_S": "k0s",
        "Lambda": "lam", "Sigma": "sig",
        "D0": "d0",
        "J/psi": "jpsi",
        "e-": "e", "mu-": "mu",
        "photon": "gamma",
    }
    return mapping.get(species, species.lower())


def _energy_code(rec):
    eg = rec.get("collision_system", {}).get("energy_gev", 0)
    try:
        e = int(float(eg))
    except (ValueError, TypeError):
        e = 0
    if e == 0:
        return "unknown"
    return f"{e}gev"


def _cent_code(rec):
    cent = rec.get("centrality", {})
    classes = cent.get("classes", [])
    if not classes:
        return "mb"
    if len(classes) == 1:
        return classes[0].replace("%", "").replace("-", "_").replace(" ", "")
    first = classes[0].replace("%", "").split("-")[0].strip()
    last = classes[-1].replace("%", "").split("-")[1].strip()
    return f"{first}_{last}"


def _hepdata_stem(rec):
    hf = rec.get("hepdata_file", "")
    stem = hf.replace(".yaml", "").replace(".yml", "")
    # Strip shell-unfriendly characters
    for ch in "{}()$\\^~`'\"!@#%&*+=|<>?[]":
        stem = stem.replace(ch, "")
    stem = stem.replace("/", "_").replace(" ", "_")
    stem = stem.replace("(", "_").replace(")", "_")
    stem = stem.replace("{", "_").replace("}", "_")
    stem = stem.strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    if not stem:
        iid = rec.get("inspire_id", "0")
        return f"ins{iid}"
    return stem


def build_informative_name(rec):
    collab = _collab_code(rec)
    obs = _short_observable(rec)
    species = _species_code(rec)
    energy = _energy_code(rec)
    cent = _cent_code(rec)
    return f"{collab}_{obs}_{species}_{energy}_{cent}"


def main():
    dry_run = "--dry-run" in sys.argv
    import yaml

    # Collect all records
    records = []  # (path, subdir_name, base_name, hepdata_stem)
    for subdir in sorted(DATA_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        for f in sorted(subdir.iterdir()):
            if f.suffix not in (".yaml", ".yml"):
                continue
            rel = str(f.relative_to(REPO_DIR))
            if rel in PRESERVE:
                continue
            record = _load_yaml(f)
            if not isinstance(record, dict):
                continue
            base = build_informative_name(record)
            hf_stem = _hepdata_stem(record)
            records.append((f, subdir.name, base, hf_stem, record))

    # Group by subdir
    by_subdir = defaultdict(list)
    for item in records:
        by_subdir[item[1]].append(item)

    # Detect within each subdir which base names collide
    # and assign final names accordingly
    renamed = 0
    hf_conflicts = 0
    numeric_conflicts = 0

    for subdir_name in sorted(by_subdir.keys()):
        items = by_subdir[subdir_name]
        dest_dir = DATA_DIR / subdir_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Group by base name
        base_groups = defaultdict(list)
        for f, sd, base, hf_stem, record in items:
            base_groups[base].append((f, hf_stem, record))

        # Build final name for each record
        name_assignments = []  # (f, final_name)
        for base, group in sorted(base_groups.items()):
            if len(group) == 1:
                f, hf_stem, record = group[0]
                name_assignments.append((f, f"{base}.yaml"))
            else:
                # Disambiguate with hepdata_file stem
                hf_groups = defaultdict(list)
                for f, hf_stem, record in group:
                    hf_groups[hf_stem].append((f, record))

                for hf_stem, subgroup in sorted(hf_groups.items()):
                    if len(subgroup) == 1:
                        f, record = subgroup[0]
                        name_assignments.append((f, f"{base}_{hf_stem}.yaml"))
                        hf_conflicts += 1
                    else:
                        for idx, (f, record) in enumerate(subgroup, start=1):
                            suffix = "" if idx == 1 else f"_{idx}"
                            name_assignments.append((f, f"{base}_{hf_stem}{suffix}.yaml"))
                            numeric_conflicts += 1

        # Resolve any remaining collisions with existing files
        used = set()
        for ef in dest_dir.iterdir():
            if ef.suffix in (".yaml", ".yml"):
                used.add(ef.name)

        for f, final_name in name_assignments:
            dest = dest_dir / final_name
            if dest.exists() and dest.resolve() != f.resolve():
                stem = final_name.replace(".yaml", "")
                suffix = 2
                while (dest_dir / f"{stem}_{suffix}.yaml").exists():
                    suffix += 1
                final_name = f"{stem}_{suffix}.yaml"
                dest = dest_dir / final_name
                numeric_conflicts += 1
            else:
                # Also check if name already taken by another record being renamed
                while final_name in used and f.name != final_name:
                    stem = final_name.replace(".yaml", "")
                    final_name = f"{stem}_2.yaml"

            used.add(final_name)

            if dry_run:
                print(f"  {f.relative_to(REPO_DIR)} → {subdir_name}/{final_name}")
            else:
                f.rename(dest)
            renamed += 1

    print(f"\nRenamed: {renamed}")
    print(f"Collisions resolved by hepdata_file: {hf_conflicts}")
    print(f"Collisions resolved by numeric suffix: {numeric_conflicts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

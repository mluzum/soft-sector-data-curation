#!/usr/bin/env python3
"""
Classify HEPData tables into observable types and produce candidate curated records.

Scans every data YAML file under hepdata_raw/records/, extracts structural
features, matches against heuristic rules for each observable schema, and writes
candidate records to data_candidates/.  Run before manual curation.

Usage:
    python scripts/classify_tables.py                          # classify all tables
    python scripts/classify_tables.py --dry-run                 # preview only
    python scripts/classify_tables.py --limit 50                # first N tables
    python scripts/classify_tables.py --stats-only              # show category counts
"""

import re
import sys
import os
import yaml
import string
from pathlib import Path
from collections import Counter, defaultdict

REPO_DIR = Path(__file__).resolve().parent.parent
HEPDATA_DIR = REPO_DIR / "hepdata_raw" / "records"
INDEX_PATH = REPO_DIR / "hepdata_raw" / "record_index.yaml"
OUTPUT_DIR = REPO_DIR / "data_candidates"

# Cache: inspire_id → submission.yaml parsed docs
_SUBMISSION_CACHE: dict[str, list | None] = {}

# Default centrality estimators per collaboration.
# Used to populate the centrality block in candidate records so that
# downstream simulation code can reproduce event selection.  Each entry
# maps to the observables/centrality.yaml schema fields.
# Unknown collaborations get {classes} only — flagging for manual fill.
CENTRALITY_ESTIMATORS: dict[str, dict] = {
    "ALICE": {
        "estimator": {"name": "V0M", "observable": "amplitude", "detector": "V0"},
        "eta": [{"min": -3.7, "max": -1.7}, {"min": 2.8, "max": 5.1}],
    },
    "ATLAS": {
        "estimator": {"name": "FCal", "observable": "ET", "detector": "FCal"},
        "eta": [{"min": -4.9, "max": -3.1}, {"min": 3.1, "max": 4.9}],
    },
    "CMS": {
        "estimator": {"name": "HF", "observable": "ET", "detector": "HF"},
        "eta": [{"min": -5.2, "max": -2.9}, {"min": 2.9, "max": 5.2}],
    },
    "STAR": {
        "estimator": {"name": "RefMult", "observable": "Nch", "detector": "TPC"},
        "eta": [{"min": -0.5, "max": 0.5}],
    },
    "PHENIX": {
        "estimator": {"name": "BBC", "observable": "amplitude", "detector": "BBC"},
        "eta": [{"min": -3.9, "max": -3.0}, {"min": 3.0, "max": 3.9}],
    },
    "LHCb": {
        "estimator": {"name": "VELO", "observable": "Nch", "detector": "VELO"},
        "eta": [{"min": -2.5, "max": 2.5}],
    },
    "PHOBOS": {
        "estimator": {"name": "Octagon", "observable": "Nch", "detector": "Octagon"},
        "eta": [{"min": -3.2, "max": 3.2}],
    },
    "BRAHMS": {
        "estimator": {"name": "TMA", "observable": "Nch", "detector": "TMA"},
        "eta": [{"min": -2.3, "max": 2.3}],
    },
}

# Default kinematic ranges per collaboration, used when qualifiers / indep
# variables don't provide explicit values.  Based on each experiment's
# standard tracking/trigger acceptance for minimum-bias charged particles.
# Only the acceptance ranges are listed; bin edges are not inferred here.
KINEMATIC_RANGE_DEFAULTS: dict[str, dict] = {
    "ALICE": {
        "eta": {"min": -0.8, "max": 0.8},
        "pT": {"min": 0.15, "max": 50},
    },
    "ATLAS": {
        "eta": {"min": -2.5, "max": 2.5},
        "pT": {"min": 0.5, "max": 20},
    },
    "CMS": {
        "eta": {"min": -2.4, "max": 2.4},
        "pT": {"min": 0.3, "max": 20},
    },
    "STAR": {
        "eta": {"min": -1.0, "max": 1.0},
        "pT": {"min": 0.15, "max": 10},
    },
    "PHENIX": {
        "eta": {"min": -0.35, "max": 0.35},
        "pT": {"min": 0.2, "max": 20},
    },
    "LHCb": {
        "eta": {"min": 2.0, "max": 5.0},
        "pT": {"min": 0.2, "max": 20},
    },
    "PHOBOS": {
        "eta": {"min": -3.2, "max": 3.2},
        "pT": {"min": 0.1, "max": 10},
    },
    "BRAHMS": {
        "eta": {"min": -0.1, "max": 3.3},
        "pT": {"min": 0.1, "max": 10},
    },
}


def _get_submission_docs(inspire_id, version):
    """Load and cache submission.yaml for a record."""
    if inspire_id in _SUBMISSION_CACHE:
        return _SUBMISSION_CACHE[inspire_id]
    sub_path = get_hepdata_file(inspire_id, version, "submission.yaml")
    if sub_path:
        try:
            with open(sub_path) as sf:
                docs = list(yaml.safe_load_all(sf))
            _SUBMISSION_CACHE[inspire_id] = docs
            return docs
        except Exception:
            pass
    _SUBMISSION_CACHE[inspire_id] = None
    return None


# ── Load record index ──────────────────────────────────────────────────

def load_record_index():
    with open(INDEX_PATH) as f:
        return yaml.safe_load(f)


# ── HEPData table reading ──────────────────────────────────────────────

def get_hepdata_file(inspire_id, version, filename):
    """Resolve the full path to a HEPData YAML file."""
    base = HEPDATA_DIR / f"ins{inspire_id}"
    if not base.exists():
        return None
    yaml_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.endswith("-yaml"))
    for d in yaml_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


def read_table(inspire_id, version, filename):
    """Load and parse a HEPData YAML table file."""
    path = get_hepdata_file(inspire_id, version, filename)
    if path is None:
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


def get_table_file_list(inspire_id, version):
    """List all data YAML files (excluding submission.yaml) for a record."""
    base = HEPDATA_DIR / f"ins{inspire_id}"
    if not base.exists():
        return []
    yaml_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.endswith("-yaml"))
    if not yaml_dirs:
        # Old-style: flat directory, YAML in the record dir itself
        files = sorted(f.name for f in base.iterdir()
                       if f.suffix in (".yaml", ".yml") and f.name != "metadata.json")
        return [('1', f) for f in files]
    version_dir = yaml_dirs[0] if version == "1" else None
    for d in yaml_dirs:
        # Match version from directory name: HEPData-ins{id}-v{version}-yaml
        m = re.search(r'-v(\d+)-yaml', d.name)
        if m and m.group(1) == str(version):
            version_dir = d
            break
    if version_dir is None:
        version_dir = yaml_dirs[0]
    files = sorted(f.name for f in version_dir.iterdir()
                   if f.suffix in (".yaml", ".yml") and f.name != "submission.yaml")
    return [(version_dir.name, f) for f in files]


# ── Feature extraction ─────────────────────────────────────────────────

def extract_features(inspire_id, record_meta, filename):
    """Extract structural features from a HEPData table.

    Returns dict with keys:
      - indep_names: set of independent variable header names (lowered, stripped)
      - dep_names: set of dependent variable header names
      - dep_header_raw: first dependent variable header (original)
      - qualifier_map: dict of qualifier name → value
      - n_indep: number of independent variables
      - n_dep_vars: number of dependent variable groups
      - n_dep_values: number of data values in first dep group
      - indep_is_binned: whether independent vars have low/high (binned) or value (point)
      - record_meta: the full record index entry
      - filename: the YAML filename
      - table_labels: keywords from submission.yaml if available
    """
    version = record_meta.get("version", "1")
    data = read_table(inspire_id, version, filename)
    if data is None:
        return None

    indep = data.get("independent_variables", [])
    dep = data.get("dependent_variables", [])

    indep_names = set()
    indep_is_binned = False
    indep_is_value = False
    indep_bin_edges: dict[str, list[float]] = {}
    for iv in indep:
        h = iv.get("header", {})
        name = h.get("name", "").lower().strip().strip("$")
        indep_names.add(name)
        vals = iv.get("values", [])
        if vals:
            if "low" in vals[0] or "high" in vals[0]:
                indep_is_binned = True
            if "value" in vals[0]:
                indep_is_value = True
        # Collect bin edges for the first independent variable
        if vals and name and name not in indep_bin_edges:
            # Low/high bins: collect all boundaries
            if "low" in vals[0] and "high" in vals[0]:
                edges = set()
                for v in vals:
                    if "low" in v:
                        edges.add(float(v["low"]))
                    if "high" in v:
                        edges.add(float(v["high"]))
                if edges:
                    indep_bin_edges[name] = sorted(edges)
            elif "value" in vals[0] and len(vals) > 1:
                # Value-based: infer bin edges from midpoint values.
                # For N midpoints, edges are at:
                #   edge[0]   = m_0 - (m_1 - m_0) / 2
                #   edge[i]   = (m_{i-1} + m_i) / 2   for i=1..N-1
                #   edge[N]   = m_{N-1} + (m_{N-1} - m_{N-2}) / 2
                try:
                    values = [float(v["value"]) for v in vals if "value" in v]
                except (ValueError, TypeError):
                    values = []
                if len(values) >= 2:
                    edges = [values[0] - (values[1] - values[0]) / 2]
                    for i in range(len(values) - 1):
                        edges.append((values[i] + values[i+1]) / 2)
                    edges.append(values[-1] + (values[-1] - values[-2]) / 2)
                    indep_bin_edges[name] = edges

    dep_names = set()
    dep_header_raw = ""
    qualifier_map = {}
    n_dep_vars = len(dep)
    n_dep_values = 0
    error_labels = set()
    for dv in dep:
        h = dv.get("header", {})
        name = h.get("name", "").strip().strip("$")
        if not dep_header_raw:
            dep_header_raw = name
        dep_names.add(name.lower().strip())
        quals = dv.get("qualifiers", [])
        for q in quals:
            qname = q.get("name", "").strip().strip("$").lower()
            qval = str(q.get("value", ""))
            if qname:
                qualifier_map[qname] = qval
        vals = dv.get("values", [])
        n_dep_values = max(n_dep_values, len(vals))
        if vals and "errors" in vals[0]:
            for e in vals[0]["errors"]:
                el = e.get("label", "").lower().strip()
                if el:
                    error_labels.add(el)

    # Try to get table label from submission.yaml
    table_label = ""
    docs = _get_submission_docs(inspire_id, version)
    if docs:
        for doc in docs[1:]:
            if isinstance(doc, dict) and doc.get("data_file") == filename:
                table_label = doc.get("description", "")
                break

    # Extract centrality from qualifiers or record meta
    centrality_info = qualifier_map.get("centrality", "")

    return {
        "indep_names": indep_names,
        "indep_is_binned": indep_is_binned,
        "indep_is_value": indep_is_value,
        "indep_bin_edges": indep_bin_edges,
        "dep_names": dep_names,
        "dep_header_raw": dep_header_raw,
        "qualifier_map": qualifier_map,
        "n_dep_vars": n_dep_vars,
        "n_dep_values": n_dep_values,
        "error_labels": error_labels,
        "record_meta": record_meta,
        "filename": filename,
        "table_label": table_label,
        "centrality_info": centrality_info,
        "inspire_id": inspire_id,
    }


# ── Helper: parse particle species from qualifiers ─────────────────────

def guess_species(features):
    """Try to extract particle species from dep_header_raw, qualifiers,
    table label, filename, and record title."""
    s = features["dep_header_raw"].lower()
    quals = " ".join(features["qualifier_map"].values()).lower()
    table_label = features.get("table_label", "").lower()
    filename = features.get("filename", "").lower()
    rec_title = features.get("record_meta", {}).get("title", "").lower()
    combined = " ".join([s, quals, table_label, filename, rec_title])

    # Check for "charged" indicators first (before "p" which would match p_T)
    if "charged" in combined or "h±" in combined or "h+" in combined:
        return "charged"

    # Multi-character tokens checked via substring (safe)
    for token, species in [
        ("proton", "p"), ("pbar", "pbar"), ("antiproton", "pbar"),
        ("pi+", "pi+"), ("pi-", "pi-"), ("pi0", "pi0"), ("pion", "charged"),
        ("k+", "K+"), ("k-", "K-"), ("k0", "K0_S"), ("kaon", "charged"),
        ("lambda", "Lambda"), ("sigma", "Sigma"),
        ("d0", "D0"), ("j/psi", "J/psi"),
        ("electron", "e-"), ("muon", "mu-"), ("photon", "photon"),
    ]:
        if token in combined:
            return species

    # Single-char "p" — use word boundaries to avoid matching p_T, Pb, EP, etc.
    if re.search(r"\bp\b", combined):
        return "p"

    return "charged"  # default


def _is_aa_reaction(reaction):
    """Check if a reaction string indicates a heavy-ion (AA) collision."""
    r = reaction.lower().replace(" ", "").replace("-", "").replace(">", "").replace("→", "")
    r = r.replace("+", "").replace("—", "").replace("–", "")
    aa_ions = ["pbpb", "auau", "xexe", "cucu", "uu",
               "agag", "sisi", "oo", "krkr", "ruru", "zrzr", "cc", "nene"]
    for ion in aa_ions:
        if ion in r:
            return True
    # Also check for multi-word ion names like "PB PB", "AU AU", "C C"
    words = reaction.lower().replace("–", "-").replace("—", "-").replace("+", "").split()
    hi = sum(1 for w in words if w in ("pb", "au", "xe", "cu", "u",
                                        "ag", "si", "o", "kr", "ru", "zr", "c", "ne"))
    if hi >= 2:
        return True
    # Check for mixed AA like "C PB"
    if 'c' in words and any(w for w in words if w in ("pb", "au", "xe", "cu")):
        return True
    return False


def guess_centrality(features):
    """Parse centrality class(es) from qualifier, dep header, table_label,
    independent variables, or record title.

    Returns a list of centrality bins (e.g. ["0-5%", "5-10%"]) or empty list.
    """

    def _parse_range(val):
        """Try to extract centrality ranges from a string value."""
        # Case-insensitive for "TO" vs "to" (HEPData uses both)
        m = re.findall(r'(\d+[\s.]*\d*)\s*[-–to]+\s*(\d+[\s.]*\d*)\s*%', val, re.IGNORECASE)
        if m:
            return [f"{a.strip()}-{b.strip()}%" for a, b in m]
        m = re.findall(r'(\d+[\s.]*\d*)\s*[-–to]+\s*(\d+[\s.]*\d*)', val, re.IGNORECASE)
        if m:
            vals = [(float(a), float(b)) for a, b in m]
            if all(0 <= a < b <= 100 for a, b in vals):
                return [f"{a}-{b}%" for a, b in vals]
        return None

    # 1. Qualifiers — exact "cent*" or "central*" match
    for qname, qval in features["qualifier_map"].items():
        ql = qname.lower().replace("_", "").replace("-", "").replace(" ", "")
        if "cent" not in ql:
            continue
        result = _parse_range(qval)
        if result:
            return result
        # Try value-list form: "0,5,10,20,40,60" (centrality boundaries)
        m2 = re.findall(r'(\d+)\s*[,\s]+(\d+)', qval)
        if m2:
            values = [int(m2[0][0])]
            for a, b in m2:
                values.append(int(b))
            return [f"{values[i]}-{values[i+1]}%" for i in range(len(values) - 1)]

    # 1b. Fallback: scan all qualifier VALUES for "%" patterns
    # (catches cases where the qualifier name doesn't contain "cent"
    #  but the value encodes centrality, e.g. "0-5%")
    for qname, qval in features["qualifier_map"].items():
        if "%" in qval:
            result = _parse_range(qval)
            if result:
                return result

    # 2. Table label (submission.yaml description)
    table_label = features.get("table_label", "")
    if table_label:
        result = _parse_range(table_label)
        if result:
            return result

    # 3. Dep header raw
    s = features["dep_header_raw"]
    result = _parse_range(s)
    if result:
        return result

    # 4. Independent variable values for centrality columns
    indep_text = " ".join(features["indep_names"]).lower()
    if "central" in indep_text or "class" in indep_text:
        edges = features.get("indep_bin_edges", {})
        # Match any indep name containing "central" or "class"
        bins = []
        for k, v in edges.items():
            kl = k.lower()
            if "central" in kl or "class" in kl:
                bins = v
                break
        if bins:
            return [f"{max(0, bins[i]):g}-{bins[i+1]:g}%"
                    for i in range(len(bins) - 1) if bins[i+1] > bins[i]]

    # 4b. Check any indep var with bin edges in [0, 100] range — likely centrality
    # This catches indep vars named "class", "mult", "event class", etc.
    if not indep_text or not any(kw in indep_text for kw in ("pt", "p_t", "eta", "rapid")):
        edges = features.get("indep_bin_edges", {})
        for k, vals in edges.items():
            if not vals or len(vals) < 2:
                continue
            if vals[0] >= 0 and vals[-1] <= 100 and all(v >= 0 and v <= 100 for v in vals):
                return [f"{max(0, vals[i]):g}-{vals[i+1]:g}%"
                        for i in range(len(vals) - 1) if vals[i+1] > vals[i]]

    # 5. Record title for AA reactions
    title = features.get("record_meta", {}).get("title", "")
    reaction = features.get("qualifier_map", {}).get("re", "")
    if not reaction:
        reaction = features.get("record_meta", {}).get("reaction", "")
    if _is_aa_reaction(reaction):
        result = _parse_range(title)
        if result:
            return result

    return []


# Canonical experiment name mapping (used for metadata.json collaborations)
CANONICAL_EXPERIMENT_NAMES = {
    "A LARGE ION COLLIDER EXPERIMENT": "ALICE",
    "BRAHMS": "BRAHMS",
    "AACHEN-BERLIN-BONN-CERN-CRACOW-HEIDELBERG-WARSAW": "NA49",
    "AACHEN-BERLIN-BONN-CERN-HEIDELBERG-LONDON-VIENNA": "NA49",
    "AACHEN-BERLIN-BONN-CERN-CRACOW-HEIDELBERG": "NA49",
    "AACHEN-HAMBURG-HEIDELBERG-MUNICH": "NA49",
    "AACHEN-BONN-CERN-MUNICH-OXFORD": "NA49",
    "AMSTERDAM-CERN-NIJMEGEN-OXFORD": "NA49",
    "CERN-MUNICH": "NA49",
    "ACCMOR": "NA22",
    "BERLIN-BUDAPEST-DUBNA-PRAGUE-SERPUKHOV-SOFIA": "NA61/SHINE",
    "NA35": "NA35",
    "E-802": "E802",
    "BESIII": "BESIII",
    "BABAR": "BaBar",
}


def resolve_experiment(record_meta) -> str | None:
    """Resolve experiment/collaboration name from record metadata.

    Reads the collaborations list from the HEPData metadata.json entry
    and returns the canonical experiment name, or None if unknown.
    """
    collabs = record_meta.get("collaborations", [])
    if not collabs:
        return None
    raw = collabs[0].strip()
    return CANONICAL_EXPERIMENT_NAMES.get(raw.upper(), raw)


def infer_centrality_estimator(record_meta) -> dict | None:
    """Look up default centrality estimator info for a record's collaboration.

    Returns a dict with keys matching the 'centrality' section of
    observables/centrality.yaml (estimator, eta, etc.), or None if
    the collaboration is unknown.
    """
    collabs = record_meta.get("collaborations", [])
    for c in collabs:
        if c in CENTRALITY_ESTIMATORS:
            est = CENTRALITY_ESTIMATORS[c]
            # Return a shallow copy so callers can add/override fields
            return dict(est)
    return None


def infer_kinematic_range_defaults(record_meta) -> dict:
    """Look up default kinematic range values for a collaboration.

    Returns a dict of kinematic_range sub-keys (e.g. {"eta": {"min": -2.5, "max": 2.5}})
    that can be used as fallback when qualifiers / indep vars provide no
    explicit values.
    """
    collabs = record_meta.get("collaborations", [])
    for c in collabs:
        if c in KINEMATIC_RANGE_DEFAULTS:
            return dict(KINEMATIC_RANGE_DEFAULTS[c])
    return {}


def guess_energy_and_reaction(features):
    """Extract energy and reaction from qualifiers, with fallback inference."""
    qm = features["qualifier_map"]
    rec = features["record_meta"]
    energy = qm.get("sqrt(s)/nucleon", qm.get("sqrt(s)", rec.get("energy_gev", "unknown")))
    reaction = qm.get("re", rec.get("reaction", ""))
    try:
        energy = float(energy)
    except (ValueError, TypeError):
        energy = rec.get("energy_gev", 0)
        try:
            energy = float(energy)
        except (ValueError, TypeError):
            energy = _infer_energy_from_reaction(reaction)
    return energy, _sanitize_reaction(reaction)


_ENERGY_LOOKUP = {
    "pb pb": 2760, "pbpb": 2760, "p b p b": 2760,
    "au au": 200, "auau": 200, "a u a u": 200,
    "u u": 193, "uu": 193,
    "xe xe": 5020, "xexe": 5020, "x e x e": 5020,
    "p p": 200, "pp": 200,
    "p pb": 5020, "ppb": 5020, "p p b": 5020,
    "d au": 200, "dau": 200,
    "cu cu": 200, "cucu": 200,
}


def _infer_energy_from_reaction(reaction):
    """Infer collision energy when metadata is missing."""
    r = reaction.lower().replace(" ", "").replace("-", "").replace(">", "").replace("→", "")
    for key, energy in _ENERGY_LOOKUP.items():
        k = key.replace(" ", "")
        if k in r:
            return energy
    return 0.0


def _sanitize_reaction(reaction):
    """Clean up LaTeX artifacts and HEPData formatting in reaction strings."""
    if not reaction:
        return reaction
    r = reaction
    # Strip LaTeX math mode
    r = re.sub(r'\$([^$]+)\$', r'\1', r)
    # Fix NB NB -> PB PB (LaTeX \mathrm{Nb} artifact)
    r = re.sub(r'\bNB\s+NB\b', 'PB PB', r, flags=re.IGNORECASE)
    # Fix CHARGED++ -> CHARGED (artifact from LaTeX exponents)
    r = re.sub(r'CHARGED\s*\+{1,}', 'CHARGED', r, flags=re.IGNORECASE)
    # Fix < CHARGED X > -> CHARGED (HEPData proton decay notation)
    r = re.sub(r'<\s*CHARGED\s+X\s*>', 'CHARGED', r, flags=re.IGNORECASE)
    # Replace underscores with spaces
    r = r.replace('_', ' ')
    # Collapse multiple spaces
    r = re.sub(r'\s+', ' ', r).strip()
    return r


# ── Classification rules ──────────────────────────────────────────────

def _reg(*patterns):
    return re.compile("|".join(patterns), re.IGNORECASE)


class Classification:
    """Result of matching a table against a schema heuristic."""
    def __init__(self, observable_type, confidence, field_values, explanation=""):
        self.observable_type = observable_type
        self.confidence = confidence  # "high", "medium", "low"
        self.field_values = field_values
        self.explanation = explanation

    def __bool__(self):
        return self.confidence is not None


def classify_table(features):
    """Try all classification rules and return the best match.

    Returns a Classification or None.
    """
    candidates = []

    for rule_name, rule_fn in CLASSIFIERS.items():
        try:
            result = rule_fn(features)
            if result and result.confidence:
                candidates.append(result)
        except Exception as e:
            pass

    if not candidates:
        return None

    # Prefer highest confidence, then best explanation match
    rank = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda c: (rank.get(c.confidence, 99), c.observable_type))
    return candidates[0]


# ── Individual classifier functions ──

def _classify_flow_auxiliary(f):
    """Match auxiliary flow tables (resolution, chi) that accompany vn."""
    dh = f["dep_header_raw"].lower()
    if any(kw in dh for kw in ("resolution", "chi", "mean(npart)")):
        rec_title = f["record_meta"]["title"].lower()
        if any(kw in rec_title for kw in ("flow", "anisotropic", "elliptic", "azimuth")):
            fields = {
                "harmonic": 2,
                "method": "event_plane",
                "value_type": dh.strip("$"),
                "dimensionality": "integrated" if "central" in f["indep_names"] else "integrated",
            }
            return Classification("flow_vn", "low", fields, "auxiliary table")
    return None


def _classify_flow_vn(f):
    """Match flow harmonics vn{2}, vn{4}, vn(EP), V2(V=EP), v2{pt}, etc."""
    dh = f["dep_header_raw"].strip("$")
    dh_lower = dh.lower()
    indep_lower = set(n.strip("$").strip().lower() for n in f["indep_names"])

    # Patterns for various flow naming conventions (match on original case):
    #   v2{2}, v4{4}              cumulant method
    #   v2(EP), v2(SP)            standard notation
    #   V2(V=EP), V2(C=EP), V2(C=LYZ)  ATLAS-style: V${n}(TYPE=METHOD)
    #   v_{1}(\%)                 directed flow in percent
    #   dv_{1}/dy                 directed flow slope
    #   v2{pt}                    some ALICE tables
    flow_pats = [
        (r'^[vV](\d+)\{(\d+)\}', 0),             # v2{2}, v4{4}  — cumulant
        (r'^[vV](\d+)\(([A-Za-z]+)\)', 1),        # v2(EP), v2(SP)
        (r'^[vV](\d+)\([A-Za-z]+=([A-Za-z]+)\)', 2),  # V2(V=EP), V2(C=LYZ)
        (r'^[vV]_?\{?(\d+)\}?\(?\s*\\?%?\s*\)?\s*$', 3),  # v_{1}(\%), v1 (terminal)
        (r'^[dv][vV]_?\{(\d+)\}/[dy][a-z]+', 4),    # dv_{1}/dy
        (r'^V_\{(\d+)\}\s*\((EP|SP|Psi|LYZ|c|V)', 5),  # V_{2}(EP)
        (r'^[vV]\{(\d+)\}_\{(\d+)\}', 0),          # v{2}_{4}  (some ALICE)
        (r'^V\(2PC\)', 7),                           # V(2PC)  (STAR 2PC flow)
        (r'^2PC\.V_\{(\d+)\}', 6),                   # 2PC.V_{2}  (ATLAS)
        (r'^2PC\.V_\{?(\d+)\}?$', 6),                # 2PC.V2  (ATLAS)
    ]

    best = None
    for pat, style in flow_pats:
        m = re.match(pat, dh, re.IGNORECASE)
        if m:
            best = (m, style)
            break

    if best is None and re.match(r'^[dv]v', dh, re.IGNORECASE) and '/dy' in dh_lower:
        hm = re.search(r'v[_{]?(\d+)[_}]?', dh_lower)
        dv_dim = "differential_eta"
        for indep_n in indep_lower:
            if "central" in indep_n or "npart" in indep_n:
                dv_dim = "integrated"
                break
        fields = {
            "harmonic": int(hm.group(1)) if hm else 1,
            "method": "event_plane",
            "value_type": dh.strip("$"),
            "dimensionality": dv_dim,
        }
        return Classification("flow_vn", "high", fields, "directed flow slope")

    if best is None:
        # Fallback: if the record title strongly indicates flow
        rec_title_lower = f["record_meta"]["title"].lower()
        if any(kw in rec_title_lower for kw in [
            "flow", "azimuthal anisotropy", "elliptic", "v2", "v3", "v4",
            "anisotropic flow", "harmonics"
        ]):
            # Check indep names for dimensionality
            fb_dim = "integrated"
            for indep_n in indep_lower:
                if "pt" in indep_n or "p_t" in indep_n:
                    fb_dim = "differential_pT"
                    break
                elif "eta" in indep_n or "rapid" in indep_n:
                    fb_dim = "differential_eta"
                    break
            # Try to extract harmonic from dep header
            fb_harm = 2
            hm = re.search(r'[vV](\d+)', dh)
            if hm:
                fb_harm = int(hm.group(1))
            fields = {
                "harmonic": fb_harm,
                "method": "event_plane",
                "value_type": dh.strip("$"),
                "dimensionality": fb_dim,
            }
            return Classification("flow_vn", "low", fields,
                                  f"flow title fallback")

    if best is None:
        return None

    m, style = best

    # Determine method and cumulant_order from the match style
    method = "event_plane"
    cumulant_order = None
    value_type = dh.strip()

    # Extract harmonic from match; V_n (n=letter) is handled per-style
    harmonic = 2  # default

    if style == 0:  # v2{2}, v2{4}
        harmonic = int(m.group(1))
        cumulant_order = int(m.group(2))
        method = "cumulant"
    elif style == 1:  # v2(EP), v2(SP)
        harmonic = int(m.group(1))
        mk = m.group(2).lower()
        if mk == "sp":
            method = "scalar_product"
        elif mk == "ep":
            method = "event_plane"
    elif style == 2:  # V2(V=EP), V2(C=LYZ)
        try:
            harmonic = int(m.group(1))
        except ValueError:
            harmonic = 2
        sub_method = m.group(2).lower()
        if sub_method == "ep":
            method = "event_plane"
        elif sub_method == "lyz":
            method = "lyz"
        elif sub_method == "c":
            method = "cumulant"
        elif sub_method == "v":
            method = ""
        if "sp" in dh.lower():
            method = "scalar_product"
    elif style == 3:  # v_{1}(\%) terminal pattern (no method group)
        harmonic = int(m.group(1))
        method = "event_plane"
    elif style == 4:  # v2 vs eta, v1 vs y (bare prefix)
        harmonic = int(m.group(1))
        method = "event_plane"
    elif style == 5:  # V_{2}(EP) or V_{2}(Psi_{2}) curly braces
        harmonic = int(m.group(1))
        mk = m.group(2).lower()
        if mk == "ep":
            method = "event_plane"
        elif mk == "sp":
            method = "scalar_product"
        elif mk.startswith("psi"):
            method = "event_plane"
        elif "lyz" in mk:
            method = "lyz"
        else:
            method = "cumulant"
        # Check for scalar_product in value_type
        if "sp" in value_type.lower():
            method = "scalar_product"
    elif style == 6:  # 2PC.V_n  (ATLAS 2-particle cumulant)
        method = "cumulant"
        cumulant_order = 2
        if m.lastindex >= 1:
            try:
                harmonic = int(m.group(1))
            except ValueError:
                harmonic = 2  # fallback for V_n (letter subscript)
    elif style == 7:  # V(2PC)  (STAR 2PC flow)
        method = "cumulant"
        cumulant_order = 2
        harmonic = 2  # default; STAR V(2PC) is v2
    # style 4 (dv/dy) — keep default
    dimension = "integrated"
    if any("pt" in n or "p_t" in n or n in ("p_t", "pt") for n in indep_lower):
        dimension = "differential_pT"
    elif any("eta" in n or "rapid" in n for n in indep_lower):
        dimension = "differential_eta"
    elif any("central" in n or "npart" in n or "npart" in n for n in indep_lower):
        dimension = "integrated"

    fields = {
        "harmonic": harmonic,
        "method": method,
        "value_type": value_type,
        "dimensionality": dimension,
    }
    if cumulant_order:
        fields["cumulant_order"] = cumulant_order

    confidence = "high"
    return Classification("flow_vn", confidence, fields)


def _classify_particle_spectra(f):
    """Match invariant yield spectra vs pT."""
    dh = f["dep_header_raw"].lower()
    indep_lower = set(n.strip("$") for n in f["indep_names"])

    has_pT = any("pt" in n or "p_t" in n or n.strip("$").startswith("p_{t}") for n in indep_lower)
    if not has_pT:
        return None

    is_spectra = any(kw in dh for kw in ["yield", "spectra", "spectrum", "d2(", "dn/dpt",
                                          "d(n)/d", "ed3n", "invariant", "d2(n)", "d2n"])
    is_ratio = any(kw in dh for kw in ["ratio"])
    if not is_spectra or is_ratio:
        return None

    # A spectra file often has many dep_vars (one per centrality bin)
    species = guess_species(f)
    fields = {
        "observable": "dN_dpT_dy",
        "value_type": "invariant_yield",
        "hepdata_file": f["filename"],
        "particle_id": {"species": species},
    }

    conf = "high" if "yield" in dh else "medium"
    return Classification("particle_spectra", conf, fields)


def _classify_particle_ratios(f):
    """Match particle ratios (K/pi, p/pi, etc.)."""
    dh = f["dep_header_raw"].lower()
    # Normalize LaTeX: strip backslashes, braces, carets
    dh_norm = dh.replace("\\", "").replace("{", "").replace("}", "").replace("^", "")
    indep_lower = set(n.strip("$") for n in f["indep_names"])

    has_pT = any("pt" in n or "p_t" in n for n in indep_lower)
    if not has_pT:
        return None

    is_ratio = any(kw in dh or kw in dh_norm for kw in
                   ["ratio", "k/π", "p/π", "k/pi", "p/pi",
                    "η/π0", "eta/pi0", "pbar/pi", "/π-", "/pi-", "/π+", "/pi+"])
    if not is_ratio:
        return None

    # Try to identify numerator/denominator
    s = f["dep_header_raw"].lower()
    ratio_map = {
        "k/π": ("K", "pi"), "k+/π+": ("K+", "pi+"), "k-/π-": ("K-", "pi-"),
        "p/π": ("p", "pi"), "p/pi": ("p", "pi"), "pbar/π": ("pbar", "pi"),
        "p/π": ("p", "pi"),
    }
    num, den = "unknown", "unknown"
    for pat, (n, d) in ratio_map.items():
        if pat in s:
            num, den = n, d
            break
    # Fallback: extract from qualifier
    if num == "unknown":
        for qv in f["qualifier_map"].values():
            ql = qv.lower()
            if "/" in ql:
                parts = ql.split("/")
                if len(parts) == 2:
                    num, den = parts[0].strip(), parts[1].strip()
                    break

    fields = {
        "observable": f"{num}_over_{den}_ratio",
        "value_type": f"{num}_over_{den}",
        "numerator": num,
        "denominator": den,
    }
    return Classification("particle_ratios", "high", fields)


def _classify_particle_yields(f):
    """Match dN/dy, dNch/dη, multiplicity."""
    dh = f["dep_header_raw"].lower()
    # Normalize LaTeX: remove $, \cmd{content} → content, remove standalone \cmd
    dh_norm_yield = dh.replace('$', '')
    dh_norm_yield = re.sub(r'\\[a-z]+\s*\{([^}]*)\}', r'\1', dh_norm_yield)
    dh_norm_yield = re.sub(r'\\[a-z]+(?:\s+|(?=[}\\]|$))', '', dh_norm_yield)
    dh_norm_yield = dh_norm_yield.replace('{', '').replace('}', '').replace('\\', '').replace('^', '').replace('_', '').strip()
    dh_stripped = dh.replace('$', '')
    dh_stripped = re.sub(r'\\[a-z]+\s*\{([^}]*)\}', r'\1', dh_stripped)
    dh_stripped = re.sub(r'\\[a-z]+(?:\s+|(?=[}\\]|$))', '', dh_stripped)
    dh_stripped = dh_stripped.replace('{', '').replace('}', '').replace('\\', '').replace('^', '').replace('_', '').strip()
    indep_names = set(n.strip("$") for n in f["indep_names"])

    is_yield = any(kw in dh or kw in dh_norm_yield or kw in dh_stripped for kw in
                   ["dN/dy", "dN/dη", "dNch", "multiplic",
                    "rapidity density", "pseudorapidity density",
                    "yield", "〈dn", "⟨dn", "d(n)/detarap",
                    "detarap", "yint", "d(n)/d(yrap)", "d(n)/dyrap",
                    "d(n)/deta", "dnch", "dn/deta", "dn/dy"])
    # Also check regex patterns for harder-to-match yield variants
    if not is_yield:
        is_yield = bool(re.search(r'\bd[ny]/d(etarap|yrap|eta|y)\b', dh)) or \
                   bool(re.search(r'\bd[ny]/d(etarap|yrap)', dh_stripped)) or \
                   bool(re.search(r'd[ny]_?[a-z]*/d(etarap|yrap|eta|y)', dh_stripped))
    if not is_yield:
        return None

    # Centrality or multiplicity as independent var (not pT)
    has_pT = any("pt" in n or "p_t" in n for n in indep_names)
    # Yint is an integrated yield plotted vs pT(a) — still a yield
    if "yint" in dh:
        has_pT = False
    if has_pT:
        return None  # this is a spectrum

    # Determine value_type
    vt = "dN_dy"
    if "pseudorapidity" in dh or "dN/dη" in dh or "dNch" in dh:
        vt = "dNch_deta"
    elif "multiplic" in dh:
        vt = "Nch"

    fields = {
        "observable": vt,
        "value_type": vt,
    }
    return Classification("particle_yields", "high", fields)


def _classify_nuclear_modification(f):
    """Match R_AA, R_CP, R_pA, n_eff."""
    dh = f["dep_header_raw"].lower()
    # Normalize curly-brace underscores for matching
    dh_norm = dh.replace("{", "").replace("}", "")
    indep_lower = set(n.strip("$") for n in f["indep_names"])

    is_mod = any(kw in dh or kw in dh_norm for kw in
                 ["r_aa", "r_cp", "r_pa", "r_dau", "r_auau",
                  "nuclear modification", "raa", "rcp",
                  "n_eff", "n_{eff}", "r_ppb", "r_pbp", "rppb"])
    if not is_mod:
        return None

    has_pT = any("pt" in n or "p_t" in n for n in indep_lower)
    vt = "R_AA"
    if "r_cp" in dh or "rcp" in dh or dh_norm.count("r_cp") > 0:
        vt = "R_CP"
    elif "r_pa" in dh or "r_dau" in dh:
        vt = "R_pA"
    elif "r_ppb" in dh or "r_pbp" in dh or "rppb" in dh:
        vt = "R_pPb"

    fields = {
        "observable": vt,
        "value_type": vt,
    }
    return Classification("nuclear_modification", "high", fields)


def _classify_mean_pT(f):
    """Match ⟨pT⟩ measurements."""
    dh = f["dep_header_raw"].lower()
    is_mean_pT = any(kw in dh for kw in ["mean p", "<p_t>", "〈p_t〉", "⟨p_t⟩",
                                         "average p", "truncated mean",
                                         "\\langle p"])
    if not is_mean_pT:
        return None

    # Distinguish from fluctuations
    is_fluct = any(kw in dh for kw in ["fluctuation", "σ", "sigma", "√⟨"])
    if is_fluct:
        return None

    fields = {
        "observable": "mean_pT",
        "value_type": "mean_pT",
    }
    return Classification("mean_pT", "high", fields)


def _classify_mean_pT_fluctuations(f):
    """Match mean pT fluctuations (Σ, σ_⟨pT⟩)."""
    dh = f["dep_header_raw"].lower()
    is_fluct = any(kw in dh for kw in ["fluctuation", "√⟨⟨δ", "√⟨⟨δp", "⟨⟨δp",
                                       "sigma", "σ⟨p", "⟨⟨Δp"])
    has_mean = any(kw in dh for kw in ["mean p", "<p_t>", "⟨p_t⟩", "p_t", "p_t1",
                                       "\\langle p"])
    if not (is_fluct or (has_mean and ("fluct" in dh or "sigma" in dh or "√" in dh))):
        return None

    fields = {
        "observable": "sigma_mean_pT",
        "value_type": "Sigma",
    }
    return Classification("mean_pT_fluctuations", "high", fields)


def _classify_flow_pT_correlations(f):
    """Match ρ_n = correlation(vn², ⟨pT⟩)."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()

    is_corr = "correlation" in dh or any(cw in dh for cw in ["c_k", "ρ_n", "rho"])
    has_flow = any(fw in dh for fw in ["v{", "v2", "v3", "flow"])
    if not (is_corr or "correlation" in rec_title and any(
            fw in dh for fw in ["c_k", "c_{", "v_n", "mean"])):
        return None

    # Specific signal: c_k [GeV²] or ρ_n
    has_pT_link = any(kw in dh for kw in ["mean p", "<p_t>", "⟨p_t⟩", "mean transverse",
                                          "\\langle p"])
    has_ck = "c_k" in dh or "c{k" in dh
    if not (has_ck or has_pT_link):
        return None

    # Try to extract harmonic from header
    m = re.search(r'[vcv](\d+)', dh)
    harmonic = int(m.group(1)) if m else 2

    fields = {
        "observable": f"rho_{harmonic}",
        "value_type": "c_k" if has_ck else "rho_n",
        "harmonic": harmonic,
    }
    return Classification("flow_pT_correlations", "high", fields)


def _classify_flow_harmonic_correlations(f):
    """Match symmetric cumulants, event-plane angle correlations."""
    dh = f["dep_header_raw"].lower()

    # Symmetric cumulants: SC(NSC)(n,m) or similar
    sc_match = re.search(r'(?:ns)?c\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', dh)
    if sc_match:
        h1, h2 = int(sc_match.group(1)), int(sc_match.group(2))
        fields = {
            "observable": f"NSC_{h1}_{h2}",
            "correlation_type": "symmetric_cumulant",
            "harmonics": [h1, h2],
            "value_type": f"NSC({h1},{h2})",
        }
        return Classification("flow_harmonic_correlations", "high", fields)

    # Event-plane angle correlations: ⟨cos[k(Ψₙ − Ψₘ)]⟩
    if re.search(r'\\?cos\s*\[', dh) and re.search(r'[ψΨ]', dh):
        # Extract harmonics from cos expression
        m = re.findall(r'[ψΨ]_?(\d+)', dh)
        if m:
            harm = [int(x) for x in m]
        else:
            harm = [2, 2]
        fields = {
            "observable": f"event_plane_corr_{'_'.join(map(str, harm))}",
            "correlation_type": "event_plane_correlation",
            "harmonics": harm,
            "value_type": f["dep_header_raw"].strip("$"),
        }
        return Classification("flow_harmonic_correlations", "high", fields)

    # ATLAS s2, s3 notation for symmetric cumulants (s2 = SC(2,3))
    if dh in ("s2", "s3"):
        h2 = 3 if dh == "s2" else 2
        h1 = 2 if dh == "s2" else 3
        fields = {
            "observable": f"SC_{h1}_{h2}",
            "correlation_type": "symmetric_cumulant",
            "harmonics": [h1, h2],
            "value_type": f"s{dh[1]}" if dh else "s2",
        }
        return Classification("flow_harmonic_correlations", "high", fields)

    # Also catch titles mentioning symmetry plane correlations
    rec_title = f["record_meta"]["title"].lower()
    if any(kw in rec_title for kw in ["symmetry plane", "symmetric cumulant",
                                       "multi-harmonic", "flow harmonic correlation"]):
        if any(kw in dh for kw in ["c{", "v{", "ρ", "cos"]):
            fields = {
                "observable": "multi_harmonic",
                "correlation_type": "multi_harmonic_correlation",
                "harmonics": [2, 3],
            }
            return Classification("flow_harmonic_correlations", "medium", fields)

    return None


def _classify_two_particle_correlations(f):
    """Match Δη-Δφ correlations and per-trigger yields."""
    dh = f["dep_header_raw"].lower()
    indep_names = set(n.strip("$") for n in f["indep_names"])
    rec_title = f["record_meta"]["title"].lower()

    # Generic "Correlation Value" from STAR — has y_t1, y_t2 indep
    if dh in ("correlation value", "correlation") and any("y_t" in n for n in indep_names):
        return Classification("two_particle_correlations", "medium", {
            "observable": "per_trigger_yield",
            "correlation_type": "delta_phi_projection",
            "value_type": "Correlation Value",
        })

    # Plain "Y" with delta-phi indep → per-trigger yield projection
    if dh.strip("$").strip() == "Y" and any("delta" in n or "phi" in n for n in indep_names):
        return Classification("two_particle_correlations", "medium", {
            "observable": "per_trigger_yield",
            "correlation_type": "delta_phi_projection",
            "value_type": "Y(Δφ)",
        })

    # "V(2PC)" should already be caught before this point (flow_vn classifier
    # runs first), but some STAR tables named just "V" may reach here.
    if re.match(r'^V\b', dh.strip("$")) and "2PC" in f.get("table_label", "").upper():
        return Classification("two_particle_correlations", "low", {
            "observable": "per_trigger_yield",
            "correlation_type": "delta_phi_delta_eta",
            "value_type": dh.strip("$"),
        })

    is_cme = any(kw in dh for kw in ["γ", "delta", "γ_os", "γ_ss", "Δγ",
                                      "charge separ", "same sign", "opposite sign",
                                      "balance function"])
    has_cme_title = "chiral magnetic" in rec_title or "charge-dependent" in rec_title

    if not (is_cme or (has_cme_title and ("γ" in dh or "delta" in dh))):
        return None

    # Determine charge combo from header
    combo = "all"
    if "same sign" in dh or "γ_ss" in dh or "ss" in dh:
        combo = "same_sign"
    elif "opposite sign" in dh or "γ_os" in dh or "os" in dh:
        combo = "opposite_sign"

    fields = {
        "observable": "gamma_os" if combo == "opposite_sign" else "gamma_ss",
        "value_type": "gamma_os" if combo == "opposite_sign" else "gamma_ss",
        "charge_combo": combo,
    }
    return Classification("charge_dependent_correlations", "high", fields)


def _classify_jet_correlations(f):
    """Match jet-hadron and dihadron jet property correlations."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()
    indep_names = set(n.strip("$") for n in f["indep_names"])

    # Fragmentation functions: DN/DZ, RDZ, RDPT with indep var 'z'
    if any(kw in dh for kw in ["dn/dz", "rdz", "dn/dz"]) or (
        dh.strip("$").strip().lower() in ("dn/dz", "rdz", "dz/dx") and "z" in indep_names):
        fields = {
            "observable": "jet_fragmentation",
            "value_type": "fragmentation_function",
            "correlation_type": "dihadron_jet_properties",
        }
        return Classification("jet_correlations", "medium", fields)

    # dpT/pT — jet fragmentation transverse momentum
    if "dpt" in dh and "/pt" in dh:
        fields = {
            "observable": "jet_fragmentation_pT",
            "value_type": "dpt_pt",
            "correlation_type": "dihadron_jet_properties",
        }
        return Classification("jet_correlations", "medium", fields)

    is_jet_corr = any(kw in dh for kw in ["jet", "dihadron", "di-hadron",
                                          "fragmentation", "away-side", "j_t"])
    has_jet_title = "jet" in rec_title or "fragmentation" in rec_title

    if not (is_jet_corr and has_jet_title):
        return None

    if any("Δ" in n or "delta" in n or "dphi" in n for n in indep_names):
        # This may be a two_particle_correlation instead — let's be careful
        pass

    # Determine sub-type
    ctype = "jet_hadron"
    if "direct photon" in rec_title:
        ctype = "direct_photon_hadron"
    elif "peak shape" in rec_title:
        ctype = "jet_peak_shape"
    elif "j_t" in dh or "jT" in dh or "fragmentation" in rec_title:
        ctype = "dihadron_jet_properties"

    fields = {
        "observable": "jet_hadron_correlation",
        "value_type": "per_trigger_yield",
        "correlation_type": ctype,
    }
    return Classification("jet_correlations", "medium", fields)


def _classify_electromagnetic_probes(f):
    """Match direct photon and dilepton measurements."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()

    is_em = any(kw in dh for kw in ["photon", "dielectron", "dimuon", "dilepton",
                                    "e+e-", "μ+μ-"])
    has_em_title = any(kw in rec_title for kw in ["photon", "dielectron", "dimuon",
                                                   "dilepton", "direct photon"])

    if not (is_em or has_em_title):
        return None

    probe = "direct_photon"
    if "dilepton" in dh or "dielectron" in dh or "e+" in dh:
        probe = "dielectron"
    elif "dimuon" in dh:
        probe = "dimuon"

    fields = {
        "observable": f"{probe}_spectra",
        "value_type": "direct_photon_yield" if probe == "direct_photon" else "dilepton_rate",
        "probe": probe,
    }
    return Classification("electromagnetic_probes", "high", fields)


def _classify_femtoscopy(f):
    """Match HBT radii, Bose-Einstein correlations."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()
    indep_names = set(n.strip("$") for n in f["indep_names"])

    is_femto = any(kw in dh for kw in ["r_out", "r_side", "r_long", "r_inv",
                                        "λ", "lambda", "hbt", "source radius",
                                        "lévy", "levy", "bose"])
    has_femto_title = any(kw in rec_title for kw in ["femtoscop", "bose-einstein",
                                                      "hbt", "interferometr", "source"])

    if not (is_femto or has_femto_title):
        return None

    vt = "R_out"
    if "side" in dh:
        vt = "R_side"
    elif "long" in dh:
        vt = "R_long"
    elif "inv" in dh or "invariant" in dh:
        vt = "R_inv"
    elif "λ" in dh or "lambda" in dh:
        vt = "lambda"
    elif "lévy" in dh or "levy" in dh:
        vt = "Levy_alpha"

    fields = {
        "observable": "hbt_radii" if "hbt" not in rec_title else "hbt_radii",
        "value_type": vt,
        "pair_type": "pi+pi+",  # default
    }
    return Classification("femtoscopy", "high", fields)


def _classify_heavy_flavor(f):
    """Match open heavy flavor and quarkonium."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()

    hf_particles = ["d0", "d+", "d-", "ds", "j/ψ", "j/psi", "ψ(2s)", "psi(2s)",
                    "χ_c", "χc", "upsilon", "ϒ", "charm", "beauty",
                    "b-jet", "c-jet", "λ_c", "lambdac", "baryon"]
    hf_title_kw = ["heavy flavor", "heavy-flavor", "open charm", "open heavy",
                   "heavy quark", "d0", "d meson", "b meson", "j/ψ", "j/psi",
                   "upsilon", "quarkonium", "charm", "beauty", "λ_c", "lambdac"]
    is_hf = any(kw in dh for kw in hf_particles)
    has_hf_title = any(kw in rec_title for kw in hf_title_kw)

    if not (is_hf or has_hf_title):
        return None

    # Identify the specific particle
    particle = "D0"
    for p in hf_particles:
        if p in dh:
            particle = p
            break

    # Determine value_type based on what's being reported
    vt = "invariant_yield"
    if "r_aa" in dh or "r_cp" in dh or "nuclear modification" in dh:
        vt = "R_AA"
    elif "v2" in dh or "v{" in dh or "v_{" in dh:
        vt = "v2"
    elif "dN/dy" in dh or "yield" in dh:
        vt = "dN_dy"

    fields = {
        "observable": f"{particle.lower().replace('/', '_').replace(' ', '_')}_{vt}",
        "value_type": vt,
        "particle": particle,
    }
    return Classification("heavy_flavor", "high", fields)


def _classify_strangeness(f):
    """Match strange hadron production."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()

    strange_particles = ["k_s", "k0_s", "ks0", "λ", "λαμβδα", "lambda", "ξ", "xi",
                         "ω", "omega", "φ", "phi", "strange", "hyperon"]
    is_strange = any(kw in dh for kw in strange_particles)
    has_strange_title = any(kw in rec_title for kw in strange_particles +
                            ["strangeness", "strange hadron"])

    if not (is_strange or has_strange_title):
        return None

    species = guess_species(f)

    fields = {
        "observable": f"{species}_spectra",
        "value_type": "invariant_yield",
        "particle_id": {"species": species},
    }
    return Classification("strangeness", "high", fields)


def _classify_fluctuations(f):
    """Match net-charge fluctuations, cumulants."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()

    is_fluct = any(kw in dh for kw in ["ν", "nu_dyn", "ν(+", "cumulant", "c{",
                                        "c1", "c2", "c3", "c4", "skewness",
                                        "kurtosis", "net charge", "net charge"])
    has_fluct_title = any(kw in rec_title for kw in ["fluctuation", "cumulant",
                                                      "net-charge", "net charge"])

    if not (is_fluct or has_fluct_title):
        return None

    # Check it's not mean_pT fluctuations (handled separately)
    if any(kw in dh for kw in ["mean p", "<p_t>", "⟨p_t⟩", "\\langle p"]):
        return None

    vt = "cumulant_C2"
    if "ν" in dh or "nu_dyn" in dh:
        vt = "nu_dyn"
    elif "skewness" in dh:
        vt = "skewness"
    elif "kurtosis" in dh:
        vt = "kurtosis"
    elif "c1" in dh:
        vt = "cumulant_C1"
    elif "c4" in dh:
        vt = "cumulant_C4"
    elif "c3" in dh:
        vt = "cumulant_C3"

    fields = {
        "observable": vt,
        "value_type": vt,
        "charge_type": "net_charge",
    }
    return Classification("fluctuations", "high", fields)


def _classify_jets(f):
    """Match jet spectra and jet observables."""
    dh = f["dep_header_raw"].lower()
    rec_title = f["record_meta"]["title"].lower()

    if not ("jet" in dh and "jet" in rec_title):
        return None

    # Don't steal from jet_correlations
    if any(kw in dh for kw in ["correlation", "dihadron", "per trigger"]):
        return None

    if "fragmentation" in dh:
        vt = "jet_fragmentation"
    elif "quenching" in dh or "suppression" in dh:
        vt = "jet_R_AA"
    elif "shape" in dh:
        vt = "jet_shape"
    else:
        vt = "jet_spectra"

    fields = {
        "observable": "jet_spectra",
        "value_type": vt,
    }
    return Classification("jets", "high", fields)


# Register all classifiers (order matters: more specific first)
CLASSIFIERS = {
    "flow_auxiliary": _classify_flow_auxiliary,
    "flow_vn": _classify_flow_vn,
    "flow_harmonic_correlations": _classify_flow_harmonic_correlations,
    "mean_pT_fluctuations": _classify_mean_pT_fluctuations,
    "flow_pT_correlations": _classify_flow_pT_correlations,
    "two_particle_correlations": _classify_two_particle_correlations,
    "jet_correlations": _classify_jet_correlations,
    "mean_pT": _classify_mean_pT,
    "particle_ratios": _classify_particle_ratios,
    "particle_spectra": _classify_particle_spectra,
    "particle_yields": _classify_particle_yields,
    "nuclear_modification": _classify_nuclear_modification,
    "femtoscopy": _classify_femtoscopy,
    "electromagnetic_probes": _classify_electromagnetic_probes,
    "heavy_flavor": _classify_heavy_flavor,
    "strangeness": _classify_strangeness,
    "fluctuations": _classify_fluctuations,
    "jets": _classify_jets,
}


# ── Output ──────────────────────────────────────────────────────────────

def _schema_has_kinematic_range(obs_type):
    """Check if an observable schema composes kinematic_range."""
    import yaml
    schema_path = REPO_DIR / "observables" / f"{obs_type}.yaml"
    if not schema_path.exists():
        return True  # default: allow
    try:
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
        if not schema:
            return True
        composes = schema.get("composes", [])
        return any("kinematic_range" in c for c in composes)
    except Exception:
        return True


def build_record_yaml(inspire_id, version, filename, classification, features):
    """Build the curated record dict from classification + features."""
    energy, reaction = guess_energy_and_reaction(features)
    centrality = guess_centrality(features)
    species = guess_species(features)

    rec = {
        "observable_type": classification.observable_type,
        "inspire_id": inspire_id,
        "version": version,
        "hepdata_file": filename,
        "experiment": resolve_experiment(features["record_meta"]),
    }

    # Table label from submission.yaml (if available)
    table_label = features.get("table_label", "")
    if table_label:
        rec["hepdata_table_label"] = table_label

    # Raw qualifiers from HEPData (preserved for downstream use)
    qm = features.get("qualifier_map", {})
    if qm:
        rec["qualifiers"] = dict(sorted(qm.items()))

    # Collision system
    rec["collision_system"] = {
        "energy_gev": energy,
        "reaction": reaction,
    }

    # Centrality (if detected)
    if centrality:
        rec["centrality"] = {"classes": centrality}
        est_info = infer_centrality_estimator(features["record_meta"])
        if est_info:
            rec["centrality"].update(est_info)
    elif _is_aa_reaction(reaction):
        # AA reaction with no parsed centrality — write minimal block with
        # collaboration default estimator and a note.
        est_info = infer_centrality_estimator(features["record_meta"])
        if est_info:
            rec["centrality"] = {"classes": [], "notes": "centrality classes not auto-parsed from HEPData qualifiers"}
            rec["centrality"].update(est_info)
    else:
        # Non-AA (pp/pA) with no centrality — set minimum-bias.
        # Only write centrality block when we have estimator info;
        # otherwise omit per schema guidance (centrality.yaml:27-28).
        est_info = infer_centrality_estimator(features["record_meta"])
        if est_info:
            rec["centrality"] = {"classes": ["MB"]}
            rec["centrality"].update(est_info)

    # Particle ID — always written (schema requires it; species "charged" is meaningful)
    pid = classification.field_values.get("particle_id")
    if not pid:
        pid = {"species": species}
        # Copy any additional particle_* keys from field_values (e.g. particle_mass)
        for k, v in classification.field_values.items():
            if k.startswith("particle") and k != "particle_id":
                pid[k] = v
    else:
        # particle_id from field_values may have extra keys; ensure species exists
        if "species" not in pid:
            pid["species"] = species
    rec["particle_id"] = pid

    # Kinematic range from qualifiers
    qm = features["qualifier_map"]
    kr = {}

    # Parse pT range from qualifiers.  Only accept simple qualifier names
    # like "pt" or "p_t"; skip ones with sub-qualifiers like "pt(jet)" or
    # "pt(leading track)" which denote different kinematic quantities.
    # Track min/max globally across qualifiers (e.g. ATLAS pt(c=a) + pt(c=b)).
    pt_min = None
    pt_max = None
    for qn, qv in qm.items():
        ql = qn.lower()
        if ql in ("pt", "p_t", "$p_t$", "p_{t}", "pt(a)", "pt(c=a)", "pt(c=b)"):
            m = re.findall(r'([\d.]+)\s*[-–to]+\s*([\d.]+)', qv, re.IGNORECASE)
            if m:
                vals = [float(x) for x in m[0]]
                if pt_min is None or min(vals) < pt_min:
                    pt_min = min(vals)
                if pt_max is None or max(vals) > pt_max:
                    pt_max = max(vals)
                continue
            # Single-value formats like "> 0.5" (min) or "< 1" (max)
            m2 = re.match(r'>=\s*([\d.]+)', qv)
            if m2:
                v = float(m2.group(1))
                if pt_min is None or v < pt_min:
                    pt_min = v
                continue
            m3 = re.match(r'>\s*([\d.]+)', qv)
            if m3:
                v = float(m3.group(1))
                if pt_min is None or v < pt_min:
                    pt_min = v
                continue
            m4 = re.match(r'<\s*([\d.]+)', qv)
            if m4:
                v = float(m4.group(1))
                if pt_max is None or v > pt_max:
                    pt_max = v
                continue
    pt_range = {}
    if pt_min is not None and pt_max is not None:
        pt_range = {"min": pt_min, "max": pt_max}
    elif pt_min is not None:
        pt_range = {"min": pt_min}
    elif pt_max is not None:
        pt_range = {"max": pt_max}

    # Parse eta range
    eta_range = {}
    for qn, qv in qm.items():
        ql = qn.lower()
        if ql in ("etarap", "abs(etarap)", "|η|", "|eta|"):
            m = re.findall(r'([\d.]+)\s*[-–to]+\s*([\d.]+)', qv)
            if m:
                vals = [float(x) for x in m[0]]
                eta_range = {"min": min(vals), "max": max(vals)}
            else:
                # Single value like "< 0.8" or "> -2.5"
                m2 = re.match(r'[<>\s]*([\d.]+)', qv.strip())
                if m2:
                    v = float(m2.group(1))
                    if "<" in qv:
                        eta_range = {"min": -v, "max": v}
                    else:
                        eta_range = {"min": min(-v, 0), "max": v}

    if pt_range:
        kr["pT"] = pt_range
    if eta_range:
        kr["eta"] = eta_range

    # Add bin edges from independent variables.  Maps known indep variable
    # names to kinematic range keys.  If the kinematic range entry doesn't
    # exist yet, create it from the edges (min/max = first/last boundary).
    _INDEP_TO_KR = {
        "pt": "pT", "p_t": "pT", "$p_t$": "pT", "p_{t}": "pT",
        "p_{t} position": "pT",
        "pt(a)": "pT", "pt(c=b)": "pT",
        "p_t$ (gev/$c$)": "pT", "$p_{t}$ (gev/$c$)": "pT",
        "eta": "eta", "η": "eta", "pseudorapidity": "eta",
        "etarap": "eta", "etarap": "eta",
        "abs(etarap)": "eta",
        "etarap(c=lab)": "eta",
        "y": "y", "rapidity": "y", "rapidity(y)": "y",
        "y*": "y", "yrap": "y",
        "abs(delta(phi))": "delta_phi", r"\delta \phi": "delta_phi",
        "delta phi": "delta_phi", "δφ": "delta_phi",
        "dphi": "delta_phi",
        "abs(detarap)": "delta_eta", "delta(eta)": "delta_eta",
        "δη": "delta_eta",
    }
    for iv_name, edges in features.get("indep_bin_edges", {}).items():
        # Try exact match first, then with qualifier suffixes stripped
        kr_key = _INDEP_TO_KR.get(iv_name)
        if kr_key is None:
            iv_clean = re.sub(r'\([^)]*\)', '', iv_name).strip()
            kr_key = _INDEP_TO_KR.get(iv_clean)
        if kr_key is None:
            iv_clean2 = iv_name.replace('\\', '').strip()
            kr_key = _INDEP_TO_KR.get(iv_clean2)
        if kr_key and edges:
            kr[kr_key] = {"min": edges[0], "max": edges[-1], "bins": edges}

    # Apply collaboration-level defaults for missing kinematic range keys.
    # Fills in missing sub-fields (e.g. missing pT.max from defaults when
    # qualifiers only gave pT.min) or adds the whole key when absent.
    if _schema_has_kinematic_range(classification.observable_type):
        kr_defaults = infer_kinematic_range_defaults(features["record_meta"])
        for key, val in kr_defaults.items():
            if key in kr:
                # Merge sub-fields (fill missing min/max from defaults)
                if isinstance(val, dict) and isinstance(kr[key], dict):
                    for sub_key, sub_val in val.items():
                        if sub_key not in kr[key]:
                            kr[key][sub_key] = sub_val
            else:
                kr[key] = val

    # For diff records with qualifier-derived ranges but no indep var bins,
    # add a single bin from the range boundaries.
    if classification.observable_type == "flow_vn":
        dim = classification.field_values.get("dimensionality", "")
        if dim == "differential_pT" and "pT" in kr and "bins" not in kr["pT"]:
            kr["pT"]["bins"] = [kr["pT"]["min"], kr["pT"]["max"]]
        if dim == "differential_eta" and "eta" in kr and "bins" not in kr["eta"]:
            kr["eta"]["bins"] = [kr["eta"]["min"], kr["eta"]["max"]]

    if kr and _schema_has_kinematic_range(classification.observable_type):
        rec["kinematic_range"] = kr

    # Add classification-specific fields
    for k, v in classification.field_values.items():
        if k not in ("hepdata_file", "particle_id"):
            rec[k] = v

    return rec


def build_output_path(inspire_id, filename, classification):
    """Determine output filename for a candidate record."""
    obs_type = classification.observable_type
    basename = f"{inspire_id}_{filename.replace('.yaml', '').replace('.yml', '')}.yaml"
    output_dir = OUTPUT_DIR / obs_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / basename


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Classify HEPData tables into curated records."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview classification without writing files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N table files (for testing)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Show classification counts and exit")
    parser.add_argument("--unclassified", action="store_true",
                        help="Show unclassified files")
    args = parser.parse_args()

    index = load_record_index()
    classification_counts = Counter()
    unclassified_list = []
    unclassified_dep_headers = Counter()
    total_files = 0

    for inspire_id, rec_meta in index.items():
        version = rec_meta.get("version", "1")
        dir_entries = get_table_file_list(inspire_id, version)
        if not dir_entries:
            continue

        for dir_name, filename in dir_entries:
            if args.limit and total_files >= args.limit:
                break
            total_files += 1

            if total_files % 200 == 0:
                print(f"  [progress] {total_files} files processed...", file=sys.stderr)

            features = extract_features(inspire_id, rec_meta, filename)
            if features is None:
                continue

            classification = classify_table(features)

            if classification:
                classification_counts[classification.observable_type] += 1
                if not args.dry_run and not args.stats_only and not args.unclassified:
                    out_path = build_output_path(inspire_id, filename, classification)
                    record = build_record_yaml(
                        inspire_id, version, filename, classification, features)
                    with open(out_path, "w") as f:
                        yaml.dump(record, f, default_flow_style=False,
                                  sort_keys=False, width=120, allow_unicode=True)
            else:
                unclassified_list.append((inspire_id, filename, features))
                dh = features["dep_header_raw"][:60]
                indep = list(features["indep_names"])[:3]
                unclassified_dep_headers[(dh, tuple(indep))] += 1

        if args.limit and total_files >= args.limit:
            break

    # ── Print summary ──
    if args.unclassified:
        print(f"\nUnclassified files ({len(unclassified_list)} total):\n")
        for (dh, indep), count in unclassified_dep_headers.most_common(30):
            print(f"  [{count:4d}x] dep='{dh}'  indep={list(indep)}")
        print()

    if args.stats_only or args.unclassified:
        print("\nClassification counts:")
        for obs_type, count in classification_counts.most_common():
            print(f"  {obs_type:40s} {count:5d}")
        print(f"  {'unclassified':40s} {len(unclassified_list):5d}")
        print(f"  {'TOTAL':40s} {total_files:5d}")
        return

    # Full output
    output_count = sum(classification_counts.values())
    print(f"\n{'='*60}")
    print(f"  Files scanned : {total_files}")
    print(f"  Classified    : {output_count}")
    print(f"  Unclassified  : {len(unclassified_list)}")
    print(f"  Written to    : {OUTPUT_DIR}")
    print()

    print("  Classification breakdown:")
    for obs_type, count in classification_counts.most_common():
        pct = count / output_count * 100 if output_count else 0
        print(f"    {obs_type:35s} {count:5d} ({pct:5.1f}%)")

    if unclassified_list:
        print(f"\n  Top unclassified patterns:")
        for (dh, indep), count in unclassified_dep_headers.most_common(10):
            print(f"    [{count:4d}x] '{dh}' indep={list(indep)}")

    if args.dry_run:
        print(f"\n  (dry-run — no files written)")


if __name__ == "__main__":
    main()

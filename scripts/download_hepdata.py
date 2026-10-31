#!/usr/bin/env python3
"""
Download all heavy-ion collision records from HEPData into a local staging area.

Strategy:
  1. Reaction-based search (existing) — catches standard HI macro data via reaction strings
  2. Collaboration-based search — catches records with non-standard reaction fields (e.g.
     isobar collisions, old fixed-target) by searching each experiment and filtering by
     heavy-ion keywords in reactions and titles.

Usage:
    python scripts/download_hepdata.py                          # Full download (both strategies)
    python scripts/download_hepdata.py --no-collaboration-search # Reaction search only (legacy)
    python scripts/download_hepdata.py --dry-run                 # Just list what would be found
    python scripts/download_hepdata.py --limit 5                 # Download first 5 records
    python scripts/download_hepdata.py --search-only             # Search + print summary, no download
    python scripts/download_hepdata.py --no-resume               # Redownload everything
"""

import argparse
import json
import os
import re
import sys
import time
import yaml
from pathlib import Path

from hepdata_cli.api import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "hepdata_raw"
RECORDS_DIR = RAW_DIR / "records"
INDEX_FILE = RAW_DIR / "record_index.yaml"
SKIPPED_FILE = RAW_DIR / "skipped_downloads.txt"  # Records that failed to download"

# Reaction patterns covering heavy-ion collisions at RHIC and LHC
REACTION_QUERIES = [
    "Au Au --> CHARGED X",
    "Au Au --> NEUTRAL X",
    "Pb Pb --> CHARGED X",
    "Pb Pb --> NEUTRAL X",
    "d Au --> CHARGED X",
    "d Au --> NEUTRAL X",
    "p Pb --> CHARGED X",
    "p Pb --> NEUTRAL X",
    "Cu Cu --> CHARGED X",
    "Cu Cu --> NEUTRAL X",
    "Xe Xe --> CHARGED X",
    "Xe Xe --> NEUTRAL X",
    "Ru Ru --> CHARGED X",
    "Ru Ru --> NEUTRAL X",
    "Zr Zr --> CHARGED X",
    "Zr Zr --> NEUTRAL X",
    "U U --> CHARGED X",
    "U U --> NEUTRAL X",
    "p Au --> CHARGED X",
    "p Al --> CHARGED X",
    "He Au --> CHARGED X",
    "d Au --> NEUTRAL X",
    "Si Si --> CHARGED X",   # historical fixed-target
    "C C --> CHARGED X",     # historical fixed-target
    "S S --> CHARGED X",     # historical fixed-target
    "O O --> CHARGED X",     # historical fixed-target
    "O O --> NEUTRAL X",
    "Ne Ne --> CHARGED X",   # historical fixed-target
    "Ne Ne --> NEUTRAL X",
]

# All experiments with collaboration-based search — catches records with
# non-standard reaction fields that the reaction-based search would miss.
COLLABORATION_QUERIES = [
    "STAR",
    "PHENIX",
    "PHOBOS",
    "BRAHMS",
    "ALICE",
    "CMS",
    "ATLAS",
    "LHCb",
]

# Heavy-ion indicator matching uses token-aware logic (see _reaction_has_hi)
# to avoid false positives from short substrings like 'u' in 'MUON'.
# These constants define the species to detect:

# Single-element ion tokens (will match as whole words)
# Also includes compound forms like 'pbpb' for compact notation (e.g. "PbPb --> X")
HI_ION_TOKENS = {
    'au', 'pb', 'xe', 'cu', 'kr', 'ru', 'zr',
    'ne', 'si', 'ag',
    'pbpb', 'auau', 'xexe', 'cucu',
    'krkr', 'ruru', 'zrzr',
}

# Multi-word reaction patterns (matched as substrings)
# These cover standard HI collisions plus old fixed-target (C, Be, nucleus target)
HI_MULTI_REACTIONS = [
    'd au', 'd+au', 'd-au',
    'he au', 'he+au', 'he-au',
    'p pb', 'p+pb', 'p-pb',
    'p au', 'p+au', 'p-au',
    'p al', 'p+al', 'p-al',
    'p si', 'p+si', 'p-si',
    'p be', 'p c', 'p nucleus', 'p target',
    'he c', 'he nucleus', 'he target',
    'deut c', 'deut nucleus',
    'nucleus nucleus',
]

# Ions that need pair matching (must appear as "x x" in reaction)
# 'c' for carbon (C C), 'u' for uranium (U U), 'o' for oxygen (O O)
HI_PAIR_IONS = {'c', 'u', 'o'}

# Heavy-ion keywords in titles
HI_TITLE_KEYWORDS = [
    # Collision systems (with and without spaces around +)
    'pb-pb', 'pbpb', 'p-pb',
    'pb+pb', 'pb + pb',
    'au-au', 'auau',
    'au+au', 'au + au',
    'xe-xe', 'xexe',
    'xe+xe', 'xe + xe',
    'cu-cu', 'cucu',
    'cu+cu', 'cu + cu',
    'u-u', 'u+u', 'uranium-uranium',
    'ru+ru', 'ru-ru', 'zr+zr', 'zr-zr', 'isobar',
    'd+au', 'd-au', 'he+au', 'he-au', 'p+au', 'p-au',
    'p+al', 'p-al', 'p+si', 'p-si',
    # Physics
    'heavy-ion', 'heavy ion', 'nuclear',
    'centrality', 'flow', 'quark-gluon', 'qgp',
    'jet quenching',
    'v2', 'v_2', 'v3', 'v_3', 'vn', 'v_n',
    'r_aa', 'raa', 'r_cp', 'rcp',
    'nuclear modification', 'nuclear matter',
    'collective flow', 'elliptic flow', 'azimuthal anisotropy', 'anisotropic',
    'lead ion', 'gold ion', 'copper ion', 'xenon ion',
    'nucleus-nucleus', 'nuclear collisions',
]

# Known records without standard reaction fields — safety net for records that
# might not be caught by reaction or filtered collaboration searches.
EXTRA_INSPIRE_IDS = [
    "1914564",  # Search for the CME with Isobar Collisions @ STAR
    "2692436",  # Upper Limit on the CME in Isobar Collisions @ STAR
    "2713075",  # Estimate of Background Baseline and Upper Limit on the CME @ STAR
    "2919644",  # Hyperon global polarization in isobar Ru+Ru and Zr+Zr @ STAR
    "2822399",  # Tracking the baryon number with nuclear collisions @ STAR
]

# Use max page size for speed
PAGE_SIZE = 100
MAX_MATCHES = 10000


def _reaction_has_hi(reaction_str):
    """Check if a single reaction string involves heavy ions.

    Uses token-aware matching to avoid false positives from short
    substrings like 'u' in 'MUON' or 'o' in 'PHOTON'.
    """
    rxn_lower = reaction_str.lower()

    # Multi-word patterns
    if any(pat in rxn_lower for pat in HI_MULTI_REACTIONS):
        return True

    # Token-based matching — split into whole words
    tokens = re.findall(r"[a-z]+", rxn_lower)
    token_set = set(tokens)

    # Single-element HI species
    if token_set & HI_ION_TOKENS:
        return True

    # Pair-matched ions (must appear as "x x" adjacent pair)
    for ion in HI_PAIR_IONS:
        if ion in token_set:
            for i, tok in enumerate(tokens):
                if tok == ion and i + 1 < len(tokens) and tokens[i + 1] == ion:
                    return True

    return False


def is_heavy_ion_record(record):
    """Check if a HEPData record is relevant to heavy-ion physics.

    Examines reactions and title fields for heavy-ion indicators.
    Used to filter collaboration search results.
    """
    # Extract reactions from either top-level or data_keywords
    reactions = record.get("reactions", [])
    if not reactions:
        dk = record.get("data_keywords", {})
        reactions = dk.get("reactions", [])

    title = (record.get("title") or "").lower()

    # Check reactions field for heavy-ion species
    for rxn in reactions:
        rxn_str = str(rxn.get("value", "")) if isinstance(rxn, dict) else str(rxn)
        if _reaction_has_hi(rxn_str):
            return True

    # Check title for heavy-ion keywords
    if any(kw in title for kw in HI_TITLE_KEYWORDS):
        return True

    return False


def gather_records(client, verbose=False, use_collaboration_search=True):
    """Search HEPData for heavy-ion records, deduplicated by inspire_id.

    Two-phase search:
      1. Reaction-based queries (fast, targeted)
      2. Collaboration-based queries (broad, catches non-standard reaction strings)
    """
    seen = {}
    from_reaction = set()
    from_collab = set()
    from_extra = set()

    # Phase 1: Reaction-based queries
    for rxn in REACTION_QUERIES:
        query = f'reactions:"{rxn}"'
        if verbose:
            print(f"  Searching: {query}")
        try:
            results = client.find(query, max_matches=MAX_MATCHES, matches_per_page=PAGE_SIZE)
        except Exception as e:
            print(f"  WARNING: Search failed for '{query}': {e}", file=sys.stderr)
            continue
        for r in results:
            if r.get("doc_type") != "publication":
                continue
            inspire_id = str(r.get("inspire_id") or "")
            if inspire_id and inspire_id not in seen:
                seen[inspire_id] = r
                from_reaction.add(inspire_id)

    # Phase 2: Collaboration-based queries with HI filtering
    if use_collaboration_search:
        for collab in COLLABORATION_QUERIES:
            query = f'collaborations:"{collab}"'
            if verbose:
                print(f"  Searching: {query}")
            try:
                results = client.find(query, max_matches=MAX_MATCHES, matches_per_page=PAGE_SIZE)
            except Exception as e:
                print(f"  WARNING: Search failed for '{query}': {e}", file=sys.stderr)
                continue
            for r in results:
                if r.get("doc_type") != "publication":
                    continue
                inspire_id = str(r.get("inspire_id") or "")
                if not inspire_id or inspire_id in seen:
                    continue
                if is_heavy_ion_record(r):
                    seen[inspire_id] = r
                    from_collab.add(inspire_id)
                    if verbose:
                        print(f"    Added from {collab}: {r.get('title', '')[:70]}")
                elif verbose:
                    print(f"    Skipped ({collab}, non-HI): {r.get('title', '')[:70]}")

    # Phase 3: Known records by inspire_id (bypasses HI filter)
    for extra_iid in EXTRA_INSPIRE_IDS:
        if extra_iid in seen:
            continue
        query = f'inspire_id:{extra_iid}'
        if verbose:
            print(f"  Fetching: {query}")
        try:
            results = client.find(query, max_matches=1, matches_per_page=1)
        except Exception as e:
            print(f"  WARNING: Failed to fetch ins{extra_iid}: {e}", file=sys.stderr)
            continue
        for r in results:
            if r.get("doc_type") != "publication":
                continue
            iid = str(r.get("inspire_id") or "")
            if iid and iid not in seen:
                seen[iid] = r
                from_extra.add(iid)
                if verbose:
                    print(f"    Found (extra): {r.get('title', '')[:70]}")

    # Print source summary in verbose mode
    if verbose:
        print()
        print(f"  Sources: {len(from_reaction)} from reaction search"
              f", {len(from_collab)} from collaboration search"
              f", {len(from_extra)} from extra IDs")

    return seen


def classify_energy(record):
    """Guess collision energy in GeV from record metadata."""
    dk = record.get("data_keywords", {})
    cm = dk.get("cmenergies", [])
    if cm:
        energies = []
        for e in cm:
            if isinstance(e, dict):
                lo = float(e.get("gte", e.get("lte", 0)))
                hi = float(e.get("lte", e.get("gte", lo)))
                energies.append((lo + hi) / 2)
            elif isinstance(e, (int, float)):
                energies.append(float(e))
        if energies:
            avg_e = sum(energies) / len(energies)
            if avg_e < 50:
                return "<50"
            elif avg_e < 400:
                return "200"
            elif avg_e < 4000:
                return "2760"
            else:
                return "5020"
    return "unknown"


def get_primary_reaction(record):
    """Get the first reaction string from record metadata."""
    dk = record.get("data_keywords", {})
    rxns = dk.get("reactions", [])
    if rxns:
        return rxns[0]
    keywords = record.get("keywords", [])
    for kw in keywords:
        if isinstance(kw, dict) and kw.get("name") == "reactions":
            val = kw.get("value", "")
            if isinstance(val, str) and val:
                return val
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Download heavy-ion HEPData records")
    parser.add_argument("--dry-run", action="store_true", help="Search + list records without downloading")
    parser.add_argument("--search-only", action="store_true", help="Search, print summary with energies, no download")
    parser.add_argument("--limit", type=int, default=None, help="Max records to download (for testing)")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Redownload existing records")
    parser.add_argument("--no-collaboration-search", action="store_false", dest="collaboration_search",
                        help="Skip collaboration-based search (reaction queries only, legacy mode)")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between downloads")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    do_download = not (args.dry_run or args.search_only)

    os.makedirs(RECORDS_DIR, exist_ok=True)

    client = Client(verbose=args.verbose)

    # Phase 1: Search
    print("=== Searching HEPData for heavy-ion records ===")
    records = gather_records(client, verbose=args.verbose, use_collaboration_search=args.collaboration_search)
    print(f"  Found {len(records)} unique publications")

    if not records:
        print("  No records found. Aborting.")
        return

    # Summary by energy
    by_energy = {}
    for rec in records.values():
        energy = classify_energy(rec)
        by_energy.setdefault(energy, []).append(rec["inspire_id"])

    # Summary by collaboration
    by_collab = {}
    for rec in records.values():
        for c in rec.get("collaborations", []):
            by_collab.setdefault(c, []).append(rec["inspire_id"])

    print()
    print("  By energy:")
    for e in sorted(by_energy, key=lambda x: str(x)):
        print(f"    {e} GeV: {len(by_energy[e])} records")

    print("  By collaboration:")
    for c in sorted(by_collab, key=lambda x: len(by_collab[x]), reverse=True):
        print(f"    {c}: {len(by_collab[c])} records")

    if args.dry_run or args.search_only:
        if args.verbose or args.dry_run:
            print()
            for inspire_id, rec in sorted(records.items()):
                energy = classify_energy(rec)
                collab = ", ".join(rec.get("collaborations", []))
                title = (rec.get("title") or "?")[:80]
                print(f"  ins{inspire_id:>8s}  [{energy:>6s}]  {collab:20s}  {title}")

        print()
        if args.dry_run:
            print(f"Dry-run complete. Would download {len(records)} records.")
        else:
            print("Search complete. Use --dry-run to preview, or run without flags to download.")
        return

    # Phase 2: Download
    print()
    print("=== Downloading records ===")

    already = set()
    if args.resume:
        for d in RECORDS_DIR.iterdir():
            if d.is_dir() and d.name.startswith("ins"):
                already.add(d.name[3:])
        if SKIPPED_FILE.exists():
            with open(SKIPPED_FILE) as f:
                for line in f:
                    iid = line.strip()
                    if iid:
                        already.add(iid)
        if already:
            print(f"  {len(already)} records already present or skipped, will skip")

    index = {}
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            index = yaml.safe_load(f) or {}

    sorted_ids = sorted(records.keys())
    count = 0
    errors = []

    for inspire_id in sorted_ids:
        if args.limit is not None and count >= args.limit:
            break
        if args.resume and inspire_id in already:
            continue

        rec = records[inspire_id]
        target_dir = RECORDS_DIR / f"ins{inspire_id}"

        collab = ", ".join(rec.get("collaborations", []))
        title = (rec.get("title") or "?")[:60]
        print(f"  [{count + 1}/{len(sorted_ids)}] ins{inspire_id}  {collab}  {title}")

        download_success = False
        for attempt in range(2):
            try:
                client.download(
                    [inspire_id],
                    file_format="yaml",
                    ids="inspire",
                    download_dir=str(target_dir),
                )
                download_success = True
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt == 0:
                    backoff = 60
                    print(f"  Rate limited. Waiting {backoff}s before retry...",
                          file=sys.stderr)
                    time.sleep(backoff)
                else:
                    print(f"  ERROR: ins{inspire_id}: {e}", file=sys.stderr)
                    errors.append((inspire_id, err_str))
                    if "not downloadable" in err_str:
                        with open(SKIPPED_FILE, "a") as f:
                            f.write(f"{inspire_id}\n")
                    time.sleep(args.delay * 3)
                    break

        if not download_success:
            continue

        count += 1

        metadata_path = target_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(rec, f, indent=2, default=str)

        energy = classify_energy(rec)
        reaction = get_primary_reaction(rec)
        arxiv = rec.get("arxiv_id") or ""
        if arxiv.startswith("arXiv:"):
            arxiv = arxiv[6:]
        index[inspire_id] = {
            "inspire_id": inspire_id,
            "hepdata_id": rec.get("id"),
            "title": rec.get("title"),
            "collaborations": rec.get("collaborations", []),
            "year": rec.get("year"),
            "energy_gev": energy,
            "reaction": reaction,
            "hepdata_doi": rec.get("hepdata_doi"),
            "doi": rec.get("doi"),
            "arxiv_id": arxiv or None,
            "version": rec.get("version", 1),
        }

        with open(INDEX_FILE, "w") as f:
            yaml.dump(index, f, sort_keys=False)

        time.sleep(args.delay)

    print()
    print(f"=== Complete ===")
    print(f"  Downloaded: {count} new records")
    print(f"  Errors:     {len(errors)}")
    print(f"  Total:      {len(index)} records in index")
    if errors:
        print("  Error list:")
        for iid, err in errors:
            print(f"    ins{iid}: {err}")


if __name__ == "__main__":
    main()

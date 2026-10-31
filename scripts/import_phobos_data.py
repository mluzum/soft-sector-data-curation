#!/usr/bin/env python3
"""
Scrape PHOBOS data tables from the BNL website with automatic file discovery.

Discovers papers from the PHOBOS publications page, crawls each paper's
index page for data file links (direct .txt/.dat links or via Data/data.htm
subdirectory), downloads missing files, infers observable types from paper
titles, and generates schema-valid curated YAML records.

Usage:
    python scripts/import_phobos_data.py                   # full run
    python scripts/import_phobos_data.py --discover-only   # just list papers + files
    python scripts/import_phobos_data.py --download-only   # fetch raw files only
    python scripts/import_phobos_data.py --dry-run         # preview without writing
    python scripts/import_phobos_data.py --paper AuAu_v1   # single paper
"""

import re
import sys
import yaml
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict
from html.parser import HTMLParser

BASE_URL = "https://www.sdcc.bnl.gov/phobos/Publications/Physics"
PUBLICATIONS_URL = f"{BASE_URL}/phobos_physics_publications.htm"
REPO_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_DIR / "hepdata_raw" / "phobos"
DATA_DIR = REPO_DIR / "data"
SCHEMA_DIR = REPO_DIR / "observables"

request_headers = {"User-Agent": "Mozilla/5.0 (compatible; PHOBOS scraper)"}

# ── Observable type inference ──────────────────────────────────────────

OBS_TYPE_PATTERNS = [
    # Flow / anisotropic flow
    (r'\bnon-flow\b', 'flow_vn'),
    (r'\bnonflow\b', 'flow_vn'),
    (r'\bflow\s+fluctuations\b', 'flow_vn'),
    (r'\belliptic\s+flow\b', 'flow_vn'),
    (r'\b(directed|v1|v2|v3|v4)\b', 'flow_vn'),
    (r'\bcollective\s+flow\b', 'flow_vn'),
    (r'\banisotropy\b', 'flow_vn'),
    (r'\bflow\b', 'flow_vn'),
    (r'\beccentricity\b', 'flow_vn'),
    # Particle spectra
    (r'\bidentified\s+hadron\s+transverse\s+momentum\s+spectra\b', 'particle_spectra'),
    (r'\btransverse\s+momentum\s+(spectra|distributions|distribution)\b', 'particle_spectra'),
    (r'\b(pT|pt)\s*spectra\b', 'particle_spectra'),
    (r'\bparticle\s+production\s+at\s+very\s+low\s+transverse\b', 'particle_spectra'),
    (r'\bcharged\s+hadron\s+transverse\s+momentum\b', 'particle_spectra'),
    (r'\b(stopping|e[ D]ep)\b', 'particle_spectra'),
    # Particle yields / multiplicity
    (r'\bpseudorapidity\s+density\b', 'particle_yields'),
    (r'\b(dN/d[ηy]|dnd[ηy]|dn[ηy])\b', 'particle_yields'),
    (r'\bmultiplicity\b', 'particle_yields'),
    (r'\btracklet\b', 'particle_yields'),
    # Particle ratios
    (r'\b(anti)?particle\s+ratios\b', 'particle_ratios'),
    (r'\bratios\s+of\s+charged\b', 'particle_ratios'),
    (r'\b(anti)?particle\s+to\s+particle\b', 'particle_ratios'),
    # HBT / femtoscopy (must come before generic "correlations")
    (r'\bHBT\b', 'femtoscopy'),
    (r'\bfemtoscopy\b', 'femtoscopy'),
    # Two-particle correlations
    (r'\btwo[- ]particle\s+correlations?\b', 'two_particle_correlations'),
    (r'\bangular\s+correlations?\b', 'two_particle_correlations'),
    (r'\bcluster\s+properties\b', 'two_particle_correlations'),
    (r'\bcorrelations?\b', 'two_particle_correlations'),
    # Nuclear modification
    (r'\bnuclear[- ]modification\b', 'nuclear_modification'),
    (r'\bR_(AA|CP)\b', 'nuclear_modification'),
    # Charge-dependent correlations / CME (must come before generic "correlations")
    (r'\bforward[- ]backward\b', 'charge_dependent_correlations'),
    # Spectator fragments
    (r'\bspectator\s+fragments\b', 'particle_yields'),
    # Tagged d+Au
    (r'\btagged\s+d\+Au\b', 'particle_yields'),
]

DEFAULT_OBS_TYPE = "particle_yields"


def infer_observable_type(title, filename="", page_text=""):
    text = f"{title} {filename} {page_text}"
    for pattern, obs_type in OBS_TYPE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return obs_type
    return DEFAULT_OBS_TYPE


# ── HTML parsers ───────────────────────────────────────────────────────

def parse_publications_page(html):
    """Extract list of paper dicts from the publications page HTML using regex.

    Each paper entry has the structure:
      <li>
        ...<b>TITLE FRAGMENTS</b>...<b>...</b><br>
        <a href="...">Journal Ref (YEAR)</a>
        ...
        <a href="SLUG/index.htm">Links: Local, arXiv, Figures, and Data</a>
      </li>
    The title is the concatenation of all <b> content between <li> and the first <br>.
    """
    papers = []
    slug_pattern = re.compile(
        r'<a\s+href="([^"]+)/index\.htm">Links:\s*Local,\s*arXiv,\s*Figures,\s*and\s*Data</a>',
        re.IGNORECASE
    )
    for m in slug_pattern.finditer(html):
        slug = m.group(1).split('/')[-1]
        before = html[:m.start()]
        li_start = before.rfind('<li')
        if li_start == -1:
            continue
        li_block = before[li_start:m.start()]
        # Find all <b>...</b> content up to the first <br> in the <li> block
        br_pos = li_block.find('<br')
        if br_pos == -1:
            continue
        title_area = li_block[:br_pos]
        b_parts = re.findall(r'<b>(.*?)</b>', title_area, re.DOTALL)
        title = ' '.join(b_parts)
        title = re.sub(r'<[^>]+>', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
        title = title.replace('\u00a0', ' ').strip()
        # Find year from the journal citation within the <li> block
        section = li_block[br_pos:]
        year_match = re.search(r'\((\d{4})\)', section)
        year = int(year_match.group(1)) if year_match else 0
        papers.append({"slug": slug, "title": title, "year": year})
    return papers


class PaperPageParser(HTMLParser):
    """Parse a paper's index.htm to find arXiv link, data file links, and Data/ subdir."""

    def __init__(self):
        super().__init__()
        self.arxiv_id = ""
        self.data_links = []  # (url, link_text)
        self.has_data_subdir = False
        self._in_a = False
        self._href = ""
        self._text = ""

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            d = dict(attrs)
            self._href = d.get('href', '')
            self._in_a = True
            self._text = ""

    def handle_endtag(self, tag):
        if tag == 'a' and self._in_a:
            href_lower = self._href.lower()
            # arXiv link
            if ('arxiv' in href_lower or 'xxx.lanl.gov' in href_lower or
                'lanl.arxiv' in href_lower or 'arxiv.org' in href_lower):
                m = re.search(r'/(?:abs|format|pdf)/([^\s<>"\']+)', self._href)
                if m:
                    self.arxiv_id = m.group(1)
            # Data subdirectory — catch Data/data.htm and any variant
            if re.search(r'data[^/]*\.htm', href_lower) or 'data tables' in self._text.lower():
                self.has_data_subdir = True
            # Direct .txt/.dat links
            if href_lower.endswith('.txt') or href_lower.endswith('.dat'):
                self.data_links.append(self._href)
            self._in_a = False

    def handle_data(self, data):
        if self._in_a:
            self._text += data


def crawl_paper_page(slug):
    """Fetch a paper's index.htm and return (arxiv_id, data_urls, has_data_subdir)."""
    url = f"{BASE_URL}/{slug}/index.htm"
    try:
        req = urllib.request.Request(url, headers=request_headers)
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return None, [], False

    parser = PaperPageParser()
    parser.feed(html)

    # Also check Data/data.htm if present
    data_urls = []
    for link in parser.data_links:
        resolved = urllib.parse.urljoin(url, link)
        data_urls.append(resolved)

    if parser.has_data_subdir:
        data_page_url = urllib.parse.urljoin(url, "Data/data.htm")
        try:
            req = urllib.request.Request(data_page_url, headers=request_headers)
            resp = urllib.request.urlopen(req, timeout=15)
            data_html = resp.read().decode('utf-8', errors='replace')
            sub_parser = PaperPageParser()
            sub_parser.feed(data_html)
            for link in sub_parser.data_links:
                resolved = urllib.parse.urljoin(data_page_url, link)
                if resolved not in data_urls:
                    data_urls.append(resolved)
        except Exception:
            pass

    return parser.arxiv_id, data_urls, parser.has_data_subdir


# ── File download ──────────────────────────────────────────────────────

def download_file(url, local_path, timeout=15):
    """Download a file. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers=request_headers)
        resp = urllib.request.urlopen(req, timeout=timeout)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception:
        return False


# ── ASCII format parser (reused from previous version) ─────────────────

_COLLISION_KEYS = {
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
}


def collision_shorthand(text):
    if not text:
        return None
    words = set(text.upper().replace("-", " ").replace("+", " ").replace(">", " ").replace(",", " ").split())
    for key_set, short in _COLLISION_KEYS.items():
        if key_set.issubset(words):
            return short
    return None


def _parse_metadata_from_comment(c):
    """Extract (collisions, energies, centrality) from a comment line."""
    coll_set = set()
    en_set = set()
    cent = ""
    coll = collision_shorthand(c)
    if coll:
        coll_set.add(coll)
    for m in re.finditer(r'(\d+[\s.]*\d*)\s*(?:GeV|GeV/c|GeV)', c, re.IGNORECASE):
        try:
            v = float(m.group(1).replace(" ", ""))
            if v > 0:
                en_set.add(v)
        except ValueError:
            pass
    cm = re.search(r'(\d+[\s.]*\d*)\s*-\s*(\d+[\s.]*\d*)\s*%', c)
    if cm:
        cent = f"{cm.group(1)}-{cm.group(2)}%"
    return coll_set, en_set, cent


def parse_phobos_file(text, obs_type, defaults):
    """Parse a PHOBOS ASCII data file into one or more datasets.
    Returns list of dicts with metadata and column info.
    """
    lines = text.split("\n")

    def _is_data_line(stripped):
        if not stripped:
            return False
        tokens = stripped.split()
        if len(tokens) < 3:
            return False
        numeric_count = 0
        for tok in tokens:
            raw = tok.strip('|,;')
            try:
                float(raw)
                numeric_count += 1
            except ValueError:
                # Handle "value+/-error" format (e.g. "700+/-27")
                if re.match(r'^[\d.]+[+]/[-][\d.]+$', raw):
                    numeric_count += 1
                continue
        return numeric_count >= 3

    header_comments = []
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    found_data = False
    for j in range(i, len(lines)):
        stripped = lines[j].strip()
        if not stripped:
            continue
        if _is_data_line(stripped):
            found_data = True
            i = j
            break
        header_comments.append(stripped.lstrip("/").strip() if stripped.startswith("//") else stripped)
    else:
        i = len(lines)
    if found_data:
        i = min(i, len(lines))

    raw_blocks = []
    section_comments = []
    current_data = []
    for line in lines[i:]:
        stripped = line.strip()
        if not stripped and current_data:
            raw_blocks.append((list(header_comments) + section_comments, current_data))
            section_comments = []
            current_data = []
        elif not stripped:
            continue
        elif _is_data_line(stripped):
            current_data.append(line)
        else:
            section_comments.append(stripped.lstrip("/").strip() if stripped.startswith("//") else stripped)

    if current_data:
        raw_blocks.append((list(header_comments) + section_comments, current_data))

    datasets = []
    for block_comments, data_lines in raw_blocks:
        if not data_lines:
            continue
        desc = block_comments[0] if block_comments else obs_type
        collisions = set()
        energies = set()
        centrality = ""
        for c in reversed(block_comments):
            if not collisions or not energies or not centrality:
                coll_set, en_set, cent = _parse_metadata_from_comment(c)
                if not collisions:
                    collisions = coll_set
                if not energies:
                    energies = en_set
                if not centrality and cent:
                    centrality = cent
            else:
                break
        if not collisions:
            for c in block_comments:
                for pat, short in [
                    ("Au-Au", "AuAu"), ("Au+Au", "AuAu"), ("AuAu", "AuAu"),
                    ("Cu-Cu", "CuCu"), ("Cu+Cu", "CuCu"), ("CuCu", "CuCu"),
                    ("d-Au", "dAu"), ("d+Au", "dAu"), ("dAu", "dAu"),
                    ("Pb-Pb", "PbPb"), ("Pb+Pb", "PbPb"), ("PbPb", "PbPb"),
                    ("p+p", "pp"), ("pp", "pp"),
                ]:
                    if pat.lower() in c.lower():
                        collisions.add(short)
                        break

        ds = _parse_data_block("\n".join(data_lines), obs_type, defaults, block_comments)
        if not ds:
            continue
        ds["collisions"] = collisions if collisions else ds.get("collisions", set())
        ds["energies"] = energies if energies else ds.get("energies", set())
        ds["centrality"] = centrality if centrality else ds.get("centrality", "")
        ds["desc"] = desc
        ds["obs_type"] = obs_type
        ds["fields"] = dict(defaults)
        datasets.append(ds)

    return datasets


def _parse_data_block(block_text, obs_type, defaults, comments):
    """Parse a single block of data (header + numeric rows)."""
    lines = block_text.strip().split("\n")
    if not lines:
        return None

    clean_lines = [l for l in lines if not l.strip().startswith("//")]

    def _is_header_line(stripped):
        if not stripped:
            return False
        tokens = stripped.split()
        alpha_count = sum(1 for t in tokens if re.search(r'[a-zA-Z]', t))
        return alpha_count >= 2 and not re.match(r'^[\d\s\-.]+$', stripped)

    header_idx = None
    for i, line in enumerate(clean_lines):
        if _is_header_line(line.strip()):
            header_idx = i
            break

    if header_idx is None and comments:
        # Search in reverse: last matching comment is closest to data → most likely header
        for ci in range(len(comments) - 1, -1, -1):
            cl = comments[ci]
            if isinstance(cl, str) and _is_header_line(cl):
                clean_lines.insert(0, cl)
                header_idx = 0
                break

    if header_idx is None:
        return None

    header_line = clean_lines[header_idx].strip()
    headers = header_line.split()
    header_lower = [h.lower() for h in headers]

    col_info = {
        "eta_idx": None, "pT_idx": None, "y_idx": None,
        "value_idx": None, "stat_idx": None,
        "syst_idx_low": None, "syst_idx_high": None,
        "npart_idx": None,
        "delta_eta_idx": None, "delta_phi_idx": None,
    }

    # Map headers to kinematic_range keys for bin inference
    _HEADER_TO_KR_KEY = {
        "eta": "eta", "η": "eta", "etarap": "eta",
        "pseudorapidity": "eta",
        "eta'": "eta", "eta`": "eta", "eta-prime": "eta",
        "ey": "eta", "e+y": "eta", "e-y": "eta", "ep": "eta",
        "pt": "pT", "ptmevc": "pT", "ptgevc": "pT",
        "transversemomentum": "pT", "p t": "pT", "ptgev": "pT",
        "y": "y", "rapidity": "y",
        "delta-eta": "delta_eta", "delta_eta": "delta_eta",
        "deltaeta": "delta_eta", "deta": "delta_eta", "delta_eta_": "delta_eta",
        "delta-phi": "delta_phi", "delta_phi": "delta_phi",
        "deltaphi": "delta_phi", "dphi": "delta_phi",
    }
    kr_idx = {}
    for i, h in enumerate(header_lower):
        h_clean = re.sub(r'[<>\-#()._]+', '', h).strip()
        if h_clean in _HEADER_TO_KR_KEY:
            key = _HEADER_TO_KR_KEY[h_clean]
            if key not in kr_idx:
                kr_idx[key] = i

    for i, h in enumerate(header_lower):
        h_clean = re.sub(r'[<>\-#()._]+', '', h).strip()
        # Use HEADER_TO_KR_KEY for kinematic column detection
        kr_key = _HEADER_TO_KR_KEY.get(h_clean)
        if kr_key == "eta" and col_info["eta_idx"] is None:
            col_info["eta_idx"] = i
        elif kr_key == "pT" and col_info["pT_idx"] is None:
            col_info["pT_idx"] = i
        elif kr_key == "y" and col_info["y_idx"] is None:
            col_info["y_idx"] = i
        elif kr_key == "delta_eta" and col_info["delta_eta_idx"] is None:
            col_info["delta_eta_idx"] = i
        elif kr_key == "delta_phi" and col_info["delta_phi_idx"] is None:
            col_info["delta_phi_idx"] = i
        elif h_clean in ("v1", "v2", "v3", "v4", "v", "flow", "value", "dn/dη", "dndη", "yield", "dn/dy"):
            col_info["value_idx"] = i
        elif h_clean.startswith("v") and len(h_clean) == 2 and h_clean[1].isdigit():
            col_info["value_idx"] = i

    if col_info["value_idx"] is None and len(headers) >= 2:
        value_candidates = []
        for i, h in enumerate(header_lower):
            if not any(x in h for x in ("err", "syst", "error", "eta", "pt", "y")):
                value_candidates.append(i)
        if len(value_candidates) == 1:
            col_info["value_idx"] = value_candidates[0]

    for i, h in enumerate(header_lower):
        if any(e in h for e in ["stat error", "stat err", "stat_err", "stat.", "statistic"]):
            if col_info["stat_idx"] is None:
                col_info["stat_idx"] = i

    for i, h in enumerate(header_lower):
        h_clean = h.strip("<>#()").replace("-", "").replace("_", "").replace(".", "")
        if any(e in h for e in ["syst error", "syst err", "syst_err", "syst.", "syst"]):
            if "-" in h or "low" in h:
                col_info["syst_idx_low"] = i
            elif "+" in h or "high" in h or "up" in h:
                col_info["syst_idx_high"] = i
            else:
                if col_info["syst_idx_low"] is None:
                    col_info["syst_idx_low"] = i
                if col_info["syst_idx_high"] is None:
                    col_info["syst_idx_high"] = i

    if col_info["syst_idx_low"] is None:
        for i, h in enumerate(header_lower):
            h_clean = h.strip("<>#()").replace("-", "").replace("_", "").replace(".", "")
            if h_clean in ("sy_l", "sy_d", "sy_low"):
                col_info["syst_idx_low"] = i
            if h_clean in ("sy_r", "sy_u", "sy_high"):
                col_info["syst_idx_high"] = i

    def _parse_val(tok):
        """Parse a token that may be a plain number or in 'val+/-err' format."""
        raw = tok.strip('|,;')
        m = re.match(r'^([\d.]+)[+-]([\d.]+)$', raw)
        if m:
            return float(m.group(1))
        try:
            return float(raw)
        except ValueError:
            return None

    data_rows = []
    for line in clean_lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 3:
            vals = [_parse_val(t) for t in tokens]
            if any(v is not None for v in vals):
                data_rows.append([v if v is not None else 0.0 for v in vals])

    if len(data_rows) == 0:
        return None

    # Extract column values as flat lists for bin inference
    n_cols = len(headers)
    col_values = [[] for _ in range(n_cols)]
    for row in data_rows:
        for ci in range(min(len(row), n_cols)):
            col_values[ci].append(row[ci])

    return {
        "col_info": col_info,
        "kr_idx": kr_idx,
        "headers": headers,
        "n_rows": len(data_rows),
        "col_values": col_values,
    }


# ── Bin edge inference from column midpoints ──────────────────────────

def _compute_bins(values):
    """Compute bin edges from sorted midpoint values.

    For interior bins uses (v[i] + v[i+1]) / 2 as the shared edge.
    Extrapolates exterior edges from the first/last bin spacing.
    Returns a flat list of N+1 edges, or None if < 2 values.
    """
    vals = sorted(set(v for v in values if v is not None))
    n = len(vals)
    if n < 2:
        return None
    edges = [vals[0] - (vals[1] - vals[0]) / 2]
    for i in range(n - 1):
        edges.append((vals[i] + vals[i + 1]) / 2)
    edges.append(vals[-1] + (vals[-1] - vals[-2]) / 2)
    return edges


# ── Record generation ──────────────────────────────────────────────────

def generate_record(paper, file_url, dataset, local_path):
    """Generate a curated YAML record dict from a parsed dataset."""
    fields = dict(dataset.get("fields", {}))
    obs_type = dataset.get("obs_type", "unknown")

    collisions = set(dataset.get("collisions", set()))
    energies = set(dataset.get("energies", set()))

    slug_lower = paper["slug"].lower()
    if not energies:
        if "19" in slug_lower:
            energies.add(19.6)
        elif "22" in slug_lower:
            energies.add(22.4)
        elif "62" in slug_lower:
            energies.add(62.4)
        elif "130" in slug_lower:
            energies.add(130)
        elif "410" in slug_lower:
            energies.add(410)
        else:
            energies.add(200)

    if not collisions:
        search_text = " ".join([slug_lower, paper.get("title", "").lower()])
        if "au+au" in search_text or "auau" in search_text or "au-au" in search_text or "au au" in search_text:
            collisions.add("AuAu")
        elif "cu+cu" in search_text or "cucu" in search_text or "cu-cu" in search_text or "cu cu" in search_text:
            collisions.add("CuCu")
        elif "d+au" in search_text or "dau" in search_text or "d-au" in search_text:
            collisions.add("dAu")
        elif "p+p" in search_text or "pp" in search_text:
            collisions.add("pp")

    coll_list = sorted(collisions)
    # B3: Normalize collision abbreviations to HEPData format (AUAU → AU AU, CUCU → CU CU)
    _NORM = {"AUAU": "AU AU", "CUCU": "CU CU", "DAU": "D AU", "PP": "P P"}
    norm_parts = []
    for c in coll_list:
        uc = c.upper().replace("-", "").replace(" ", "")
        norm_parts.append(_NORM.get(uc, uc))
    reaction = " ".join(norm_parts) + " --> CHARGED X" if norm_parts else "? --> CHARGED X"

    centrality_str = dataset.get("centrality", "")
    centrality_classes = []
    if centrality_str and "-" in centrality_str and "%" in centrality_str:
        centrality_classes = [centrality_str]

    particle = fields.get("particle_id", {"species": "charged"})

    col_info = dataset.get("col_info", {})
    kr = {}
    phobos_eta = {"min": -3.2, "max": 3.2}
    phobos_pt = {"min": 0.1, "max": 2.0}

    has_eta_col = col_info.get("eta_idx") is not None
    has_pt_col = col_info.get("pT_idx") is not None

    if has_eta_col or obs_type == "flow_vn":
        kr["eta"] = dict(phobos_eta)
    if has_pt_col:
        kr["pT"] = dict(phobos_pt)

    dimensionality = fields.get("dimensionality", "")
    if obs_type == "flow_vn" and dimensionality in ("differential_eta", "integrated") and "pT" not in kr:
        kr["pT"] = dict(phobos_pt)

    # Infer dimensionality from detected kinematic columns
    if obs_type == "flow_vn" and dimensionality != "":
        has_eta = col_info.get("eta_idx") is not None
        has_pt = col_info.get("pT_idx") is not None
        if not has_eta and not has_pt:
            dimensionality = "integrated"
        elif has_eta:
            dimensionality = "differential_eta"
        elif has_pt:
            dimensionality = "differential_pT"
        fields["dimensionality"] = dimensionality

    # Compute bin edges from column values
    kr_idx = dataset.get("kr_idx", {})
    col_values = dataset.get("col_values", [])
    for kr_key, col_idx in kr_idx.items():
        if col_idx < len(col_values) and len(col_values[col_idx]) >= 2:
            raw_vals = [v for v in col_values[col_idx] if v is not None]
            if len(raw_vals) < 2:
                continue
            vals = list(raw_vals)
            if kr_key == "delta_phi":
                vmin, vmax = min(vals), max(vals)
                if vmax - vmin > 6.28:
                    vals = [v * 3.141592653589793 / 180.0 for v in vals]
            bins = _compute_bins(vals)
            if bins:
                entry = kr.setdefault(kr_key, {})
                entry["bins"] = bins
                entry["min"] = min(bins)
                entry["max"] = max(bins)

    # Also add bins for delta_eta / delta_phi from col_info (2p correlations)
    for col_key, kr_key in [("delta_eta_idx", "delta_eta"),
                             ("delta_phi_idx", "delta_phi")]:
        idx = col_info.get(col_key)
        if idx is not None and kr_key not in kr and idx < len(col_values) and len(col_values[idx]) >= 2:
            raw_vals = [v for v in col_values[idx] if v is not None]
            if not raw_vals or len(raw_vals) < 2:
                continue
            vals = list(raw_vals)
            if kr_key == "delta_phi":
                vmin, vmax = min(vals), max(vals)
                if vmax - vmin > 6.28:
                    vals = [v * 3.141592653589793 / 180.0 for v in vals]
            bins = _compute_bins(vals)
            if bins:
                kr[kr_key] = {"bins": bins, "min": min(bins), "max": max(bins)}

    energy_gev = list(energies)[0] if len(energies) == 1 else sorted(energies)

    arxiv_id = paper.get("arxiv_id", "")
    inspire_id = f"PHOBOS_{arxiv_id}" if arxiv_id else f"PHOBOS_{paper['slug']}"

    record = {
        "observable_type": obs_type,
        "inspire_id": inspire_id,
        "version": 1,
        "particle_id": particle,
        "collision_system": {
            "energy_gev": energy_gev,
            "reaction": reaction,
        },
    }

    if centrality_classes:
        record["centrality"] = {
            "classes": centrality_classes,
            "estimator": {"name": "Octagon", "observable": "Nch", "detector": "Octagon"},
            "eta": [{"min": -3.2, "max": 3.2}],
        }
    else:
        record["centrality"] = {
            "classes": ["MB"],
            "notes": "centrality derived from minimum-bias event selection by the Octagon detector",
            "estimator": {"name": "Octagon", "observable": "Nch", "detector": "Octagon"},
            "eta": [{"min": -3.2, "max": 3.2}],
        }

    if kr:
        record["kinematic_range"] = kr

    for k, v in fields.items():
        if k not in ("particle_id",):
            record[k] = v

    return record


# ── Slug generation ────────────────────────────────────────────────────

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
    r")",
    re.IGNORECASE,
)

_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "for", "at", "to", "and", "by", "with"})


def generate_slug(title):
    if not title:
        return "unknown"
    t = re.sub(r'\$[^$]*\$', '', title)
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


# ── Schema validation ──────────────────────────────────────────────────

def _resolve_schema(schema, schema_dir):
    """Minimal schema resolver for validation."""
    sections = {}
    top_level = {}
    common_path = schema_dir / "_common.yaml"
    try:
        common = yaml.safe_load(open(common_path))
    except Exception:
        common = {}
    for name, defn in schema.get("required_fields", {}).items():
        defn["_required"] = defn.get("required", True)
        top_level[name] = defn
    for name, defn in schema.get("optional_fields", {}).items():
        defn["_required"] = False
        top_level[name] = defn
    for compose_ref in schema.get("composes", []):
        parts = compose_ref.split("#", 1)
        file_name, section_name = parts[0], parts[1]
        source = common if file_name == "_common" else yaml.safe_load(open(schema_dir / f"{file_name}.yaml"))
        if section_name and section_name in source:
            section_def = source[section_name]
            if "fields" in section_def:
                sections[section_name] = section_def["fields"]
                sections.setdefault("_section_flags", {})[section_name] = {"optional": False}
    for compose_ref in schema.get("optional_composes", []):
        parts = compose_ref.split("#", 1)
        file_name, section_name = parts[0], parts[1]
        source = common if file_name == "_common" else yaml.safe_load(open(schema_dir / f"{file_name}.yaml"))
        if section_name and section_name in source:
            section_def = source[section_name]
            if "fields" in section_def:
                sections[section_name] = section_def["fields"]
                sections.setdefault("_section_flags", {})[section_name] = {"optional": True}
    return {
        "sections": sections,
        "top_level_fields": top_level,
        "conditional_fields": schema.get("conditional_fields", {}),
        "dimensionality_rules": schema.get("dimensionality_rules", {}),
    }


def _validate_record(record, resolved):
    """Minimal validation against resolved schema."""
    errors = []
    for name, defn in resolved["top_level_fields"].items():
        if defn.get("_required") and name not in record:
            errors.append(f"'{name}': missing required field")
    flags = resolved["sections"].pop("_section_flags", {})
    for section_name, fields in resolved["sections"].items():
        section_data = record.get(section_name)
        if section_data is None:
            if flags.get(section_name, {}).get("optional"):
                continue
            if any(f.get("required", False) for f in fields.values()):
                errors.append(f"'{section_name}': missing required section")
            continue
        if not isinstance(section_data, dict):
            errors.append(f"'{section_name}': expected object")
            continue
        for field_name, defn in fields.items():
            if defn.get("required") and field_name not in section_data:
                errors.append(f"'{section_name}.{field_name}': missing required field")
    for field_name, defn in resolved["conditional_fields"].items():
        required_if = defn.get("required_if")
        if required_if:
            if record.get(required_if.get("field")) == required_if.get("value") and field_name not in record:
                errors.append(f"'{field_name}': required when '{required_if.get('field')}' = '{required_if.get('value')}'")
    return errors


# ── Observable-type specific defaults ──────────────────────────────────

def build_defaults(obs_type, title, filename, headers=None):
    """Build sensible default fields based on observable type and paper info."""
    title_lower = title.lower()
    defaults = {}

    # B1: Link back to raw PHOBOS data file
    defaults["hepdata_file"] = f"phobos/{filename}"

    if obs_type == "flow_vn":
        defaults["particle_id"] = {"species": "charged"}
        defaults["method"] = "event_plane"
        defaults["value_type"] = "v2"
        defaults["dimensionality"] = "differential_eta"
        # B4: PHOBOS forward/backward octagon gives |Δη| ~ 2.0
        defaults["eta_gap"] = 2.0

        if re.search(r'\bv1\b|directed\s+flow', title_lower):
            defaults["harmonic"] = 1
            defaults["value_type"] = "v1"
        elif re.search(r'\bv3\b', title_lower):
            defaults["harmonic"] = 3
            defaults["value_type"] = "v3"
        elif re.search(r'\bv4\b', title_lower):
            defaults["harmonic"] = 4
            defaults["value_type"] = "v4"
        else:
            defaults["harmonic"] = 2
            defaults["value_type"] = "v2"

    elif obs_type == "particle_yields":
        defaults["particle_id"] = {"species": "charged"}
        # Infer observable type from header names if available
        if headers:
            h = " ".join(h.lower() for h in headers)
            if "dN/dy" in h or "dndy" in h or "dn/dy" in h:
                defaults["observable"] = "dN_dy"
                defaults["value_type"] = "dN_dy"
            elif "dN/dη" in h or "dndeta" in h or "dn/dη" in h or "dN/deta" in h:
                defaults["observable"] = "dNch_deta"
                defaults["value_type"] = "dNch_deta"
            elif "multiplicity" in h or "nch" in h:
                defaults["observable"] = "Nch"
                defaults["value_type"] = "Nch"
            else:
                defaults["observable"] = "dNch_deta"
                defaults["value_type"] = "dNch_deta"
        else:
            defaults["observable"] = "dNch_deta"
            defaults["value_type"] = "dNch_deta"

    elif obs_type == "particle_ratios":
        defaults["particle_id"] = {"species": "charged"}
        defaults["observable"] = "ratio"
        defaults["value_type"] = "ratio"
        defaults["numerator"] = "charged"
        defaults["denominator"] = "charged"
        if "pbar" in title_lower or "antiparticle" in title_lower:
            defaults["ratio_type"] = "yield_ratio"
            defaults["is_antiparticle_ratio"] = True

    elif obs_type == "particle_spectra":
        defaults["particle_id"] = {"species": "charged"}
        defaults["observable"] = "invariant_yield"
        defaults["value_type"] = "invariant_yield"

    elif obs_type == "two_particle_correlations":
        defaults["particle_id"] = {"species": "charged"}
        defaults["observable"] = "per_trigger_yield"
        defaults["correlation_type"] = "delta_phi_projection"
        defaults["value_type"] = "Y(Δφ)"

    elif obs_type == "femtoscopy":
        defaults["particle_id"] = {"species": "charged"}
        defaults["observable"] = "HBT_correlation"
        defaults["value_type"] = "C(q_inv)"
        defaults["pair_type"] = "pion_pion"

    elif obs_type == "charge_dependent_correlations":
        defaults["particle_id"] = {"species": "charged"}
        defaults["observable"] = "gamma_correlator"
        defaults["value_type"] = "γ"

    elif obs_type == "nuclear_modification":
        defaults["particle_id"] = {"species": "charged"}
        defaults["observable"] = "R_AA"
        defaults["value_type"] = "R_AA"

    return defaults


# ── Main ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scrape PHOBOS data with automatic file discovery.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing")
    parser.add_argument("--discover-only", action="store_true",
                        help="Only discover papers and files, don't download or parse")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download raw files, skip record generation")
    parser.add_argument("--paper", type=str, default=None,
                        help="Only process a specific paper (slug)")
    parser.add_argument("--rate-limit", type=float, default=0.5,
                        help="Seconds between requests (default: 0.5)")
    args = parser.parse_args()

    # ── Phase 1: Fetch publications page ──
    print("=== Phase 1: Scraping publications page ===")
    req = urllib.request.Request(PUBLICATIONS_URL, headers=request_headers)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        pubs_html = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"ERROR fetching publications page: {e}")
        return 1

    papers = parse_publications_page(pubs_html)
    print(f"Found {len(papers)} papers")

    # Filter to specific paper if requested
    if args.paper:
        papers = [p for p in papers if p["slug"] == args.paper]
        if not papers:
            print(f"Paper '{args.paper}' not found on publications page")
            return 1

    # ── Phase 2: Crawl each paper page ──
    print("\n=== Phase 2: Crawling paper pages ===")
    paper_index = {}
    for paper in papers:
        slug = paper["slug"]
        if args.rate_limit > 0:
            time.sleep(args.rate_limit)
        arxiv_id, data_urls, has_data = crawl_paper_page(slug)
        paper["arxiv_id"] = arxiv_id or ""
        paper["data_urls"] = data_urls
        paper["has_data"] = has_data or len(data_urls) > 0
        paper_index[slug] = paper

        status = "✓" if data_urls else ("✗ (no data)" if not has_data else "?")
        print(f"  {status} {slug}: {len(data_urls)} data files, arxiv={arxiv_id or 'NONE'}")

    papers_with_data = [p for p in papers if p["data_urls"]]
    papers_without_data = [p for p in papers if not p["data_urls"]]

    print(f"\n  Papers with data: {len(papers_with_data)}")
    print(f"  Papers without data: {len(papers_without_data)}")
    if papers_without_data:
        for p in papers_without_data:
            print(f"    - {p['slug']} ({p.get('year', '?')})")

    if args.discover_only:
        print("\n=== Discovery summary ===")
        for p in papers_with_data:
            print(f"\n{p['slug']} ({p.get('year', '?')}): {len(p['data_urls'])} files")
            for url in p['data_urls']:
                fname = url.rsplit('/', 1)[-1]
                print(f"    {fname}")
        return 0

    # ── Phase 3: Download files ──
    print("\n=== Phase 3: Downloading files ===")
    download_stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for paper in papers_with_data:
        slug = paper["slug"]
        paper_raw_dir = RAW_DIR / slug
        paper_raw_dir.mkdir(parents=True, exist_ok=True)

        paper["local_files"] = []
        for url in paper["data_urls"]:
            fname = url.rsplit('/', 1)[-1]
            local = paper_raw_dir / fname
            if local.exists():
                download_stats["skipped"] += 1
                paper["local_files"].append((url, fname, local))
                continue
            if args.dry_run:
                paper["local_files"].append((url, fname, local))
                continue
            success = download_file(url, local)
            if success:
                download_stats["downloaded"] += 1
                paper["local_files"].append((url, fname, local))
                print(f"  ✓ {slug}/{fname}")
            else:
                download_stats["failed"] += 1
                print(f"  ✗ {slug}/{fname}")

    print(f"\n  Downloaded: {download_stats['downloaded']}")
    print(f"  Skipped (exists): {download_stats['skipped']}")
    print(f"  Failed: {download_stats['failed']}")

    if args.download_only:
        return 0

    # ── Phase 4: Parse and generate records ──
    print("\n=== Phase 4: Parsing and generating records ===")
    total_records = 0
    parse_errors = 0
    validation_errors = 0
    papers_with_records = 0

    for paper in papers_with_data:
        slug = paper["slug"]
        if not paper.get("local_files"):
            continue

        # Infer observable type from paper title + data filenames
        obs_type = infer_observable_type(
            paper["title"],
            page_text=" ".join(f for _, f, _ in paper["local_files"]),
        )
        paper_obs_types = set()

        paper_records = []
        for url, fname, local_path in paper["local_files"]:
            if not local_path.exists():
                continue
            try:
                text = local_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                parse_errors += 1
                continue

            # Check if filename suggests nuclear_modification
            file_obs = obs_type
            fname_lower = fname.lower()
            if any(x in fname_lower for x in ['raa', 'r_aa', 'rpc', 'r_cp', 'rcp']):
                file_obs = "nuclear_modification"

            # First parse to detect headers, then build defaults with header info
            datasets = parse_phobos_file(text, file_obs, {})
            if not datasets:
                parse_errors += 1
                continue

            for ds in datasets:
                headers = ds.get("headers", [])
                defaults = build_defaults(file_obs, paper["title"], fname, headers)
                ds["fields"] = dict(defaults)
                rec = generate_record(paper, url, ds, local_path)
                paper_obs_types.add(file_obs)

                schema_path = SCHEMA_DIR / f"{rec.get('observable_type', '')}.yaml"
                if schema_path.exists():
                    try:
                        schema = yaml.safe_load(open(schema_path))
                        resolved = _resolve_schema(schema, SCHEMA_DIR)
                        errs = _validate_record(rec, resolved)
                        if errs:
                            validation_errors += 1
                            continue
                    except Exception:
                        pass
                paper_records.append(rec)

        if not paper_records:
            continue

        papers_with_records += 1

        # Write multi-doc YAML
        for obs_t in paper_obs_types:
            obs_records = [r for r in paper_records if r.get("observable_type") == obs_t]
            if not obs_records:
                continue

            rec_slug = generate_slug(paper["title"])
            all_collisions = set()
            all_energies = set()
            for r in obs_records:
                cs = r.get("collision_system", {})
                if not isinstance(cs, dict):
                    continue
                eg = cs.get("energy_gev", 0)
                if isinstance(eg, (int, float)):
                    all_energies.add(eg)
                elif isinstance(eg, (list, tuple)):
                    all_energies.update(eg)
                reaction_str = cs.get("reaction", "")
                for key_set, short in _COLLISION_KEYS.items():
                    if key_set.issubset(set(reaction_str.upper().replace("-", " ").replace("+", " ").replace(">", " ").replace(",", " ").split())):
                        all_collisions.add(short)

            coll_part = ""
            if all_collisions:
                coll_part = "_" + "_".join(sorted(all_collisions))

            if len(all_energies) == 1:
                e = list(all_energies)[0]
                if isinstance(e, float) and e == int(e):
                    e_str = f"{int(e)}gev"
                else:
                    e_str = f"{e}gev"
                energy_part = f"_{e_str}"
            else:
                energy_part = "_multigev"

            obs_dir = DATA_DIR / obs_t
            obs_dir.mkdir(parents=True, exist_ok=True)

            fname_out = f"phobos_{obs_t}_{rec_slug}{coll_part}{energy_part}.yaml"
            dest = obs_dir / fname_out

            if args.dry_run:
                print(f"  would write: {dest.relative_to(REPO_DIR)} ({len(obs_records)} docs)")
            else:
                with open(dest, "w") as f:
                    for rec in obs_records:
                        f.write("---\n")
                        yaml.dump(rec, f, default_flow_style=False,
                                  sort_keys=False, width=120, allow_unicode=True)
                print(f"  ✓ wrote: {dest.relative_to(REPO_DIR)} ({len(obs_records)} docs) for {slug}")

            total_records += len(obs_records)

    print(f"\nPapers with records: {papers_with_records}")
    print(f"Total records created: {total_records}")
    if parse_errors:
        print(f"Parse errors: {parse_errors}")
    if validation_errors:
        print(f"Validation errors: {validation_errors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

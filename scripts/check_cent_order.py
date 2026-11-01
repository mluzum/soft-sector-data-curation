"""Audit: do curated centrality.classes follow HEPData table order?

Values live positionally in raw tables, so classes MUST be listed in raw
table order (central-first AND peripheral-first tables both occur, e.g.
STAR data29a/data30 list 70-80% .. 0-5%).  Sorting classes without
reordering values silently mislabels every bin (found Aug 2026 via the
JETSCAPE cross-validation: v2 labels were swapped central<->peripheral).

For each curated doc with >1 cent class and a resolvable raw table whose
independent variable is Centrality: compare the listed order against the
raw order (normalized).  Report mismatches.
"""
import glob
import os
import re
import sys

import yaml

sys.path.insert(0, '.')


def norm_cent(s):
    return re.sub(r'\s+', '', str(s).replace('\t', ' ').replace('%', ''))


def parse_range(s):
    """'0-5' (or '0.0-5.0') -> (0.0, 5.0); None if not a range."""
    parts = norm_cent(s).split('-')
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def parse_center(s):
    """'2.5' -> 2.5; None if not a plain number."""
    try:
        return float(norm_cent(s))
    except ValueError:
        return None


def main():
    n_checked = n_ok = n_mismatch = n_unresolvable = 0
    mismatches = []
    for fp in sorted(glob.glob('data/**/*.yaml', recursive=True)):
        try:
            docs = list(yaml.safe_load_all(open(fp)))
        except Exception:
            continue
        for di, d in enumerate(docs):
            if not d:
                continue
            classes = (d.get('centrality', {}) or {}).get('classes', []) or []
            if len(classes) < 2:
                continue
            ins = str(d.get('inspire_id', ''))
            hf = d.get('hepdata_file', '')
            ver = d.get('version', 1)
            if not hf or ins.startswith('PHOBOS'):
                n_unresolvable += 1
                continue
            cands = glob.glob(os.path.join(
                'hepdata_raw/records', f'ins{ins}', 'HEPData-*-yaml', hf))
            if not cands:
                n_unresolvable += 1
                continue
            try:
                raw = yaml.safe_load(open(cands[0]))
            except Exception:
                n_unresolvable += 1
                continue
            raw_cent = None
            for v in raw.get('independent_variables', []):
                hn = ((v.get('header') or {}).get('name')) or ''
                if 'ent' in hn:
                    vals = []
                    for x in v.get('values', []):
                        if isinstance(x, dict) and 'value' in x:
                            vals.append(x['value'])
                        elif isinstance(x, dict) and 'low' in x:
                            vals.append(f"{x['low']}-{x['high']}")
                    raw_cent = vals
                    break
            if not raw_cent or len(raw_cent) != len(classes):
                n_unresolvable += 1
                continue
            n_checked += 1
            # Positional correspondence: each raw entry must be the same
            # bin as the listed class (range==range, or center-in-range).
            # This holds regardless of ascending/descending table order.
            # Raw plain numbers that fit NEITHER pattern (Npart centers,
            # eta/pT values under a *Centrality-named header, ...) are
            # unresolvable by construction - skip, don't flag.
            # A raw point outside its listed class IS a mismatch when every raw
            # entry is a plausible percentile (0-100): reversed or invented
            # classes of point-valued tables (STAR 2017211, ATLAS 1728935)
            # used to pass as "unresolvable".
            centers = [parse_center(v) for v in raw_cent]
            percentiles = all(c is not None and 0.0 <= c <= 100.0 for c in centers)
            bad = []
            resolvable = True
            for pos, (raw_v, listed) in enumerate(zip(raw_cent, classes)):
                rr = parse_range(raw_v)
                lr = parse_range(listed)
                if rr is not None and lr is not None:
                    if abs(rr[0] - lr[0]) > 1e-9 or abs(rr[1] - lr[1]) > 1e-9:
                        bad.append(pos)
                elif rr is None:
                    rc = parse_center(raw_v)
                    if rc is None or lr is None:
                        resolvable = False
                        break
                    if not (lr[0] - 1e-9 <= rc <= lr[1] + 1e-9):
                        if not percentiles:
                            resolvable = False
                            break
                        bad.append(pos)
                else:
                    bad.append(pos)
            if not resolvable:
                n_unresolvable += 1
                continue
            if not bad:
                n_ok += 1
            else:
                n_mismatch += 1
                mismatches.append((fp, di, d.get('hepdata_file'),
                                   [str(c) for c in classes[:3]],
                                   [str(c) for c in raw_cent[:3]]))
    print(f"checked={n_checked} ok={n_ok} mismatch={n_mismatch} "
          f"unresolvable={n_unresolvable}", flush=True)
    for fp, di, hf, listed, raw in mismatches:
        print(f"  MISMATCH {fp} doc{di} {hf}: listed={listed} raw={raw}",
              flush=True)
    return mismatches


if __name__ == '__main__':
    main()

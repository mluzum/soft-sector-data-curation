#!/usr/bin/env python3
"""Cross-field metadata consistency checker for curated records.

Usage:
    python scripts/validate_metadata.py                     # all types
    python scripts/validate_metadata.py --type flow_vn      # one type
    python scripts/validate_metadata.py --fix               # auto-fix certain issues
    python scripts/validate_metadata.py --summary-only      # counts only
"""
import sys, os, argparse, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import defaultdict, Counter
from analysis.dispatcher import iter_records

KNOWN_ISSUES = [
    'integrated_with_multi_bins',
    'missing_kr_eta_with_cent_eta',
    'missing_all_eta_y_cent',
    'duplicate_cent_classes',
    'integrated_with_single_bin_as_range',
    'missing_kr_pT',
    'diff_eta_no_particles',
    'dimensionality_none',
    'kr_eta_from_y_no_eta',
    'kr_eta_from_y_no_y',
]


# A column header that pre-multiplies the value: "(%)", "(\\%)", "*100",
# "*1000", "*10^4".  HEPData stores these with units:none, so a record
# without `value_scale` silently compares x100 too large.
# Only an EXPLICIT multiplier counts.  A bare \*\d+ is far too loose:
# "SQRT(MEAN(V2**2)) / MEAN(V2)" is exponentiation, not a scale.
# Order matters -- 1000 before 100.
_SCALE_MARKER = re.compile(
    r'\(\s*\\?%\s*\)|\*\s*(?:1000|100|10\s*\^\s*\{?\s*\d)')
# ...but a '%' attached to a centrality/acceptance word is a label, not a
# unit -- "[F] 00 < Centrality (%) < 15" has v2 values of 0.04-0.13, i.e.
# already fractions (verified against ins585347).
_CENT_LABEL = re.compile(r'centrality|\bcent\b|\bMB\b|<\s*Centrality', re.I)


_EXPLICIT_MULTIPLIER = re.compile(r'\*\s*(?:1000|100|10\s*\^\s*\{?\s*\d)')
_PERCENT_UNIT = re.compile(r'\(\s*\\?%\s*\)')


def check_value_scale(doc):
    """value_scale must be present exactly when the header pre-scales."""
    if doc.get('value_scale') is not None:
        return False
    # source_column holds the verbatim header; value_type is the cleaned
    # label.  Check both so a stale value_type is still caught.
    for field in ('source_column', 'value_type'):
        s = str(doc.get(field) or '')
        if not s:
            continue
        # An explicit numeric multiplier is decisive even when the header
        # also names a centrality bin: "A*100 (cent 0-5%)" is scaled, while
        # "[F] 00 < Centrality (%) < 15" is not.
        if _EXPLICIT_MULTIPLIER.search(s):
            return True
        # A bare "(%)" unit is a scale only when it is not attached to a
        # centrality/acceptance word (verified against ins585347, whose
        # "Centrality (%)" v2 values of 0.04-0.13 are already fractions).
        if _PERCENT_UNIT.search(s) and not _CENT_LABEL.search(s):
            return True
    return False


def check_centrality_classes(doc):
    """Delegate to the shared definition in analysis.dispatcher.

    One definition, so the validator and the pipeline can never disagree
    about which records have a bogus centrality field.
    """
    from analysis.dispatcher import centrality_classes_suspicious
    classes = (doc.get('centrality') or {}).get('classes') or []
    return centrality_classes_suspicious(classes)


def classify(doc, observable_type=None):
    """Return a dict of issue -> bool for a single record."""
    ot = observable_type or doc.get('observable_type', '?')
    dim = doc.get('dimensionality', None)
    kr = doc.get('kinematic_range', {})
    if not isinstance(kr, dict):
        kr = {}
    cent = doc.get('centrality', {})
    if not isinstance(cent, dict):
        cent = {}
    classes = cent.get('classes', []) or []
    cent_eta = cent.get('eta', []) or []
    method = doc.get('method')
    particles = doc.get('particles')

    eta = kr.get('eta', {})
    has_kr_eta = isinstance(eta, dict) and (eta.get('min') is not None or eta.get('bins', []) or eta.get('points', []) or eta.get('bin_ranges', []))
    y = kr.get('y', {})
    has_kr_y = isinstance(y, dict) and (y.get('min') is not None or y.get('bins', []) or y.get('points', []) or y.get('bin_ranges', []))
    pT = kr.get('pT', {})
    has_kr_pT = isinstance(pT, dict) and (pT.get('min') is not None or pT.get('bins', []) or pT.get('points', []) or pT.get('bin_ranges', []))
    pT_bins = pT.get('bins', []) if isinstance(pT, dict) else []
    has_kr_pT_bins = len(pT_bins) > 2

    result = {}

    result['duplicate_cent_classes'] = len(classes) != len(set(classes))
    result['integrated_with_multi_bins'] = (dim == 'integrated' and has_kr_pT_bins)
    result['integrated_with_single_bin_as_range'] = (dim == 'integrated' and len(pT_bins) == 2)
    result['missing_kr_eta_with_cent_eta'] = (not has_kr_eta and bool(cent_eta))
    result['missing_all_eta_y_cent'] = (not has_kr_eta and not has_kr_y and not cent_eta)
    result['missing_kr_pT'] = (not has_kr_pT and ot != 'mean_pT')
    result['dimensionality_none'] = (dim is None)
    result['diff_eta_no_particles'] = (
        method == 'cumulant' and dim == 'differential_eta'
        and not particles and doc.get('cumulant_order') == 2
        and not doc.get('eta_gap')
    )

    return result


def fix_record(doc, issue, dry_run=True):
    """Attempt to auto-fix known issues. Returns modified doc or None."""
    if issue == 'integrated_with_multi_bins':
        if dry_run:
            return None
        doc['dimensionality'] = 'differential_pT'
        return doc
    return None


def hfmt(s, w=30):
    return str(s)[:w] if s else '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default=None, help='Observable type subdirectory')
    ap.add_argument('--fix', action='store_true', help='Apply auto-fixes')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    if args.type:
        dirs = [os.path.join('data', args.type)]
    else:
        dirs = [os.path.join('data', d) for d in sorted(os.listdir('data'))
                if os.path.isdir(os.path.join('data', d))]

    issues = defaultdict(list)
    total = 0
    fixable = 0
    fix_candidates = []

    for d in dirs:
        for fpath, doc in iter_records(d):
            total += 1
            ot = doc.get('observable_type', os.path.basename(d))
            result = classify(doc, observable_type=ot)
            if check_value_scale(doc):
                result['value_scale_missing'] = True
            if check_centrality_classes(doc):
                result['centrality_classes_suspicious'] = True
            for iss, flagged in result.items():
                if flagged:
                    hf = doc.get('hepdata_file', '?')
                    vt = doc.get('value_type', '?')
                    issues[iss].append((fpath, ot, hf, vt))
                    if iss == 'integrated_with_multi_bins':
                        fixable += 1
                        fix_candidates.append((fpath, doc))

    if args.summary_only:
        print(f'Scanned {total} records')
        for iss, items in sorted(issues.items()):
            by_ot = Counter(ot for _, ot, _, _ in items)
            ot_str = ', '.join(f'{ot}: {n}' for ot, n in sorted(by_ot.items()))
            print(f'  {iss}: {len(items)} ({ot_str})')
        return

    print(f'Scanned {total} records\n')

    for iss, items in sorted(issues.items()):
        by_ot = Counter(ot for _, ot, _, _ in items)
        ot_str = ', '.join(f'{ot}: {n}' for ot, n in sorted(by_ot.items()))
        print(f'{"="*60}')
        print(f'{iss}: {len(items)} records')
        print(f'  Types: {ot_str}')
        print(f'{"="*60}')
        for fpath, ot, hf, vt in items[:8]:
            print(f'  {fpath.split("/")[-1][:30]:30s} [{ot:25s}] {hfmt(hf,20):20s} {hfmt(vt,20)}')
        if len(items) > 8:
            print(f'  ... ({len(items)-8} more)')
        print()

    if args.fix and fix_candidates:
        print(f'\nApplying auto-fixes to {len(fix_candidates)} records...')
        for fpath, doc in fix_candidates:
            fix_record(doc, 'integrated_with_multi_bins', dry_run=False)
        print('Done.')
        print(f'  {fixable} records fixed (set dimensionality=differential_pT)')


if __name__ == '__main__':
    main()

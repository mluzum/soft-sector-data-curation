"""Populate `value_scale` / `source_column` and de-decorate `value_type`.

The importer set `value_type` from the HEPData dependent-variable column
header, verbatim.  That header routinely carries decoration that is not part
of the observable:

  * a VALUE SCALE  -- "v_3(TPC) (%)", "v_3*100 (TPC)", "v_3^2*10^4",
                     "v_1(\\%)".  These tables are x100 (or x10^4) relative
                     to the dimensionless fraction the simulation produces.
                     HEPData reports `units: none`, so nothing downstream can
                     notice: a percent table compared against a fraction is
                     wrong by 100x and still looks like a plausible number.
  * a CENTRALITY LABEL -- "v2 in centrality 0-5%".  Not a scale.
  * a detector/method tag -- "v_3(TPC) (%)", "v_2(C=EP)".

Verified against the raw data for ins1210062: the "v_3(TPC) (%)" columns run
1.4-6.95 while the plain "v_3 (TPC)" columns run 0.05-0.067, so the (%) tables
really are x100.

This script:
  1. reads every raw HEPData table and classifies each dependent-variable
     header as SCALED / SCALE_UNKNOWN / PLAIN;
  2. matches headers to curated records by (inspire_id, hepdata_file,
     value_type), which is the only provenance the record carries;
  3. writes `value_scale` (the factor to canonical units) and `source_column`
     (the verbatim header) into the record;
  4. rewrites `value_type` to a normalised label, moving the original text
     to `source_column` so nothing is lost;
  5. records SCALE_UNKNOWN columns (e.g. a bare "Scaled") WITHOUT guessing a
     factor, and flags them for manual attention.

Run:  python scripts/fix_value_scale.py --dry-run
      python scripts/fix_value_scale.py
"""
import argparse
import glob
import json
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------------------------------------------------------------- scales --
# Ordered most specific first.  Each entry: (regex, factor, canonical note).
SCALE_RULES = [
    (re.compile(r'\*\s*10\s*\^\s*\{?\s*6'), 1e-6, '10^6'),
    (re.compile(r'\*\s*10\s*\^\s*\{?\s*4'), 1e-4, '10^4'),
    (re.compile(r'\*\s*10\s*\^\s*\{?\s*3'), 1e-3, '10^3'),
    (re.compile(r'\*\s*1000'), 1e-3, '1000'),
    (re.compile(r'\*\s*100\b'), 1e-2, '100'),
]

# A column labelled only "Scaled" / "scaled_V2" states no factor.  It is
# reported, never guessed.
# No trailing \b: "scaled_V2" has an underscore, which is a word
# character, so \bscaled\b does not match it.
UNKNOWN_SCALE = re.compile(r'(?<![A-Za-z])scaled', re.I)

# Only these rules express an explicit numeric multiplier (rule 1 above).
MULTIPLIER_ONLY = re.compile(r'\*')
# A percent unit applied to the VALUE, e.g. "v_3(TPC) (%)", "v_1(\%)".
PERCENT_UNIT = re.compile(r'\(\s*\\?%\s*\)|\(\s*percent', re.I)

# Centrality labels containing '%' are NOT value scales.
CENTRALITY_LABEL = re.compile(
    r'centrality|\bcent\b|\b\d+\s*-\s*\d+\s*%|\bMB\b|<\s*Centrality', re.I)


def classify_header(name):
    """Return (factor_or_None, kind, canonical_label).

    factor is the multiplier that converts the tabulated number to the
    canonical (dimensionless) unit, or None when no scale is stated.
    kind is 'plain' | 'scaled' | 'scale_unknown' | 'centrality_label'.
    """
    s = str(name or '')
    # ORDER MATTERS.
    #  1. An explicit numeric multiplier is decisive, even when the header
    #     also carries a centrality label ("A*100 (cent 0-5%)").
    #  2. A bare "Scaled" states no factor -> flag, never guess.
    #  3. A "(%)" unit is a value scale ONLY if it is not attached to a
    #     centrality/acceptance word.  "[F] 00 < Centrality (%) < 15" is a
    #     label; its v2 values are 0.04-0.13, i.e. already fractions
    #     (verified against ins585347 raw tables), so scaling it by 0.01
    #     would be a 100x error introduced by the cleaner.
    for rx, factor, _tag in SCALE_RULES:
        if MULTIPLIER_ONLY.search(rx.pattern) and rx.search(s):
            return factor, 'scaled', s
    if UNKNOWN_SCALE.search(s):
        return None, 'scale_unknown', s
    if re.search(r'%', s):
        if CENTRALITY_LABEL.search(s):
            return None, 'centrality_label', s
        if PERCENT_UNIT.search(s):
            return 1e-2, 'scaled', s
        return None, 'scaled_but_unquantified', s
    return None, 'plain', s


# ----------------------------------------------------------- value_type ---
# Decoration to strip from `value_type` once it is carried in
# `source_column`.  Conservative: only remove things that are provably not
# part of the observable's identity.
_DECOR = [
    re.compile(r'\(\s*\\?%\s*\)'),                       # (%)
    re.compile(r'\*\s*10\s*\^\s*\{?\s*\d+\}?'),          # *10^4
    re.compile(r'\*\s*\d+\s*'),                          # *100
    re.compile(r'\bin\s+centrality\b.*$', re.I),         # "in centrality 0-5%"
    re.compile(r'\(cent[^)]*\)', re.I),                  # (cent 0-5%)
    re.compile(r'\bfor\s+\$?p\^\{?ref\}.*$', re.I),      # "for p_ref = 0.5-2 GeV"
    re.compile(r'\s+'),                                  # collapse spaces
]


def normalise_value_type(vt, header):
    """Strip scale/centrality decoration, keeping the observable label."""
    s = str(vt or header or '').strip()
    for rx in _DECOR:
        s = rx.sub(' ', s)
    s = s.strip().strip('$').strip()
    s = re.sub(r'\s{2,}', ' ', s)
    return s or str(vt or header or '').strip()


# ------------------------------------------------------------------ main --
def load_headers():
    """(inspire_id, table_basename) -> [(header_name, factor, kind)]."""
    out = {}
    for path in sorted(glob.glob('hepdata_raw/records/*/HEPData-*/*.yaml')):
        parts = path.split(os.sep)
        ins = parts[2]
        table = os.path.basename(path)
        rows = []
        try:
            # submission.yaml is a multi-document stream; a single-document
            # load raises and would silently drop the whole table.
            docs_t = list(yaml.safe_load_all(open(path)))
        except Exception:
            continue
        for t in docs_t:
          if not isinstance(t, dict):
            continue
          for dv in (t.get('dependent_variables') or []):
              h = (dv.get('header') or {})
              name = h.get('name')
              if not name:
                  continue
              factor, kind, canon = classify_header(name)
              rows.append((str(name), factor, kind))
        if rows:
            out[(ins, table)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', default='data_validation/value_scale_report.json')
    args = ap.parse_args()

    headers = load_headers()
    print(f'raw tables with dependent variables: {len(headers)}')

    n_scaled = n_unknown = n_plain = 0
    changed_files = 0
    flagged = []
    report = {}

    for path in sorted(glob.glob(os.path.join(args.data, '*', '*.yaml'))):
        docs = list(yaml.safe_load_all(open(path)))
        touched = False
        base = os.path.basename(path)
        for i, doc in enumerate(docs):
            if not doc:
                continue
            ins = str(doc.get('inspire_id') or '').replace('ins', '')
            hf = str(doc.get('hepdata_file') or '')
            table = os.path.basename(hf) if hf else None
            cand = headers.get((f'ins{ins}', table)) or headers.get((ins, table))
            if not cand:
                continue
            vt = str(doc.get('value_type') or '')
            # Match the record to the column it was built from.
            match = None
            for name, factor, kind in cand:
                if name == vt:
                    match = (name, factor, kind)
                    break
            if match is None:
                # Some importers mangled whitespace/case; fall back to a
                # normalised comparison rather than guessing.
                # Raw headers are LaTeX-wrapped ("$v_{2}(\\%)$") while the
                # importer stored the record with the $ signs stripped, so
                # compare on a normalised form rather than exact text.
                def _n(x):
                    x = str(x)
                    x = x.replace('$', '').replace('\\left', '')
                    x = x.replace('\\right', '').replace('{', '')
                    x = x.replace('}', '')
                    return re.sub(r'\s+', '', x).lower()
                for name, factor, kind in cand:
                    if _n(name) == _n(vt):
                        match = (name, factor, kind)
                        break
            if match is None:
                continue
            name, factor, kind = match
            report.setdefault(base, {})[str(i)] = {
                'value_type': vt, 'header': name, 'kind': kind,
                'factor': factor,
            }
            if kind == 'scaled':
                n_scaled += 1
            elif kind == 'scale_unknown':
                n_unknown += 1
                flagged.append((path, i, vt))
            else:
                n_plain += 1

            # ---- edits ----
            new_scale = None
            if kind == 'scaled' and factor is not None:
                new_scale = factor
            new_vt = normalise_value_type(vt, name)
            if new_scale is not None or new_vt != vt:
                if not args.dry_run:
                    if new_scale is not None:
                        doc['value_scale'] = new_scale
                    if new_vt != vt:
                        doc['source_column'] = name
                        doc['value_type'] = new_vt
                    else:
                        doc['source_column'] = name
                touched = True
        if touched and not args.dry_run:
            with open(path, 'w') as fh:
                yaml.safe_dump_all(docs, fh, sort_keys=True,
                                   default_flow_style=False,
                                   allow_unicode=True)
            changed_files += 1

    print(f'\nscaled columns matched        : {n_scaled}')
    print(f'plain columns matched         : {n_plain}')
    print(f'scale stated but UNKNOWN      : {n_unknown}   (not guessed)')
    print(f'files rewritten               : {changed_files}'
          f'{"   (dry run)" if args.dry_run else ""}')
    if flagged:
        print('\nSCALE UNKNOWN -- need a human to read the paper:')
        for p, i, vt in flagged[:40]:
            print(f'   {p} doc{i}: {vt!r}')
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w') as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f'\nreport -> {args.report}')


if __name__ == '__main__':
    main()
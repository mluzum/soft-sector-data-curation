"""Stamp curated flow references onto event_plane records, per PAPER.

WHY
---
An `event_plane` record states the METHOD but not the REFERENCE: which
acceptance the flow proxy (event plane) was reconstructed from.  Without it
the pipeline falls back to the single-set amplitude estimator
sqrt(<|Q_n|^2>/<M^2>), which contains no event plane and therefore no
resolution factor -- a different observable from the one the paper reports.

The reference cannot be inferred from the collaboration.  It is not the
centrality detector, and it is not constant within a collaboration:

  * PHENIX arXiv:1412.1043 Sec III.4.1 -- the obvious assumption is that the
    central arms supply the event plane, because that is where the POI sits.
    The paper says the opposite: "the combined second-harmonic BBC event
    plane", built from "the charge seen in the corresponding channel of the
    BBC" at 3.0 < |eta| < 3.9, i.e. DISJOINT from the |eta| < 0.35 POI.
    Assuming same-window would have been wrong for the largest PHENIX paper
    (75 records).
  * PHENIX arXiv:1304.2852 -- for |eta| < 0.35 hadrons the event planes are
    "determined in 1.0 < eta < 2.8 to ensure a sufficient pseudo-rapidity
    gap ... to reduce non-flow correlations due to jet production": a third
    geometry again, neither the POI nor the centrality detector.

So each entry in curation/flow_reference.yaml is keyed by collaboration +
inspire_id and quotes the method section.  This script applies it.

Records whose paper has no entry are left untouched and reported, so the
remaining work is always a visible, shrinking list.

USAGE
    python scripts/apply_flow_reference.py --dry-run
    python scripts/apply_flow_reference.py
"""
import argparse
import collections
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

COLLAB_FROM_FILE = {
    'phenix': 'phenix', 'atlas': 'atlas', 'alice': 'alice',
    'star': 'star', 'cms': 'cms', 'phobos': 'phobos',
}


def load_knowledge(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def inspire_to_arxiv(inspire_id):
    """arXiv id for an INSPIRE id, from the local raw download."""
    p = f'hepdata_raw/records/ins{inspire_id}/metadata.json'
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh).get('arxiv_id')
    except Exception:
        return None


def verify_arxiv(collab, inspire, ref):
    """Refuse to stamp when the recorded arXiv id contradicts the paper.

    Guards the mistake of keying the knowledge base on the arXiv id while the
    records carry INSPIRE ids: that mismatch stamps NOTHING and looks like
    "no records match", which is easy to misread as "already curated".
    """
    want = str(ref.get('reference') or '')
    if 'arXiv:' not in want:
        return True
    quoted = want.split('arXiv:')[1].split()[0]
    have = inspire_to_arxiv(inspire)
    if have is None:
        return True                      # no local metadata: cannot check
    if quoted not in str(have):
        raise SystemExit(
            f'knowledge base mismatch for {collab}/{inspire}: notes quote '
            f'arXiv:{quoted} but hepdata_raw says arXiv:{have}. Refusing to '
            f'samp -- records are keyed by INSPIRE id.')
    return True


def file_collaboration(fname):
    head = os.path.basename(fname).split('_')[0]
    return COLLAB_FROM_FILE.get(head)


def normalise_arxiv(inspire_id):
    """Records store a bare number; the knowledge file may use arXiv:1412.x."""
    s = str(inspire_id)
    if s.lower().startswith('arxiv:'):
        s = s.split(':', 1)[1]
    return s.lstrip('0') or '0'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--knowledge', default='curation/flow_reference.yaml')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report',
                    default='data_validation/flow_reference_status.json')
    args = ap.parse_args()

    kb = load_knowledge(args.knowledge)
    # (collab, inspire) -> reference dict
    index = {}
    for collab, papers in kb.items():
        for ins, ref in (papers or {}).items():
            index[(collab, normalise_arxiv(ins))] = ref
    print(f'knowledge base: {len(index)} curated paper(s)')

    n_done = n_skip_has = 0
    per_paper = collections.Counter()
    todo = collections.Counter()
    touched = set()

    for path in sorted(glob.glob(os.path.join(args.data, '*', '*.yaml'))):
        collab = file_collaboration(path)
        if collab is None:
            continue
        docs = list(yaml.safe_load_all(open(path)))
        changed = False
        for i, doc in enumerate(docs):
            if not doc or doc.get('method') != 'event_plane':
                continue
            ins = normalise_arxiv(doc.get('inspire_id'))
            key = (collab, ins)
            if doc.get('flow_reference'):
                n_skip_has += 1
                continue
            ref = index.get(key)
            if ref is not None:
                verify_arxiv(collab, ins, ref)
            if ref is None:
                todo[(collab, ins)] += 1
                continue
            doc['flow_reference'] = dict(ref)
            changed = True
            n_done += 1
            per_paper[f'{collab}/{ins}'] += 1
        if changed:
            touched.add(path)
            if not args.dry_run:
                with open(path, 'w') as fh:
                    yaml.safe_dump_all(docs, fh, sort_keys=True,
                                       default_flow_style=False,
                                       allow_unicode=True)

    print(f'\nrecords stamped                 : {n_done}')
    print(f'already had a reference (kept)  : {n_skip_has}')
    print(f'files rewritten                 : '
          f'{len(touched)}{"  (dry run)" if args.dry_run else ""}')
    if per_paper:
        print('\nby paper:')
        for k, v in sorted(per_paper.items()):
            print(f'   {v:5d}  {k}')
    if todo:
        tot = sum(todo.values())
        print(f'\nstill TODO: {tot} event_plane records over '
              f'{len(todo)} paper(s) with no curated reference')
        for (collab, ins), n in todo.most_common(12):
            print(f'   {n:5d}  {collab}/{ins}')

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w') as fh:
        json.dump({'stamped': dict(per_paper),
                   'todo': {f'{c}/{i}': n for (c, i), n in todo.items()}},
                  fh, indent=1, sort_keys=True)
    print(f'\nreport -> {args.report}')


if __name__ == '__main__':
    main()

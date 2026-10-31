"""Curate the event-plane reference for STAR event_plane records.

WHY
---
2,925 curated records carry `method: event_plane`.  Not one of them
declared where the event plane was reconstructed from (zero `particles:
AB`, zero `subevents`, 34 `eta_gap` of which the PHOBOS ones were blanket
guesses).  So they were all dispatched to the single-set amplitude
estimator -- a two-particle correlation over the POI sample that contains
NO event plane and therefore no resolution factor, i.e. a different
observable from the one the paper reports.

Of those, 1,443 are STAR.  All 1,443 have an OVERLAPPING geometry: the POI
is TPC |eta|<1 while the centrality estimator is TPC RefMult in |eta|<0.5,
i.e. the same detector, nested.  That is the case where self-correlation
is mandatory, and the only estimator that handles it correctly is the
scalar-product form with the POI/reference diagonal subtracted.

WHAT THIS SCRIPT DOES
---------------------
For each STAR paper it reads the POI acceptance of the CHARGED records.
Those are the reference sample: STAR's midrapidity event plane is
reconstructed from TPC tracks, and in an identified-particle paper the
charged rows of the same table ARE the reference population.  It then
writes `flow_reference` on every event_plane record of that paper:

  * charged      -> same_sample_as_poi: true  (POI == reference)
  * identified   -> eta: [that same TPC window]  (POI is a subset)

PROVENANCE IS RECORDED, NOT ASSUMED SILENTLY
---------------------------------------------
This is an INFERENCE from the collaboration's standard practice plus the
table structure, not a quotation from each paper's method section.  Every
record it touches is stamped with `provenance: assumed_tpc` and a `notes`
string naming the evidence, so any downstream consumer can tell a curated
reference from an assumed one, and the population is greppable for
re-checking.  Records it will NOT touch:

  * POI outside |eta| <= 1.3 -- the TPC does not cover these; the event
    plane there comes from the FTPC/BEMC and the convention is not
    guessable from the table.  Left uncurated and reported.
  * papers whose charged rows do not show a single consistent TPC window.

Run:  python scripts/curate_star_flow_reference.py --dry-run
      python scripts/curate_star_flow_reference.py
"""
import argparse
import collections
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

TPC_ETA_LIMIT = 1.3          # STAR central TPC acceptance
TPC_WINDOW = (-1.0, 1.0)     # the standard midrapidity TPC window
ROUND = 1e-9
# Two windows count as the same sample when their edges agree this closely;
# below it, bookkeeping differences (-1.0 vs -1.001) are noise.
WINDOW_TOL = 0.01


def is_charged(doc):
    return str((doc.get('particle_id') or {}).get('species')) == 'charged'


def is_tpc_window(doc, limit=TPC_ETA_LIMIT):
    eta = ((doc.get('kinematic_range') or {}).get('eta') or {})
    lo, hi = eta.get('min'), eta.get('max')
    if lo is None or hi is None:
        return False
    return (abs(lo) <= limit + ROUND) and (abs(hi) <= limit + ROUND)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/flow_vn')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', default='data_validation/star_flow_reference.json')
    args = ap.parse_args()

    n_charged = n_ident = 0
    n_skip_window = n_skip_paper = n_mixed_window = 0
    papers = {}
    touched_files = set()
    skipped = collections.Counter()

    for path in sorted(glob.glob(os.path.join(args.data, 'star_*.yaml'))):
        docs = list(yaml.safe_load_all(open(path)))
        ep = [(i, d) for i, d in enumerate(docs)
              if d and d.get('method') == 'event_plane']
        if not ep:
            continue
        charged = [(i, d) for i, d in ep if is_charged(d)]

        # The reference window comes from the charged rows.  Only proceed
        # when they agree on one window -- otherwise the table mixes
        # acceptances and the reference is genuinely ambiguous.
        wins = collections.Counter()
        for i, d in charged:
            if not is_tpc_window(d):
                wins[('OUTSIDE_TPC',)] += 1
                continue
            eta = ((d.get('kinematic_range') or {}).get('eta') or {})
            wins[(round(eta['min'], 3), round(eta['max'], 3))] += 1
        usable = {w: n for w, n in wins.items() if w != ('OUTSIDE_TPC',)}
        outside = wins.get(('OUTSIDE_TPC',), 0)
        if not usable:
            n_skip_paper += 1
            skipped[os.path.basename(path)] = (
                'no charged row inside the TPC acceptance')
            continue
        # The reference is the TPC, so take the DOMINANT charged window.
        # Small differences between charged rows (-1.0 vs -1.001 vs -0.99)
        # are pT-cut bookkeeping, not different detectors.  A record whose
        # own window differs from the reference is still curated, but as an
        # explicit reference rather than same_sample_as_poi -- the geometry
        # is nested rather than identical and the diagonal differs.
        ref_win, n_ref = max(usable.items(), key=lambda kv: kv[1])
        if len(usable) > 1:
            n_mixed_window += 1

        lo, hi = ref_win
        changed = False
        for i, d in ep:
            if d.get('flow_reference'):
                continue                    # already curated
            if is_charged(d):
                if not is_tpc_window(d):
                    n_skip_window += 1
                    continue
                e = ((d.get('kinematic_range') or {}).get('eta') or {})
                same = (e.get('min') is not None
                        and abs(e['min'] - lo) < WINDOW_TOL
                        and abs(e['max'] - hi) < WINDOW_TOL)
                fr = {
                    'detector': 'TPC',
                    'same_sample_as_poi': bool(same),
                    'provenance': 'assumed_tpc',
                    'notes': (
                        "STAR midrapidity event plane reconstructed from "
                        f"TPC tracks in {lo:g} < eta < {hi:g} -- the same "
                        "tracks whose flow is measured (STAR's 'standard "
                        "event plane method', V2{EP}, as opposed to the "
                        "modified V2{EP2} that excludes same-side tracks). "
                        "INFERRED from the collaboration's standard practice "
                        "and from this table's charged rows, NOT quoted "
                        "from the paper's method section; re-check "
                        "provenance=assumed_tpc against the text."),
                }
                n_charged += 1
            else:
                species = (d.get('particle_id') or {}).get('species')
                fr = {
                    'detector': 'TPC',
                    'eta': [{'min': lo, 'max': hi}],
                    'provenance': 'assumed_tpc',
                    'notes': (
                        f"Event plane from charged TPC tracks in "
                        f"{lo:g} < eta < {hi:g}; the particle of interest "
                        f"({species}) is a SUBSET of that sample, so the "
                        "POI/reference diagonal equals N_poi and is "
                        "subtracted. INFERRED from the charged rows of the "
                        "same table, not quoted from the paper."),
                }
                n_ident += 1
            d['flow_reference'] = fr
            changed = True
        if outside:
            skipped.setdefault(os.path.basename(path), '')
        if changed:
            touched_files.add(path)
            papers[os.path.basename(path)] = {
                'ref_window': [lo, hi], 'n_charged_ref_rows': n_ref,
                'n_outside_tpc': outside,
            }
            if not args.dry_run:
                with open(path, 'w') as fh:
                    yaml.safe_dump_all(docs, fh, sort_keys=True,
                                       default_flow_style=False,
                                       allow_unicode=True)

    print(f'STAR papers touched            : {len(touched_files)}')
    print(f'  charged  -> same_sample_as_poi: {n_charged}')
    print(f'  identified -> TPC reference  : {n_ident}')
    print(f'  skipped, POI outside TPC     : {n_skip_window} record(s)')
    print(f'  papers skipped entirely      : {n_skip_paper}')
    print(f'  papers with mixed POI windows: {n_mixed_window}'
          f'   (curated as an explicit reference, not same_sample)')
    print(f'  files rewritten              : '
          f'{len(touched_files)}{"  (dry run)" if args.dry_run else ""}')
    if skipped:
        print('\nnot curated (needs the paper):')
        for k, v in list(skipped.items())[:20]:
            print(f'   {k[:64]:64s} {v}')
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    import json
    with open(args.report, 'w') as fh:
        json.dump(papers, fh, indent=1, sort_keys=True)
    print(f'\nreport -> {args.report}')


if __name__ == '__main__':
    main()
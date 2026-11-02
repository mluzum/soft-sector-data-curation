#!/usr/bin/env python3
"""Curate STAR v2 from two- and four-particle correlations in Au+Au at 130 GeV, INSPIRE 587825.

C. Adler et al. (STAR), "Elliptic flow from two- and four-particle
correlations in Au+Au collisions at sqrt(sNN) = 130 GeV", PRC 66, 034904
(2002), arXiv:nucl-ex/0206001.

From the paper (Sec. II, Table I):
- charged tracks |eta| < 1.3, 0.1 < pT < 2.0 GeV/c (the flow vector);
  particles over a wider range in eta and pT correlated with it;
- centrality: eight bins of n_ch/n_max, n_ch in |eta| < 0.75; Table I maps
  them to 53-77, 41-53, 31-41, 24-31, 16-24, 10-16, 5-10, 0-5% of the total
  geometric cross section;
- methods: the standard event plane (full events, resolution from random
  subevents), the second- and fourth-order cumulants (generating
  function), the four-subevent fourth-order cumulant;
- errors: statistical (the event-plane v2 vs centrality carries the
  estimated non-flow systematic in its lower error).

Records (public): v2 vs centrality from the event plane and v2{4} (Table
7), v2(eta) and v2(pT) per class from the event plane, v2{2} and v2{4}
(Tables 8a-h, 9a-h; points with edges at the midpoints), the event plane
computed with the scalar-product estimator (TPC reference, self-pairs
removed), the cumulants with reference particles 0.1 < pT < 2.0.  Staged
(--staged OUT): the four-subevent v2{4} (four subevents: no computation).
Not curated: the event-plane correlations and resolutions (Tables 1-2),
the method studies (3-6), the quarter-event v2, the averages over bins
2-7 (Tables 10-11: weighting not stated), ratios (12-13) and v2/epsilon
(14).  The 5 old records had percent classes such as 2.0-7.0% (the
n_ch/n_max axis read as percent), a 200 GeV file name and mixed methods.

Run from the repository root:
    python scripts/curate_star_v2_cumulants_130.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins587825/HEPData-ins587825-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Adler et al. (STAR), PRC 66, 034904 (2002), arXiv:nucl-ex/0206001'
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.75, 'max': 0.75}],
       'notes': 'Charged multiplicity n_ch in |eta| < 0.75 (n_ch/n_max bins, Table I).'}
# n_ch/n_max bin -> class (Table I), peripheral first as in the tables
CLASSES = ['53-77%', '41-53%', '31-41%', '24-31%', '16-24%', '10-16%', '5-10%', '0-5%']
FLOW_PT = {'min': 0.1, 'max': 2.0}
TPC = {'detector': 'TPC', 'eta': [{'min': -1.3, 'max': 1.3}], 'pT': dict(FLOW_PT),
       'notes': 'Full-event TPC event plane, |eta| < 1.3, 0.1 < pT < 2 GeV/c, the particle of interest removed.'}
OLD = 'data/flow_vn/star_flow_vn_elliptic_flow_from_two_and_AuAu_200gev.yaml'


def edges(pts, floor=None):
    mids = [round(0.5 * (a + b), 4) for a, b in zip(pts[:-1], pts[1:])]
    lo = pts[0] - (mids[0] - pts[0])
    return [round(lo if floor is None else max(floor, lo), 4)] + mids + [round(pts[-1] + (pts[-1] - mids[-1]), 4)]


def method(h):
    """(fields, description) from a column header; None for columns not curated."""
    if 'stand' in h or 'conventional $v_{2}$' == h.strip():
        return {'method': 'event_plane', 'flow_reference': dict(TPC)}, 'event plane (full events)'
    if '2nd order cumulant' in h:
        return {'method': 'cumulant', 'cumulant_order': 2, 'flow_reference': {'pT': dict(FLOW_PT)}}, 'v2{2}'
    if 'four-subevent' in h:
        return {'method': 'cumulant', 'cumulant_order': 4, 'staging_reason':
                'four-subevent v2{4}: no computation'}, 'v2{4}, four subevents'
    if '4' in h and 'cumulant' in h:
        return {'method': 'cumulant', 'cumulant_order': 4, 'flow_reference': {'pT': dict(FLOW_PT)}}, 'v2{4}'
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    plan = [('Table7.yaml', 'cent')] + [(f'Table8{c}.yaml', 'eta') for c in 'abcdefgh'] + \
           [(f'Table9{c}.yaml', 'pt') for c in 'abcdefgh']
    for hf, kind in plan:
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        cls = None
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            mf = method(h)
            for q in dv.get('qualifiers', []):
                m = re.search(r'(\d+)\\% - (\d+)\\%', str(q.get('value')))
                if m:
                    cls = f'{m.group(1)}-{m.group(2)}%'
            if mf is None:
                continue
            f, desc = mf
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            if not rows:
                continue
            if kind == 'cent':
                classes = [CLASSES[j] for j in rows]
                kr, dim, where = {'eta': {'min': -1.3, 'max': 1.3}, 'pT': dict(FLOW_PT)}, 'integrated', 'vs centrality'
                f = {k: v for k, v in f.items() if k != 'flow_reference' or f['method'] == 'event_plane'}
            else:
                pts = [float(v['value']) for v in iv['values']]
                e = edges(pts, floor=0.0 if kind == 'pt' else None)
                pairs = [[e[j], e[j + 1]] for j in rows]
                classes = [cls]
                if kind == 'eta':
                    kr = {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}, 'pT': dict(FLOW_PT)}
                    dim, where = 'differential_eta', f'vs eta, {cls}'
                else:
                    kr = {'eta': {'min': -1.3, 'max': 1.3}, 'pT': {'min': pairs[0][0], 'max': pairs[-1][1],
                                                                    'bin_ranges': pairs}}
                    dim, where = 'differential_pT', f'vs pT, {cls}'
            d = {'observable_type': 'flow_vn', 'inspire_id': '587825', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'v2 {desc} {where} ({hf})',
                 'collision_system': {'energy_gev': 130.0, 'reaction': 'AU AU --> CHARGED X'},
                 'centrality': {'classes': classes, **NCH}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': kr, 'harmonic': 2, 'value_type': h, 'dimensionality': dim,
                 **{k: v for k, v in f.items() if k != 'staging_reason'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. v2 of charged particles {where}, {desc}'
                                   + ('; computed with the scalar-product estimator' if f['method'] == 'event_plane' else
                                      '; reference particles |eta| < 1.3, 0.1 < pT < 2 GeV/c')
                                   + '; classes from Table I (n_ch/n_max bins). Errors: statistical'
                                   + (' (the lower error includes the estimated non-flow)' if kind == 'cent' and
                                      f['method'] == 'event_plane' else '') + '.'}
            if 'staging_reason' in f:
                d['staging_reason'] = f['staging_reason']
                staged.append(d)
            else:
                public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '587825' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_v2_cumulants_AuAu_130gev_ins587825.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['dimensionality'], d['method'], d.get('cumulant_order')) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate STAR identified-particle v2 in Au+Au at 130 GeV, INSPIRE 559609.

C. Adler et al. (STAR), "Identified particle elliptic flow in Au+Au
collisions at sqrt(sNN) = 130 GeV", PRL 87, 182301 (2001),
arXiv:nucl-ex/0107003.

From the paper:
- event plane from TPC tracks 0.1 < pT <= 2.0 GeV/c, |eta| < 1.0, each
  particle correlated with the plane of all the other particles; flow
  tracks |eta| < 1.3; pi, K, p + pbar from dE/dx at |y| <= 1.0, 0.8, 0.5;
- centrality from the charged multiplicity (|eta| < 0.75 in the STAR 130
  GeV analyses): 0-11, 11-45, 45-85% of the measured cross section;
  minimum bias 0-85%, v2 the multiplicity-weighted average over centrality;
- errors: statistical only (systematic 13% minimum bias; 20, 8, 22% for
  the three classes).

Records (public): v2(pT) of pi+ + pi- and p + pbar in 0-11, 11-45, 45-85%
(Fig. 3), TPC event plane computed with the scalar-product estimator, in
|eta| < 1.3 (rapidity windows are implemented for spectra only), pT points
with edges at the midpoints.  Staged (--staged OUT): minimum-bias v2(pT) of
pi, K, p (Fig. 2: averaged over centrality with multiplicity weights, the
bins not stated).  The 5 old records had no reference and the class
records the class name in value_type.

Run from the repository root:
    python scripts/curate_star_pid_v2_130.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins559609/HEPData-ins559609-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Adler et al. (STAR), PRL 87, 182301 (2001), arXiv:nucl-ex/0107003'
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.75, 'max': 0.75}],
       'notes': 'Charged multiplicity in the TPC, |eta| < 0.75 (STAR 130 GeV); classes of the measured cross section.'}
REF = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.1, 'max': 2.0},
       'notes': 'TPC event plane, 0.1 < pT < 2 GeV/c, |eta| < 1, the particle of interest removed.'}
SPECIES = {'data1.yaml': 'pi+ + pi-', 'data2.yaml': 'p + pbar', 'data3.yaml': 'K+ + K-', 'data4.yaml': 'pi+ + pi-',
           'data5.yaml': 'p + pbar'}
YCUT = {'pi+ + pi-': 1.0, 'K+ + K-': 0.8, 'p + pbar': 0.5}
OLD = 'data/flow_vn/star_flow_vn_identified_particle_elliptic_flow_AuAu_130gev.yaml'


def edges(pts):
    mids = [round(0.5 * (a + b), 4) for a, b in zip(pts[:-1], pts[1:])]
    return [round(max(0.0, pts[0] - (mids[0] - pts[0])), 4)] + mids + [round(pts[-1] + (pts[-1] - mids[-1]), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf, sp in SPECIES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            m = re.search(r'(\d+-\d+)%', h)
            cls = m.group(1) + '%' if m else '0-85%'
            # blanks are 'none'; the classes of one table use different (interleaved) pT points,
            # so each column takes edges from its own points
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            e = edges([pts[j] for j in rows])
            pairs = [[e[k], e[k + 1]] for k in range(len(rows))]
            d = {'observable_type': 'flow_vn', 'inspire_id': '559609', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'v2(pT) of {sp}, {cls} ({hf})',
                 'collision_system': {'energy_gev': 130.0, 'reaction': 'AU AU --> X'},
                 'centrality': {'classes': [cls], **NCH}, 'particle_id': {'species': sp},
                 'kinematic_range': {'eta': {'min': -1.3, 'max': 1.3},
                                     'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(REF), 'value_type': h,
                 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}. v2(pT) of {sp}, {cls}, |y| <= {YCUT[sp]:g} within the flow tracks '
                                    '|eta| < 1.3 (computed in |eta| < 1.3), TPC event plane, computed with the '
                                    'scalar-product estimator; pT points, edges at the midpoints. Errors: statistical '
                                    '(systematic in the paper).'}}
            if rows != list(range(len(pts))):
                d['table_rows'] = rows
            if not m:
                d['staging_reason'] = ('minimum bias (0-85%): v2 averaged over centrality with multiplicity weights, '
                                       'the centrality bins not stated')
                staged.append(d)
            else:
                public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '559609' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_pid_v2_AuAu_130gev_ins559609.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

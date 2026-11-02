#!/usr/bin/env python3
"""Curate STAR K0_S and Lambda v2 in Au+Au at 200 GeV, INSPIRE 620309.

J. Adams et al. (STAR), "Particle-type dependence of azimuthal anisotropy
and nuclear modification of particle production in Au+Au collisions at
sqrt(sNN) = 200 GeV", PRL 92, 052302 (2004), arXiv:nucl-ex/0306007.

From the paper:
- K0_S and Lambda + anti-Lambda at |y| < 1; the event plane from primary
  tracks 0.1 < pT < 2.0 GeV/c, |eta| < 1.0, the decay daughters excluded;
  centrality as in STAR 200 GeV (RefMult |eta| < 0.5);
- classes 0-80% (minimum bias), 0-5, 5-30, 30-70%;
- errors: statistical and point-to-point background systematic (0-80%),
  statistical (the classes).

Records (public): v2(pT) of K0_S, Lambda + Lambdabar and charged hadrons
in 0-80% (data_11-13) and of K0_S, Lambda in 0-5, 5-30, 30-70% (data_21,
22), TPC event plane computed with the scalar-product estimator, in |eta|
< 1 (rapidity windows are implemented for spectra only); pT points with
edges at the midpoints.  Not curated: R_CP (nuclear modification: out of
scope) and the quark-number scaled v2 (the same data).  The 2 old records
lacked the class tables.

Run from the repository root:  python scripts/curate_star_strange_v2_200.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins620309/HEPData-ins620309-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRL 92, 052302 (2004), arXiv:nucl-ex/0306007'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.5, 'max': 0.5}]}
REF = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.1, 'max': 2.0},
       'notes': 'TPC event plane, 0.1 < pT < 2 GeV/c, |eta| < 1, the decay daughters excluded.'}
SPECIES = {'K0s': 'K0_S', 'Lambda': 'Lambda + Lambdabar', 'h^+': 'charged'}
TABLES = {'data_11.yaml': None, 'data_12.yaml': None, 'data_13.yaml': None, 'data_21.yaml': 'classes',
          'data_22.yaml': 'classes'}
OLD = 'data/flow_vn/star_flow_vn_particle_dependence_of_azimuthal_anisotropy_AuAu_200gev.yaml'


def edges(pts):
    mids = [round(0.5 * (a + b), 4) for a, b in zip(pts[:-1], pts[1:])]
    return [round(max(0.0, pts[0] - (mids[0] - pts[0])), 4)] + mids + [round(pts[-1] + (pts[-1] - mids[-1]), 4)]


def main():
    public = []
    for hf, kind in TABLES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        sp_text = ''
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            sp_text = next((k for k in SPECIES if k in h), sp_text)
            sp = SPECIES[sp_text]
            if kind:
                m = re.search(r'(\d+-\d+)%', h)
                cls = m.group(1) + '%' if m else ['0-5%', '5-30%', '30-70%'][i]
            else:
                cls = '0-80%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            e = edges([float(iv['values'][j]['value']) for j in rows])
            pairs = [[e[k], e[k + 1]] for k in range(len(rows))]
            d = {'observable_type': 'flow_vn', 'inspire_id': '620309', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'v2(pT) of {sp}, {cls} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                 'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': sp},
                 'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                     'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(REF), 'value_type': h,
                 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}. v2(pT) of {sp}, {cls}, '
                                    + ('|eta| < 1' if sp == 'charged' else '|y| < 1 (computed in |eta| < 1)')
                                    + ', TPC event plane, computed with the scalar-product estimator; pT points, edges '
                                      'at the midpoints. Errors: '
                                    + ('statistical (non-flow systematic in the paper).' if kind else
                                       'statistical and point-to-point background systematic.')}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '620309' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_strange_v2_AuAu_200gev_ins620309.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [(d['particle_id']['species'], d['centrality']['classes'][0]) for d in public])


if __name__ == '__main__':
    main()

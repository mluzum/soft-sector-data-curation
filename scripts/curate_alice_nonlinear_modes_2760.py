#!/usr/bin/env python3
"""Curate ALICE v4, v5, v6 and non-linear flow modes in Pb-Pb at 2.76 TeV, INSPIRE 1599396.

S. Acharya et al. (ALICE), "Linear and non-linear flow modes in Pb-Pb
collisions at sqrt(sNN) = 2.76 TeV", PLB 773 (2017) 68, arXiv:1705.04377.

From the paper (Secs. 2-4):
- charged particles |eta| < 0.8, 0.2 < pT < 5.0 GeV/c; two subevents
  -0.8 < eta < -0.4 and 0.4 < eta < 0.8 (|Delta eta| > 0.8);
- v_n = v_n{2} of the two subevents; linear (v_n^L) and non-linear
  (v_{n,mk}) modes, rho_{n,mk} and chi_{n,mk} from multi-particle
  correlations of the subevents; Fig. 1 tests of the decomposition;
- V0 centrality (V0M).

Records (public): v4, v5, v6 vs centrality (fig2LeftA, MiddleA, RightA),
two-subevent v_n{2}.  Staged (--staged OUT): v_n^L, v_{n,mk}, rho, chi and
the Fig. 1 ratios (no computation; correlation_type
multi_harmonic_correlation).  The old records filed rho, chi and the modes
as v2{EP}, and v4, v5, v6 as single-acceptance v_n{2}.

Run from the repository root:
    python scripts/curate_alice_nonlinear_modes_2760.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1599396/HEPData-ins1599396-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PLB 773 (2017) 68, arXiv:1705.04377'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
PT = {'min': 0.2, 'max': 5.0}
SUB = {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.4}, 'pT': dict(PT)}},
       'B': {'kinematic_range': {'eta': {'min': 0.4, 'max': 0.8}, 'pT': dict(PT)}}}
OLD = ['data/flow_vn/alice_flow_vn_linear_and_non-linear_flow_modes_PbPb_2760gev.yaml',
       'data/flow_vn/alice_flow_vn_linear_and_nonlinear_flow_modes_PbPb_2760gev.yaml']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    public, staged = [], []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
        assert q.get('PT') == '0.2 - 5.0 GeV', (hf, q)
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
        name = dv['header']['name']
        d = {'inspire_id': '1599396', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f"{' '.join(x['description'].split())[:100]} ({name})",
             'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
             'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT)}}
        m = re.fullmatch(r'v(\d)2', name)
        if m:
            n = int(m.group(1))
            d = {'observable_type': 'flow_vn', **d, 'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2,
                 'particles': 'AB', 'eta_gap': 0.8,
                 'subevents': {k: {'kinematic_range': dict(v['kinematic_range'])} for k, v in SUB.items()},
                 'value_type': name, 'dimensionality': 'integrated',
                 'notes': {'value': f'{CITE}. v{n}{{2}} of charged particles, 0.2 < pT < 5.0 GeV/c, subevents '
                                    '-0.8 < eta < -0.4 and 0.4 < eta < 0.8. Errors: statistical and systematic.'}}
            public.append(d)
        else:
            harm = [int(c) for c in re.findall(r'\d', re.sub(r'Gap|overv.*|Tv.*', '', name))[:4]]
            d = {'observable_type': 'flow_harmonic_correlations', **d, 'observable': name,
                 'correlation_type': 'multi_harmonic_correlation', 'harmonics': harm or [0], 'value_type': name,
                 'notes': {'value': f'{CITE}. {name} ({" ".join(x["description"].split())[:90]}), charged '
                                    'particles, 0.2 < pT < 5.0 GeV/c, subevents -0.8 < eta < -0.4 and 0.4 < eta < 0.8.'},
                 'staging_reason': 'linear / non-linear flow modes, symmetry-plane correlations or their ratios: '
                                   'no computation'}
            staged.append(d)
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
    for f in OLD:
        if os.path.exists(f):
            os.remove(f)
    out = 'data/flow_vn/alice_flow_vn_v4_v6_PbPb_2760gev_ins1599396.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [d['harmonic'] for d in public])
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

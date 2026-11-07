#!/usr/bin/env python3
"""Curate ALICE identified-hadron v2 in high-multiplicity pp and p-Pb, INSPIRE 2848254.

ALICE, "Observation of partonic flow in proton-proton and proton-nucleus
collisions", Nature Commun. 17 (2026) 2585, arXiv:2411.09323.

From the paper (Results; Methods A.1-A.4):
- pp at 13 TeV, 0-0.07% V0M (V0A + V0C) with Nch(|eta| < 0.8, 0.2 < pT <
  3 GeV/c) > 25; p-Pb at 5.02 TeV, 0-20% V0A (the Pb-going side);
- pi+-, K+-, p + pbar at |eta| < 0.8 (TPC), K0S and Lambda + Lambdabar at
  |y| < 0.5;
- v2 from three sets of two-particle correlations (Eq. A.2): the POI with
  the FMD at forward (FMD1,2, 1.7 < eta < 5.1) and backward (FMD3, -3.1 <
  eta < -1.7) rapidity, and FMD1,2 with FMD3, v2 = sqrt(V(TPC, FMD1,2)
  V(TPC, FMD3) / V(FMD1,2, FMD3)); non-flow removed from each correlation
  by a template fit to low-multiplicity events (residual 5-7%);
- the tabulated pT is each bin's mean; errors stat and sys.

Records (public): flow_vn, method three_subevent_2pc (analysis/
three_sub_2pc.py; references all pT; a model without non-flow compares to
the template-fitted values).  p-Pb windows are in the lab frame
(collision_system frame lab, rapidity_shift -0.465, Pb toward +eta as for
ALICE's V0A-side Pb beam).  pT bins: midpoints between the tabulated
points.  The quoted |Delta eta| ranges (1.1-7.8 p-Pb, 1.1-6.4 pp) reflect
each period's FMD performance; the regions above are those of Methods
A.2 for both.

Run from the repository root:  python scripts/curate_alice_partonic_flow_small_systems.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2848254/HEPData-ins2848254-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'ALICE, Nature Commun. 17 (2026) 2585, arXiv:2411.09323'
OUT = 'data/flow_vn/alice_flow_vn_identified_3x2pc_pp_pPb_ins2848254.yaml'
SPECIES = {'pion': ('pi+ + pi-', 'eta'), 'kaon': ('K+ + K-', 'eta'), 'proton': ('p + pbar', 'eta'),
           'K0S': ('K0_S', 'y'), 'Lambda': ('Lambda + Lambdabar', 'y')}
REFS = {'B': {'eta': [{'min': 1.7, 'max': 5.1}], 'pT': {'min': 0.0, 'max': 100.0}},
        'C': {'eta': [{'min': -3.1, 'max': -1.7}], 'pT': {'min': 0.0, 'max': 100.0}}}
SYSTEMS = {
    'pPb': {'collision_system': {'energy_gev': 5020.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'lab',
                                 'rapidity_shift': -0.465, 'forward_beam': 'Pb'},
            'centrality': {'classes': ['0-20%'], 'estimator': {'name': 'V0A', 'observable': 'amplitude',
                                                               'detector': 'V0A'},
                           'eta': [{'min': 2.8, 'max': 5.1}], 'notes': 'V0A amplitude (Pb-going side), lab frame.'},
            'label': 'p-Pb 5.02 TeV 0-20% V0A'},
    'pp': {'collision_system': {'energy_gev': 13000.0, 'reaction': 'P P --> CHARGED X'},
           'centrality': {'classes': ['0-0.07%'], 'estimator': {'name': 'V0M', 'observable': 'amplitude',
                                                                'detector': 'V0'},
                          'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}],
                          'notes': 'V0M amplitude (high-multiplicity trigger), with Nch(|eta| < 0.8, 0.2 < pT < 3 '
                                   'GeV/c) > 25.'},
           'label': 'pp 13 TeV 0-0.07% V0M'}}


def main():
    subs = {x['data_file']: ' '.join(x['description'].split())
            for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x}
    public = []
    for hf, desc in subs.items():
        m = re.fullmatch(r'(pPb|pp)_v2(\w+)_pt\.yaml', hf)
        assert m, hf
        system, sp_key = m.groups()
        species, window = SPECIES[sp_key]
        sysd = SYSTEMS[system]
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        assert q['$|\\eta|$'] == '< 0.8', (hf, q)
        rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'sys')}, hf
        mids = [(a + b) / 2 for a, b in zip(pts, pts[1:])]
        edges = [round(2 * pts[0] - mids[0], 4)] + [round(x, 4) for x in mids] + [round(2 * pts[-1] - mids[-1], 4)]
        rb = [[edges[r], edges[r + 1]] for r in rows]
        kr = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}}
        if window == 'y':
            kr['y'] = {'min': -0.5, 'max': 0.5}
        d = {'observable_type': 'flow_vn', 'inspire_id': '2848254', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': desc,
             'collision_system': dict(sysd['collision_system']), 'centrality': dict(sysd['centrality']),
             'particle_id': {'species': species}, 'kinematic_range': kr,
             'harmonic': 2, 'method': 'three_subevent_2pc', 'dimensionality': 'differential_pT',
             'flow_reference': {'subevents': {k: {'eta': [dict(e) for e in v['eta']], 'pT': dict(v['pT'])}
                                              for k, v in REFS.items()},
                                'detector': 'FMD',
                                'notes': 'B: FMD1,2 (1.7 < eta < 5.1), C: FMD3 (-3.1 < eta < -1.7), all pT; '
                                         'v2 = sqrt(V(POI, B) V(POI, C) / V(B, C)).'},
             'value_type': 'v2'}
        if rows != list(range(len(pts))):
            d['table_rows'] = rows
        acc = '|y| < 0.5' if window == 'y' else '|eta| < 0.8'
        d['notes'] = {'value': f'{CITE}. v2(pT) of {species} ({acc}) from three sets of two-particle correlations '
                               f'with the FMD (Eq. A.2), non-flow removed by a template fit to low-multiplicity '
                               f'events; {sysd["label"]}; pT bins from the midpoints between the tabulated mean pT. '
                               'Errors: stat and sys.'}
        public.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')


if __name__ == '__main__':
    main()

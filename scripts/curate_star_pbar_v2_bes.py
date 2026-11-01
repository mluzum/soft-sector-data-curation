#!/usr/bin/env python3
"""Curate STAR proton and antiproton v2(pT) at 7.7-39 GeV, INSPIRE 1210463.

L. Adamczyk et al. (STAR), "Observation of an energy-dependent difference
in elliptic flow between particles and antiparticles in relativistic heavy
ion collisions", PRL 110, 142301 (2013), arXiv:1301.2347.

The analysis is that of the companion paper PRC 88, 014902 (INSPIRE
1210464, scripts/curate_star_bes_identified_v2.py): v2 from the eta-sub
event plane, the second-harmonic plane of TPC tracks (0.15 < pT < 5 GeV/c,
|eta| < 1, pT weighted) in the hemisphere opposite the particle, with a gap
of +-0.05; p and pbar identified by dE/dx and time of flight in |eta| < 1;
0-80% centrality from the multiplicity in |eta| < 0.5 (RefMult); errors:
statistical, systematic (asymmetric) and a global systematic.

Records: v2(pT) of p and of pbar at 7.7, 11.5, 27 and 39 GeV (Fig. 1),
0-80%; pT points as tabulated (0.1 GeV/c windows).  method event_plane
with the TPC eta-sub reference (both hemispheres 0.05 < |eta| < 1 here),
computed with the scalar-product estimator.  Not curated: the p - pbar
differences, v2 vs m_T - m_0 and its n_q scaling (the PRC data re-plotted),
the particle-antiparticle differences vs energy.

Run from the repository root:  python scripts/curate_star_pbar_v2_bes.py
"""
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1210463/HEPData-ins1210463-v1-yaml'
CITE = 'L. Adamczyk et al. (STAR), PRL 110, 142301 (2013), arXiv:1301.2347'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
WINDOW = 0.1


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        desc = ' '.join(x['description'].split())
        if not desc.startswith('The elliptic flow $v_{2}$ of protons and anti-protons'):
            continue
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        pts_all = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            if dv['header']['name'] != '$v_{2}$':
                continue                               # the p - pbar differences (lower panels)
            q = str(dv['qualifiers'][0]['value'])
            m = re.fullmatch(r'\$(p|\\bar\{p\})\$, ([0-9.]+) GeV', q)
            assert m and dv['header']['name'] == '$v_{2}$', (hf, q)
            sp = 'p' if m.group(1) == 'p' else 'pbar'
            e = float(m.group(2))
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pts = [pts_all[j] for j in rows]
            docs.append({
                'observable_type': 'flow_vn', 'inspire_id': '1210463', 'version': 1, 'hepdata_file': hf,
                'dv_index': i, 'hepdata_table_label': f'v2(pT) of {sp}, 0-80% Au+Au at {e:g} GeV (Fig. 1)',
                'collision_system': {'energy_gev': e, 'reaction': 'AU AU --> X'},
                'centrality': {'classes': ['0-80%'], **REFMULT},
                'particle_id': {'species': sp},
                'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                    'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                           'points': pts, 'point_window': WINDOW}},
                'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'dimensionality': 'differential_pT',
                'flow_reference': {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': -0.05}, {'min': 0.05, 'max': 1.0}],
                                   'pT': {'min': 0.15, 'max': 5.0},
                                   'notes': 'eta-sub event plane: TPC tracks of the hemisphere opposite the particle, '
                                            'gap +-0.05, pT weighted (both hemispheres here).'},
                'notes': {'value': (f'{CITE}, Fig. 1. v2(pT) of {sp} from the eta-sub event plane (the analysis of PRC '
                                    f'88, 014902), |eta| < 1; pT points ({WINDOW} GeV/c windows). Computed with the '
                                    'scalar-product estimator. Errors: statistical, systematic and a global '
                                    'systematic.')},
            })
    out = 'data/flow_vn/star_flow_vn_p_pbar_v2_AuAu_7.7-39gev_ins1210463.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', [(d['collision_system']['energy_gev'], d['particle_id']['species'],
                                           len(d['kinematic_range']['pT']['points'])) for d in docs])


if __name__ == '__main__':
    main()

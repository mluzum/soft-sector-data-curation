#!/usr/bin/env python3
"""Curate STAR v2, v3, v4 of identified hadrons in Au+Au at 200 GeV, INSPIRE 2051708.

M. S. Abdallah et al. (STAR), "Centrality and transverse momentum
dependence of higher-order flow harmonics of identified hadrons in Au+Au
collisions at sqrt(sNN) = 200 GeV", PRC 105, 064911 (2022),
arXiv:2203.07204.

From the paper (Sec. II):
- eta-sub event plane: TPC tracks 0.2 < pT < 2 GeV/c, east -1 <= eta <=
  -0.05 and west 0.05 <= eta <= 1, the hemisphere opposite the particle;
  each v_n relative to the same-order plane; resolution from the two
  subevents;
- pi+-, K+-, p, pbar (TPC + ToF), K0_S, Lambda, phi at mid-rapidity
  (|y| < 1.0); 0-80% and 0-10, 10-40, 40-80%; pT-integrated v_n (pT < 2
  GeV/c) vs centrality;
- errors: statistical and systematic.

Records: every v_n(pT) column of Figs. 1-4 (particles and antiparticles
separately, and summed, 0-80%; per class for Figs. 3-4) and the
integrated v_n of pi, K, p vs centrality (Fig. 8; classes from the
tabulated centres; 0.2 < pT < 2 GeV/c).  event_plane with the TPC eta-sub
reference as two subevents (scripts/eta_sub.py), computed as the two-subevent
v_n; |y| < 1 computed in |eta| < 1 (rapidity windows are implemented
for spectra only).  pT points (0.1 GeV/c windows).  The particle -
antiparticle differences (Delta v_n) are not curated.

Run from the repository root:  python scripts/curate_star_pid_vn_200.py
"""
import collections
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eta_sub import as_eta_sub  # noqa: E402

RAW = 'hepdata_raw/records/ins2051708/HEPData-ins2051708-v2-yaml'
CITE = 'M. S. Abdallah et al. (STAR), PRC 105, 064911 (2022), arXiv:2203.07204'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
REF = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': -0.05}, {'min': 0.05, 'max': 1.0}],
       'pT': {'min': 0.2, 'max': 2.0},
       'notes': 'eta-sub event plane: TPC tracks of the hemisphere opposite the particle (both hemispheres here).'}
CENTRES = {2.5: '0-5', 7.5: '5-10', 15.0: '10-20', 25.0: '20-30', 35.0: '30-40', 45.0: '40-50', 55.0: '50-60',
           65.0: '60-70'}
WINDOW = 0.1


def species(text):
    t = re.sub(r'[${}\\ ]', '', text)
    table = {'pi^+': 'pi+', 'pi^-': 'pi-', 'K^+': 'K+', 'k^+': 'K+', 'K^-': 'K-', 'k^-': 'K-', 'p^+': 'p',
             'p^-': 'pbar', 'pi^++pi^-': 'pi+ + pi-', 'K^++K^-': 'K+ + K-', 'p^++p^-': 'p + pbar',
             'K^0_S': 'K0_S', 'Lambda^++Lambda^+': 'Lambda + Lambdabar', 'phi': 'phi',
             'pi': 'pi+ + pi-', 'K': 'K+ + K-', 'p': 'p + pbar'}
    return table[t]


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            m = re.fullmatch(r'\$v_\{(\d)\}\$ \$(.+)\$', dv['header']['name'].strip())
            if not m:
                assert 'Delta' in dv['header']['name'], (hf, dv['header'])
                continue
            n, sp = int(m.group(1)), species(m.group(2))
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            label = f"{' '.join(x['description'].split())[:150]} | {dv['header']['name']}"
            if hf.startswith('Fig_8'):
                cents = [float(v['value']) for v in iv['values']]
                classes = [f'{CENTRES[cents[j]]}%' for j in rows]
                kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 2.0}}
                dim, what = 'integrated', f'pT-integrated v{n} of {sp} (pT < 2 GeV/c) vs centrality (Fig. 8)'
            else:
                q = str(dv['qualifiers'][0]['value'])
                cm = re.search(r'(\d+)\s*-\s*(\d+)', q)
                classes = [f'{cm.group(1)}-{cm.group(2)}%']
                pts = [float(iv['values'][j]['value']) for j in rows]
                kr = {'eta': {'min': -1.0, 'max': 1.0},
                      'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                             'points': pts, 'point_window': WINDOW}}
                dim, what = 'differential_pT', f'v{n}(pT) of {sp}, {classes[0]} ({x.get("location", "")})'
            docs.append({
                'observable_type': 'flow_vn', 'inspire_id': '2051708', 'version': 2, 'hepdata_file': hf,
                'dv_index': i, 'hepdata_table_label': label,
                'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                'centrality': {'classes': classes, **REFMULT},
                'particle_id': {'species': sp}, 'kinematic_range': kr,
                'harmonic': n, 'method': 'event_plane', 'value_type': dv['header']['name'], 'dimensionality': dim,
                'flow_reference': dict(REF),
                'notes': {'value': (f'{CITE}. {what}; eta-sub event plane (TPC, 0.2 < pT < 2 GeV/c, gap 0.1), '
                                    'computed with the scalar-product estimator; |y| < 1 computed in |eta| < 1'
                                    + ('' if dim == 'integrated' else f'; pT points ({WINDOW} GeV/c windows)')
                                    + '. Errors: statistical and systematic.')},
            })
            if dim == 'integrated' and rows != list(range(len(iv['values']))):
                docs[-1]['table_rows'] = rows
    out = 'data/flow_vn/star_flow_vn_centrality_and_transverse_momentum_dependence_200gev.yaml'
    for d in docs:                    # eta-sub event plane: two subevents (scripts/eta_sub.py)
        as_eta_sub(d, 0.05, 1.0, d['flow_reference']['pT'])
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['harmonic'], d['dimensionality']) for d in docs)))


if __name__ == '__main__':
    main()

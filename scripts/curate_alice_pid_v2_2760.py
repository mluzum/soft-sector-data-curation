#!/usr/bin/env python3
"""Curate ALICE v2(pT) of identified hadrons in Pb-Pb at 2.76 TeV, INSPIRE 1297103.

B. Abelev et al. (ALICE), "Elliptic flow of identified hadrons in Pb-Pb
collisions at sqrt(sNN) = 2.76 TeV", JHEP 06 (2015) 190, arXiv:1405.4632.

From the paper (Sec. 4):
- scalar product with three subevents: the particles of interest in the
  central barrel (B, |y| < 0.5), the reference Q-vectors of the VZERO-A
  (A, 2.8 < eta < 5.1) and VZERO-C (C, -3.7 < eta < -1.7), channel
  amplitudes as weights: v2 = sqrt(<<u.Q^A*/M_A>><<u.Q^C*/M_C>> /
  <Q^A.Q^C*/(M_A M_C)>) (Eq. 3); |Delta eta| > 0.9;
- pi+-, K+-, p+pbar from TPC and TOF, K0_S, phi, Lambda+Lambdabar,
  Xi+Xibar, Omega+Omegabar from their decays, all in |y| < 0.5;
- centrality from the VZERO amplitude (V0M), 0-5% to 60-70%;
- errors: statistical and systematic.

Records: v2(pT) per species and class (Figs. 4 and 12): method
scalar_product with the VZERO-A and VZERO-C reference (charged particles,
unit weights, both windows summed here; no pT threshold).  The rapidity
window |y| < 0.5 is not implemented for flow: computed in the tracking
acceptance |eta| < 0.8.  pT points as tabulated (0.1 GeV/c windows).  Not
curated: the K+- / K0_S average ("Average kaons", derived from two
curated series).

Run from the repository root:  python scripts/curate_alice_pid_v2_2760.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1297103/HEPData-ins1297103-v1-yaml'
CITE = 'B. Abelev et al. (ALICE), JHEP 06 (2015) 190, arXiv:1405.4632'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SPECIES = {'Charged pions': 'pi+ + pi-', 'Charged kaons': 'K+ + K-', 'K0s': 'K0_S',
           'Protons+AntiProtons': 'p + pbar', 'Phi': 'phi', 'Lambda+AntiLambda': 'Lambda + Lambdabar',
           'Xi+AntiXi': 'Xi- + Xibar+', 'Omega+AntiOmega': 'Omega- + Omegabar+'}
VZERO_REF = {'detector': 'VZERO', 'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}],
             'pT': {'min': 0.0, 'max': 10.0},
             'notes': 'Q-vectors of the VZERO-A (2.8 < eta < 5.1) and VZERO-C (-3.7 < eta < -1.7), channel amplitudes '
                      'as weights, three-subevent normalization (Eq. 3); here charged particles with unit weights, '
                      'both windows summed, no pT threshold.'}
WINDOW = 0.1


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        desc = ' '.join(x['description'].split())
        m = re.fullmatch(r'(.+) v2 as a function of pT for centrality: ([0-9]+-[0-9]+)%\.', desc)
        assert m, desc
        if m.group(1) == 'Average kaons':
            continue
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert len(t['dependent_variables']) == 1, hf
        dv = t['dependent_variables'][0]
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        assert q['Particle'] == m.group(1) and q['Centrality'] == f'{m.group(2)}%' and q['ABS(Y(RF=LAB))'] == '<0.5', (hf, q)
        sp, cls = SPECIES[m.group(1)], f'{m.group(2)}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pts = [float(t['independent_variables'][0]['values'][j]['value']) for j in rows]
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '1297103', 'version': 1, 'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': f'{desc} ({x.get("location", "")})',
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> X'},
            'centrality': {'classes': [cls], **V0M},
            'particle_id': {'species': sp},
            'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8},
                                'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                       'points': pts, 'point_window': WINDOW}},
            'harmonic': 2, 'method': 'scalar_product', 'value_type': 'v2', 'dimensionality': 'differential_pT',
            'flow_reference': dict(VZERO_REF),
            'notes': {'value': (f'{CITE}, {x.get("location", "")}. v2(pT) of {sp}, |y| < 0.5 within the tracking '
                                'acceptance |eta| < 0.8 (computed in |eta| < 0.8: rapidity windows are implemented '
                                'for spectra only); scalar product with the VZERO-A and VZERO-C Q-vectors, |Delta '
                                f'eta| > 0.9; pT points ({WINDOW} GeV/c windows). Errors: statistical and '
                                'systematic.')},
        })
    out = 'data/flow_vn/alice_flow_vn_identified_hadrons_PbPb_2760gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(d['particle_id']['species'] for d in docs)))


if __name__ == '__main__':
    main()

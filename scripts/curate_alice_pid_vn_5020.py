#!/usr/bin/env python3
"""Curate ALICE v2, v3, v4 of identified hadrons in Pb-Pb at 5.02 TeV, INSPIRE 1672822.

S. Acharya et al. (ALICE), "Anisotropic flow of identified particles in
Pb-Pb collisions at sqrt(sNN) = 5.02 TeV", JHEP 09 (2018) 006,
arXiv:1805.04390.

From the paper (Sec. 2):
- scalar product, v_n{SP} = <<u_n Q_n*>> / sqrt(<Q_n Q_n^A*><Q_n Q_n^B*> /
  <Q_n^A Q_n^B*>): the identified hadron's unit vector against the Q-vector
  of the V0A (2.8 < eta < 5.1, channel amplitudes as weights), normalized
  with the V0C (-3.7 < eta < -1.7) and the TPC tracks (|eta| < 0.8,
  0.2 < pT < 5 GeV/c); |Delta eta| > 2;
- pi+-, K+-, p+pbar, phi from ITS+TPC tracks (|eta| < 0.8), K0_S and
  Lambda+Lambdabar from V0 decays; all in |y| < 0.5; particles and
  antiparticles averaged;
- centrality from the V0 amplitude (V0M), 0-1% and 0-5% to 60-70%;
- errors: statistical and systematic.

Records: v_n(pT) per species, harmonic and class -- pi, K, p (v2, v3, v4),
K0_S and Lambda (v2, v3), phi (v2, 5-10% to 50-60%).  method
scalar_product with the V0A reference (charged particles, unit weights
here; no pT threshold).  The rapidity window |y| < 0.5 is not
implemented for flow: computed in the tracking acceptance |eta| < 0.8 (a
boost-invariant model is unaffected).  Not curated: the pT of the v_n
maximum vs centrality (derived).

Run from the repository root:  python scripts/curate_alice_pid_vn_5020.py
"""
import collections
import os

import yaml

RAW = 'hepdata_raw/records/ins1672822/HEPData-ins1672822-v1-yaml'
CITE = 'S. Acharya et al. (ALICE), JHEP 09 (2018) 006, arXiv:1805.04390'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SPECIES = {'CH_PIONS': 'pi+ + pi-', 'CH_KAONS': 'K+ + K-', 'PROTONS': 'p + pbar', 'K0S': 'K0_S',
           'LAMBDA': 'Lambda + Lambdabar', 'PHI': 'phi'}
V0A_REF = {'detector': 'V0A', 'eta': [{'min': 2.8, 'max': 5.1}], 'pT': {'min': 0.0, 'max': 10.0},
           'notes': 'Q_n of the V0A (2.8 < eta < 5.1), channel amplitudes as weights; normalized with the V0C and '
                    'the TPC (three subevents). Charged particles with unit weights and no pT threshold here.'}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        if iv['header']['name'] != 'PT':
            continue                                   # pT of the v_n maximum vs centrality
        assert len(t['dependent_variables']) == 1, hf
        dv = t['dependent_variables'][0]
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        sp = SPECIES[q['RE'].split('-->')[1].strip()]
        n = int(dv['header']['name'].split('_v')[1][0])
        assert q['ABS(Y)'].startswith('< 0.5'), (hf, q)
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(len(iv['values']))), hf
        edges = [float(iv['values'][0]['low'])] + [float(v['high']) for v in iv['values']]
        cls = q['CENTRALITY']
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '1672822', 'version': 1, 'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': f'v{n}(pT) of {sp}, {cls} Pb-Pb at 5.02 TeV ({x.get("location", "")})',
            'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> X'},
            'centrality': {'classes': [cls], **V0M},
            'particle_id': {'species': sp},
            'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8},
                                'pT': {'min': edges[0], 'max': edges[-1], 'bins': edges}},
            'harmonic': n, 'method': 'scalar_product', 'value_type': dv['header']['name'],
            'dimensionality': 'differential_pT',
            'flow_reference': dict(V0A_REF),
            'notes': {'value': (f'{CITE}, {x.get("location", "")}. v{n}(pT) of {sp} (particles and antiparticles '
                                'averaged), |y| < 0.5 within the tracking acceptance |eta| < 0.8 (computed in |eta| < '
                                '0.8: rapidity windows are implemented for spectra only); scalar product with the V0A '
                                'Q-vector, |Delta eta| > 2. Errors: statistical and systematic.')},
        })
    order = list(SPECIES.values())
    docs.sort(key=lambda d: (order.index(d['particle_id']['species']), d['harmonic'],
                             float(d['centrality']['classes'][0].split('-')[0]),
                             float(d['centrality']['classes'][0].split('-')[1].rstrip('%'))))
    out = 'data/flow_vn/alice_flow_vn_anisotropic_flow_of_identified_particles_PbPb_5020gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;',
          dict(collections.Counter((d['particle_id']['species'], d['harmonic']) for d in docs)))


if __name__ == '__main__':
    main()

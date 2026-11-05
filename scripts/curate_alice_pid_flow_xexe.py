#!/usr/bin/env python3
"""Curate ALICE v2, v3 of identified hadrons in Xe-Xe at 5.44 TeV, INSPIRE 1889989.

S. Acharya et al. (ALICE), "Anisotropic flow of identified hadrons in Xe-Xe
collisions at sqrt(sNN) = 5.44 TeV", JHEP 10 (2021) 152, arXiv:2107.10592.

From the paper (Sec. 2-3):
- pi+-, K+-, p + pbar, K0_S and Lambda + Lambdabar at |y| < 0.5 (TPC tracks
  |eta| < 0.8);
- v_n{SP, |Delta eta| > 2.0}: the scalar product with the V0A Q_n
  (2.8 < eta < 5.1, channel amplitudes), normalized with the V0C and the
  TPC (three subevents); encoded with a V0A flow_reference, as ALICE
  1672822;
- centrality from the V0 amplitude (V0M); v2 in 0-5, 5-10, 10-20 ... 50-60%,
  v3 in three classes;
- errors: stat and sys.

Records (public): v2(pT), v3(pT) per species and class.  Not curated: the
pT of the v2 maximum vs centrality (a derived quantity).

Run from the repository root:  python scripts/curate_alice_pid_flow_xexe.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1889989/HEPData-ins1889989-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), JHEP 10 (2021) 152, arXiv:2107.10592'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
V0A = {'detector': 'V0A', 'eta': [{'min': 2.8, 'max': 5.1}], 'pT': {'min': 0.0, 'max': 100.0},
       'notes': 'Q_n of the V0A (2.8 < eta < 5.1), channel amplitudes as weights; normalized with the V0C and the TPC '
                '(three subevents). Charged particles with unit weights and no pT threshold here.'}
SPECIES = {'pi': ('pi+ + pi-', 'PI+ PI-'), 'K': ('K+ + K-', 'K+ K-'), 'p': ('p + pbar', 'P PBAR'),
           'K0': ('K0_S', 'K0S'), 'L': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR')}
OUT = 'data/flow_vn/alice_flow_vn_pid_v2_v3_XeXe_5440gev_ins1889989.yaml'


def main():
    docs = []
    pat = re.compile(r'v([23])(pi|K0|K|p|L)_(\d+)\.yaml')
    files = [os.path.basename(p) for p in glob.glob(os.path.join(RAW, 'v*.yaml'))]
    tables = sorted((pat.fullmatch(f) for f in files if pat.fullmatch(f)),
                    key=lambda m: (m.group(1), m.group(2), int(m.group(3))))
    for m in tables:
        hf = m.group(0)
        path = os.path.join(RAW, hf)
        n, sp = int(m.group(1)), m.group(2)
        species, prod = SPECIES[sp]
        t = yaml.safe_load(open(path))
        dv = t['dependent_variables'][0]
        q = {x['name']: str(x['value']) for x in dv['qualifiers']}
        assert dv['header']['name'] == f'$v_{n}$' and q['SQRT(S)'] == '5440' and q['ABS(Y)'] == '< 0.5', (hf, q)
        assert [e['label'] for e in dv['values'][0]['errors']] == ['stat', 'sys'], hf
        cls = q['CENTRALITY']
        assert re.fullmatch(r'\d+-\d+%', cls), cls
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        d = {'observable_type': 'flow_vn', 'inspire_id': '1889989', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'v{n}(pT) of {species}, Xe-Xe 5.44 TeV, {cls} ({hf})',
             'collision_system': {'energy_gev': 5440.0, 'reaction': f'XE XE --> {prod} X'},
             'centrality': {'classes': [cls], **V0M},
             'particle_id': {'species': species},
             'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8},
                                 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
             'harmonic': n, 'method': 'scalar_product', 'value_type': f'v{n}', 'dimensionality': 'differential_pT',
             'flow_reference': dict(V0A)}
        d['notes'] = {'value': f'{CITE}. v{n}{{SP, |Delta eta| > 2.0}}(pT) of {species}, |y| < 0.5 (computed in |eta| < '
                               f'0.8), Xe-Xe 5.44 TeV {cls}; V0A reference. Errors: stat and sys.'}
        docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate CMS v2{4} and v2{SP} of charged hadrons, K0_S and Lambda in PbPb at 5.02 TeV, INSPIRE 2075415.

CMS Collaboration, "Strange hadron collectivity in pPb and PbPb collisions",
JHEP 05 (2023) 007, arXiv:2205.00080.

From the paper (Sec. 2-4):
- charged hadrons |eta| < 1, K0_S and Lambda + Lambdabar (p pi-, pbar pi+)
  at |y| < 1 (computed in |eta| < 1 here), v2(pT);
- v2{4}: the four-particle Q-cumulant of a particle of interest with
  three reference particles, charged hadrons |eta| < 2.4, 0.5 < pT < 5
  GeV (flow_reference eta and pT);
- v2{SP}: the scalar product with the HF Q-vectors (3 < |eta| < 5),
  |Delta eta| > 2; encoded as CMS 1511868 (the HF opposite the particle);
- PbPb centrality from the HF transverse energy: 5-10, 10-30, 30-50,
  50-80%;
- errors: stat and sys.

Records (public): v2{4}(pT) and v2{SP}(pT) per species and class (Fig. 2).
Not curated: v2{6}, v2{8}(pT) (differential v{6}, v{8} of a particle of
interest are not computed), v2{4, Sub} (a subevent four-particle cumulant
of a particle of interest, not computed), sigma/<v2> and the cumulant
ratios (derived).  The pPb tables (N_trk^offline classes):
scripts/curate_cms_strange_v2_ppb.py.

Run from the repository root:  python scripts/curate_cms_strange_v2_pbpb.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2075415/HEPData-ins2075415-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'CMS Collaboration, JHEP 05 (2023) 007, arXiv:2205.00080'
HF_EST = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
          'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
REF_CUMULANT = {'eta': [{'min': -2.4, 'max': 2.4}], 'pT': {'min': 0.5, 'max': 5.0},
                'notes': 'reference particles of the cumulants: charged hadrons |eta| < 2.4, 0.5 < pT < 5 GeV.'}
REF_HF = {'detector': 'HF', 'mirror': True, 'eta': [{'min': 3.0, 'max': 5.0}], 'pT': {'min': 0.0, 'max': 100.0},
          'notes': 'HF on the side opposite the particle, 3 < |eta| < 5 (E_T-weighted in the paper; unit weights '
                   'here). Reference pT: all particles.'}
TABLES = {1: ('charged', 'CHARGED', 4), 4: ('charged', 'CHARGED', 'SP'), 5: ('K0_S', 'K0S', 4), 8: ('K0_S', 'K0S', 'SP'),
          9: ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR', 4), 12: ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR', 'SP')}
OUT = 'data/flow_vn/cms_flow_vn_strange_v2_PbPb_5020gev_ins2075415.yaml'


def main():
    docs = []
    for n, (species, prod, meth) in TABLES.items():
        hf = f'table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        for k, dv in enumerate(t['dependent_variables']):
            q = {x['name']: str(x['value']) for x in dv['qualifiers']}
            assert dv['header']['name'] == ('$v_2\\{4\\}$' if meth == 4 else '$v_2\\{SP\\}$'), (hf, dv['header'])
            assert q['SQRT(S_NN)'] == '5020.0' and q['RE'].startswith('PB PB'), q
            assert [e['label'] for e in dv['values'][0]['errors']] == ['stat', 'sys'], hf
            cls = q['CENTRALITY']
            assert re.fullmatch(r'\d+-\d+%', cls), cls
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            label = 'v2{4}' if meth == 4 else 'v2{SP}'
            d = {'observable_type': 'flow_vn', 'inspire_id': '2075415', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': k, 'hepdata_table_label': f'{label}(pT) of {species}, PbPb 5.02 TeV, {cls} ({hf})',
                 'collision_system': {'energy_gev': 5020.0, 'reaction': f'PB PB --> {prod} X'},
                 'centrality': {'classes': [cls], **HF_EST},
                 'particle_id': {'species': species},
                 'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                     'pT': {'min': bins[rows[0]][0], 'max': bins[rows[-1]][1],
                                            'bin_ranges': [bins[j] for j in rows]}},
                 'harmonic': 2, 'value_type': 'v2', 'dimensionality': 'differential_pT'}
            if len(rows) < len(bins):
                d['table_rows'] = rows
            if meth == 4:
                d.update(method='cumulant', cumulant_order=4, flow_reference=dict(REF_CUMULANT))
            else:
                d.update(method='scalar_product', flow_reference=dict(REF_HF))
            d['notes'] = {'value': f'{CITE}, Fig. 2. {label}(pT) of {species}, '
                                   f'{"|eta|" if species == "charged" else "|y|"} < 1, PbPb 5.02 TeV {cls}; '
                                   + ('reference charged hadrons |eta| < 2.4, 0.5 < pT < 5 GeV' if meth == 4 else
                                      'HF reference, |Delta eta| > 2') + '. Errors: stat and sys.'}
            docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()

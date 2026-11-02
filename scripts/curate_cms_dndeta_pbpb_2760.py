#!/usr/bin/env python3
"""Curate CMS dNch/deta at eta = 0 vs centrality in Pb-Pb at 2.76 TeV, INSPIRE 919733.

S. Chatrchyan et al. (CMS), "Dependence on pseudorapidity and centrality
of charged hadron production in PbPb collisions at sqrt(sNN) = 2.76 TeV",
JHEP 08 (2011) 141, arXiv:1107.4800.

From the paper and tables: dNch/deta of charged hadrons from pixel
tracklets (all pT), centrality from the HF energy (2.9 < |eta| < 5.2);
Table 1 at eta = 0 (the eta bins of Table 2 are 0.5 wide: |eta| < 0.5
used here); errors as tabulated.

Records (public): dNch/deta at eta = 0 vs centrality, 0-5 ... 85-90%
(Table 1).  Not curated: dNch/deta per participant pair vs eta and vs
N_part (Tables 2, 3: Glauber).  The old record read Table 3 (per
participant pair) as dN/dy in 0-5%.

Run from the repository root:  python scripts/curate_cms_dndeta_pbpb_2760.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins919733/HEPData-ins919733-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Chatrchyan et al. (CMS), JHEP 08 (2011) 141, arXiv:1107.4800'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
OLD = 'data/particle_yields/cms_particle_yields_dependence_on_pseudorapidity_and_centrality_PbPb_2760gev.yaml'


def main():
    t = yaml.safe_load(open(os.path.join(RAW, 'Table1.yaml')))
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
    classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
    d = {'observable_type': 'particle_yields', 'inspire_id': '919733', 'version': VERSION, 'hepdata_file': 'Table1.yaml',
         'dv_index': 0, 'hepdata_table_label': 'dNch/deta at eta = 0 vs centrality (Table1.yaml)',
         'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
         'centrality': {'classes': classes, **HF}, 'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}},
         'observable': 'dNch_deta', 'value_type': 'dNch_deta',
         'notes': {'value': f'{CITE}, Table 1. dNch/deta of charged hadrons at eta = 0 (|eta| < 0.5 used) vs '
                            'centrality, HF centrality, pixel tracklets. Errors as tabulated.'}}
    if rows != list(range(len(iv['values']))):
        d['table_rows'] = rows
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '919733' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/particle_yields/cms_particle_yields_dNch_deta_PbPb_2760gev_ins919733.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all([d], fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: 1 record, {len(classes)} classes')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate ALICE dET/deta at mid-rapidity in Pb-Pb at 2.76 TeV, INSPIRE 1427723.

J. Adam et al. (ALICE), "Measurement of transverse energy at midrapidity
in Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", PRC 94, 034903 (2016),
arXiv:1603.04775.

From the paper and tables: dE_T/deta at |eta| < 0.6 from identified
single-particle tracks (cross-checked with the calorimeters); centrality
from the V0 amplitude (V0M); errors stat and sys (total).

Records (public): dE_T/deta vs centrality (Table 1, the E_T column): the
2.5% and 5% classes 0-2.5 ... 75-80% and, separately, 0-5% (overlapping
the first two).  Not curated: <N_part>, dE_T/deta per participant or
quark pair (Glauber), the Glauber table, and dE_T/deta / dNch/deta (Table
3, a ratio).  The old record read Table 3 (the ratio) as dN/dy.

Run from the repository root:  python scripts/curate_alice_et_pbpb_2760.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1427723/HEPData-ins1427723-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (ALICE), PRC 94, 034903 (2016), arXiv:1603.04775'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
OLD = 'data/particle_yields/alice_particle_yields_transverse_energy_PbPb_2760gev.yaml'


def main():
    t = yaml.safe_load(open(os.path.join(RAW, 'Table1.yaml')))
    iv = t['independent_variables'][0]
    i = [dv['header']['name'] for dv in t['dependent_variables']].index('$E_{T}$')
    classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
    public = []
    for rows in ([j for j, c in enumerate(classes) if c != '0-5%'], [classes.index('0-5%')]):
        public.append({
            'observable_type': 'transverse_energy', 'inspire_id': '1427723', 'version': VERSION,
            'hepdata_file': 'Table1.yaml', 'dv_index': i, 'hepdata_table_label': 'dE_T/deta vs centrality (Table1.yaml)',
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> X'},
            'centrality': {'classes': [classes[j] for j in rows], **V0M}, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -0.6, 'max': 0.6}, 'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'dET_deta', 'value_type': 'dET_deta', 'table_rows': rows,
            'notes': {'value': f'{CITE}. dE_T/deta at |eta| < 0.6 vs centrality (V0M), from identified single-particle '
                               'tracks (cross-checked with the calorimeters). Errors: stat and sys.'}})
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1427723' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/transverse_energy/alice_transverse_energy_dET_deta_PbPb_2760gev_ins1427723.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate ALICE dNch/deta(eta) per centrality class in p-Pb at 5.02 TeV, INSPIRE 1335350.

J. Adam et al. (ALICE), "Centrality dependence of particle production in
p-Pb collisions at sqrt(sNN) = 5.02 TeV", PRC 91 (2015) 064905,
arXiv:1412.6828.

From the paper (Sec. 2, 3, 5):
- 2013 run, 4 TeV protons on 1.58 TeV/nucleon Pb, the nucleon-nucleon cms
  moving by 0.465 toward the proton; eta is eta_lab, the Pb beam toward
  the A side, positive eta (the tables: in 0-5% dNch/deta is larger at
  positive eta, in 80-100% at negative eta);
- classes are percentiles of the visible (V0A and V0C) cross section, not
  corrected for trigger inefficiency (mostly in 80-100%); estimators CL1
  (SPD outer-layer clusters, |eta| < 1.4), V0A (2.8 < eta < 5.1, Pb side),
  V0M (V0A + V0C, -3.7 < eta < -1.7), ZNA (neutron calorimeter, Pb side);
- Tables 1-4: dNch/deta_lab of primary charged particles, -2 < eta < 2 in
  0.1 bins, classes 0-5, 5-10, 10-20, 20-40, 40-60, 60-80, 80-100%;
  errors stat and sys;
- Tables 5-10: Q_pPb, the nuclear modification factor with the hybrid
  <N_coll>.

Records (public, 21): dNch/deta(eta_lab) for the CL1, V0M and V0A classes
(dimensionality differential_eta; frame lab, Pb toward +eta,
rapidity_shift -0.465; the estimator windows as in the paper, lab).
Staged (--staged-out): the ZNA classes (spectator neutrons: models have no
spectators) and Q_pPb (nuclear modification, a ratio to pp with Glauber
<N_coll>; not computed).

Run from the repository root:
    python scripts/curate_alice_ppb_centrality_dndeta_5020.py [--staged-out <collection root>]
"""
import argparse
import copy
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1335350/HEPData-ins1335350-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (ALICE), PRC 91 (2015) 064905, arXiv:1412.6828'
SYSTEM = {'energy_gev': 5020.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'lab', 'rapidity_shift': -0.465,
          'forward_beam': 'Pb'}
ESTIMATORS = {
    'CL1': {'estimator': {'name': 'CL1', 'observable': 'Nch', 'detector': 'SPD'}, 'eta': [{'min': -1.4, 'max': 1.4}],
            'notes': 'clusters in the outer SPD layer, |eta_lab| < 1.4.'},
    'V0M': {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
            'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}],
            'notes': 'V0A + V0C amplitude (eta_lab; V0A on the Pb side).'},
    'V0A': {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
            'eta': [{'min': 2.8, 'max': 5.1}], 'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side).'},
    'ZNA': {'estimator': {'name': 'ZNA', 'observable': 'spectator_neutrons', 'detector': 'ZNA'},
            'notes': 'neutron zero-degree calorimeter on the Pb side.'}}
TABLES = {'Table1.yaml': 'CL1', 'Table2.yaml': 'V0M', 'Table3.yaml': 'V0A', 'Table4.yaml': 'ZNA'}
QPPB = {'Table5.yaml': 'CL1', 'Table6.yaml': 'V0M', 'Table7.yaml': 'V0A', 'Table8.yaml': 'ZNA',
        'Table9.yaml': 'hybrid (Pb side)', 'Table10.yaml': 'hybrid (multiplicity)'}
OUT = 'data/particle_yields/alice_particle_yields_dNch_deta_eta_pPb_5020gev_ins1335350.yaml'
STAGED_ZNA = 'centrality from the ZNA (spectator neutrons): models have no spectators'
STAGED_Q = 'Q_pPb, a nuclear modification factor (ratio to pp with the hybrid <N_coll>): not computed'


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def cls(dv):
    return dv['header']['name'].strip('()')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged-out', help='collection root for the staged records (data/<type>/ below it)')
    args = ap.parse_args()
    public, staged_y, staged_q = [], [], []
    for hf, est in TABLES.items():
        t = load(hf)
        iv = t['independent_variables'][0]['values']
        assert all(abs(a['high'] - b['low']) < 1e-9 for a, b in zip(iv, iv[1:])), hf
        edges = [round(float(iv[0]['low']), 6)] + [round(float(v['high']), 6) for v in iv]
        for i, dv in enumerate(t['dependent_variables']):
            c = cls(dv)
            assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
            d = {'observable_type': 'particle_yields', 'inspire_id': '1335350', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'dNch/deta_lab vs eta_lab, {est} {c} ({hf})',
                 'collision_system': dict(SYSTEM),
                 'centrality': {'classes': [c], **copy.deepcopy(ESTIMATORS[est])},
                 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': edges[0], 'max': edges[-1], 'bins': edges},
                                     'pT': {'min': 0.0, 'max': 100.0}},
                 'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'dimensionality': 'differential_eta',
                 'notes': {'value': f'{CITE}. dNch/deta_lab of primary charged particles vs eta_lab, {est} class {c} '
                                    '(percentiles of the visible cross section, not corrected for trigger '
                                    'inefficiency); lab frame, Pb toward +eta. Errors: stat and sys.'}}
            if est == 'ZNA':
                d['staging_reason'] = STAGED_ZNA
                staged_y.append(d)
            else:
                public.append(d)
    for hf, est in QPPB.items():
        t = load(hf)
        iv = t['independent_variables'][0]['values']
        pairs = [[float(v['low']), float(v['high'])] for v in iv]
        for i, dv in enumerate(t['dependent_variables']):
            c = cls(dv)
            staged_q.append({
                'observable_type': 'nuclear_modification', 'inspire_id': '1335350', 'version': VERSION,
                'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'Q_pPb vs pT, {est} {c} ({hf})',
                'collision_system': dict(SYSTEM),
                'centrality': {'classes': [c], **copy.deepcopy(ESTIMATORS.get(est, {'estimator': {'name': est}}))},
                'particle_id': {'species': 'charged'},
                'kinematic_range': {'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                'value_type': 'Q_pPb', 'notes': {'value': f'{CITE}. Q_pPb vs pT, {est} class {c}.'},
                'staging_reason': STAGED_Q})
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')
    if args.staged_out:
        for otype, docs in (('particle_yields', staged_y), ('nuclear_modification', staged_q)):
            path = os.path.join(args.staged_out, 'data', otype, f'alice_{otype}_pPb_5020gev_ins1335350.yaml')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as fh:
                yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
            print(f'{path}: {len(docs)} staged records')


if __name__ == '__main__':
    main()

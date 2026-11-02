#!/usr/bin/env python3
"""Curate ALICE dNch/deta in p-Pb and Pb-Pb at 5.02 TeV, INSPIRE 2070408.

S. Acharya et al. (ALICE), "System-size dependence of the charged-particle
pseudorapidity density at sqrt(sNN) = 5.02 TeV for pp, p-Pb, and Pb-Pb
collisions", PLB 845 (2023) 137730, arXiv:2204.10210.

From the paper (Sec. 2):
- dNch/deta from the SPD (|eta| < 2) and the FMD (-3.5 < eta < -1.8,
  1.8 < eta < 5.0), primary charged particles, all pT;
- Pb-Pb: centrality from V0M, 0-90%;
- p-Pb: both beam configurations, the one with Pb toward -eta mirrored, so
  that the results are in the laboratory frame with Pb toward +eta_lab
  (the nucleon-nucleon cms at y = -0.465 in this orientation); centrality
  from the V0 array on the Pb-going side (V0A, 2.8 < eta_lab < 5.1);
- pp: INEL > 0 events;
- errors: stat and sys (asymmetric).

Records (public): dNch/deta(eta) in Pb-Pb (Table 3, ten classes) and
dNch/deta(eta_lab) in p-Pb (Table 2, seven V0A classes; frame lab,
forward_beam Pb, rapidity_shift -0.465), differential_eta (computed on 3D
containers).  Staged (--staged OUT): pp INEL > 0 (no centrality class).
Not curated: the derived quantities (Tables 4-6: total N_ch, N_part,
energy density estimates).  The 2 old records gave one value per table
with a wrong class (80-100%, 80-90%).

Run from the repository root:
    python scripts/curate_alice_dndeta_systems_5020.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2070408/HEPData-ins2070408-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PLB 845 (2023) 137730, arXiv:2204.10210'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
V0A = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'}, 'eta': [{'min': 2.8, 'max': 5.1}],
       'notes': 'V0 array on the Pb-going side (V0A here, the Pb beam toward +eta_lab).'}
OLD = 'data/particle_yields/alice_particle_yields_system-size_dependence_of_the_charged-particle_PbPb_5020gev.yaml'


def records(hf, system, cent, label, extra_note=''):
    out = []
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        lo, hi = (float(x) for x in re.findall(r'[\d.]+', q['CENTRALITY IN PCT']))
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'particle_yields', 'inspire_id': '2070408', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': i, 'hepdata_table_label': f'dNch/deta(eta), {label}, {cls} ({hf})',
             'collision_system': dict(system), 'centrality': {'classes': [cls], **cent},
             'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs},
                                 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'dimensionality': 'differential_eta',
             'notes': {'value': f'{CITE}. dNch/deta of primary charged particles vs eta, {label}, {cls}'
                                f'{extra_note}. Errors: stat and sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public = records('tab3.yaml', {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'}, V0M, 'Pb-Pb', '; V0M')
    public += records('tab2.yaml', {'energy_gev': 5020.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'lab',
                                    'rapidity_shift': -0.465, 'forward_beam': 'Pb'}, V0A, 'p-Pb',
                      '; eta_lab with Pb toward +eta (both beam configurations, one mirrored); V0A centrality')
    t = yaml.safe_load(open(os.path.join(RAW, 'tab1.yaml')))
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
    staged = [{'observable_type': 'particle_yields', 'inspire_id': '2070408', 'version': VERSION,
               'hepdata_file': 'tab1.yaml', 'dv_index': 0, 'hepdata_table_label': 'dNch/deta(eta), pp INEL>0 (tab1.yaml)',
               'collision_system': {'energy_gev': 5023.0, 'reaction': 'P P --> CHARGED X'},
               'centrality': {'classes': ['INEL>0']}, 'particle_id': {'species': 'charged'},
               'kinematic_range': {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs},
                                   'pT': {'min': 0.0, 'max': 100.0}},
               'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'dimensionality': 'differential_eta',
               'notes': {'value': f'{CITE}. dNch/deta in pp INEL>0 at 5.02 TeV.'},
               'staging_reason': 'pp INEL > 0: no centrality class'}]
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '2070408' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/particle_yields/alice_particle_yields_dNch_deta_pPb_PbPb_5020gev_ins2070408.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

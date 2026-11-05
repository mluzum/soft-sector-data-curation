#!/usr/bin/env python3
"""Curate ALICE rho(v_n^2, [pT]) in Pb-Pb at 5.02 TeV and Xe-Xe at 5.44 TeV, INSPIRE 1966120.

S. Acharya et al. (ALICE), "Characterizing the initial conditions of
heavy-ion collisions at the LHC with mean transverse momentum and
anisotropic flow correlations", PLB 834 (2022) 137393, arXiv:2111.06106.

From the paper:
- rho(v_n^2, [pT]) = Cov(v_n^2, [pT]) / sqrt(Var(v_n^2) c_k) (Eq. 2),
  Var(v_n^2) = v_n{2}^4 - v_n{4}^4;
- charged tracks, 0.2 < pT < 3.0 GeV/c, |eta| < 0.8, three subevents
  A (-0.8 < eta < -0.4), B (|eta| < 0.4), C (0.4 < eta < 0.8): v_n from A
  and C (the cumulants and the flow pair of the covariance), [pT] from B
  (the pT particle and c_k); the generic framework with the 3-subevent
  method, corrected for acceptance and efficiency;
- centrality from the V0 amplitude (V0A 2.8 < eta < 5.1, V0C -3.7 < eta <
  -1.7); classes from Fig. 1 (HEPData gives the class centres): Pb-Pb
  0-5 ... 55-60%, Xe-Xe 0-5, 5-10, 10-15, 15-20, 20-30 ... 60-70% (rho_3:
  to 40-50%);
- errors: stat and full syst.

Records (public, 4): Table 1 rho_2 and rho_3 in Pb-Pb, Table 2 rho_2 and
Table 3 rho_3 in Xe-Xe.  Staged (--staged OUT): Table 4, rho(v_2^2, v_3^2,
[pT]) (Eq. 3: a three-observable correlator with SC(2,3), not computed).
Not curated: the model curves.

Run from the repository root:  python scripts/curate_alice_rho_vn_pt.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1966120/HEPData-ins1966120-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PLB 834 (2022) 137393, arXiv:2111.06106'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
PT = {'min': 0.2, 'max': 3.0}
SUB = {'A': (-0.8, -0.4), 'B': (-0.4, 0.4), 'C': (0.4, 0.8)}
PBPB = [f'{lo}-{lo + 5}%' for lo in range(0, 60, 5)]
XEXE = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%']
SYSTEMS = {'Pb': (5020.0, 'PB PB --> CHARGED X', 'Pb-Pb 5.02 TeV'), 'Xe': (5440.0, 'XE XE --> CHARGED X',
                                                                         'Xe-Xe 5.44 TeV')}
# (table, dv_index, harmonic, system, classes)
PUBLIC = [(1, 0, 2, 'Pb', PBPB), (1, 1, 3, 'Pb', PBPB), (2, 0, 2, 'Xe', XEXE), (3, 0, 3, 'Xe', XEXE[:7])]
OUT = 'data/flow_pT_correlations/alice_flow_pT_correlations_rho_PbPb_XeXe_5tev_ins1966120.yaml'


def load(n):
    return yaml.safe_load(open(os.path.join(RAW, f'Table_{n}.yaml')))


def centre(c):
    lo, hi = (float(x) for x in c.rstrip('%').split('-'))
    return 0.5 * (lo + hi)


def record(n, k, harm, sysname, classes, header):
    energy, reaction, label = SYSTEMS[sysname]
    return {'observable_type': 'flow_pT_correlations', 'inspire_id': '1966120', 'version': VERSION,
            'hepdata_file': f'Table_{n}.yaml', 'dv_index': k,
            'hepdata_table_label': f'{header} vs centrality, {label} (Table_{n}.yaml)',
            'collision_system': {'energy_gev': energy, 'reaction': reaction},
            'centrality': {'classes': list(classes), **V0M},
            'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT)},
            'particles': 'ABC',
            'subevents': {s: {'kinematic_range': {'eta': {'min': lo, 'max': hi}, 'pT': dict(PT)}}
                          for s, (lo, hi) in SUB.items()},
            'observable': 'rho_vn2_pT', 'value_type': 'rho_n', 'correlation_type': 'rho_vn2_pT',
            'harmonic': harm, 'covariance_method': 'three_subevent',
            'variance_terms': {'c4': 'two_subevent', 'c2': 'two_subevent'}, 'ck_window': 'B'}


def check(t, k, classes, sysname, header_part):
    cent = [float(v['value']) for v in t['independent_variables'][0]['values']]
    assert cent == [centre(c) for c in classes], (cent, classes)
    dv = t['dependent_variables'][k]
    q = {x['name']: str(x['value']) for x in dv['qualifiers']}
    assert q['RE'].startswith(f'{sysname} {sysname}') and q['ABS(ETA)'] == '$< 0.8$', q
    assert q['PT'].startswith('$0.2 - 3.0$'), q
    assert header_part in dv['header']['name'], dv['header']
    assert [e['label'] for e in dv['values'][0]['errors']] == ['stat', 'full syst.'], dv['values'][0]
    return dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged', help='write the staged records to this file')
    a = ap.parse_args()
    docs = []
    for n, k, harm, sysname, classes in PUBLIC:
        check(load(n), k, classes, sysname, f'v_{{{harm}}}^{{2}}')
        d = record(n, k, harm, sysname, classes, f'rho(v_{harm}^2, [pT])')
        d['notes'] = {'value': f'{CITE}, Fig. 1. rho(v_{harm}^2, [pT]) of charged particles, 0.2 < pT < 3 GeV/c, '
                               f'{SYSTEMS[sysname][2]}, V0M classes (HEPData: their centres). Three subevents: '
                               'flow from A (-0.8 < eta < -0.4) and C (0.4 < eta < 0.8), [pT] and c_k from B (|eta| '
                               '< 0.4); Var(v_n^2) = v_n{2}^4 - v_n{4}^4 from A and C. Errors: stat and full syst.'}
        docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')
    if a.staged:
        t = load(4)
        check(t, 0, PBPB[:10], 'Pb', 'v_{3}^{2}')
        d = record(4, 0, 2, 'Pb', PBPB[:10], 'rho(v_2^2, v_3^2, [pT])')
        d.update(observable='rho_v22_v32_pT', value_type='rho_mn', harmonics=[2, 3])
        d['staging_reason'] = 'rho(v_2^2, v_3^2, [pT]) (Eq. 3, with SC(2,3)): no computation'
        d['notes'] = {'value': f'{CITE}, Fig. 2. Errors: stat and full syst.'}
        with open(a.staged, 'w') as fh:
            yaml.safe_dump_all([d], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{a.staged}: 1 staged record')


if __name__ == '__main__':
    main()

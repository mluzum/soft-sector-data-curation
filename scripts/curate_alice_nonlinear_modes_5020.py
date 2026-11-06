#!/usr/bin/env python3
"""Curate ALICE v2 ... v9 and non-linear flow modes in Pb-Pb at 5.02 TeV, INSPIRE 1778342.

S. Acharya et al. (ALICE), "Higher harmonic non-linear flow modes of
charged hadrons in Pb-Pb collisions at sqrt(sNN) = 5.02 TeV", JHEP 05
(2020) 085, arXiv:2002.00633.

From the paper (abstract, Secs. 2-3, Fig. 2):
- charged particles 0.2 < pT < 5.0 GeV/c in 0.4 < |eta| < 0.8 (the
  HEPData qualifier says 0.2 - 0.5 GeV: a typo), two subevents
  -0.8 < eta < -0.4 and 0.4 < eta < 0.8;
- v_n = v_n{2} of the two subevents; v_{n,mk} = Re<V_n V_m* V_k*> /
  sqrt(<|V_m|^2 |V_k|^2>) (Eq. 6, A.1; three lower harmonics for v_{6,222},
  v_{7,223}) from multi-particle correlations, particle 1 in one subevent
  and the others in the other, the normalization's first half in one, the
  results of the two orientations averaged; rho_{n,mk} = v_{n,mk} / v_n
  (Eq. A.2); chi_{n,mk} = v_{n,mk} / sqrt(<|V_m|^2 |V_k|^2>) (Eq. A.3),
  except chi_{6,24} (Eq. A.4: a projection that removes the V_2^3 part);
- V0 centrality (V0M), 0-60%.

Records (public): v2 ... v9 vs centrality (v2.yaml ... v9.yaml), two-subevent
v_n{2}; v_{n,mk}, rho_{n,mk}, chi_{n,mk} as correlation_type mode_coupling
(analysis/mode_coupling.py; harmonics [n, m, k(, l)], quantity
v_mode_coupled | rho | chi; the two orientations pooled before the ratios).
Staged (--staged OUT): chi_{6,24} (Eq. A.4, not computed).  The old records
filed all 17 as v2{EP} with pT < 0.5 GeV/c.

Run from the repository root:
    python scripts/curate_alice_nonlinear_modes_5020.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1778342/HEPData-ins1778342-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), JHEP 05 (2020) 085, arXiv:2002.00633'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
PT = {'min': 0.2, 'max': 5.0}
SUB = {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.4}, 'pT': dict(PT)}},
       'B': {'kinematic_range': {'eta': {'min': 0.4, 'max': 0.8}, 'pT': dict(PT)}}}
PT_FIX = {'pT': "HEPData qualifier PT '0.2 - 0.5 GeV'; the paper (abstract, Sec. 3, Fig. 2): 0.2 < pT < 5.0 GeV/c"}
NAMES = {'v': 'non-linear flow mode v_{n,mk}', 'rho': 'symmetry-plane correlation rho_{n,mk}',
         'chi': 'non-linear response coefficient chi_{n,mk}'}
QUANTITY = {'v': 'v_mode_coupled', 'rho': 'rho', 'chi': 'chi'}
EQ = {'v': 'Re<V_n V_m* V_k*> / sqrt(<|V_m|^2 |V_k|^2>) (Eqs. 6, A.1)', 'rho': 'v_{n,mk} / v_n{2} (Eq. A.2)',
      'chi': 'v_{n,mk} / sqrt(<|V_m|^2 |V_k|^2>) (Eq. A.3)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    public, modes, staged = [], [], []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        assert q['PT'] == '0.2 - 0.5 GeV' and q['ABS(ETA)'].startswith('< 0.8'), (hf, q)
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
        name = dv['header']['name']
        d = {'inspire_id': '1778342', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f"{' '.join(x['description'].split())} ({name})",
             'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
             'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT)},
             'hepdata_corrections': dict(PT_FIX)}
        m = re.fullmatch(r'v(\d)', name)
        if m:
            n = int(m.group(1))
            d = {'observable_type': 'flow_vn', **d, 'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2,
                 'particles': 'AB', 'subevents': {k: {'kinematic_range': dict(v['kinematic_range'])}
                                                  for k, v in SUB.items()},
                 'value_type': name, 'dimensionality': 'integrated',
                 'notes': {'value': f'{CITE}. v{n}{{2}} of charged particles, 0.2 < pT < 5.0 GeV/c, subevents '
                                    '-0.8 < eta < -0.4 and 0.4 < eta < 0.8. Errors: statistical and systematic.'}}
            public.append(d)
        else:
            kind, digits = re.fullmatch(r'(v|rho|chi)(\d+)', name).groups()
            harms = [int(c) for c in digits]
            assert harms[0] == sum(harms[1:]), name
            if name == 'chi624':
                d = {'observable_type': 'flow_harmonic_correlations', **d, 'observable': name,
                     'correlation_type': 'multi_harmonic_correlation', 'harmonics': harms, 'value_type': name,
                     'notes': {'value': f'{CITE}. chi_{{6,24}} (Eq. A.4), charged particles, 0.2 < pT < 5.0 '
                                        'GeV/c, subevents -0.8 < eta < -0.4 and 0.4 < eta < 0.8.'},
                     'staging_reason': 'chi_{6,24} of Eq. A.4 (a projection removing the V_2^3 part of V_4): '
                                       'not computed (mode_coupling chi is v_{6,24} / sqrt(<|V_2|^2 |V_4|^2>))'}
                staged.append(d)
            else:
                d = {'observable_type': 'flow_harmonic_correlations', **d, 'observable': name,
                     'correlation_type': 'mode_coupling', 'harmonics': harms, 'quantity': QUANTITY[kind],
                     'particles': 'AB', 'subevents': {k: {'kinematic_range': dict(v['kinematic_range'])}
                                                      for k, v in SUB.items()},
                     'dimensionality': 'integrated', 'value_type': name,
                     'notes': {'value': f'{CITE}. {NAMES[kind]} ({name}) = {EQ[kind]}, charged particles, 0.2 < '
                                        'pT < 5.0 GeV/c, two subevents -0.8 < eta < -0.4 and 0.4 < eta < 0.8 '
                                        '(particle 1 in one, the others in the other; the two orientations '
                                        'averaged). Errors: statistical and systematic.'}}
                modes.append(d)
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
    old = 'data/flow_vn/alice_flow_vn_higher_harmonic_non-linear_flow_modes_PbPb_5020gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/alice_flow_vn_v2_v9_PbPb_5020gev_ins1778342.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [d['harmonic'] for d in public])
    out = 'data/flow_harmonic_correlations/alice_flow_harmonic_correlations_nonlinear_modes_PbPb_5020gev_ins1778342.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(modes, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(modes)} records;', [d['value_type'] for d in modes])
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

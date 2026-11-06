#!/usr/bin/env python3
"""Curate ALICE pT-dependent flow-vector fluctuations in Pb-Pb at 5.02 TeV, INSPIRE 2771093.

S. Acharya et al. (ALICE), "Systematic study of flow vector fluctuations in
sqrt(sNN) = 5.02 TeV Pb-Pb collisions", PRC 109 (2024) 065202,
arXiv:2403.15213.

From the paper (Secs. 2, 3):
- charged particles |eta| < 0.8; reference particles 0.2 < pT < 5.0 GeV/c;
  POI in the tabulated pT bins (0.2-4 GeV/c);
- v_n{2}(pT) / v_n[2](pT) (Eqs. 5, 7, 8), the factorization ratio r_n(pT^a,
  pT^t) (Eq. 9): two-particle correlations with |Delta eta| >= 0.8 (here two
  subevents -0.8 < eta < -0.4 and 0.4 < eta < 0.8, ALICE's generic-framework
  gap, both orientations pooled);
- A_2^f (Eq. 10) and M_2^f (Eq. 14): four-particle correlations with a
  subevent method and no gap (here eta < 0 and eta > 0, the positive
  harmonics in one subevent and the negative in the other);
- the lower limit of <cos n(Psi_n(pT) - Psi_n)>, sqrt((A^f + 1)/2) (Eq. 11),
  and the upper limit of the magnitude fluctuations, (v{2}/v[2]) /
  sqrt((A^f + 1)/2) (Eq. 12);
- V0 centrality (V0A + V0C amplitude); errors stat and 'full syst.'.

Records (public): correlation_type pT_flow_vector_fluctuations
(analysis/flow_vector_fluct.py), one per centrality class (Tables 1-3,
11-14) or per trigger interval (r_n: Tables 4-6, 8-10, one class each).
Staged (--staged OUT): Table 7, r_2 vs pT^a for 0.2 < pT^t < 0.6 GeV/c in
six classes: it disagrees with the same quantity in Tables 4-6 (0-5%, pT^a
0.6-1 GeV/c: 1.015 vs 0.991; up to 1.06), unexplained in the paper.

Run from the repository root:
    python scripts/curate_alice_flow_vector_fluctuations.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2771093/HEPData-ins2771093-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PRC 109 (2024) 065202, arXiv:2403.15213'
OUT = 'data/flow_harmonic_correlations/alice_flow_harmonic_correlations_flow_vector_fluctuations_PbPb_5020gev_ins2771093.yaml'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
# table -> (harmonic, quantity, columns: 'classes' | 'triggers')
TABLES = {'Table_1.yaml': (2, 'v_ratio', 'classes'), 'Table_2.yaml': (3, 'v_ratio', 'classes'),
          'Table_3.yaml': (4, 'v_ratio', 'classes'), 'Table_4.yaml': (2, 'r', 'triggers'),
          'Table_5.yaml': (2, 'r', 'triggers'), 'Table_6.yaml': (2, 'r', 'triggers'),
          'Table_7.yaml': (2, 'r', 'staged'), 'Table_8.yaml': (3, 'r', 'triggers'),
          'Table_9.yaml': (3, 'r', 'triggers'), 'Table_10.yaml': (3, 'r', 'triggers'),
          'Table_11.yaml': (2, 'A_f', 'classes'), 'Table_12.yaml': (2, 'M_f', 'classes'),
          'Table_13.yaml': (2, 'angle_lower_limit', 'classes'),
          'Table_14.yaml': (2, 'magnitude_upper_limit', 'classes')}
WHAT = {'v_ratio': 'v_{n}{2}/v_{n}[2] (Eq. 8)', 'r': 'factorization ratio r_{n}(pT^a, pT^t) (Eq. 9)',
        'A_f': 'flow angle fluctuations A_{n}^f (Eq. 10)', 'M_f': 'flow magnitude fluctuations M_{n}^f (Eq. 14)',
        'angle_lower_limit': 'lower limit of <cos n(Psi_n(pT) - Psi_n)>, sqrt((A^f + 1)/2) (Eq. 11)',
        'magnitude_upper_limit': 'upper limit of the flow magnitude fluctuations, (v{2}/v[2]) / '
                                 'sqrt((A^f + 1)/2) (Eq. 12)'}
HOW = ('two-particle correlators with particle 1 in -0.8 < eta < -0.4 and particle 2 in 0.4 < eta < 0.8 '
       '(|Delta eta| >= 0.8; both orientations pooled)')
HOW4 = ('four-particle correlators with the positive harmonics in eta < 0 and the negative in eta > 0 (the '
        'subevent method without gap; both orientations pooled)')


def pt_range(label):
    """'0.2-0.6 GeV/$c$' -> (0.2, 0.6)."""
    m = re.match(r'\$?([\d.]+)\$?\s*-\s*\$?([\d.]+)\$?', label)
    assert m, label
    return float(m.group(1)), float(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    subs = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
            if x and 'data_file' in x}
    assert set(subs) == set(TABLES), sorted(subs)
    public, staged = [], []
    for hf, (n, quantity, cols) in TABLES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        desc = ' '.join(subs[hf]['description'].split())
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            assert q['SQRT(S)'] == '5020.0 GEV' and q['ETARAP(CM)'] == '$|\\eta| < 0.8$', (hf, q)
            if cols == 'staged':
                staged.append({'observable_type': 'flow_harmonic_correlations', 'inspire_id': '2771093',
                               'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f'{desc} ({name})', 'value_type': name,
                               'notes': {'value': f'{CITE}. {desc}.'},
                               'staging_reason': 'r_2 for 0.2 < pT^t < 0.6 GeV/c disagrees with the same quantity '
                                                 'in Tables 4-6 (0-5%, pT^a 0.6-1 GeV/c: 1.015 vs 0.991), '
                                                 'unexplained in the paper'})
                continue
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'full syst.')}
            if cols == 'classes':
                cls = name
                trig = None
            else:
                cls = q['CENTRALITY'].replace(' pct', '%')
                trig = pt_range(name)
            assert re.fullmatch(r'\d+-\d+%', cls), (hf, cls)
            rb = [bins[r] for r in rows]
            d = {'observable_type': 'flow_harmonic_correlations', 'inspire_id': '2771093', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{desc} ({name})',
                 'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': [cls], **V0M}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8},
                                     'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}},
                 'observable': f'{quantity}_{n}', 'correlation_type': 'pT_flow_vector_fluctuations',
                 'harmonics': [n], 'quantity': quantity, 'dimensionality': 'differential_pT',
                 'flow_reference': {'eta': [{'min': -0.8, 'max': 0.8}], 'pT': {'min': 0.2, 'max': 5.0},
                                    'notes': 'reference particles |eta| < 0.8, 0.2 < pT < 5.0 GeV/c.'},
                 'particles': 'AB',
                 'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.4}}},
                               'B': {'kinematic_range': {'eta': {'min': 0.4, 'max': 0.8}}}},
                 'four_particle_subevents': {'A': {'eta': {'min': -0.8, 'max': 0.0}},
                                             'B': {'eta': {'min': 0.0, 'max': 0.8}}},
                 'value_type': name}
            if trig:
                d['trigger_pT'] = {'min': trig[0], 'max': trig[1]}
            if rows != list(range(len(bins))):
                d['table_rows'] = rows
            uses = {'v_ratio': HOW, 'r': HOW, 'A_f': HOW4, 'M_f': HOW4, 'angle_lower_limit': HOW4,
                    'magnitude_upper_limit': f'{HOW}; {HOW4}'}[quantity]
            d['notes'] = {'value': f'{CITE}. {WHAT[quantity].replace("{n}", str(n)).replace("_n", f"_{n}")} of charged '
                                   f'particles vs pT, Pb-Pb 5.02 TeV {cls}'
                                   + (f', trigger {trig[0]:g} < pT^t < {trig[1]:g} GeV/c' if trig else '')
                                   + f'; {uses}; reference 0.2 < pT < 5.0 GeV/c. Errors: stat and full syst.'}
            public.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {len(staged)} staged')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

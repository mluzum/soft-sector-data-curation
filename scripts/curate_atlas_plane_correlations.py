#!/usr/bin/env python3
"""Curate ATLAS event-plane correlations in Pb+Pb at 2.76 TeV, INSPIRE 1283339.

G. Aad et al. (ATLAS), "Measurement of event-plane correlations in
sqrt(sNN) = 2.76 TeV lead-lead collisions with the ATLAS detector", PRC 90
(2014) 024905, arXiv:1403.0489.

From the paper:
- centrality from the FCal Sigma E_T (3.2 < |eta| < 4.9): five 1% classes
  over 0-5% and 5% classes to 75% (Table I gives their <N_part>, the
  tables' only axis; each table's points are a subset of these classes);
- eight two-plane correlators <cos k(Phi_n - Phi_m)> and six three-plane
  correlators <cos(c_a a Phi_a + c_b b Phi_b + c_c c Phi_c)>, each with the
  event-plane (EP) and the scalar-product (SP) method (Eqs. 9, 10);
- two planes: the subevents ECal+FCal at 0.5 < eta < 4.8 and -4.8 < eta <
  -0.5, both assignments averaged, the resolution from the two
  (Eqs. 12, 14, 16); three planes: ECalP (0.5 < eta < 2.7), FCal (3.3 <
  |eta| < 4.8), ECalN (-2.7 < eta < -0.5), the three types (which plane in
  the FCal) statistically combined, the FCal resolution from three
  subevents (Eqs. 13, 15, 17);
- the calorimeter flow vectors are E_T-weighted towers; errors stat and
  sys.

Records (public): the SP-method correlators (14), correlation_type
plane_correlation (analysis/plane_correlations.py: distinct-particle
flow-vector powers, charged particles with unit weights in the model, the
E_T weighting not modelled), classes from the <N_part> of Table I.  The
second column of the -10 Phi_2 + 4 Phi_4 + 6 Phi_6 table is labelled 'EP
mathod': the SP column by its position (the other tables give SP then EP).
Staged (--staged OUT): the EP-method columns (they depend on the
event-plane resolution of the detector) and the Glauber-model tables.

Run from the repository root:  python scripts/curate_atlas_plane_correlations.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1283339/HEPData-ins1283339-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'G. Aad et al. (ATLAS), PRC 90 (2014) 024905, arXiv:1403.0489'
OUT = 'data/flow_harmonic_correlations/atlas_flow_harmonic_correlations_plane_correlations_PbPb_2760gev_ins1283339.yaml'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}],
        'notes': 'FCal Sigma E_T, 3.2 < |eta| < 4.9.'}
NPART = {400.6: '0-1%', 392.6: '1-2%', 383.2: '2-3%', 372.6: '3-4%', 361.8: '4-5%', 382.2: '0-5%',
         330.3: '5-10%', 281.9: '10-15%', 239.5: '15-20%', 202.6: '20-25%', 170.2: '25-30%', 141.7: '30-35%',
         116.8: '35-40%', 95.0: '40-45%', 76.1: '45-50%', 59.9: '50-55%', 46.1: '55-60%', 34.7: '60-65%',
         25.4: '65-70%', 18.0: '70-75%'}
TWO = {'P': [{'min': 0.5, 'max': 4.8}], 'N': [{'min': -4.8, 'max': -0.5}]}
THREE = {'A': [{'min': 0.5, 'max': 2.7}], 'B': [{'min': -4.8, 'max': -3.3}, {'min': 3.3, 'max': 4.8}],
         'C': [{'min': -2.7, 'max': -0.5}]}


def planes_of(name):
    """[(n, c)] with sum c n = 0 from the column header."""
    s = name.replace('$', '').replace('\\langle', '').replace('\\rangle', '').replace(' ', '')
    s = re.sub(r'\\Phi_\{(\d)\}', r'P\1', s)
    m = re.match(r'cos(\d+)\(P(\d)-P(\d)\)', s)
    if m:
        k, n, mm = (int(x) for x in m.groups())
        return [(n, k // n), (mm, -(k // mm))]
    m = re.match(r'cos\(([-+]?\d*)P(\d)([-+]\d*)P(\d)([-+]\d*)P(\d)\)', s)
    assert m, name
    out = []
    for coef, n in zip(m.groups()[0::2], m.groups()[1::2]):
        coef = int(coef + '1') if coef in ('', '+', '-') else int(coef)
        n = int(n)
        assert coef % n == 0, (name, coef, n)
        out.append((n, coef // n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        glauber = 'Glauber' in desc
        npart = [round(float(v['value']), 1) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name'].strip()
            sp = i == 0                        # SP then EP in every table (Table 25 mislabels SP 'EP mathod')
            assert name.endswith('SP method') if (sp and 'mathod' not in name) else True, name
            if glauber or not sp:
                staged.append({'observable_type': 'flow_harmonic_correlations', 'inspire_id': '1283339',
                               'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f'{name} ({hf} column {i})', 'value_type': name,
                               'notes': {'value': f'{CITE}. {desc.strip()}'},
                               'staging_reason': ('Glauber-model eccentricity correlators' if glauber else
                                                  'EP method: depends on the event-plane resolution of the '
                                                  'detector (the SP column is public)')})
                continue
            planes = planes_of(name)
            assert sum(n * c for n, c in planes) == 0, planes
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'sys')}
            classes = [NPART[npart[r]] for r in rows]
            three = len(planes) == 3
            label = re.sub(r'\s+', ' ', name.replace('$', '').replace('\\langle', '<').replace('\\rangle', '>')
                           .replace('\\Phi_', 'Phi_').replace(' EP mathod', ' SP method')).strip()
            d = {'observable_type': 'flow_harmonic_correlations', 'inspire_id': '1283339', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{label} ({hf} column {i})',
                 'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': classes, **FCAL}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': -4.8, 'max': 4.8}, 'pT': {'min': 0.0, 'max': 100.0}},
                 'observable': label, 'correlation_type': 'plane_correlation',
                 'harmonics': [n for n, _ in planes],
                 'planes': [{'harmonic': n, 'coefficient': c} for n, c in planes],
                 'estimator': 'SP', 'plane_subevents': THREE if three else TWO, 'value_type': name,
                 'hepdata_corrections': {'centrality': "HEPData's axis is <N_part>; the classes are those of "
                                                       'Table I with these <N_part>'}}
            if rows != list(range(len(npart))):
                d['table_rows'] = rows
            how = ('three subevents: ECalP 0.5 < eta < 2.7, FCal 3.3 < |eta| < 4.8, ECalN -2.7 < eta < -0.5; '
                   'the three types (the plane in the FCal) averaged' if three else
                   'two subevents 0.5 < |eta| < 4.8 (ECal+FCal), both assignments averaged')
            mis = (" HEPData labels this column 'EP mathod'; it is the SP column (the first, as in every table)."
                   if 'mathod' in name else '')
            d['notes'] = {'value': f'{CITE}. {label}: scalar-product method, {how}; the E_T-weighted calorimeter '
                                   'flow vectors computed from charged particles with unit weights. Pb+Pb 2.76 TeV '
                                   f'vs centrality (classes of Table I from the tabulated <N_part>).{mis} Errors: '
                                   'stat and sys.'}
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

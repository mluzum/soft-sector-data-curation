#!/usr/bin/env python3
"""Curate ALICE v2-v5 of identified hadrons in Pb-Pb at 2.76 TeV, INSPIRE 1471285.

J. Adam et al. (ALICE), "Higher harmonic flow coefficients of identified
hadrons in Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", JHEP 09 (2016) 164,
arXiv:1606.06057.

From the paper:
- scalar product: identified hadrons (POI) in one TPC subevent
  (-0.8 < eta < 0 or 0 < eta < 0.8) with the charged reference particles of
  the other, v_n^a = <<u_n . Q_n^b*/M^b>> / sqrt(<Q_n^a/M^a . Q_n^b*/M^b>>),
  averaged over a and b;
- non-flow subtracted with pp collisions: the published v_n^sub;
- pi+- (0.3-6 GeV/c), K+- (0.3-4), p+pbar (0.4-6) in |eta| < 0.8; V0
  centrality (V0A + V0C, -3.7 < eta < -1.7 and 2.8 < eta < 5.1);
- errors: statistical and systematic.

Records: v_n(pT) per species, harmonic and class (0-1% to 40-50%), and the
pT-integrated v_n vs centrality (classes from the tabulated centres, 5% and
10% wide).  Two-subevent POI vs charged reference (vn2_gap_poi; the
scalar product differs from the two-particle cumulant only in event
weights); reference pT range not stated here: 0.2-5 GeV/c (ALICE
identified-flow analyses) assumed -- with factorized flow it cancels.
pT points as tabulated (mean pT, 0.1 GeV/c windows).  Not curated: <uQ>
(numerators), delta_n (the pp non-flow terms) and the pp tables.

Run from the repository root:  python scripts/curate_alice_pid_vn_2760.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1471285/HEPData-ins1471285-v1-yaml'
CITE = 'J. Adam et al. (ALICE), JHEP 09 (2016) 164, arXiv:1606.06057'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SPECIES = {'pion': 'pi+ + pi-', 'kaon': 'K+ + K-', 'proton': 'p + pbar'}
REF_PT = {'min': 0.2, 'max': 5.0}
SUBEVENTS = {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': 0.0}, 'pT': dict(REF_PT)},
                   'particle_id': {'species': 'charged'}},
             'B': {'kinematic_range': {'eta': {'min': 0.0, 'max': 0.8}, 'pT': dict(REF_PT)},
                   'particle_id': {'species': 'charged'}}}
WINDOW = 0.1
CENTRES = {2.5: '0-5', 7.5: '5-10', 15.0: '10-20', 25.0: '20-30', 35.0: '30-40', 45.0: '40-50'}


def rows(dv):
    return [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]


def record(hf, sp, n, classes, kr, dim, label, note, table_rows=None):
    d = {'observable_type': 'flow_vn', 'inspire_id': '1471285', 'version': 1, 'hepdata_file': hf, 'dv_index': 0,
         'hepdata_table_label': label,
         'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> X'},
         'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': SPECIES[sp]},
         'kinematic_range': kr, 'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2,
         'value_type': f'v{n}', 'dimensionality': dim, 'particles': 'AB', 'subevents': SUBEVENTS,
         'notes': {'value': (f'{CITE}. {note} Scalar product: the identified hadron in one TPC subevent (-0.8 < eta '
                             '< 0 or 0 < eta < 0.8), charged reference particles in the other; non-flow subtracted '
                             'with pp (v_n^sub). Reference pT range not stated here: 0.2-5 GeV/c assumed. Errors: '
                             'statistical and systematic.')}}
    if table_rows is not None:
        d['table_rows'] = table_rows
    return d


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        desc = ' '.join(x['description'].split())
        hf = x['data_file']
        m = re.match(r'(pion|kaon|proton) v(\d) as a function of pT for centrality: ([0-9]+-[0-9]+)%\.$', desc)
        g = re.match(r'(pion|kaon|proton) Integrated v(\d) as a function of centrality percentile', desc)
        if not (m or g):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = t['dependent_variables'][0]
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        prange = re.match(r'([0-9.]+) < pT < ([0-9.]+)', q['Range'])
        plo, phi_ = float(prange.group(1)), float(prange.group(2))
        r = rows(dv)
        if m:
            sp, n, cls = m.group(1), int(m.group(2)), m.group(3)
            assert q['Centrality'] == f'{cls}%' and q['Particle'] == sp, (hf, q)
            pts = [float(t['independent_variables'][0]['values'][j]['value']) for j in r]
            kr = {'eta': {'min': -0.8, 'max': 0.8},
                  'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                         'points': pts, 'point_window': WINDOW}}
            docs.append(record(hf, sp, n, [f'{cls}%'], kr, 'differential_pT', desc,
                               f'v{n}(pT) of {SPECIES[sp]}, |eta| < 0.8; pT points (mean pT, {WINDOW} GeV/c windows).'))
        else:
            sp, n = g.group(1), int(g.group(2))
            cents = [float(v['value']) for v in t['independent_variables'][0]['values']]
            classes = [f'{CENTRES[c]}%' for j, c in enumerate(cents) if j in r]
            kr = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': plo, 'max': phi_}}
            docs.append(record(hf, sp, n, classes, kr, 'integrated', desc,
                               f'pT-integrated v{n} of {SPECIES[sp]} ({plo:g} < pT < {phi_:g} GeV/c) vs centrality; '
                               'classes from the tabulated centres (2.5, 7.5, 15, ... -> 0-5, 5-10, 10-20, ...%).',
                               table_rows=r if r != list(range(len(cents))) else None))
    out = 'data/flow_vn/alice_flow_vn_identified_hadrons_higher_harmonics_PbPb_2760gev_ins1471285.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['dimensionality'], d['harmonic']) for d in docs)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate PHENIX pi, K, p, pbar spectra in Au+Au at 130 GeV, INSPIRE 568437.

K. Adcox et al. (PHENIX), "Centrality dependence of pi+-, K+-, p and pbar
production from sqrt(sNN) = 130 GeV Au+Au collisions at RHIC", PRL 88,
242301 (2002), arXiv:nucl-ex/0112006.

From the tables: 1/(2 pi pT N) d2N/(dpT dy) of pi+, pi-, K+, K-, p, pbar
in the central arms (|eta| < 0.35) for 0-5, 15-30, 60-92% (BBC); pT
points.  Errors as tabulated.

Records: the 18 spectra, species from each table's reaction qualifier
(the old records called 16 of them pi+); per unit rapidity at
midrapidity: |y| < 0.35 (the central-arm acceptance; a boost-invariant
model is flat there); pT points with windows of the point spacing (the
old records had invented bin edges).  Not curated: <pT> vs N_part
(Tables 19-21: classes not tabulated).

Run from the repository root:  python scripts/curate_phenix_pikp_spectra_130.py
"""
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins568437/HEPData-ins568437-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'K. Adcox et al. (PHENIX), PRL 88, 242301 (2002), arXiv:nucl-ex/0112006'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
SPECIES = {'PI+': 'pi+', 'PI-': 'pi-', 'K+': 'K+', 'K-': 'K-', 'P': 'p', 'PBAR': 'pbar'}


def main():
    docs = []
    for n in range(1, 19):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        sp = SPECIES[q['RE'].split('-->')[1].split()[0]]
        lo, hi = (float(x) for x in q['CENTRALITY'].replace(' PCT', '').split(' TO '))
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pts = [float(iv['values'][j]['value']) for j in rows]
        w = round(min(b - a for a, b in zip(pts[:-1], pts[1:])), 4)
        d = {'observable_type': 'particle_spectra', 'inspire_id': '568437', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'Transverse momentum spectra for {sp}, {cls}',
             'collision_system': {'energy_gev': 130.0, 'reaction': f"AU AU --> {q['RE'].split('-->')[1].strip()}"},
             'centrality': {'classes': [cls], **BBC}, 'particle_id': {'species': sp},
             'kinematic_range': {'y': {'min': -0.35, 'max': 0.35},
                                 'pT': {'min': round(pts[0] - w / 2, 4), 'max': round(pts[-1] + w / 2, 4),
                                        'points': pts, 'point_window': w}},
             'observable': 'invariant_yield', 'value_type': 'invariant_yield',
             'notes': {'value': f'{CITE}. 1/(2 pi pT N) d2N/(dpT dy) of {sp}, {cls}, midrapidity (central arms, '
                                '|eta| < 0.35; |y| < 0.35 here); pT points. Errors as tabulated.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        docs.append(d)
    out = 'data/particle_spectra/phenix_particle_spectra_centrality_dependence_of_pi_k_AuAu_130gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(d['particle_id']['species'] for d in docs)))


if __name__ == '__main__':
    main()

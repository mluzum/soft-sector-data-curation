#!/usr/bin/env python3
"""Curate STAR v3 of charged particles in Au+Au at 200 GeV, INSPIRE 1210062.

L. Adamczyk et al. (STAR), "Third harmonic flow of charged particles in
Au+Au collisions at sqrt(sNN) = 200 GeV", PRC 88, 014904 (2013),
arXiv:1301.2187.

From the paper (Secs. II-IV):
- TPC tracks |eta| < 1.0, 0.15 < pT < 2.0 GeV/c for the eta and centrality
  dependence; RefMult centrality (|eta| < 0.5);
- v3{TPC}: eta-subevent plane from the TPC hemisphere opposite the
  particle, with an extra gap of +-0.05; v3{FTPC}: full plane from both
  FTPCs, 2.5 < |eta| < 4.0; event-plane tracks weighted by pT (2 GeV/c
  above 2 GeV/c), not modelled here (unit weights);
- errors: statistical only (HEPData), systematic bands in the paper.

Records: v3{TPC} and v3{FTPC} vs eta (Tables 5-10), vs pT (Tables 13-18,
and 0-5% from Table 26), vs centrality (Table 19: the first and fourth
columns; HEPData labels both "TPC", the fourth holds the FTPC values of
Fig. 8), and v2{TPC}(pT) (Tables 23-25); event_plane computed with the
scalar-product estimator; points with windows.  Columns in percent ("(%)",
"*100") are scaled by the loader.  Not curated: the wide-Gaussian fit
parameters and v3 (Tables 1-4, 11-12, 19 col. 2), the |Delta eta| > 1
two-particle v3 (Table 19 col. 3: pairs, not two windows), v3^2 vs
Delta eta (Table 20), v3{4}^4 and its ratio (Tables 21-22, with ALICE
columns), and Tables 27-28 (repeat Tables 15-16).

Run from the repository root:  python scripts/curate_star_v3_auau200.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1210062/HEPData-ins1210062-v1-yaml'
CITE = 'L. Adamczyk et al. (STAR), PRC 88, 014904 (2013), arXiv:1301.2187'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
EP_PT = {'min': 0.15, 'max': 100.0}
REFS = {
    'TPC': {'detector': 'TPC', 'mirror': True, 'eta': [{'min': 0.05, 'max': 1.0}], 'pT': dict(EP_PT),
            'notes': 'eta-subevent plane from the TPC hemisphere opposite the particle (gap +-0.05); tracks '
                     'weighted by pT in the paper (unit weights here).'},
    'FTPC': {'detector': 'FTPC', 'eta': [{'min': -4.0, 'max': -2.5}, {'min': 2.5, 'max': 4.0}], 'pT': dict(EP_PT),
             'notes': 'Full event plane from both FTPCs, 2.5 < |eta| < 4.0; tracks weighted by pT in the paper '
                      '(unit weights here).'},
}
CENTRES = {2.5: '0-5%', 7.5: '5-10%', 15.0: '10-20%', 25.0: '20-30%', 35.0: '30-40%', 45.0: '40-50%',
           55.0: '50-60%', 65.0: '60-70%', 75.0: '70-80%'}
ETA_WINDOW, PT_WINDOW = 0.125, 0.1


def cls_of(desc):
    return re.search(r'for (\d+-\d+%) centrality', desc).group(1)


def record(hf, i, desc, h, n, ref, classes, kr, dim, what, rows=None):
    d = {'observable_type': 'flow_vn', 'inspire_id': '1210062', 'version': 1, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': f'{desc[:150]} | {h}',
         'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
         'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': 'charged'},
         'kinematic_range': kr, 'harmonic': n, 'method': 'event_plane', 'value_type': h, 'dimensionality': dim,
         'flow_reference': dict(REFS[ref]),
         'notes': {'value': f'{CITE}. v{n}{{{ref}}} {what}; computed with the scalar-product estimator. Errors: '
                            'statistical (systematic uncertainties only in the paper).'}}
    if rows is not None:
        d['table_rows'] = rows
    return d


def points(iv, rows, window):
    pts = [float(iv['values'][j]['value']) for j in rows]
    return {'min': round(pts[0] - window / 2, 4), 'max': round(pts[-1] + window / 2, 4), 'points': pts,
            'point_window': window}


def main():
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    docs = []
    for n_t in list(range(5, 11)) + list(range(13, 20)) + [23, 24, 25, 26]:
        hf = f'table{n_t}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        desc = ' '.join(sub[hf]['description'].split())
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-', 'none')]
            if n_t == 19:
                if i not in (0, 3):
                    continue
                ref = 'TPC' if i == 0 else 'FTPC'
                classes = [CENTRES[float(iv['values'][j]['value'])] for j in rows]
                kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.15, 'max': 2.0}}
                docs.append(record(hf, i, desc, h, 3, ref, classes, kr, 'integrated',
                                   'vs centrality, 0.15 < pT < 2 GeV/c, |eta| < 1 (Fig. 8)',
                                   rows if rows != list(range(len(iv['values']))) else None))
                continue
            ref = 'FTPC' if 'FTPC' in h else 'TPC'
            n = 2 if n_t in (23, 24, 25) else 3
            cls = cls_of(desc)
            if n_t <= 10:
                kr = {'eta': points(iv, rows, ETA_WINDOW), 'pT': {'min': 0.15, 'max': 2.0}}
                dim, what = 'differential_eta', f'(eta), {cls}, 0.15 < pT < 2 GeV/c (Fig. 5)'
            else:
                kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': points(iv, rows, PT_WINDOW)}
                dim, what = 'differential_pT', f'(pT), {cls}, |eta| < 1, pT points ({PT_WINDOW} GeV/c windows)'
            assert rows == list(range(rows[0], rows[-1] + 1)), (hf, i)
            docs.append(record(hf, i, desc, h, n, ref, [cls], kr, dim, what,
                               rows if rows != list(range(len(iv['values']))) else None))
    out = 'data/flow_vn/star_flow_vn_third_harmonic_flow_of_charged_AuAu_200gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(
        (d['harmonic'], d['dimensionality'], d['flow_reference']['detector']) for d in docs)))


if __name__ == '__main__':
    main()

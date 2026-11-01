#!/usr/bin/env python3
"""Curate STAR identified-hadron v2(pT) in Au+Au at 62.4 GeV, INSPIRE 741917.

B. I. Abelev et al. (STAR), "Mass, quark-number, and sqrt(sNN) dependence of
the second and fourth flow harmonics in ultra-relativistic nucleus-nucleus
collisions", PRC 75, 054906 (2007), arXiv:nucl-ex/0701010.

From the paper:
- v2 from the event plane of charged TPC tracks; pions, K0_S, protons and
  Lambda with eta-subevents (particles at eta > 0 against the plane of
  tracks at eta < -0.15, and vice versa), |eta| < 1;
- identification by dE/dx (pions, kaons, (anti)protons), time of flight
  (pions, kaons, p+pbar) and invariant mass (K0_S, Lambda+Lambdabar,
  Xi+Xibar, Omega+Omegabar);
- centrality from the charged tracks in |eta| < 0.5 (pT > 0.2 GeV/c):
  minimum bias 0-80%, and 0-10, 10-40, 40-80%;
- the tables give statistical errors; the systematic estimates are separate
  per-panel tables (not attached to the records).

Records: one per series of Figs. 1 (minimum bias) and 2 (centrality), a
series per species and identification method; pT points (0.1 GeV/c
windows).  method event_plane with the TPC eta-sub reference (both halves
0.15 < |eta| < 1 here; the opposite one per particle in the paper), pT of
the plane tracks 0.15-2 GeV/c assumed (STAR's standard of the time);
computed with the scalar-product estimator.  Not curated: v4 (relative to
the second-harmonic plane), v4/v2^2, the anti-baryon/baryon ratios, v2/n_q
and v2/(n_q eps) scalings, data/fit and the 200/62.4 ratios.

Run from the repository root:  python scripts/curate_star_pid_v2_62gev.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins741917/HEPData-ins741917-v1-yaml'
CITE = 'B. I. Abelev et al. (STAR), PRC 75, 054906 (2007), arXiv:nucl-ex/0701010'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.2}}
SPECIES = [(r'charged hadrons', 'charged'), (r'pions', 'pi+ + pi-'), (r'K0short|Kshort', 'K0_S'),
           (r'charged kaons', 'K+ + K-'), (r'anti-?protons?', 'pbar'), (r'p\+pbar', 'p + pbar'),
           (r'Lambda\+Antilambda', 'Lambda + Lambdabar'), (r'Xi\+Xibar', 'Xi- + Xibar+'),
           (r'Omega\+Omegabar', 'Omega- + Omegabar+')]
WINDOW = 0.1


def species(text):
    for pat, sp in SPECIES:
        if re.search(pat, text, re.I):
            return sp
    raise ValueError(text)


def centrality(text):
    if text.lower().startswith('minbias'):
        return '0-80'
    m = re.search(r'\(([0-9]+)%-([0-9]+)%?\)', text)
    return f'{m.group(1)}-{m.group(2)}'


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        if not hf.startswith(('figure-1_', 'figure-2_')) or 'sys_err' in hf:
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = t['dependent_variables'][0]
        text = str(dv['qualifiers'][0]['value'])
        sp, cls = species(text), centrality(text)
        method = re.search(r'TOF|r?dE/dx', text)
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pts = [float(t['independent_variables'][0]['values'][j]['value']) for j in rows]
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '741917', 'version': 1, 'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': f'v2(pT): {text}',
            'collision_system': {'energy_gev': 62.4, 'reaction': 'AU AU --> X'},
            'centrality': {'classes': [f'{cls}%'], **REFMULT},
            'particle_id': {'species': sp},
            'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                       'points': pts, 'point_window': WINDOW}},
            'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'dimensionality': 'differential_pT',
            'flow_reference': {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': -0.15}, {'min': 0.15, 'max': 1.0}],
                               'pT': {'min': 0.15, 'max': 2.0},
                               'notes': 'eta-subevents: the plane of the tracks in the opposite hemisphere, beyond '
                                        '|eta| = 0.15 (both halves here); plane-track pT range assumed.'},
            'notes': {'value': (f'{CITE}, {hf.split("_")[0].replace("figure-", "Fig. ")}. v2(pT) of {sp}'
                                + (f' ({method.group(0)} identification)' if method else '')
                                + f', |eta| < 1, eta-subevent event plane; pT points ({WINDOW} GeV/c windows). '
                                'Computed with the scalar-product estimator. Statistical errors (systematic '
                                'estimates in separate tables).')},
        })
    out = 'data/flow_vn/star_flow_vn_mass_quark-number_and_sqrt_s_62gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['centrality']['classes'][0], d['particle_id']['species']) for d in docs)))


if __name__ == '__main__':
    main()

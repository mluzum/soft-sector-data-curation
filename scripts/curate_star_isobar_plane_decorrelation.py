#!/usr/bin/env python3
"""Curate STAR's flow-plane decorrelation in the isobars, INSPIRE 2928639.

STAR Collaboration, "Measurement of the longitudinal flow-plane
decorrelation using multi-plane cumulants in sqrt(sNN) = 200 GeV Au+Au,
Ru+Ru, and Zr+Zr collisions", PRC 113 (2026) 024908, arXiv:2506.02330.

From the paper (Secs. II-III):
- windows a: EPD-east -5.1 < eta < -2.1 (or BBC), b: TPC-east -1 < eta <
  -0.1, c: TPC-west 0.1 < eta < 1, d: EPD-west 2.1 < eta < 5.1; TPC tracks
  0.2 < pT < 2.0 GeV/c;
- r_n(eta_a->c/b) = <cos n(phi_c - psi_a)> / <cos n(phi_b - psi_a)> (Eq. 2,
  particle level: tracks in b and c, the EPD plane of a), and the mirror
  r_n(eta_d->b/c) (Eq. 22); r_n their mean (Eq. 23);
- T_n{ba; dc} (Eqs. 5-21): a four-plane cumulant of event-plane angles
  over the product of resolutions, particle level (tracks in b, c) or Q
  level (TPC event planes).

Records (public): r_2, r_3 vs centrality in Ru+Ru and Zr+Zr (Fig. 7):
correlation_type longitudinal_decorrelation, quantity r_nn (key
rnn_decorrelation): the TPC tracks at 0.1 < |eta| < 1 against the EPD at
2.1 < |eta| < 5.1 (charged particles, all pT; the EPD's nMIP weights not
modelled), opposite- over same-side correlation, both orientations pooled
(STAR averages the two ratios, Eq. 23: equal for symmetric systems).  The
event-plane angle of a enters numerator and denominator alike, so its
resolution cancels; the scalar product is its low-resolution limit.
Centrality: STAR's isobar convention (RefMult, F50 normalization, as
1914564; Table III gives no actual ranges for these 5% classes).
Staged (--staged OUT): T_2, T_3 and r_n^Psi (Figs. 8-10: only the Ru+Ru
and Zr+Zr combination, or Au+Au at Q level from TPC event planes, whose
resolution is not low), the resolution-uncorrected T_n^obs terms and the
resolutions (Figs. 2-6, detector level).

Run from the repository root:  python scripts/curate_star_isobar_plane_decorrelation.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

IID = '2928639'
RAW = glob.glob(f'hepdata_raw/records/ins{IID}/HEPData-ins{IID}-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'STAR Collaboration, PRC 113 (2026) 024908, arXiv:2506.02330'
OUT = f'data/flow_harmonic_correlations/star_flow_harmonic_correlations_plane_decorrelation_isobar_ins{IID}.yaml'
F50 = {'Ru': 0.4712, 'Zr': 0.4590}
NORM_NOTE = ('STAR isobar centrality (RefMult |eta| < 0.5, pT > 0.2 GeV/c): percentiles of a Case-3 Glauber total in '
             'which Ntrk >= 50 holds fraction_above (F50 as for 1914564, derived by the isobar analysis); actual '
             'ranges of these 5% classes not published.  Efficiency 0.87 (TPC 85-90%).')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs, staged = [], []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        desc = ' '.join(x['description'].split())
        classes = [f"{float(b['low']):g}-{float(b['high']):g}%" for b in iv['values']]
        if hf in ('fig7_left.yaml', 'fig7_right.yaml'):
            n = 2 if 'left' in hf else 3
            for j, dv in enumerate(t['dependent_variables']):
                nuc = 'Ru' if 'Ru+Ru' in dv['header']['name'] else 'Zr'
                assert dv['header']['name'] == f'$r_{{{n}}}$ for {nuc}+{nuc}', dv['header']['name']
                assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
                docs.append({
                    'observable_type': 'flow_harmonic_correlations', 'inspire_id': IID, 'version': VERSION,
                    'hepdata_file': hf, 'dv_index': j, 'hepdata_table_label': f"{x['name']}: {desc}",
                    'collision_system': {'energy_gev': 200.0, 'reaction': f'{nuc}+{nuc}'},
                    'centrality': {'classes': classes,
                                   'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
                                   'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.2, 'max': 100.0},
                                   'normalization': {'estimator_threshold': 50, 'fraction_above': F50[nuc],
                                                     'efficiency': 0.87, 'notes': NORM_NOTE}},
                    'particle_id': {'species': 'charged'},
                    'kinematic_range': {'eta': {'min': 0.1, 'max': 1.0, 'bin_ranges': [[0.1, 1.0]]},
                                        'pT': {'min': 0.2, 'max': 2.0}},
                    'observable': f'r_{n}', 'correlation_type': 'longitudinal_decorrelation', 'harmonics': [n],
                    'quantity': 'r_nn', 'dimensionality': 'differential_eta',
                    'decorrelation_reference': {'eta': {'min': 2.1, 'max': 5.1}, 'pT': {'min': 0.0, 'max': 100.0},
                                                'detector': 'EPD',
                                                'notes': 'EPD at 2.1 < |eta| < 5.1 (nMIP-weighted tiles in the '
                                                         'paper; charged particles with unit weights here), both '
                                                         'sides.'},
                    'value_type': dv['header']['name'],
                    'notes': {'value': f'{CITE}, Fig. 7. r_{n} = <cos {n}(phi_c - psi_a)> / <cos {n}(phi_b - psi_a)> '
                                       '(Eqs. 2, 22, 23): TPC tracks 0.1 < |eta| < 1, 0.2 < pT < 2 GeV/c, against the '
                                       f'EPD plane at 2.1 < |eta| < 5.1, {nuc}+{nuc} 200 GeV. Errors: stat and syst.'}})
            continue
        reason = ('only for Ru+Ru and Zr+Zr combined (or Au+Au at Q level, TPC event planes): no single system, and '
                  'the Q-level planes have no low-resolution limit' if hf.startswith(('fig8', 'fig9', 'fig10'))
                  else 'event-plane resolutions and resolution-uncorrected T_n^obs terms: detector level')
        for j, dv in enumerate(t['dependent_variables']):
            staged.append({'observable_type': 'flow_harmonic_correlations', 'inspire_id': IID, 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': j, 'hepdata_table_label': f"{x['name']}: {desc}",
                           'value_type': dv['header']['name'],
                           'collision_system': {'energy_gev': 200.0, 'reaction': 'Ru+Ru'},
                           'centrality': {'classes': classes}, 'correlation_type': 'longitudinal_decorrelation',
                           'harmonics': [3 if '_{3}' in dv['header']['name'] or 'right' in hf else 2],
                           'staging_reason': reason, 'notes': {'value': f"{CITE}, {x['name']}."}})
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

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
Records (public), Ru+Ru and Zr+Zr combined (Figs. 8-10, particle level):
T_2, T_3 with the EPD and T_2 with the BBC (3.3 < |eta| < 5.0), and r_2, r_3:
observable_ratios records of ratio_type system_combination (combination:
mean) holding the values, linked to one record per system: for r_n the
Fig. 7 records, for T_n side records (experimental_values: false; quantity
T_n, key tn_four_plane: the scalar-product analog of Eqs. 17-19, the EPD or
BBC planes a, d as flow vectors of charged particles, tracks in b, c).  The
mean of the two systems stands for STAR's combined sample (about equal
numbers of Ru+Ru and Zr+Zr events).
Staged (--staged OUT): the Q-level r_n^Psi and T_n (Figs. 8-10: TPC event
planes, whose resolution is not low, and Au+Au only at Q level), the
resolution-uncorrected T_n^obs terms and the resolutions (Figs. 2-6,
detector level).

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
SIDES = f'data/flow_harmonic_correlations/star_flow_harmonic_correlations_four_plane_sides_isobar_ins{IID}.yaml'
COMB = f'data/observable_ratios/star_observable_ratios_isobar_combined_ins{IID}.yaml'
REF = {'EPD': ((2.1, 5.1), 'EPD at 2.1 < |eta| < 5.1 (nMIP-weighted tiles in the paper; charged particles with unit '
                           'weights here), both sides.'),
       'BBC': ((3.3, 5.0), 'BBC at 3.3 < |eta| < 5.0 (tiles weighted by their signal in the paper; charged '
                           'particles with unit weights here), both sides.')}
# the combined tables: (file, dv index) -> (quantity, harmonic, reference detector)
COMBINED = {('fig8_left.yaml', 0): ('r_nn', 2, 'EPD'), ('fig8_right.yaml', 0): ('r_nn', 3, 'EPD'),
            ('fig9_left.yaml', 0): ('T_n', 2, 'EPD'), ('fig9_right.yaml', 0): ('T_n', 3, 'EPD'),
            ('fig10.yaml', 0): ('T_n', 2, 'BBC')}
CENT_BASE = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
             'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.2, 'max': 100.0}}


def centrality(classes, nuc=None):
    c = {'classes': classes, **CENT_BASE}
    if nuc:
        c['normalization'] = {'estimator_threshold': 50, 'fraction_above': F50[nuc], 'efficiency': 0.87,
                              'notes': NORM_NOTE}
    return c


def reference(det):
    (lo, hi), note = REF[det]
    return {'eta': {'min': lo, 'max': hi}, 'pT': {'min': 0.0, 'max': 100.0}, 'detector': det, 'notes': note}


def tn_side(nuc, n, det, classes):
    """T_n of one system: a side of the combined record (no values of its own)."""
    return {'observable_type': 'flow_harmonic_correlations', 'inspire_id': IID, 'version': VERSION,
            'experimental_values': False,
            'hepdata_table_label': f'T_{n}{{ba; dc}}, particle level, TPC/{det}, {nuc}+{nuc} 200 GeV (a side of '
                                   'the Ru+Ru and Zr+Zr combination; no values per system published)',
            'collision_system': {'energy_gev': 200.0, 'reaction': f'{nuc}+{nuc}'},
            'centrality': centrality(classes, nuc), 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': 0.1, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 2.0}},
            'observable': f'T_{n}', 'correlation_type': 'longitudinal_decorrelation', 'harmonics': [n],
            'quantity': 'T_n', 'dimensionality': 'integrated', 'decorrelation_reference': reference(det),
            'value_type': f'T_{n} {nuc}+{nuc} TPC/{det}',
            'notes': {'value': f'{CITE}, Eqs. 17-19: T_{n}{{ba; dc}} = <<sin {n}(phi_b - psi_a) sin {n}(psi_d - phi_c)>> '
                               f'/ (<cos {n}(phi_b - psi_a)><cos {n}(psi_d - phi_c)>), tracks b, c at -1 < eta < -0.1 '
                               f'and 0.1 < eta < 1 (0.2 < pT < 2 GeV/c), planes a, d from the {det} east and west; '
                               f'{nuc}+{nuc} 200 GeV.  The side of a combined record (the values).'}}
F50 = {'Ru': 0.4712, 'Zr': 0.4590}
NORM_NOTE = ('STAR isobar centrality (RefMult |eta| < 0.5, pT > 0.2 GeV/c): percentiles of a Case-3 Glauber total in '
             'which Ntrk >= 50 holds fraction_above (F50 as for 1914564, derived by the isobar analysis); actual '
             'ranges of these 5% classes not published.  Efficiency 0.87 (TPC 85-90%).')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs, staged, sides, comb = [], [], [], []
    fig7 = {}                       # (n, nucleus) -> document index in OUT
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
                fig7[(n, nuc)] = len(docs)
                docs.append({
                    'observable_type': 'flow_harmonic_correlations', 'inspire_id': IID, 'version': VERSION,
                    'hepdata_file': hf, 'dv_index': j, 'hepdata_table_label': f"{x['name']}: {desc}",
                    'collision_system': {'energy_gev': 200.0, 'reaction': f'{nuc}+{nuc}'},
                    'centrality': centrality(classes, nuc),
                    'particle_id': {'species': 'charged'},
                    'kinematic_range': {'eta': {'min': 0.1, 'max': 1.0, 'bin_ranges': [[0.1, 1.0]]},
                                        'pT': {'min': 0.2, 'max': 2.0}},
                    'observable': f'r_{n}', 'correlation_type': 'longitudinal_decorrelation', 'harmonics': [n],
                    'quantity': 'r_nn', 'dimensionality': 'differential_eta',
                    'decorrelation_reference': reference('EPD'),
                    'value_type': dv['header']['name'],
                    'notes': {'value': f'{CITE}, Fig. 7. r_{n} = <cos {n}(phi_c - psi_a)> / <cos {n}(phi_b - psi_a)> '
                                       '(Eqs. 2, 22, 23): TPC tracks 0.1 < |eta| < 1, 0.2 < pT < 2 GeV/c, against the '
                                       f'EPD plane at 2.1 < |eta| < 5.1, {nuc}+{nuc} 200 GeV. Errors: stat and syst.'}})
            continue
        reason = ('Q level: TPC event planes, whose resolution is not low (no scalar-product analog); Au+Au only '
                  'at Q level' if hf.startswith(('fig8', 'fig9', 'fig10'))
                  else 'event-plane resolutions and resolution-uncorrected T_n^obs terms: detector level')
        for j, dv in enumerate(t['dependent_variables']):
            if (hf, j) in COMBINED:
                q, n, det = COMBINED[(hf, j)]
                assert 'Particle-level' in dv['header']['name'] or dv['header']['name'] == \
                    f'$r_{{{n}}}$ for Ru+Ru and Zr+Zr combined', dv['header']['name']
                assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
                quals = {qq['name']: str(qq['value']) for qq in dv.get('qualifiers') or []}
                assert quals.get('Collision species') == 'Ru+Ru and Zr+Zr', quals
                if q == 'T_n':
                    link = {}
                    for nuc in ('Ru', 'Zr'):
                        link[nuc] = f'{SIDES}#{len(sides)}'
                        sides.append(tn_side(nuc, n, det, classes))
                    what = (f'T_{n}{{ba; dc}} (Eqs. 17-19, particle level): tracks b, c at -1 < eta < -0.1 and '
                            f'0.1 < eta < 1, planes a, d from the {det} east and west')
                else:
                    link = {nuc: f'{OUT}#{fig7[(n, nuc)]}' for nuc in ('Ru', 'Zr')}
                    what = f'r_{n} (Eqs. 2, 22, 23, particle level), as Fig. 7 per system'
                comb.append({
                    'observable_type': 'observable_ratios', 'inspire_id': IID, 'version': VERSION,
                    'hepdata_file': hf, 'dv_index': j, 'hepdata_table_label': f"{x['name']}: {desc}",
                    'collision_system': {'energy_gev': 200.0, 'reaction': 'Ru+Ru and Zr+Zr'},
                    'centrality': centrality(classes), 'particle_id': {'species': 'charged'},
                    'kinematic_range': {'eta': {'min': 0.1, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 2.0}},
                    'observable': f'T_{n}' if q == 'T_n' else f'r_{n}', 'value_type': dv['header']['name'],
                    'numerator': 'Ru+Ru', 'denominator': 'Zr+Zr', 'ratio_type': 'system_combination',
                    'combination': 'mean', 'numerator_system': 'Ru+Ru', 'denominator_system': 'Zr+Zr',
                    'numerator_record': link['Ru'], 'denominator_record': link['Zr'],
                    'notes': {'value': f'{CITE}, {x["name"]}: {what}, TPC tracks 0.2 < pT < 2 GeV/c; Ru+Ru and '
                                       'Zr+Zr combined, computed as the mean of the two systems (each on its own '
                                       'simulation, with its own centrality normalization).  Errors: stat and '
                                       'syst.'}})
                continue
            staged.append({'observable_type': 'flow_harmonic_correlations', 'inspire_id': IID, 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': j, 'hepdata_table_label': f"{x['name']}: {desc}",
                           'value_type': dv['header']['name'],
                           'collision_system': {'energy_gev': 200.0, 'reaction': 'Ru+Ru'},
                           'centrality': {'classes': classes}, 'correlation_type': 'longitudinal_decorrelation',
                           'harmonics': [3 if '_{3}' in dv['header']['name'] or 'right' in hf else 2],
                           'staging_reason': reason, 'notes': {'value': f"{CITE}, {x['name']}."}})
    for path, recs in ((OUT, docs), (SIDES, sides), (COMB, comb)):
        with open(path, 'w') as fh:
            yaml.safe_dump_all(recs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(recs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

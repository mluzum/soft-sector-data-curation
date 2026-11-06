#!/usr/bin/env python3
"""Curate PHENIX phi -> mu+ mu- at forward and backward rapidity in Cu+Au at 200 GeV, INSPIRE 1394228.

A. Adare et al. (PHENIX), "Measurement of phi-meson production at forward
rapidity in Cu+Au collisions at sqrt(sNN) = 200 GeV", PRC 93 (2016)
024904, arXiv:1509.06337.

From the paper:
- phi -> mu+ mu- in the muon arms, Cu-going 1.2 < y < 2.2 and Au-going
  -2.2 < y < -1.2, 1 < pT < 5 GeV/c;
- centrality from the combined charge of both BBCs (3.1 < |eta| < 3.9),
  classes 0-20, 20-40, 40-93% (the minimum-bias trigger: 93% of the
  inelastic cross section);
- yields tabulated times the branching ratio, BR(phi -> mu mu) = (2.89 +-
  0.19) x 10^-4 (Eq. 8); errors: stat. + type A sys., type B sys.
  (point-to-point correlated) and type C (3% global); the forward/backward
  ratio has no type C.

Records (public): BR dN/dy in 1 < pT < 5 GeV/c vs class (Fig. 5 / Table
IV, Cu-going and Au-going), BR d2N/(dy dpT) in two pT bins, 0-93% (Fig.
6), BR dN/dy in two |y| bins, 0-93% (Fig. 7), each with value_scale =
2.89e-4 (the paper's BR; its 6.6% uncertainty is not in the tabulated
errors); the forward/backward ratio vs class (Fig. 11), a rapidity_ratio
linked to the two Fig. 5 records (the BR cancels).  Cu+Au in the nn frame
with Cu toward +y (rapidity_shift 0).  Staged (--staged OUT): R_CuAu
(Figs. 8-10, need the pp reference).

Run from the repository root:  python scripts/curate_phenix_phi_mumu_cuau.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1394228/HEPData-ins1394228-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 93 (2016) 024904, arXiv:1509.06337'
OUT_Y = 'data/particle_yields/phenix_particle_yields_phi_mumu_CuAu_200gev_ins1394228.yaml'
OUT_S = 'data/particle_spectra/phenix_particle_spectra_phi_mumu_CuAu_200gev_ins1394228.yaml'
OUT_R = 'data/observable_ratios/phenix_observable_ratios_phi_fwd_bwd_CuAu_200gev_ins1394228.yaml'
BR = 2.89e-4
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.1}, {'min': 3.1, 'max': 3.9}],
       'notes': 'combined charge of both BBCs, 3.1 < |eta| < 3.9.'}
SYSTEM = {'energy_gev': 200.0, 'reaction': 'CU AU --> PHI X', 'frame': 'nn_cms', 'rapidity_shift': 0.0,
          'forward_beam': 'Cu'}
ARMS = {'Cu-going': (1.2, 2.2), 'Au-going': (-2.2, -1.2)}
CLASSES = ['0-20%', '20-40%', '40-93%']
ERRS = ('stat. + A sys.', 'B sys.', 'C sys.')
BR_NOTE = (f'Tabulated times BR(phi -> mu+ mu-); value_scale = {BR:g}, the paper\'s BR (+-0.19e-4, not in the '
           'tabulated errors). Errors: stat. + type A sys., type B sys. (point-to-point correlated), type C (3% '
           'global).')


def arm(dv):
    q = {k['name']: k['value'] for k in dv['qualifiers']}['Rapidity']
    name = {'1.2 < $y$ < 2.2': 'Cu-going', '-2.2 < $y$ < -1.2': 'Au-going'}[q]
    assert 'Cu-going' not in dv['header']['name'] or name == 'Cu-going', dv['header']
    return name


def head(hf, i, label, otype, classes):
    return {'observable_type': otype, 'inspire_id': '1394228', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf} column {i})', 'collision_system': dict(SYSTEM),
            'centrality': {'classes': classes, **BBC}, 'particle_id': {'species': 'phi'}}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    yields, spectra, ratios, staged = [], [], [], []
    side = {}
    t = load('fig5.yaml')
    assert [f"{v['low']:g}-{v['high']:g}%" for v in t['independent_variables'][0]['values']] == CLASSES
    for i, dv in enumerate(t['dependent_variables']):
        if 'N_{part}' in dv['header']['name']:
            continue
        name = arm(dv)
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {ERRS}
        lo, hi = ARMS[name]
        d = head('fig5.yaml', i, f'Fig. 5: BR dN/dy of phi, {name}, vs class', 'particle_yields', list(CLASSES))
        d['kinematic_range'] = {'y': {'min': lo, 'max': hi}, 'pT': {'min': 1.0, 'max': 5.0}}
        d.update(observable='dN_dy', value_type='dN_dy', value_scale=BR)
        d['notes'] = {'value': f'{CITE}, Fig. 5 and Table IV. dN/dy of phi (mu+ mu-) in 1 < pT < 5 GeV/c, '
                               f'{lo:g} < y < {hi:g} ({name}), Cu+Au 200 GeV vs class. {BR_NOTE}'}
        side[name] = (OUT_Y, len(yields))
        yields.append(d)
    t = load('fig6.yaml')
    bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
    assert bins == [[1.0, 2.5], [2.5, 5.0]], bins
    for i, dv in enumerate(t['dependent_variables']):
        if 'langle p_T' in dv['header']['name']:
            continue
        name = arm(dv)
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {ERRS}
        lo, hi = ARMS[name]
        d = head('fig6.yaml', i, f'Fig. 6: BR d2N/(dy dpT) of phi, {name}, 0-93%', 'particle_spectra', ['0-93%'])
        d['kinematic_range'] = {'y': {'min': lo, 'max': hi}, 'pT': {'min': 1.0, 'max': 5.0, 'bin_ranges': bins}}
        d.update(observable='dN_dpT_dy', value_type='dN_dpT_dy', value_scale=BR)
        d['notes'] = {'value': f'{CITE}, Fig. 6 and Table V. d2N/(dy dpT) of phi (mu+ mu-), {lo:g} < y < {hi:g} '
                               f'({name}), Cu+Au 200 GeV 0-93%, in two pT bins. {BR_NOTE}'}
        spectra.append(d)
    t = load('fig7.yaml')
    absy = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
    assert absy == [[1.8, 2.2], [1.2, 1.8]], absy
    for i, dv in enumerate(t['dependent_variables']):
        if 'langle |y|' in dv['header']['name']:
            continue
        name = arm(dv)
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {ERRS}
        sgn = 1 if name == 'Cu-going' else -1
        ybins = [sorted([sgn * a, sgn * b]) for a, b in absy]
        lo, hi = ARMS[name]
        d = head('fig7.yaml', i, f'Fig. 7: BR dN/dy of phi vs |y|, {name}, 0-93%', 'particle_yields', ['0-93%'])
        d['kinematic_range'] = {'y': {'min': lo, 'max': hi, 'bin_ranges': ybins}, 'pT': {'min': 1.0, 'max': 5.0}}
        d.update(observable='dN_dy', value_type='dN_dy', dimensionality='differential_y', value_scale=BR)
        d['notes'] = {'value': f'{CITE}, Fig. 7 and Table VI. dN/dy of phi (mu+ mu-) in 1 < pT < 5 GeV/c per '
                               f'rapidity bin ({name}; the table rows run 1.8 < |y| < 2.2, then 1.2 < |y| < 1.8), '
                               f'Cu+Au 200 GeV 0-93%. {BR_NOTE}'}
        yields.append(d)
    t = load('fig11.yaml')
    assert [f"{v['low']:g}-{v['high']:g}%" for v in t['independent_variables'][0]['values']] == CLASSES
    assert len(t['dependent_variables']) == 1
    num, den = side['Cu-going'], side['Au-going']
    d = head('fig11.yaml', 0, 'Fig. 11: phi forward (Cu-going) / backward (Au-going) yield ratio vs class',
             'observable_ratios', list(CLASSES))
    d['kinematic_range'] = {'pT': {'min': 1.0, 'max': 5.0}}
    d.update(observable='dN_dy', value_type='forward/backward ratio', ratio_type='rapidity_ratio',
             numerator='1.2 < y < 2.2 (Cu-going)', denominator='-2.2 < y < -1.2 (Au-going)',
             numerator_system='Cu+Au', denominator_system='Cu+Au',
             numerator_record=f'{num[0]}#{num[1]}', denominator_record=f'{den[0]}#{den[1]}')
    d['notes'] = {'value': f'{CITE}, Fig. 11 and Table IV. Ratio of the phi dN/dy (1 < pT < 5 GeV/c) at 1.2 < y < '
                           '2.2 (Cu-going) to -2.2 < y < -1.2 (Au-going), Cu+Au 200 GeV vs class. Errors: stat. + '
                           'type A sys. (asymmetric), type B sys.; no type C.'}
    ratios.append(d)
    for hf in ('fig8.yaml', 'fig9.yaml', 'fig10.yaml'):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            staged.append({'observable_type': 'nuclear_modification', 'inspire_id': '1394228', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                           'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. R_CuAu of phi.'},
                           'staging_reason': 'R_CuAu: needs the pp reference'})
    for out, docs in ((OUT_Y, yields), (OUT_S, spectra), (OUT_R, ratios)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

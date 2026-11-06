#!/usr/bin/env python3
"""Curate BRAHMS charged-particle densities in Au+Au at 130 GeV, INSPIRE 561518.

I. G. Bearden et al. (BRAHMS), "Charged particle densities from Au+Au
collisions at sqrt(sNN) = 130 GeV", PLB 523 (2001) 227,
arXiv:nucl-ex/0108016.

From the paper:
- dNch/deta over -4.7 <= eta <= 4.7 from the Si and tile multiplicity
  arrays (SiMA, TMA: |eta| <= 2.2 at the nominal vertex, 3.0 with the
  vertex range), the beam-beam counters (2.1 <= |eta| <= 4.7) and TPM1;
- centrality from the total multiplicity of the multiplicity array (SiMA
  + TMA; the BBC analysis from the summed BBC hits), classes 0-5, 5-10,
  10-20, 20-30, 30-40, 40-50%;
- Table 1: dNch/deta at eta = 0, 1.5, 3.0, 4.5 and the integrated
  multiplicity in -4.7 <= eta <= 4.7, total uncertainties; Fig. 4:
  dNch/deta symmetrized and averaged over the detector systems (total
  uncertainties) for 0-5, 5-10, 20-30, 40-50%.

Records (public): Table 1 (four eta points, windows of 0.2 around each,
and the integrated multiplicity as the mean dNch/deta over the 9.4 units
of eta: value_scale 9.4) and Fig. 4 (|eta| points, 0.2 windows).  The
centrality estimator is the multiplicity array, |eta| < 2.2.  Staged
(--staged OUT): the per-detector distributions (Fig. 2, the constituents
of Fig. 4 and Table 1; statistical errors only), N_part and dNch/deta per
participant pair (Fig. 3; Glauber).

Run from the repository root:  python scripts/curate_brahms_dndeta_auau130.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins561518/HEPData-ins561518-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'I. G. Bearden et al. (BRAHMS), PLB 523 (2001) 227, arXiv:nucl-ex/0108016'
OUT = 'data/particle_yields/brahms_particle_yields_dNch_deta_AuAu_130gev_ins561518.yaml'
WINDOW = 0.2
MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'SiMA+TMA'},
      'eta': [{'min': -2.2, 'max': 2.2}],
      'notes': 'total multiplicity of the multiplicity array (SiMA + TMA), |eta| <= 2.2 at the nominal vertex; the '
               'BBC points select on the summed BBC hits (2.1 <= |eta| <= 4.7).'}
SYSTEM = {'energy_gev': 130.0, 'reaction': 'AU AU --> CHARGED X'}


def cls_of(v):
    return f"{float(v['low']):g}-{float(v['high']):g}%"


def head(hf, i, label, classes):
    return {'observable_type': 'particle_yields', 'inspire_id': '561518', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf} column {i})', 'collision_system': dict(SYSTEM),
            'centrality': {'classes': classes, **MA}, 'particle_id': {'species': 'charged'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    hf = 'tab1.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    classes = [cls_of(v) for v in t['independent_variables'][0]['values']]
    assert classes == ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%'], classes
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name']
        q = {k['name']: k['value'] for k in dv['qualifiers']}
        if name == 'NPART':
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '561518', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'N_part ({hf} column {i})',
                           'value_type': name, 'notes': {'value': f'{CITE}. Table 1.'},
                           'staging_reason': 'N_part: Glauber (HIJING)'})
            continue
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('tot',)}, (hf, i)
        if name == 'MULT':
            assert q['ETARAP'] == '-4.7 TO 4.7', q
            d = head(hf, i, 'integrated Nch in -4.7 <= eta <= 4.7 vs class', list(classes))
            d['kinematic_range'] = {'eta': {'min': -4.7, 'max': 4.7}, 'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='dNch_deta', value_type='dNch_deta', value_scale=9.4)
            d['notes'] = {'value': f'{CITE}, Table 1. The integrated charged multiplicity in -4.7 <= eta <= 4.7, Au+Au '
                                   '130 GeV vs class, compared as the mean dNch/deta over the window (value_scale = '
                                   '9.4, its width). Errors: total.'}
            public.append(d)
            continue
        assert name == 'DN/DETARAP', name
        x = float(q['ETARAP'])
        lo, hi = (-WINDOW / 2, WINDOW / 2) if x == 0 else (round(x - WINDOW / 2, 4), round(x + WINDOW / 2, 4))
        d = head(hf, i, f'dNch/deta at eta = {x:g} vs class', list(classes))
        d['kinematic_range'] = {'eta': {'min': lo, 'max': hi}, 'pT': {'min': 0.0, 'max': 100.0}}
        d.update(observable='dNch_deta', value_type='dNch_deta')
        d['notes'] = {'value': f'{CITE}, Table 1. dNch/deta at eta = {x:g} (here {lo:g} < eta < {hi:g}), Au+Au 130 GeV '
                               'vs class. Errors: total.'}
        public.append(d)
    hf = 'tab4.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
    assert pts == sorted(pts) and pts[0] > 0, pts
    for i, dv in enumerate(t['dependent_variables']):
        q = {k['name']: k['value'] for k in dv['qualifiers']}
        lo, hi = (float(x) for x in q['CENTRALITY'].split(' TO '))
        cls = f'{lo:g}-{hi:g}%'
        rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('tot',)}, (hf, i)
        p = [pts[r] for r in rows]
        d = head(hf, i, f'dNch/deta(|eta|), symmetrized detector average, {cls}', [cls])
        d['kinematic_range'] = {'eta': {'min': round(p[0] - WINDOW / 2, 4), 'max': round(p[-1] + WINDOW / 2, 4),
                                        'points': p, 'point_window': WINDOW},
                                'pT': {'min': 0.0, 'max': 100.0}}
        d.update(observable='dNch_deta', value_type='dNch_deta', dimensionality='differential_eta')
        if rows != list(range(len(pts))):
            d['table_rows'] = rows
        d['notes'] = {'value': f'{CITE}, Fig. 4. dNch/deta vs |eta|, symmetrized and averaged over the detector '
                               f'systems, Au+Au 130 GeV {cls}; 0.2-wide windows at the tabulated |eta| (positive '
                               'side). Errors: total.'}
        public.append(d)
    for hf in ('tab2a.yaml', 'tab2b.yaml', 'tab2c.yaml', 'tab2d.yaml', 'tab3.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            q = ', '.join(str(k['value']) for k in dv['qualifiers'])
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '561518', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']} ({q}) ({hf} column {i})",
                           'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}.'},
                           'staging_reason': ('dNch/deta per participant pair: Glauber' if hf == 'tab3.yaml' else
                                              'one detector system (Fig. 2; statistical errors only): the averaged '
                                              'Fig. 4 and Table 1 are public')})
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

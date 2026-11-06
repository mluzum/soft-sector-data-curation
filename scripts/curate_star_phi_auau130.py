#!/usr/bin/env python3
"""Curate STAR phi production at midrapidity in Au+Au at 130 GeV, INSPIRE 584631.

C. Adler et al. (STAR), "Midrapidity phi production in Au+Au collisions at
sqrt(sNN) = 130 GeV", PRC 65 (2002) 041901(R) (proceedings with the same
analysis: E. Yamamoto for STAR, hep-ph/0112017).

From the paper:
- phi -> K+ K- at |y| < 0.5, invariant yields 1/(2 pi mT) d2N/(dmT dy)
  vs mT - m_phi in three classes, 0-11, 11-26, 26-85% of the measured
  cross section, from the raw total charged multiplicity in |eta| < 0.75;
- Table 1: the inverse slope T, dN/dy (extrapolated with the exponential
  fit) and N_phi/N_h-; systematics 12% (T, dN/dy) and 22% (N_phi/N_h-),
  not tabulated; errors statistical only.

Records (public): the spectra (Fig. 2; 1/(2 pi mT) d2N/(dmT dy) = 1/(2 pi
pT) d2N/(dpT dy), in pT bins from mT - m bins 0.1 GeV wide around the
points; the isolated last point at 0.95 GeV of the 11-26 and 26-85%
spectra, after a gap, is left out: its bin is not given) and dN/dy vs
class.  Staged (--staged OUT): T (a fit parameter) and N_phi/N_h- (ratio
to the negative-hadron yield, not tabulated here).

Run from the repository root:  python scripts/curate_star_phi_auau130.py --staged OUT.yaml
"""
import argparse
import glob
import math
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins584631/HEPData-ins584631-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Adler et al. (STAR), PRC 65 (2002) 041901(R)'
OUT_S = 'data/particle_spectra/star_particle_spectra_phi_AuAu_130gev_ins584631.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_phi_AuAu_130gev_ins584631.yaml'
M_PHI = 1.019461
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': -0.75, 'max': 0.75}], 'notes': 'raw total charged multiplicity in |eta| < 0.75.'}
SYSTEM = {'energy_gev': 130.0, 'reaction': 'AU AU --> PHI X'}
CLASSES = ['0-11%', '11-26%', '26-85%']


def pt_of(dm):
    return round(math.sqrt((dm + M_PHI) ** 2 - M_PHI ** 2), 4)


def head(hf, i, label, otype, classes):
    return {'observable_type': otype, 'inspire_id': '584631', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf} column {i})', 'collision_system': dict(SYSTEM),
            'centrality': {'classes': classes, **NCH}, 'particle_id': {'species': 'phi'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, staged = [], [], []
    for k, cls in enumerate(CLASSES):
        hf = f'HEPData-gpc8-fig2.{k}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = t['dependent_variables'][0]
        q = {x['name']: x['value'] for x in dv['qualifiers']}
        lo, hi = (float(x) for x in q['Centrality'].split(' TO '))
        assert f'{lo:g}-{hi:g}%' == cls and q['RAPIDITY'] == '-0.5 TO 0.5', q
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        rows = [r for r, p in enumerate(pts) if r == 0 or abs(p - pts[r - 1] - 0.1) < 1e-6]
        assert rows == list(range(len(rows))) and len(pts) - len(rows) <= 1, (hf, pts)
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat',)}
        bins = [[pt_of(pts[r] - 0.05), pt_of(pts[r] + 0.05)] for r in rows]
        d = head(hf, 0, f'Fig. 2: phi invariant yield vs mT - m, {cls}', 'particle_spectra', [cls])
        d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5},
                                'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
        d.update(observable='invariant_yield', value_type='invariant_yield')
        if len(rows) < len(pts):
            d['table_rows'] = rows
        gap = (' The last point (mT - m = 0.95 GeV), after a gap, is left out: its bin is not given.'
               if len(rows) < len(pts) else '')
        d['notes'] = {'value': f'{CITE}, Fig. 2. 1/(2 pi mT) d2N/(dmT dy) = 1/(2 pi pT) d2N/(dpT dy) of phi (K+ K-), '
                               f'|y| < 0.5, Au+Au 130 GeV {cls}; pT bins from mT - m bins 0.1 GeV wide around the '
                               f'points.{gap} Errors: statistical.'}
        spectra.append(d)
    hf = 'HEPData-gpc8-tab1.0.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    classes = [f"{float(a):g}-{float(b):g}%" for a, b in (v['value'].split(' TO ') for v in
                                                         t['independent_variables'][0]['values'])]
    assert classes == CLASSES, classes
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name']
        if name == 'dN/dy':
            d = head(hf, i, 'Table 1: phi dN/dy vs class', 'particle_yields', list(CLASSES))
            d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='dN_dy', value_type='dN_dy')
            d['notes'] = {'value': f'{CITE}, Table 1. dN/dy of phi at |y| < 0.5, extrapolated to all pT with the '
                                   'exponential mT fit, Au+Au 130 GeV vs class. Errors: statistical (12% systematic '
                                   'not tabulated).'}
            yields.append(d)
            continue
        staged.append({'observable_type': 'particle_yields', 'inspire_id': '584631', 'version': VERSION,
                       'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{name} ({hf} column {i})',
                       'value_type': name, 'notes': {'value': f'{CITE}. Table 1.'},
                       'staging_reason': 'inverse slope T: a fit parameter' if 'Slope' in name else
                       'N_phi/N_h-: ratio to the negative-hadron yield, not tabulated here'})
    for out, docs in ((OUT_S, spectra), (OUT_Y, yields)):
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

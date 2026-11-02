#!/usr/bin/env python3
"""Curate STAR multi-strange baryon production in Au+Au at 130 GeV, INSPIRE 624566.

J. Adams et al. (STAR), "Multi-strange baryon production in Au-Au collisions
at sqrt(sNN) = 130 GeV", PRL 92, 182301 (2004), arXiv:nucl-ex/0307024.

From the paper:
- Xi-, anti-Xi+ and Omega- + anti-Omega+ at |y| < 0.75;
- centrality from the mid-rapidity negative-particle multiplicity
  (h-, |eta| < 0.5): 0-10, 10-25, 25-75% of the total hadronic cross
  section (Omega: 0-10% only);
- m_T spectra 1/(2 pi m_T) d2N/(dm_T dy) vs m_T - m (= the invariant yield
  1/(2 pi pT) d2N/(dpT dy), m_T dm_T = pT dpT); errors: statistical and
  pT-dependent systematic in quadrature (pT-independent 10% not included);
- Table I: dN/dy from exponential and Boltzmann fits in m_T.

Records (public): invariant yields of Xi-, anti-Xi+ (three classes) and
Omega- + anti-Omega+ (0-10%) (Fig. 1); the m_T - m points are converted to
pT per species, the bin edges at the m_T - m midpoints; dN/dy of each
species vs class from the exponential fit (Table I, first column; classes
from the h- values).  The HEPData Table I values are used (they differ
slightly from arXiv v1, e.g. Xi- 0-10% 2.04 vs 2.16).  Not curated: the
fit temperatures, particle ratios and model predictions (Fig. 2, Table II),
the freeze-out fits (Fig. 3a) and the <pT> compilation vs mass (Fig. 3b:
pi, K, p from other analyses, charge states not stated).  The 5 old records
filed the spectra as charged dN/dy in |eta| < 1 and the <pT> compilation
as Lambda.

Run from the repository root:  python scripts/curate_star_multistrange_130.py
"""
import collections
import glob
import math
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins624566/HEPData-ins624566-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRL 92, 182301 (2004), arXiv:nucl-ex/0307024'
HM = {'estimator': {'name': 'h-', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.5, 'max': 0.5}],
      'notes': 'Negative-particle multiplicity in |eta| < 0.5 (charged multiplicity used here).'}
Y = {'min': -0.75, 'max': 0.75}
MASS = {'Xi-': 1.32171, 'Xibar+': 1.32171, 'Omega- + Omegabar+': 1.67245}
HM_CLASS = {260.3: '0-10%', 163.6: '10-25%', 42.5: '25-75%'}
OLD = ['data/particle_yields/star_particle_yields_multi-strange_baryon_production_AuAu_130gev.yaml',
       'data/mean_pT/star_mean_pT_multi-strange_baryon_production_AuAu_200gev.yaml']


def species_of(q):
    if 'Omega' in q:
        return 'Omega- + Omegabar+'
    if 'bar{\\Xi}' in q:
        return 'Xibar+'
    if 'Xi' in q:
        return 'Xi-'
    raise KeyError(q)


def pt_of(x, m):
    return math.sqrt(x * x + 2.0 * x * m)


def system():
    return {'energy_gev': 130.0, 'reaction': 'AU AU --> X'}


def main():
    out = collections.defaultdict(list)
    for hf in sorted(os.listdir(RAW)):
        if not hf.startswith('fig1_'):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        xs = [float(v['value']) for v in t['independent_variables'][0]['values']]
        mids = [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:])]
        xe = [max(0.0, xs[0] - (mids[0] - xs[0]))] + mids + [xs[-1] + (xs[-1] - mids[-1])]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            sp = species_of(q['PARTICLES'])
            lo, hi = (float(x) for x in re.findall(r'[\d.]+', q['CENTRALITY']))
            cls = f'{lo:g}-{hi:g}%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            pairs = [[round(pt_of(xe[j], MASS[sp]), 4), round(pt_of(xe[j + 1], MASS[sp]), 4)] for j in rows]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '624566', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'm_T spectrum of {sp}, {cls} ({hf})',
                 'collision_system': system(), 'centrality': {'classes': [cls], **HM}, 'particle_id': {'species': sp},
                 'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}, Fig. 1. 1/(2 pi m_T) d2N/(dm_T dy) of {sp} (= the invariant yield), {cls}, '
                                    '|y| < 0.75; pT bins from the m_T - m points (edges at the midpoints) with m = '
                                    f'{MASS[sp]} GeV. Errors: statistical and pT-dependent systematic (pT-independent '
                                    '10% not included).'}}
            if rows != list(range(len(xs))):
                d['table_rows'] = rows
            out['particle_spectra'].append(d)
    for hf in ('tab1_xi.yaml', 'tab1_axi.yaml', 'tab1_om.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        sp = species_of(str(dv['qualifiers'][0]['value']))
        classes = [HM_CLASS[float(v['value'])] for v in iv['values']]
        out['particle_yields'].append({
            'observable_type': 'particle_yields', 'inspire_id': '624566', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': f'dN/dy of {sp} vs centrality, exponential fit ({hf})',
            'collision_system': system(), 'centrality': {'classes': classes, **HM}, 'particle_id': {'species': sp},
            'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'dN_dy', 'value_type': 'dN_dy',
            'notes': {'value': f'{CITE}, Table I. dN/dy of {sp} at |y| < 0.75 from an exponential fit in m_T (the '
                               'measured range holds ~75% (Xi) and ~66% (Omega) of the yield); classes from the h- '
                               'values. Errors: statistical and pT-dependent systematic (10% pT-independent not '
                               'included).'}})
    for f in OLD:
        if os.path.exists(f):
            assert all(str(x['inspire_id']) == '624566' for x in yaml.safe_load_all(open(f)) if x), f
            os.remove(f)
    for otype, docs in out.items():
        path = f'data/{otype}/star_{otype}_multistrange_AuAu_130gev_ins624566.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records;', [(d['particle_id']['species'], d['centrality']['classes']) for d in docs])


if __name__ == '__main__':
    main()

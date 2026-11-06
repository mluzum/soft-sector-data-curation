#!/usr/bin/env python3
"""Curate STAR proton and antiproton production in Au+Au at 130 GeV, INSPIRE 621642.

J. Adams et al. (STAR), "Rapidity and centrality dependence of proton and
anti-proton production from 197Au+197Au collisions at sqrt(sNN) = 130
GeV", PRC 70 (2004) 041901, arXiv:nucl-ex/0306029.

From the paper:
- centrality from the raw charged multiplicity in |eta| < 0.75, eight
  classes (the most central 6%): 0-6, 6-11, 11-18, 18-26, 26-34, 34-45,
  45-58, 58-85%;
- p and pbar from TPC dE/dx at |y| < 0.5, 0.35 < pT < 1.0 GeV/c; corrected
  for efficiency, background protons and pbar absorption; NOT for the
  feed-down of hyperon decays (acceptance > 95% for those);
- Fig. 1: 1/(2 pi pT) dN/(dy dpT) per class (statistical errors; ~10%
  efficiency systematics not tabulated); Fig. 2: dN/dy vs y (bins of 0.1,
  thermal-fit extrapolation in pT; stat and the bin-to-bin systematics)
  and <pT> vs y of p + pbar (their average); Fig. 3: <pT> and dN/dy
  (fiducial 0.35 < pT < 1.0 GeV/c, and extrapolated) vs class;
  extrapolation systematics up to 20% (dN/dy) and 6% (<pT>), not
  tabulated.

Records (public): spectra of p and pbar per class (16; pT bins midway
between the tabulated points), dN/dy(y) per class and species (16), <pT>(y)
of p + pbar per class (8), <pT> vs class of p, pbar, p + pbar (3), dN/dy
vs class fiducial and extrapolated of p, pbar (4).  Staged (--staged
OUT): the same vs the number of negative hadrons (Fig. 3 right panels).

Run from the repository root:  python scripts/curate_star_p_pbar_130.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins621642/HEPData-ins621642-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRC 70 (2004) 041901, arXiv:nucl-ex/0306029'
OUT_S = 'data/particle_spectra/star_particle_spectra_p_pbar_AuAu_130gev_ins621642.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_p_pbar_AuAu_130gev_ins621642.yaml'
OUT_M = 'data/mean_pT/star_mean_pT_p_pbar_AuAu_130gev_ins621642.yaml'
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': -0.75, 'max': 0.75}], 'notes': 'raw charged multiplicity in |eta| < 0.75.'}
SPECIES = {'$p$': ('p', 'P'), '$\\bar{p}$': ('pbar', 'PBAR'), '$p+\\bar{p}$': ('p + pbar', 'P PBAR')}
FEED = 'Not corrected for the feed-down of hyperon decays.'
CLASSES = ['0-6%', '6-11%', '11-18%', '18-26%', '26-34%', '34-45%', '45-58%', '58-85%']


def cls_of(s):
    lo, hi = (float(x) for x in re.fullmatch(r'([\d.]+) TO ([\d.]+)', str(s)).groups())
    return f'{lo:g}-{hi:g}%'


def bins_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    edges = [pts[0] - (mids[0] - pts[0])] + mids + [pts[-1] + (pts[-1] - mids[-1])]
    return [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]


def head(hf, i, label, otype, classes, species, prod):
    return {'observable_type': otype, 'inspire_id': '621642', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf} column {i})',
            'collision_system': {'energy_gev': 130.0, 'reaction': f'AU AU --> {prod} X'},
            'centrality': {'classes': classes, **NCH}, 'particle_id': {'species': species}}


def errs(dv):
    labels = {tuple(e['label'] for e in v['errors']) for v in dv['values']}
    assert len(labels) == 1, labels
    return {'$stat$': 'statistical', '$stat+syst$': 'statistical and bin-to-bin systematic'}[list(labels)[0][0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, means, staged = [], [], [], []
    for hf in ('fig1a.yaml', 'fig1b.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        bins = bins_from_points(pts)
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            assert q['RAPIDITY'] == '-0.5 TO 0.5' and q['SQRT(S)/NUCLEON'] == 130, q
            species, prod = SPECIES[q['PARTICLES']]
            cls = cls_of(q['CENTRALITY'])
            d = head(hf, i, f'Fig. 1: {species} invariant yield, {cls}', 'particle_spectra', [cls], species, prod)
            d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            d.update(observable='invariant_yield', value_type='invariant_yield')
            d['notes'] = {'value': f'{CITE}, Fig. 1. 1/(2 pi pT) dN/(dy dpT) of {species}, |y| < 0.5, Au+Au 130 GeV '
                                   f'{cls}. {FEED} pT bins midway between the tabulated points. Errors: {errs(dv)} '
                                   '(the ~10% efficiency systematics are not tabulated).'}
            spectra.append(d)
    for hf in ('fig2a.yaml', 'fig2b.yaml', 'fig2c.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        ys = [float(v['value']) for v in t['independent_variables'][0]['values']]
        assert all(abs(b - a - 0.1) < 1e-6 for a, b in zip(ys, ys[1:])), ys
        ybins = [[round(y - 0.05, 4), round(y + 0.05, 4)] for y in ys]
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            species, prod = SPECIES[q['PARTICLES']]
            cls = cls_of(q['CENTRALITY'])
            mean = hf == 'fig2c.yaml'
            d = head(hf, i, f"Fig. 2: {'<pT>' if mean else 'dN/dy'} of {species} vs y, {cls}",
                     'mean_pT' if mean else 'particle_yields', [cls], species, prod)
            d['kinematic_range'] = {'y': {'min': ybins[0][0], 'max': ybins[-1][1], 'bin_ranges': ybins},
                                    'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='mean_pT' if mean else 'dN_dy', value_type='mean_pT' if mean else 'dN_dy',
                     dimensionality='differential_y')
            d['notes'] = {'value': f'{CITE}, Fig. 2. {"<pT> (the average of p and pbar)" if mean else "dN/dy"} of '
                                   f'{species} per rapidity bin of 0.1, Au+Au 130 GeV {cls}; extrapolated in pT with '
                                   f'a thermal fit. {FEED} Errors: {errs(dv)} (extrapolation systematics up to '
                                   f'{"6" if mean else "20"}% not tabulated).'}
            (means if mean else yields).append(d)
    for hf in ('fig3.0.yaml', 'fig3.1.yaml', 'fig3.2.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        classes = [cls_of(v['value']) for v in t['independent_variables'][0]['values']]
        assert classes == CLASSES, classes
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            species, prod = SPECIES[q['PARTICLES']]
            name = dv['header']['name']
            mean = hf == 'fig3.0.yaml'
            fiducial = 'Fiducial' in name
            d = head(hf, i, f'Fig. 3: {name} of {species} vs class', 'mean_pT' if mean else 'particle_yields',
                     list(classes), species, prod)
            d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5},
                                    'pT': {'min': 0.35, 'max': 1.0} if fiducial else {'min': 0.0, 'max': 100.0}}
            d.update(observable='mean_pT' if mean else 'dN_dy', value_type='mean_pT' if mean else 'dN_dy')
            what = '<pT>' if mean else ('dN/dy in 0.35 < pT < 1.0 GeV/c' if fiducial else
                                        'dN/dy extrapolated to all pT (thermal fit)')
            d['notes'] = {'value': f'{CITE}, Fig. 3. {what} of {species}, |y| < 0.5, Au+Au 130 GeV vs class. {FEED} '
                                   f'Errors: {errs(dv)} (systematics: {"6" if mean else "10-22"}%, not tabulated).'}
            (means if mean else yields).append(d)
    for hf in ('fig3.0_nhm.yaml', 'fig3.1_nhm.yaml', 'fig3.2_nhm.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '621642', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']} vs the number of negative hadrons ({hf})",
                           'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. Fig. 3 (right).'},
                           'staging_reason': 'the Fig. 3 values again, vs the number of negative hadrons'})
    for out, docs in ((OUT_S, spectra), (OUT_Y, yields), (OUT_M, means)):
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

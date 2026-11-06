#!/usr/bin/env python3
"""Curate STAR K0_S and Lambda production in Au+Au at 3 GeV (fixed target), INSPIRE 2807679.

STAR Collaboration, "Strangeness production in sqrt(sNN) = 3 GeV Au+Au
collisions at RHIC", JHEP 10 (2024) 139, arXiv:2407.10110.

From the paper:
- fixed-target Au+Au at sqrt(sNN) = 3 GeV (2018); rapidities in the
  centre-of-momentum frame with the beam toward +y, the target at y_cm =
  -1.05; TPC acceptance -2 < eta_lab < 0;
- centrality from the number of reconstructed charged tracks in the TPC
  (all of its acceptance) with a Glauber model; classes 0-10, 10-20,
  20-30, 30-40, 40-60, 60-80%;
- K0_S and Lambda from their weak decays; Lambda corrected for the Xi- and
  Xi0 feed-down (Xi0 as Xi-; < 4% in 0-10%), Sigma0 decays count as prompt;
- spectra d2N/(2 pi pT dpT dy) per rapidity slice of 0.1 from -0.9 (K0_S
  -0.8) to 0; dN/dy per slice with a blast-wave extrapolation in pT; <pT>
  at -0.2 < y_cm < 0 vs <N_part> (the six classes);
- errors: stat and sys.

Records (public): the spectra per class and rapidity slice (Figs. 2, A),
dN/dy vs y per class (Tables 2, 3 of the paper; HEPData tables 1, 2) and
<pT> vs class (HEPData table 4).  Windows in the paper's frame: the
estimator's -2 < eta_lab < 0, beam toward -eta_lab, is -1.046 < eta_cm <
0.954 with the beam toward +y (a symmetric system: no frame fields).
HEPData names the pT axis of the spectra 'rapidity' and the rapidity axis
of the dN/dy tables 'Collision Centrality' (noted in each record).
pT bins midway between the tabulated points.  Staged (--staged OUT):
Lambda/p vs y (the protons from STAR PRC 110 054911), the 4pi yields over
<N_part> (three-Gaussian extrapolation in y), the mid-rapidity yield
ratios of 0-10% (Table 6), <pT> vs hadron mass (other papers' hadrons).

Run from the repository root:
  python scripts/curate_star_strange_3gev.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.curate_star_pt_correlations_fxt import y_cm  # noqa: E402

RAW = glob.glob('hepdata_raw/records/ins2807679/HEPData-ins2807679-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'STAR Collaboration, JHEP 10 (2024) 139, arXiv:2407.10110'
OUT_S = 'data/particle_spectra/star_particle_spectra_strange_AuAu_3gev_ins2807679.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_strange_AuAu_3gev_ins2807679.yaml'
OUT_M = 'data/mean_pT/star_mean_pT_strange_AuAu_3gev_ins2807679.yaml'
YC = y_cm(3.0)
FXT = {'estimator': {'name': 'FXTMult', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': round(-YC, 3), 'max': round(2.0 - YC, 3)}],
       'notes': f'reconstructed charged tracks in the TPC, -2 < eta_lab < 0 with the beam toward -eta_lab; in the '
                f'paper\'s centre-of-momentum frame (beam toward +y, target at y_cm = -{YC:.3f}) -{YC:.3f} < eta_cm < '
                f'{2.0 - YC:.3f}.'}
CLASSES = ['0-10%', '10-20%', '20-30%', '30-40%', '40-60%', '60-80%']
SPECIES = {'7': ('Lambda', 'LAMBDA'), '8': ('K0_S', 'K0S'), 1: ('Lambda', 'LAMBDA'), 2: ('K0_S', 'K0S')}
FEED = {'Lambda': ' Corrected for the Xi- and Xi0 weak-decay feed-down (Xi0 as Xi-); Sigma0 decays count as prompt.',
        'K0_S': ''}
FRAME = 'y_cm with the beam toward +y (the target at y_cm = -1.05)'


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rows_of(dv):
    return [i for i, v in enumerate(dv['values']) if num(v['value']) is not None]


def errors_of(dv, rows):
    labels = {tuple(e['label'] for e in dv['values'][r].get('errors', [])) for r in rows}
    assert labels == {('stat', 'sys')}, labels
    return 'stat and sys'


def bins_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    edges = [max(0.0, pts[0] - (mids[0] - pts[0]))] + mids + [pts[-1] + (pts[-1] - mids[-1])]
    return [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]


def head(species, prod, hf, i, label, otype, classes):
    return {'observable_type': otype, 'inspire_id': '2807679', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': 3.0, 'reaction': f'AU AU --> {prod} X'},
            'centrality': {'classes': classes, **FXT}, 'particle_id': {'species': species}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, means, staged = [], [], [], []
    for path in sorted(glob.glob(os.path.join(RAW, 'table_[78]_*.yaml'))):
        hf = os.path.basename(path)
        tab, cls = re.fullmatch(r'table_([78])_(\d+-\d+%)\.yaml', hf).groups()
        species, prod = SPECIES[tab]
        t = yaml.safe_load(open(path))
        assert t['independent_variables'][0]['header']['name'] == 'rapidity', hf      # pT, mislabelled
        x = [num(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            y0, y1 = (float(v) for v in re.fullmatch(r'\[(-?[\d.]+), ?(-?[\d.]+)\]', dv['header']['name']).groups())
            rows = rows_of(dv)
            pts = [x[r] for r in rows]
            assert pts == sorted(pts) and len(pts) >= 2, (hf, i, pts)
            bins = bins_from_points(pts)
            d = head(species, prod, hf, i, f'{species} invariant yield, {y0:g} < y_cm < {y1:g}, Au+Au 3 GeV, {cls} '
                     f'({hf} column {i})', 'particle_spectra', [cls])
            d['kinematic_range'] = {'y': {'min': y0, 'max': y1},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            d['observable'] = 'invariant_yield'
            d['value_type'] = 'invariant_yield'
            if rows != list(range(len(x))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. 1/(2 pi pT) d2N/(dpT dy) of {species} per event, {y0:g} < y_cm < {y1:g} '
                                   f'({FRAME}), Au+Au 3 GeV {cls}.{FEED[species]} pT bins midway between the '
                                   'tabulated points (HEPData names the pT axis \'rapidity\'; the column is the '
                                   f'rapidity slice). Errors: {errors_of(dv, rows)}.'}
            spectra.append(d)
    for tab in (1, 2):
        hf = f'table_{tab}.yaml'
        species, prod = SPECIES[tab]
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert t['independent_variables'][0]['header']['name'] == 'Collision Centrality', hf    # y, mislabelled
        yc = [num(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            cls = dv['header']['name']
            assert cls in CLASSES, (hf, cls)
            rows = rows_of(dv)
            ys = [yc[r] for r in rows]
            assert all(abs(b - a - 0.1) < 1e-6 for a, b in zip(ys, ys[1:])), ys
            bins = [[round(y - 0.05, 4), round(y + 0.05, 4)] for y in ys]
            d = head(species, prod, hf, i, f'{species} dN/dy vs y_cm, Au+Au 3 GeV, {cls} ({hf} column {i})',
                     'particle_yields', [cls])
            d['kinematic_range'] = {'y': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins},
                                    'pT': {'min': 0.0, 'max': 100.0}}
            d['observable'] = 'dN_dy'
            d['value_type'] = 'dN_dy'
            d['dimensionality'] = 'differential_y'
            if rows != list(range(len(yc))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. dN/dy of {species} per rapidity slice of 0.1 ({FRAME}), Au+Au 3 GeV '
                                   f'{cls}; pT-integrated with a blast-wave extrapolation.{FEED[species]} HEPData '
                                   'names the rapidity axis \'Collision Centrality\' and each column by its class. '
                                   f'Errors: {errors_of(dv, rows)}.'}
            yields.append(d)
    hf = 'table_4.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    npart = [num(v['value']) for v in t['independent_variables'][0]['values']]
    assert len(npart) == len(CLASSES) and npart == sorted(npart, reverse=True), npart
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name']
        species, prod = {'$\\Lambda$': ('Lambda', 'LAMBDA'), '$K^{0}_{S}$': ('K0_S', 'K0S')}[name]
        rows = rows_of(dv)
        assert rows == list(range(len(CLASSES))), rows
        d = head(species, prod, hf, i, f'{species} <pT> vs <N_part>, -0.2 < y_cm < 0, Au+Au 3 GeV ({hf} column {i})',
                 'mean_pT', CLASSES)
        d['centrality']['notes'] = (FXT['notes'] + ' Rows by <N_part> (' + ', '.join(f'{n:g}' for n in npart)
                                    + '): the six classes in order.')
        d['kinematic_range'] = {'y': {'min': -0.2, 'max': 0.0}, 'pT': {'min': 0.0, 'max': 100.0}}
        d['observable'] = 'mean_pT'
        d['value_type'] = 'mean_pT'
        d['notes'] = {'value': f'{CITE}. <pT> of {species}, -0.2 < y_cm < 0 ({FRAME}), Au+Au 3 GeV, vs class '
                               f'(HEPData: vs <N_part>); blast-wave extrapolation in pT.{FEED[species]} Errors: '
                               f'{errors_of(dv, rows)}.'}
        means.append(d)
    reasons = {'table_3.yaml': 'Lambda/p vs y: the protons are STAR PRC 110 054911\'s',
               'table_5.yaml': '4pi yields over <N_part>: extrapolated in y with three Gaussians (not computed)',
               'table_6.yaml': 'mid-rapidity yield ratios Lambda/p, Xi/Lambda, K0_S/Lambda, 0-10% (protons and Xi '
                               'from other papers)',
               'table_9.yaml': '<pT> vs hadron mass (p, Xi, phi, K- from other papers)'}
    for hf, reason in reasons.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '2807679', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                           'collision_system': {'energy_gev': 3.0, 'reaction': 'AU AU --> X'},
                           'value_type': dv['header']['name'],
                           'notes': {'value': f'{CITE}. {hf}: {dv["header"]["name"]}.'}, 'staging_reason': reason})
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

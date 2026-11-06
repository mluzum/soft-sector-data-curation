#!/usr/bin/env python3
"""Curate STAR phi production in Au+Au at 200 GeV (2004), INSPIRE 651461.

J. Adams et al. (STAR), "phi meson production in Au+Au and p+p collisions
at sqrt(sNN) = 200 GeV", PLB 612 (2005) 181, arXiv:nucl-ex/0406003.

From the paper:
- phi -> K+ K- at |y| < 0.5 (event mixing); centrality from the raw
  hadron multiplicity in |eta| <= 0.5, classes 50-80, 30-50, 10-30%, top
  10% and top 5% (the 0-5% from the central-trigger data set);
- Fig. 1: 1/(2 pi mT) dN/(dmT dy) = 1/(2 pi pT) d2N/(dpT dy) in the pT
  bins HEPData gives, statistical errors only (systematics 11% in the
  yield, not tabulated);
- Table 1: T, dN/dy and <pT> from single-exponential fits (Au+Au), stat
  and sys.

Records (public): the Au+Au spectra (5) and dN/dy and <pT> vs class (the
table's 0-5% and 0-10% rows overlap: one record with 0-5, 10-30, 30-50,
50-80%, another with 0-10%; 4).  Staged (--staged OUT): p+p (no
centrality), T (a fit parameter), R_CP and R_AA (Fig. 3).

Run from the repository root:  python scripts/curate_star_phi_auau200_2004.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins651461/HEPData-ins651461-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PLB 612 (2005) 181, arXiv:nucl-ex/0406003'
OUT_S = 'data/particle_spectra/star_particle_spectra_phi_AuAu_200gev_ins651461.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_phi_AuAu_200gev_ins651461.yaml'
OUT_M = 'data/mean_pT/star_mean_pT_phi_AuAu_200gev_ins651461.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': 'raw hadron multiplicity in |eta| <= 0.5.'}
SYSTEM = {'energy_gev': 200.0, 'reaction': 'AU AU --> PHI X'}
ROW_SETS = ([0, 2, 3, 4], [1])


def head(hf, i, label, otype, classes):
    return {'observable_type': otype, 'inspire_id': '651461', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf} column {i})', 'collision_system': dict(SYSTEM),
            'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': 'phi'}}


def stage(hf, i, name, reason):
    return {'observable_type': 'particle_yields', 'inspire_id': '651461', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{name} ({hf} column {i})', 'value_type': name,
            'notes': {'value': f'{CITE}.'}, 'staging_reason': reason}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, means, staged = [], [], [], []
    hf = 'figure_1.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        q = dv['qualifiers'][0]['value']
        if q == 'pp NSD':
            staged.append(stage(hf, i, 'phi spectrum, p+p NSD', 'p+p: no centrality classes'))
            continue
        cls = q
        assert dv['header']['name'] == f'{cls} Centrality', dv['header']
        rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat',)}
        rb = [bins[r] for r in rows]
        d = head(hf, i, f'Fig. 1: phi invariant yield, Au+Au 200 GeV {cls}', 'particle_spectra', [cls])
        d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5},
                                'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}}
        d.update(observable='invariant_yield', value_type='invariant_yield')
        if rows != list(range(len(bins))):
            d['table_rows'] = rows
        trig = ' (central-trigger data set)' if cls == '0-5%' else ''
        d['notes'] = {'value': f'{CITE}, Fig. 1. 1/(2 pi mT) dN/(dmT dy) = 1/(2 pi pT) d2N/(dpT dy) of phi (K+ K-), '
                               f'|y| < 0.5, Au+Au 200 GeV {cls}{trig}. Errors: statistical (11% systematic not '
                               'tabulated).'}
        spectra.append(d)
    hf = 'table_1.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    ivs = t['independent_variables'][0]['values']
    labels = [f"{v['low']:g}-{v['high']:g}%" for v in ivs[:5]]
    assert labels == ['0-5%', '0-10%', '10-30%', '30-50%', '50-80%'] and ivs[5]['value'] == 'ppNSD', ivs
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name']
        if 'Slope' in name:
            staged.append(stage(hf, i, name, 'inverse slope T: a fit parameter'))
            continue
        mean = name == '$<p_T>$'
        assert mean or name == '$dN/dy$', name
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat', 'sys')}
        for rows in ROW_SETS:
            classes = [labels[r] for r in rows]
            d = head(hf, i, f"Table 1: phi {'<pT>' if mean else 'dN/dy'} vs class ({', '.join(classes)})",
                     'mean_pT' if mean else 'particle_yields', classes)
            d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='mean_pT' if mean else 'dN_dy', value_type='mean_pT' if mean else 'dN_dy',
                     table_rows=rows)
            d['notes'] = {'value': f"{CITE}, Table 1. {'<pT>' if mean else 'dN/dy'} of phi at |y| < 0.5, extrapolated "
                                   'to all pT with a single-exponential mT fit, Au+Au 200 GeV (the 0-5% and 0-10% '
                                   'rows overlap: one record each). Errors: stat and sys.'}
            (means if mean else yields).append(d)
    hf = 'figure_3.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    for i, dv in enumerate(t['dependent_variables']):
        s = stage(hf, i, dv['header']['name'], 'R_CP / R_AA: N_bin scaled, or needs the pp reference')
        s['observable_type'] = 'nuclear_modification'
        staged.append(s)
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

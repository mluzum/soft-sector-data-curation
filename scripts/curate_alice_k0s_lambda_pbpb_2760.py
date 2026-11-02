#!/usr/bin/env python3
"""Curate ALICE K0_S and Lambda production in Pb-Pb at 2.76 TeV, INSPIRE 1243863.

B. Abelev et al. (ALICE), "K0_S and Lambda production in Pb-Pb collisions
at sqrt(sNN) = 2.76 TeV", PRL 111, 222301 (2013), arXiv:1307.5530.

From the paper:
- K0_S and Lambda at |y| < 0.5 (decay daughters in |eta| < 0.8);
  centrality from the VZERO amplitudes (V0M), 0-90%;
- Lambda corrected for the feed-down from Xi- and Xi0 decays;
- d2N/(dpT dy) per event; dN/dy with the spectra extrapolated below 0.4
  (K0_S) and 0.6 (Lambda) GeV/c (the fraction tabulated);
- errors: stat and sys.

Records (public): K0_S and Lambda d2N/(dpT dy) in 0-5, 5-10, 10-20,
20-40, 40-60, 60-80, 80-90% (Tables 1-14, value_type dN_dpT_dy) and dN/dy
vs centrality (Tables 22, 23).  Not curated: the Lambda/K0_S ratios
(Tables 15-21: ratios of curated spectra) and the extrapolated fractions.
The 2 old records filed dN/dy without the y window and Lambda as
inclusive.

Run from the repository root:  python scripts/curate_alice_k0s_lambda_pbpb_2760.py
"""
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1243863/HEPData-ins1243863-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. Abelev et al. (ALICE), PRL 111, 222301 (2013), arXiv:1307.5530'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
Y = {'min': -0.5, 'max': 0.5}
OLD = 'data/particle_yields/alice_particle_yields_k_0_s_and_lambda_PbPb_2760gev.yaml'
FEED = '; corrected for the feed-down from Xi- and Xi0'


def load(n):
    return yaml.safe_load(open(os.path.join(RAW, f'Table{n}.yaml')))


def main():
    out = collections.defaultdict(list)
    for n in range(1, 15):
        sp = 'K0_S' if n <= 7 else 'Lambda'
        t = load(n)
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'particle_spectra', 'inspire_id': '1243863', 'version': VERSION,
             'hepdata_file': f'Table{n}.yaml', 'dv_index': 0,
             'hepdata_table_label': f'd2N/(dpT dy) of {sp}, {cls} (Table{n}.yaml)',
             'collision_system': {'energy_gev': 2760.0, 'reaction': f'PB PB --> {sp.upper()} X'},
             'centrality': {'classes': [cls], **V0M}, 'particle_id': {'species': sp},
             'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy', 'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}. 1/N_ev d2N/(dpT dy) of {sp}, {cls}, |y| < 0.5'
                                + (FEED if sp == 'Lambda' else '') + '. Errors: stat and sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        out['particle_spectra'].append(d)
    for n, sp, ext in ((22, 'K0_S', 0.4), (23, 'Lambda', 0.6)):
        t = load(n)
        iv = t['independent_variables'][0]
        classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
        out['particle_yields'].append({
            'observable_type': 'particle_yields', 'inspire_id': '1243863', 'version': VERSION,
            'hepdata_file': f'Table{n}.yaml', 'dv_index': 0,
            'hepdata_table_label': f'dN/dy of {sp} vs centrality (Table{n}.yaml)',
            'collision_system': {'energy_gev': 2760.0, 'reaction': f'PB PB --> {sp.upper()} X'},
            'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': sp},
            'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'dN_dy', 'value_type': 'dN_dy',
            'notes': {'value': f'{CITE}. dN/dy of {sp} vs centrality, |y| < 0.5, the spectrum extrapolated below '
                               f'{ext} GeV/c' + (FEED if sp == 'Lambda' else '') + '. Errors: stat and sys.'}})
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1243863' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    for otype, docs in out.items():
        path = f'data/{otype}/alice_{otype}_k0s_lambda_PbPb_2760gev_ins1243863.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records')


if __name__ == '__main__':
    main()

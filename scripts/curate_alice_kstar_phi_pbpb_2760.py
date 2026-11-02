#!/usr/bin/env python3
"""Curate ALICE K*0 and phi production in Pb-Pb at 2.76 TeV, INSPIRE 1288320.

B. Abelev et al. (ALICE), "K*(892)0 and phi(1020) production in Pb-Pb
collisions at sqrt(sNN) = 2.76 TeV", PRC 91, 024609 (2015),
arXiv:1404.0495.

From the paper and tables:
- K*0 -> K pi (particle and antiparticle averaged) and phi -> K+ K- at
  |y| < 0.5; centrality from the V0 amplitude (V0M);
- d2N/(dpT dy) per event, dN/dy and <pT> (fits extrapolated to all pT);
- errors: stat, sys uncorrelated, normalization and pT- (centrality-)
  correlated sys, disjoint.

Records (public): phi d2N/(dpT dy) in 0-5, 5-10, 0-10, 10-20 ... 80-90%
(Tables 5-15, value_type dN_dpT_dy; 0-10% overlaps 0-5 and 5-10%, a
separate record), phi dN/dy and <pT> vs centrality (Tables 21, 25).
Staged (--staged OUT): K*0 spectra, dN/dy, <pT> (a resonance not provided
by the pipeline: Tables 1-4, 20, 24).  Not curated: masses and widths
(Tables 16-19), particle ratios (22, 23, 26-33, 36: ratios to yields of
other papers) and the enhancement relative to pp per N_part (34, 35).  The
4 old records filed K*0, phi and the phi and Lambda enhancements as dN/dy.

Run from the repository root:
    python scripts/curate_alice_kstar_phi_pbpb_2760.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1288320/HEPData-ins1288320-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. Abelev et al. (ALICE), PRC 91, 024609 (2015), arXiv:1404.0495'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
Y = {'min': -0.5, 'max': 0.5}
OLD = 'data/particle_yields/alice_particle_yields_k_892_0_and_phi_PbPb_2760gev.yaml'
KSTAR = {'species': 'K*0 + K*0bar', 'charge_average': True}
ERRORS = 'stat, sys uncorrelated, normalization and correlated sys'


def load(n):
    return yaml.safe_load(open(os.path.join(RAW, f'Table{n}.yaml')))


def cls_of(v):
    return f"{float(v['low']):g}-{float(v['high']):g}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out, staged = collections.defaultdict(list), []

    def keep(d, kstar):
        if kstar:
            d['staging_reason'] = 'K*(892)0, a resonance not provided by the pipeline'
            staged.append(d)
        else:
            out[d['observable_type']].append(d)

    for n in range(1, 16):                                   # spectra: K*0 1-4, phi 5-15
        kstar = n <= 4
        t = load(n)
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        sp = '(K*0 + K*0bar)/2' if kstar else 'phi'
        d = {'observable_type': 'particle_spectra', 'inspire_id': '1288320', 'version': VERSION,
             'hepdata_file': f'Table{n}.yaml', 'dv_index': 0,
             'hepdata_table_label': f'd2N/(dpT dy) of {sp}, {cls} (Table{n}.yaml)',
             'collision_system': {'energy_gev': 2760.0, 'reaction': q['RE'].split(' < ')[0].replace('(', '').strip() + ' X'
                                  if not kstar else 'PB PB --> K*0 X'},
             'centrality': {'classes': [cls], **V0M},
             'particle_id': dict(KSTAR) if kstar else {'species': 'phi'},
             'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy', 'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}. 1/N_ev d2N/(dpT dy) of {sp}, {cls}, |y| < 0.5. Errors: {ERRORS}.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        keep(d, kstar)
    for n, otype, kstar in ((20, 'particle_yields', True), (21, 'particle_yields', False),
                            (24, 'mean_pT', True), (25, 'mean_pT', False)):
        t = load(n)
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        classes = [cls_of(iv['values'][j]) for j in rows]
        sp = '(K*0 + K*0bar)/2' if kstar else 'phi'
        what, vt = ('dN/dy', 'dN_dy') if otype == 'particle_yields' else ('<pT>', 'mean_pT')
        d = {'observable_type': otype, 'inspire_id': '1288320', 'version': VERSION, 'hepdata_file': f'Table{n}.yaml',
             'dv_index': 0, 'hepdata_table_label': f'{what} of {sp} vs centrality (Table{n}.yaml)',
             'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> K*0 X' if kstar else 'PB PB --> PHI X'},
             'centrality': {'classes': classes, **V0M},
             'particle_id': dict(KSTAR) if kstar else {'species': 'phi'},
             'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': vt, 'value_type': vt,
             'notes': {'value': f'{CITE}. {what} of {sp} vs centrality, |y| < 0.5, all pT (fits extrapolated). '
                                f"Errors: {ERRORS if otype == 'particle_yields' else 'stat and sys'}."}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        keep(d, kstar)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1288320' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    for otype, docs in out.items():
        path = f'data/{otype}/alice_{otype}_phi_PbPb_2760gev_ins1288320.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

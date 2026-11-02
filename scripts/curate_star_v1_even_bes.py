#!/usr/bin/env python3
"""Curate STAR rapidity-even v1 in Au+Au at 7.7-200 GeV, INSPIRE 1669807 (staged only).

J. Adam et al. (STAR), "Beam energy dependence of rapidity-even dipolar
flow in Au+Au collisions", PLB 784 (2018) 26, arXiv:1804.08647.

From the paper:
- BES-I; TPC tracks |eta| < 1, 0.2 < pT < 4 GeV/c; centrality from the
  multiplicity (RefMult, |eta| < 0.5) with a Glauber calculation;
- v11(pTa, pTb) = <cos Delta phi> from same/mixed-event pair
  correlation functions, |Delta eta| > 0.7;
- v1^even(pT) and K from a simultaneous fit of the v11 matrix,
  v11 = v1(pTa) v1(pTb) - K pTa pTb (K: global momentum conservation);
- errors: stat and syst (v11: total).

Every record is staged (--staged OUT): v11 vs pTb for five pTa windows,
0-5% at 200 GeV (Table 1: a pair coefficient, not a flow coefficient);
v1^even(pT), 0-10%, eight energies (Table 2); |v1^even| vs centrality at
0.4 < pT < 0.7 GeV/c (Table 3a; the third energy is 27 GeV in HEPData,
19.6 GeV in the caption); v1^even and v3 vs energy at 0.4 < pT < 0.7
GeV/c (Table 4, one record per energy; the class is 10-20% in HEPData,
0-10% in the caption).  Rapidity-even v1 from the momentum-conservation fit
is not implemented; pT points carry no bin edges (midpoints here).  Not
curated: K vs <N_ch> (Table 3b, a fit parameter).  The old public record
read Table 1 as v2 with reaction 'unknown'.

Run from the repository root:  python scripts/curate_star_v1_even_bes.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1669807/HEPData-ins1669807-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (STAR), PLB 784 (2018) 26, arXiv:1804.08647'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
ETA = {'min': -1.0, 'max': 1.0}
OLD = 'data/flow_vn/star_flow_vn_beam_energy_dependence_of_rapidity-even_200gev.yaml'
EVEN = ('rapidity-even v1 from a simultaneous fit of v11(pTa, pTb) = v1(pTa) v1(pTb) - K pTa pTb (global momentum '
        'conservation): not implemented')
BLANK = ('', '-', '--', 'none')


def edges(x, lo=0.2):
    e = [lo] + [round((a + b) / 2, 4) for a, b in zip(x, x[1:])] + [round(x[-1] + (x[-1] - x[-2]) / 2, 4)]
    return [[a, b] for a, b in zip(e, e[1:])]


def base(hf, i, label, energy, harmonic=1):
    return {'observable_type': 'flow_vn', 'inspire_id': '1669807', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf})',
            'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> X'}, 'particle_id': {'species': 'charged'},
            'method': 'cumulant', 'cumulant_order': 2, 'eta_gap': 0.7, 'harmonic': harmonic}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    staged = []
    # Table 1: v11 vs pTb for pTa windows, 0-5% at 200 GeV
    t = yaml.safe_load(open(os.path.join(RAW, 'table1.yaml')))
    x = [float(v['value']) for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        a, b = (float(s) for s in re.search(r'\[([\d.]+)-([\d.]+)\]', h).groups())
        bins = edges(x)
        d = base('table1.yaml', i, f'v11 vs pTb, pTa {a:g}-{b:g} GeV/c, 0-5%', 200.0)
        d.update({'value_type': h, 'observable': 'v11', 'centrality': {'classes': ['0-5%'], **REFMULT},
                  'dimensionality': 'differential_pT',
                  'kinematic_range': {'eta': dict(ETA), 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                  'flow_reference': {'pT': {'min': a, 'max': b}, 'notes': 'The pTa window (the other pair member).'},
                  'notes': {'value': f'{CITE}, Fig. 1. v11 = <cos(phi_a - phi_b)> vs pTb for {a:g} < pTa < {b:g} GeV/c, '
                                     '0-5%, |eta| < 1, |Delta eta| > 0.7; pTb points at their mean (edges: midpoints).'},
                  'staging_reason': 'two-particle Fourier coefficient v11(pTa, pTb), not a flow coefficient: no '
                                    'computation; pTb bin edges not given'})
        staged.append(d)
    # Table 2: v1^even(pT), 0-10%, per energy
    t = yaml.safe_load(open(os.path.join(RAW, 'table2.yaml')))
    x = [float(v['value']) for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        e = float(re.search(r'([\d.]+)\s*GeV', h).group(1))
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        bins = [edges(x)[j] for j in rows]
        d = base('table2.yaml', i, f'v1^even(pT), 0-10%, {e:g} GeV', e)
        d.update({'value_type': h, 'centrality': {'classes': ['0-10%'], **REFMULT}, 'dimensionality': 'differential_pT',
                  'kinematic_range': {'eta': dict(ETA), 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                  'notes': {'value': f'{CITE}, Fig. 2. v1^even(pT), 0-10%, {e:g} GeV, |eta| < 1, from the fit of '
                                     'v11 (|Delta eta| > 0.7); pT points at their mean (edges: midpoints). Errors: stat '
                                     'and syst.'},
                  'staging_reason': EVEN + '; pT bin edges not given'})
        if rows != list(range(len(x))):
            d['table_rows'] = rows
        staged.append(d)
    # Table 3a: |v1^even| vs centrality at 0.4-0.7 GeV/c
    t = yaml.safe_load(open(os.path.join(RAW, 'table3_a.yaml')))
    c = [float(v['value']) for v in t['independent_variables'][0]['values']]
    cls_all = []                                     # centres 2.5, 7.5, 15, ... -> 0-5, 5-10, 10-20, ...
    prev = 0.0
    for cc in c:
        hi = 2 * cc - prev
        cls_all.append(f'{prev:g}-{hi:g}%')
        prev = hi
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        e = float(re.search(r'([\d.]+)\s*GeV', h).group(1))
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        d = base('table3_a.yaml', i, f'|v1^even| vs centrality, 0.4-0.7 GeV/c, {e:g} GeV', e)
        d.update({'value_type': h, 'centrality': {'classes': [cls_all[j] for j in rows], **REFMULT},
                  'dimensionality': 'integrated', 'kinematic_range': {'eta': dict(ETA), 'pT': {'min': 0.4, 'max': 0.7}},
                  'notes': {'value': f'{CITE}, Fig. 3a. pT-weighted |v1^even| at 0.4 < pT < 0.7 GeV/c vs centrality, '
                                     f'{e:g} GeV (HEPData; the caption lists 200, 39 and 19.6 GeV). Errors: stat and '
                                     'syst.'},
                  'staging_reason': EVEN})
        if rows != list(range(len(c))):
            d['table_rows'] = rows
        staged.append(d)
    # Table 4: v1^even and v3 vs energy at 0.4-0.7 GeV/c, one record per energy
    t = yaml.safe_load(open(os.path.join(RAW, 'table4.yaml')))
    es = [float(v['value']) for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        n = 3 if 'v_{3}' in h else 1
        for j, e in enumerate(es):
            if str(dv['values'][j]['value']).strip().lower() in BLANK:
                continue
            d = base('table4.yaml', i, f"{'v3' if n == 3 else 'v1^even'}, 0.4-0.7 GeV/c, {e:g} GeV", e, n)
            d.update({'value_type': h, 'centrality': {'classes': ['10-20%'], **REFMULT}, 'dimensionality': 'integrated',
                      'kinematic_range': {'eta': dict(ETA), 'pT': {'min': 0.4, 'max': 0.7}}, 'table_rows': [j],
                      'notes': {'value': f"{CITE}, Fig. 4. {'v3' if n == 3 else '|v1^even|'} at 0.4 < pT < 0.7 GeV/c, "
                                         f'{e:g} GeV, from the pair correlations (|Delta eta| > 0.7); class 10-20% in '
                                         'HEPData, 0-10% in the caption. Errors: stat and syst.'},
                      'staging_reason': ('the class is ambiguous (10-20% in HEPData, 0-10% in the caption)'
                                         if n == 3 else EVEN + '; the class is ambiguous (10-20% in HEPData, 0-10% in '
                                                               'the caption)')})
            staged.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1669807' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    print(f'{len(staged)} staged records; removed {OLD}' if not os.path.exists(OLD) else '')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

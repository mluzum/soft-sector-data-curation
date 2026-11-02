#!/usr/bin/env python3
"""Curate STAR v2{2} and v2{4} of charged particles up to high pT in Au+Au at 200 GeV, INSPIRE 654226.

J. Adams et al. (STAR), "Azimuthal anisotropy and correlations at large
transverse momenta in p+p and Au+Au collisions at sqrt(sNN) = 200 GeV",
PRL 93, 252301 (2004), nucl-ex/0407007.

From the paper:
- TPC tracks (-1.3 < eta < 1.3); centrality from the mid-rapidity
  multiplicity; the flow-vector (generating-function) tracks with the cuts
  of the 130 GeV analysis (PRC 66, 034904) but pT > 0.15 GeV/c;
- v2{2} and v2{4}(pT) from cumulants for 20-60% of the cross section; v2{2}
  also after subtracting the p+p correlations;
- v2{4} at 3 < pT < 6 GeV/c vs impact parameter (Glauber);
- errors: statistical.

Records (public): v2{2}(pT) and v2{4}(pT) of charged particles, 20-60%
(Fig. 2), |eta| < 1.3, the reference particles 0.15 < pT < 2 GeV/c (the
130 GeV analysis's integrated range with the 0.15 GeV/c cutoff).  Staged
(--staged OUT): v2{2} with the p+p correlations subtracted (Fig. 2) and
v2{4} at 3 < pT < 6 GeV/c vs impact parameter (Fig. 4: the classes are
Glauber b ranges; the matching percentile classes 5-10 ... 60-70% are
inferred, not stated).  Not curated: the pair-correlation sums (Fig. 1)
and the dihadron distributions (Fig. 3: jet correlations).  The old record
read the b ranges of Fig. 4 as percent classes.

Run from the repository root:
    python scripts/curate_star_v2_high_pt_200.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins654226/HEPData-ins654226-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRL 93, 252301 (2004), nucl-ex/0407007'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
ETA = {'min': -1.3, 'max': 1.3}
REF = {'pT': {'min': 0.15, 'max': 2.0},
       'notes': 'Generating-function particles as in the 130 GeV analysis (0.1 < pT < 2 GeV/c), with 0.15 GeV/c here.'}
OLD = 'data/flow_vn/star_flow_vn_azimuthal_anisotropy_and_correlations_200gev.yaml'
OUT = 'data/flow_vn/star_flow_vn_v2_cumulants_high_pt_AuAu_200gev_ins654226.yaml'
B_CLASSES = ['60-70%', '50-60%', '40-50%', '30-40%', '20-30%', '10-20%', '5-10%']   # fig4 rows, b decreasing
BLANK = ('', '-', '--', 'none')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    t = yaml.safe_load(open(os.path.join(RAW, 'fig2.yaml')))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        order = 4 if '{4}' in h else 2
        sub = 'pp' in h
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        what = 'v2{2} with the p+p correlations subtracted' if sub else f'v2{{{order}}}'
        d = {'observable_type': 'flow_vn', 'inspire_id': '654226', 'version': VERSION, 'hepdata_file': 'fig2.yaml',
             'dv_index': i, 'hepdata_table_label': f'{what}(pT), 20-60% (fig2.yaml)',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'}, 'particle_id': {'species': 'charged'},
             'value_type': h, 'method': 'cumulant', 'cumulant_order': order, 'harmonic': 2,
             'flow_reference': dict(REF), 'centrality': {'classes': ['20-60%'], **REFMULT},
             'dimensionality': 'differential_pT',
             'kinematic_range': {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'notes': {'value': f'{CITE}, Fig. 2. {what}(pT) of charged particles, 20-60%, |eta| < 1.3, reference '
                                'particles 0.15 < pT < 2 GeV/c. Errors: statistical.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        if sub:
            d['staging_reason'] = 'v2{2} minus the p+p correlations (a subtraction using another system): not computed'
            staged.append(d)
        else:
            public.append(d)
    t = yaml.safe_load(open(os.path.join(RAW, 'fig4.yaml')))
    iv = t['independent_variables'][0]
    dv = t['dependent_variables'][0]
    b = [f"{float(v['low']):g}-{float(v['high']):g} fm" for v in iv['values']]
    staged.append({'observable_type': 'flow_vn', 'inspire_id': '654226', 'version': VERSION, 'hepdata_file': 'fig4.yaml',
                   'dv_index': 0, 'hepdata_table_label': 'v2{4} at 3 < pT < 6 GeV/c vs impact parameter (fig4.yaml)',
                   'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                   'particle_id': {'species': 'charged'}, 'value_type': dv['header']['name'], 'method': 'cumulant',
                   'cumulant_order': 4, 'harmonic': 2, 'flow_reference': dict(REF),
                   'centrality': {'classes': list(B_CLASSES), **REFMULT}, 'dimensionality': 'integrated',
                   'kinematic_range': {'eta': dict(ETA), 'pT': {'min': 3.0, 'max': 6.0}},
                   'notes': {'value': f"{CITE}, Fig. 4. v2{{4}} at 3 < pT < 6 GeV/c vs impact parameter ({', '.join(b)}; "
                                      'Glauber); the percentile classes are inferred from the b values. Errors: '
                                      'statistical.'},
                   'staging_reason': 'centrality given as Glauber impact-parameter ranges; the percentile classes are '
                                     'inferred, not stated'})
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '654226' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

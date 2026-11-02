#!/usr/bin/env python3
"""Curate ALICE v_n(eta) over -3.5 < eta < 5 in Pb-Pb at 2.76 TeV, INSPIRE 1456145.

J. Adam et al. (ALICE), "Pseudorapidity dependence of the anisotropic flow
of charged particles in Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", PLB 762
(2016) 376, arXiv:1605.02035.

From the paper (Sec. 2):
- particles of interest: SPD and FMD hits, -3.5 < eta < 5.0 (no pT
  measurement: all pT);
- reference particles: TPC tracks |eta| < 0.8, 0.2 < pT < 5.0 GeV/c, taken
  in the half opposite the hit ("for SPD hits with eta < 0, tracks are
  required to have eta > 0 and vice versa"; the same for the FMD);
- v_n{2}(eta) and v2{4}(eta) from Q-cumulants; V0 centrality (V0M), 0-80%;
- eta points at the centres of 0.5-wide bins.

Records (public): v2{2}, v3{2}, v4{2} (eta) per class (Tables 1-9), the
mirrored TPC reference (computed with the scalar-product estimator).
Staged (--staged OUT): v2{4}(eta) (the cumulant keys take the reference
in the record acceptance; a separate reference window is not expressed).
The 9 old records held v2{2} only, with the whole TPC as a subevent
overlapping the hits.

Run from the repository root:
    python scripts/curate_alice_vn_eta_2760.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1456145/HEPData-ins1456145-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (ALICE), PLB 762 (2016) 376, arXiv:1605.02035'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
TPC_REF = {'detector': 'TPC', 'mirror': True, 'eta': [{'min': 0.0, 'max': 0.8}], 'pT': {'min': 0.2, 'max': 5.0},
           'notes': 'TPC tracks |eta| < 0.8, 0.2 < pT < 5 GeV/c, in the half opposite the hit.'}
OLD = 'data/flow_vn/alice_flow_vn_pseudorapidity_dependence_of_the_anisotropic_PbPb_2760gev.yaml'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for n in range(1, 10):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            m = re.fullmatch(r'v\$_\{(\d)\}\\\{(\d)\\\}\$', h)
            harm, order = int(m.group(1)), int(m.group(2))
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
            cls = f'{lo:g}-{hi:g}%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pairs = [[round(float(iv['values'][j]['value']) - 0.25, 3), round(float(iv['values'][j]['value']) + 0.25, 3)]
                     for j in rows]
            d = {'observable_type': 'flow_vn', 'inspire_id': '1456145', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'v{harm}{{{order}}}(eta), {cls} ({hf})',
                 'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': [cls], **V0M}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs},
                                     'pT': {'min': 0.0, 'max': 100.0}},
                 'harmonic': harm, 'value_type': h, 'dimensionality': 'differential_eta'}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if order == 2:
                d.update({'method': 'scalar_product', 'flow_reference': dict(TPC_REF),
                          'notes': {'value': f'{CITE}. v{harm}{{2}}(eta) of SPD/FMD hits (all pT), {cls}, against TPC '
                                             'tracks (|eta| < 0.8, 0.2 < pT < 5 GeV/c) in the opposite half; computed '
                                             'with the scalar-product estimator. Errors as tabulated.'}})
                public.append(d)
            else:
                d.update({'method': 'cumulant', 'cumulant_order': 4,
                          'notes': {'value': f'{CITE}. v2{{4}}(eta) of SPD/FMD hits, {cls}, TPC reference.'},
                          'staging_reason': 'differential v2{4} with a reference window apart from the particles of '
                                            'interest (the cumulant keys take the record acceptance)'})
                staged.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1456145' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/alice_flow_vn_vn_eta_PbPb_2760gev_ins1456145.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(d['harmonic'] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

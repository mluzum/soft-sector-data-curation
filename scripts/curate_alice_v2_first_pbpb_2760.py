#!/usr/bin/env python3
"""Curate ALICE elliptic flow of charged particles in Pb-Pb at 2.76 TeV, INSPIRE 877822.

K. Aamodt et al. (ALICE), "Elliptic flow of charged particles in Pb-Pb
collisions at 2.76 TeV", PRL 105, 252302 (2010), arXiv:1011.3914.

From the paper:
- charged particles |eta| < 0.8, 0.2 < pT < 5.0 GeV/c (integrated flow);
- centrality from the uncorrected charged multiplicity in the TPC, |eta| <
  0.8: nine classes 0-5, 5-10, 10-20 ... 70-80% (the table gives centres);
- 2- and 4-particle cumulants (v2{2}: all pairs, no gap), also from
  same-charge particles; a fit of the flow-vector distribution (q-dist)
  and Lee-Yang zeros for the integrated flow;
- the 20-30% v2{4} integrated over 0 < pT < 5 GeV/c (Table 4, the
  spectrum extrapolated below 0.2).

Records (public): v2{2}, v2{4} vs centrality (Table 3), v2{2}(pT) and
v2{4}(pT) at 40-50% (Table 1), v2{4}(pT) at 10-20, 20-30, 30-40% (Table 2;
reference particles 0.2 < pT < 5), v2{4} at 20-30% for 0 < pT < 5 (Table
4).  pT points with edges at the midpoints.  Staged (--staged OUT): the
same-charge cumulants, v2{q-dist} and v2{LYZ} (no computation).  Not
curated: the uncorrected multiplicity distribution (Table 5).  The 3 old
records had no reference and a V0 estimator.

Run from the repository root:
    python scripts/curate_alice_v2_first_pbpb_2760.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins877822/HEPData-ins877822-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'K. Aamodt et al. (ALICE), PRL 105, 252302 (2010), arXiv:1011.3914'
TPC = {'estimator': {'name': 'TPC multiplicity', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': -0.8, 'max': 0.8}], 'notes': 'Uncorrected charged multiplicity in the TPC, |eta| < 0.8.'}
ETA = {'min': -0.8, 'max': 0.8}
PT = {'min': 0.2, 'max': 5.0}
CENTRES = {2.5: '0-5%', 7.5: '5-10%', 15.0: '10-20%', 25.0: '20-30%', 35.0: '30-40%', 45.0: '40-50%', 55.0: '50-60%',
           65.0: '60-70%', 75.0: '70-80%'}
OLD = 'data/flow_vn/alice_flow_vn_charged_particles_PbPb_2760gev_877822.yaml'


def edges(pts):
    mids = [round(0.5 * (a + b), 4) for a, b in zip(pts[:-1], pts[1:])]
    return [round(max(0.0, pts[0] - (mids[0] - pts[0])), 4)] + mids + [round(pts[-1] + (pts[-1] - mids[-1]), 4)]


def method(h):
    if 'same charge' in h:
        return None, 'same-charge particles only: no computation'
    if 'q-dist' in h:
        return None, 'v2 from a fit of the flow-vector distribution: no computation'
    if 'LYZ' in h:
        return None, 'Lee-Yang zeros: no computation'
    return int(h.split('{')[1][0]), None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf in ('Table1.yaml', 'Table2.yaml', 'Table3.yaml', 'Table4.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            order, why = method(h)
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            if hf in ('Table1.yaml', 'Table2.yaml'):
                lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
                cls = f'{lo:g}-{hi:g}%'
                e = edges([float(v['value']) for v in iv['values']])
                pairs = [[e[j], e[j + 1]] for j in rows]
                classes, dim, where = [cls], 'differential_pT', f'vs pT, {cls}'
                kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
            elif hf == 'Table3.yaml':
                classes = [CENTRES[float(iv['values'][j]['value'])] for j in rows]
                dim, where, kr = 'integrated', 'vs centrality, 0.2 < pT < 5 GeV/c', {'eta': dict(ETA), 'pT': dict(PT)}
            else:
                classes, dim = ['20-30%'], 'integrated'
                where, kr = '20-30%, 0 < pT < 5 GeV/c', {'eta': dict(ETA), 'pT': {'min': 0.0, 'max': 5.0}}
            d = {'observable_type': 'flow_vn', 'inspire_id': '877822', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{h} {where} ({hf})',
                 'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': classes, **TPC}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': kr, 'harmonic': 2, 'value_type': h, 'dimensionality': dim, 'method': 'cumulant',
                 'cumulant_order': order or 2}
            if dim == 'differential_pT':
                d['flow_reference'] = {'pT': dict(PT)}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. {h} of charged particles {where}, |eta| < 0.8'
                                   + ('; reference particles 0.2 < pT < 5 GeV/c' if dim == 'differential_pT' else '')
                                   + ('; classes from the tabulated centres' if hf == 'Table3.yaml' else '')
                                   + ('; integrated over the spectrum extrapolated to pT = 0' if hf == 'Table4.yaml' else '')
                                   + '. Errors as tabulated.'}
            if why:
                d['staging_reason'] = why
                staged.append(d)
            else:
                public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '877822' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/alice_flow_vn_v2_PbPb_2760gev_ins877822.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

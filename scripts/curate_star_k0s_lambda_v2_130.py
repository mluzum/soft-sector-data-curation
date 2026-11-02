#!/usr/bin/env python3
"""Curate STAR K0_S and Lambda v2 in Au+Au at 130 GeV, INSPIRE 587154.

C. Adler et al. (STAR), "Azimuthal anisotropy of K0_S and Lambda +
anti-Lambda production at mid-rapidity from Au+Au collisions at sqrt(sNN)
= 130 GeV", PRL 89, 132301 (2002), arXiv:hep-ex/0205072.

From the paper:
- K0_S and Lambda + anti-Lambda at |y| < 1.0 (TPC, decay topology);
- event plane from tracks with 0.1 < pT <= 2.0 GeV/c, |eta| < 1.0 (the
  decay daughters removed); classes 0-11%, 11-45% and minimum bias;
- centrality from the charged multiplicity (|eta| < 0.75 in the STAR 130
  GeV analyses);
- errors: statistical (systematic +-0.005 for identification and
  background).

Records (public): v2(pT) of K0_S and Lambda + Lambdabar in 0-11% and
11-45%, TPC event plane computed with the scalar-product estimator, in
|eta| < 1.0 (rapidity windows are implemented for spectra only); pT points
with edges at the midpoints.  Staged (--staged OUT): minimum-bias v2(pT)
of K0_S, Lambda and charged hadrons (no class of the model events).  Not
curated: v2 vs mass with the hydrodynamic curve (Fig. 4).  The 4 old
records called Lambda K0_S and had no reference.

Run from the repository root:
    python scripts/curate_star_k0s_lambda_v2_130.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins587154/HEPData-ins587154-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Adler et al. (STAR), PRL 89, 132301 (2002), arXiv:hep-ex/0205072'
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.75, 'max': 0.75}],
       'notes': 'Charged multiplicity in the TPC, |eta| < 0.75 (STAR 130 GeV).'}
REF = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.1, 'max': 2.0},
       'notes': 'TPC event plane, 0.1 < pT <= 2 GeV/c, |eta| < 1.'}
TABLES = {'v2k0s_0-11.yaml': ('K0_S', '0-11%'), 'v2k0s_11-45.yaml': ('K0_S', '11-45%'),
          'v2lam_0-11.yaml': ('Lambda + Lambdabar', '0-11%'), 'v2lam_11-45.yaml': ('Lambda + Lambdabar', '11-45%'),
          'v2k0s_mb.yaml': ('K0_S', 'MB'), 'v2lam_mb.yaml': ('Lambda + Lambdabar', 'MB'), 'v2ch_mb.yaml': ('charged', 'MB')}
OLD = 'data/flow_vn/star_flow_vn_azimuthal_anisotropy_of_k0_s_AuAu_130gev.yaml'


def edges(pts):
    mids = [round(0.5 * (a + b), 4) for a, b in zip(pts[:-1], pts[1:])]
    return [round(max(0.0, pts[0] - (mids[0] - pts[0])), 4)] + mids + [round(pts[-1] + (pts[-1] - mids[-1]), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf, (sp, cls) in TABLES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
        e = edges([float(iv['values'][j]['value']) for j in rows])
        pairs = [[e[k], e[k + 1]] for k in range(len(rows))]
        eta = {'min': -0.75, 'max': 0.75} if sp == 'charged' else {'min': -1.0, 'max': 1.0}
        d = {'observable_type': 'flow_vn', 'inspire_id': '587154', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f'v2(pT) of {sp}, {cls} ({hf})',
             'collision_system': {'energy_gev': 130.0, 'reaction': 'AU AU --> X'},
             'centrality': {'classes': [cls], **NCH}, 'particle_id': {'species': sp},
             'kinematic_range': {'eta': eta, 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(REF), 'value_type': 'v_2',
             'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}. v2(pT) of {sp}, {cls}, '
                                + ('|eta| < 0.75' if sp == 'charged' else '|y| < 1 (computed in |eta| < 1)')
                                + ', TPC event plane, computed with the scalar-product estimator; pT points, edges at '
                                  'the midpoints. Errors: statistical (systematic +-0.005 in the paper).'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        if cls == 'MB':
            d['staging_reason'] = 'minimum bias: no centrality class of the model events'
            staged.append(d)
        else:
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '587154' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_k0s_lambda_v2_AuAu_130gev_ins587154.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

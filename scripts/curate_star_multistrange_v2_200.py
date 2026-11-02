#!/usr/bin/env python3
"""Curate STAR v2 of multi-strange hadrons and phi in Au+Au at 200 GeV, INSPIRE 1383879.

L. Adamczyk et al. (STAR), "Centrality and transverse momentum dependence
of elliptic flow of multistrange hadrons and phi meson in Au+Au collisions
at sqrt(sNN) = 200 GeV", PRL 116, 062301 (2016), arXiv:1507.05247.

From the paper:
- |y| < 1.0; eta-subevent plane (TPC, |Delta eta| > 0.1 between the
  positive and negative subevents, the plane from the opposite one);
  RefMult centrality; classes 0-30, 30-80, 0-80%;
- phi, Xi, Omega v2 carry an "event bias correction": v2 x <eps>_ch /
  <eps>_POI from a Glauber model, 1.002, 1.053, 1.028 (phi), 1.019, 1.054,
  1.091 (Xi), 1.068, 1.067, 1.177 (Omega) for 0-30, 30-80, 0-80%;
- errors: statistical and systematic.

Records: v2(pT) of phi, Xi, Omega (Tables 1-8; value_scale = the
correction factor, which undoes it: the measured, yield-weighted v2,
without the Glauber factor), pi, p, K (Tables 9-13, 16-18) and Lambda
(Table 20), species from the HEPData keywords; eta-subevent reference
mirrored 0.05 < |eta| < 1, event-plane pT 0.15-2 GeV/c assumed (not
stated); |y| < 1 computed in |eta| < 1; pT points.  Staged (--staged OUT):
Tables 14, 15, 19 (no species in HEPData; likely K0_S and Lambda).  Not
curated: v2(phi)/v2(p) (Tables 21-23).  The old records called phi, Xi,
Omega 'charged' and pi+-, p+pbar 'pi+', 'pbar'.  HEPData labels Table 13
(K, 0-30% by its values and position) 30-80%: corrected.

Run from the repository root:
    python scripts/curate_star_multistrange_v2_200.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1383879/HEPData-ins1383879-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'L. Adamczyk et al. (STAR), PRL 116, 062301 (2016), arXiv:1507.05247'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
REF = {'detector': 'TPC', 'mirror': True, 'eta': [{'min': 0.05, 'max': 1.0}], 'pT': {'min': 0.15, 'max': 2.0},
       'notes': 'eta-subevent plane from the opposite TPC half (|Delta eta| > 0.1); event-plane pT 0.15-2 '
                'GeV/c assumed (not stated).'}
SPECIES = {'PHI': 'phi', 'XI-': 'Xi- + Xibar+', 'OMEGA-': 'Omega- + Omegabar+', 'PI+': 'pi+ + pi-',
           'P': 'p + pbar', 'K+': 'K+ + K-', 'LAMBDA': 'Lambda + Lambdabar'}
CORRECTION = {'phi': {'0-30%': 1.002, '30-80%': 1.053, '0-80%': 1.028},
              'Xi- + Xibar+': {'0-30%': 1.019, '30-80%': 1.054, '0-80%': 1.091},
              'Omega- + Omegabar+': {'0-30%': 1.068, '30-80%': 1.067, '0-80%': 1.177}}
CLASSES = {'0.0-30.0': '0-30%', '30.0-80.0': '30-80%', '0.0-80.0': '0-80%'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    public, staged = [], []
    for x in sub:
        hf = x['data_file']
        n = int(hf[5:-5])
        if n >= 21:
            continue                                                # v2(phi)/v2(p)
        reactions = [r for k in x.get('keywords', []) if k['name'] == 'reactions' for r in k['values']]
        sp = SPECIES[reactions[0].split('-->')[1].split()[0]] if reactions else None
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = str(dv['qualifiers'][0]['value'])
        cls = CLASSES.get(q, {'centrality central': '0-30%', '30-80% centrality': '30-80%'}.get(q))
        if n == 13:
            # HEPData labels Table 13 (K, after the 0-30% pi and p of Tables
            # 11-12) 30.0-80.0; its v2 is the central one (0.007, 0.021, 0.040
            # vs Table 18's 30-80% 0.022, 0.046, 0.076): 0-30%.
            cls = '0-30%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pts = [float(iv['values'][j]['value']) for j in rows]
        w = round(min(b - a for a, b in zip(pts[:-1], pts[1:])), 4)
        d = {'observable_type': 'flow_vn', 'inspire_id': '1383879', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'{x.get("name")} | v2 {cls} {sp or "(species not stated)"}',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
             'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': sp or 'unknown'},
             'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                 'pT': {'min': round(pts[0] - w / 2, 4), 'max': round(pts[-1] + w / 2, 4),
                                        'points': pts, 'point_window': w}},
             'harmonic': 2, 'method': 'event_plane', 'value_type': 'v2', 'dimensionality': 'differential_pT',
             'flow_reference': dict(REF)}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        if n == 13:
            d['hepdata_corrections'] = {'centrality': 'HEPData labels Table 13 (K) 30.0-80.0; its v2 (0.007, 0.021, '
                                                      '0.040 vs 0.022, 0.046, 0.076 in Table 18, K 30-80%) and its '
                                                      'place after the 0-30% pi and p tables make it 0-30%'}
        corr = CORRECTION.get(sp, {}).get(cls)
        if corr:
            d['value_scale'] = corr
        d['notes'] = {'value': f'{CITE}. v2(pT) of {sp}, {cls}, |y| < 1 (computed in |eta| < 1), eta-subevent plane; '
                               'computed with the scalar-product estimator; pT points'
                               + (f'; the published Glauber event-bias correction ({corr}) undone by value_scale'
                                  if corr else '') + '. Errors: statistical and systematic.'}
        if sp is None:
            d['staging_reason'] = 'HEPData names no species for this table (likely K0_S or Lambda).'
            staged.append(d)
        else:
            public.append(d)
    old = 'data/flow_vn/star_flow_vn_centrality_and_transverse_momentum_dependence_AuAu_200gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/star_flow_vn_multistrange_phi_v2_AuAu_200gev_ins1383879.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['particle_id']['species'], d['centrality']['classes'][0]) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

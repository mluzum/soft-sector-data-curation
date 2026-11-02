#!/usr/bin/env python3
"""Curate STAR identified-particle v2(pT) vs centrality at 7.7-62.4 GeV, INSPIRE 1395151.

L. Adamczyk et al. (STAR), "Centrality dependence of identified particle
elliptic flow in relativistic heavy ion collisions at sqrt(sNN) = 7.7-62.4
GeV", PRC 93, 014907 (2016), arXiv:1509.08397; analysis as in PRC 88,
014902 (2013) (arXiv:1301.2348).

From the papers:
- v2 from the eta-sub event plane: the plane of TPC tracks (0.15 < p < 5
  GeV/c, |eta| < 1) in the hemisphere opposite the particle, with an
  extra gap of +-0.05 (subevents -1 < eta < -0.05 and 0.05 < eta < 1),
  averaged over the two subevents;
- pi, K, p in |eta| < 1; K0_S, Lambda, Xi, Omega, phi by invariant mass in
  |y| < 1 (here the subevent eta windows for all species);
- centrality from RefMult (|eta| < 0.5): 0-10, 10-40, 40-80, 0-80%;
- errors: stat and sys.

Records (public): v2(pT) of each species, energy and class (Tables
1-392 with two or more points), method event_plane with the two
subevents (particles AB; computed as the two-subevent v2 of the particle
of interest against the charged reference, the event-plane result at
low resolution).  The tables give the mean pT of each bin: bin edges at
the midpoints between points (the outer ones mirrored).  Staged
(--staged OUT): tables with one point (no bin width).  Not curated: the
particle-antiparticle differences vs energy (Tables 6a-7).  The old
records had a full-TPC reference "inferred from standard practice", and
'Reclassified integrated' single bins.

Run from the repository root:
    python scripts/curate_star_bes_pid_v2_centrality.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1395151/HEPData-ins1395151-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = ('L. Adamczyk et al. (STAR), PRC 93, 014907 (2016), arXiv:1509.08397 (analysis: PRC 88, 014902 (2013), '
        'arXiv:1301.2348)')
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
REF_PT = {'min': 0.15, 'max': 5.0}
SUBEVENTS = {'A': {'kinematic_range': {'eta': {'min': -1.0, 'max': -0.05}, 'pT': dict(REF_PT)}},
             'B': {'kinematic_range': {'eta': {'min': 0.05, 'max': 1.0}, 'pT': dict(REF_PT)}}}
SPECIES = {'PI+': 'pi+', 'PI-': 'pi-', 'K+': 'K+', 'K-': 'K-', 'P': 'p', 'PBAR': 'pbar', 'K0S': 'K0_S',
           'LAMBDA': 'Lambda', 'LAMBDABAR': 'Lambdabar', 'XI-': 'Xi-', 'XIBAR+': 'Xibar+', 'OMEGA-': 'Omega-',
           'OMEGABAR+': 'Omegabar+', 'PHI': 'phi'}
OLD = 'data/flow_vn/star_flow_vn_centrality_dependence_of_identified_particles_AuAu_multigev.yaml'


def edges_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    lo = max(0.0, pts[0] - (mids[0] - pts[0]))
    hi = pts[-1] + (pts[-1] - mids[-1])
    return [round(x, 4) for x in [lo] + mids + [hi]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the single-point tables here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for n in range(1, 393):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        if not iv['values']:
            continue
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        assert iv['header']['name'] == 'PT' and dv['header']['name'] == 'v2', (hf, iv['header'], dv['header'])
        sp = SPECIES[q['RE'].split('-->')[1].strip().rsplit(' X', 1)[0]]
        energy = float(q['SQRT(S)/NUCLEON'])
        lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pts_all = [float(v['value']) for v in iv['values']]
        d = {'observable_type': 'flow_vn', 'inspire_id': '1395151', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'v2(pT) of {sp}, {cls}, {energy:g} GeV ({hf})',
             'collision_system': {'energy_gev': energy, 'reaction': q['RE']},
             'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': sp},
             'harmonic': 2, 'method': 'event_plane', 'value_type': 'v2', 'particles': 'AB',
             'subevents': {k: {'kinematic_range': {kk: dict(vv) for kk, vv in v['kinematic_range'].items()}}
                           for k, v in SUBEVENTS.items()},
             'eta_gap': 0.1}
        note = (f'{CITE}. v2(pT) of {sp}, {cls}, eta-sub event plane: the particle in one TPC half, the plane from '
                'the other (gap 0.1; tracks 0.15 < p < 5 GeV/c), averaged; computed as the two-subevent v2. ')
        if len(pts_all) < 2:
            d.update({'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                          'pT': {'min': pts_all[0], 'max': pts_all[0], 'points': pts_all}},
                      'dimensionality': 'differential_pT',
                      'notes': {'value': note + 'One pT point, no bin edges. Errors: stat and sys.'},
                      'staging_reason': 'one pT point (the mean pT of a bin whose edges are not given)'})
            staged.append(d)
            continue
        edges = edges_from_points(pts_all)
        pairs = [[edges[j], edges[j + 1]] for j in rows]
        d.update({'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                      'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                  'dimensionality': 'differential_pT',
                  'notes': {'value': note + 'Bin edges at the midpoints between the tabulated mean pT (edges not '
                                            'published). Errors: stat and sys.'}})
        if rows != list(range(len(pts_all))):
            d['table_rows'] = rows
        public.append(d)
    if os.path.exists(OLD):
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_pid_v2_centrality_AuAu_7_62gev_ins1395151.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(d['particle_id']['species'] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

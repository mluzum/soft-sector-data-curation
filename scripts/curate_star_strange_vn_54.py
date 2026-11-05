#!/usr/bin/env python3
"""Curate STAR v2, v3 of (multi-)strange hadrons in Au+Au at 54.4 GeV, INSPIRE 2635688.

M. S. Abdallah et al. (STAR), "Azimuthal anisotropy measurement of
(multi)strange hadrons in Au+Au collisions at sqrt(sNN) = 54.4 GeV",
PRC 107 (2023) 024912, arXiv:2205.11073.

From the paper (Sec. II-III):
- K0_S, Lambda, Lambdabar, phi, Xi-, Xibar+, Omega-, Omegabar+ at |y| < 1;
- eta-sub event plane: particles at 0 < eta < 1 against the plane of
  -1 < eta < -0.05 and vice versa (pT-weighted tracks; their pT range is
  not stated, here STAR's usual 0.15-2 GeV/c), divided by the subevent
  resolution; encoded as two subevents (scripts/eta_sub.py);
- centrality from the reference multiplicity (TPC, |eta| < 0.5); classes
  0-10, 10-40, 40-80, 0-80%;
- errors: statistical and systematic; HEPData gives the mean pT of each
  bin without edges (bins here run midway between neighbouring points).

Records (public): v2(pT) and v3(pT) of each species and class.  Not
curated: v2(phi)/v2(pbar), particle - antiparticle differences (derived).

Run from the repository root:  python scripts/curate_star_strange_vn_54.py
"""
import glob
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eta_sub import subevents  # noqa: E402

RAW = glob.glob('hepdata_raw/records/ins2635688/HEPData-ins2635688-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'M. S. Abdallah et al. (STAR), PRC 107 (2023) 024912, arXiv:2205.11073'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'reference multiplicity: TPC tracks in |eta| < 0.5.'}
SPECIES = {'k0s': ('K0_S', 'K0S'), 'lambda': ('Lambda', 'LAMBDA'), 'antilambda': ('Lambdabar', 'LAMBDABAR'),
           'phi': ('phi', 'PHI'), 'xiM': ('Xi-', 'XI-'), 'xiP': ('Xibar+', 'XIBAR+'),
           'omegaM': ('Omega-', 'OMEGA-'), 'omegaP': ('Omegabar+', 'OMEGABAR+')}
OUT = 'data/flow_vn/star_flow_vn_strange_AuAu_54.4gev_ins2635688.yaml'


def bins_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    edges = [max(0.0, pts[0] - (mids[0] - pts[0]))] + mids + [pts[-1] + (pts[-1] - mids[-1])]
    return [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]


def main():
    docs = []
    for sp in SPECIES:
        for n in (2, 3):
            for lo, hi in ((0, 10), (10, 40), (40, 80), (0, 80)):
                hf = f'{sp}_v{n}_pt_invmass_Cen_{lo}_{hi}.yaml'
                path = os.path.join(RAW, hf)
                if not os.path.exists(path):
                    continue
                species, prod = SPECIES[sp]
                t = yaml.safe_load(open(path))
                dv = t['dependent_variables'][0]
                cls = f'{lo}-{hi}%'
                assert dv['header']['name'] == f'v_{{{n}}} ({cls})', (hf, dv['header'])
                assert [e['label'] for e in dv['values'][0]['errors']] == ['statistical', 'systematic'], hf
                pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
                assert pts == sorted(pts) and len(pts) >= 2, hf
                bins = bins_from_points(pts)
                d = {'observable_type': 'flow_vn', 'inspire_id': '2635688', 'version': VERSION, 'hepdata_file': hf,
                     'dv_index': 0, 'hepdata_table_label': f'v{n}(pT) of {species}, Au+Au 54.4 GeV, {cls} ({hf})',
                     'collision_system': {'energy_gev': 54.4, 'reaction': f'AU AU --> {prod} X'},
                     'centrality': {'classes': [cls], **REFMULT},
                     'particle_id': {'species': species},
                     'harmonic': n, 'method': 'event_plane', 'value_type': f'v{n}',
                     'particles': 'AB', 'subevents': subevents(0.05, 1.0, {'min': 0.15, 'max': 2.0}), 'eta_gap': 0.1,
                     'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                         'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                     'dimensionality': 'differential_pT'}
                d['notes'] = {'value': f'{CITE}. v{n}(pT) of {species}, |y| < 1, Au+Au 54.4 GeV {cls}; eta-sub event '
                                       'plane: the particle in one TPC half, the plane from the other (-1 < eta < '
                                       '-0.05, 0.05 < eta < 1; plane tracks pT-weighted, their pT range not stated: '
                                       '0.15-2 GeV/c assumed), computed as the two-subevent v_n. Bin edges midway '
                                       'between the tabulated mean pT. Errors: statistical and systematic.'}
                docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()

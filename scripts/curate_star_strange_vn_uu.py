#!/usr/bin/env python3
"""Curate STAR v_n of strange and multi-strange hadrons in U+U at 193 GeV, INSPIRE 1852040.

J. Adam et al. (STAR), "Azimuthal anisotropy measurements of strange and
multistrange hadrons in U+U collisions at sqrt(sNN) = 193 GeV at the
BNL Relativistic Heavy Ion Collider", PRC 103 (2021) 064907,
arXiv:2103.09451.

From the paper (Sec. II-III):
- K0_S, Lambda + Lambdabar, Xi + Xibar, Omega + Omegabar, phi at |y| < 1;
- v_2, v_3, v_4 by the eta-sub event plane: two subevents -1 < eta < -0.05
  and 0.05 < eta < 1 (gap 0.1), the planes from charged tracks with pT <
  2 GeV/c, the particle correlated with the plane of the other subevent,
  divided by the subevent resolution (encoded as two subevents,
  scripts/eta_sub.py: the two-subevent v_n of the particle against the
  charged reference in the opposite half);
- centrality from the reference multiplicity (TPC, |eta| < 0.5), classes
  0-80% (minimum bias), 0-10, 10-40, 40-80%;
- errors: stat and sys; HEPData gives pT points without edges (bins here
  run midway between neighbouring points, the outer ones mirrored).

Records (public): v_n(pT) of each species and harmonic in 0-80% (Fig. 5)
and in 0-10, 10-40, 40-80% (Fig. 6).  Not curated: v_n vs KE_T/n_q (Fig.
8), v_n scaled by epsilon_n or n_q (Figs. 9, 11), v_3/v_2, v_4/v_2 ratios
(Fig. 16), the event-plane resolutions (Fig. 3).

Run from the repository root:  python scripts/curate_star_strange_vn_uu.py
"""
import glob
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eta_sub import subevents  # noqa: E402

RAW = glob.glob('hepdata_raw/records/ins1852040/HEPData-ins1852040-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (STAR), PRC 103 (2021) 064907, arXiv:2103.09451'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'reference multiplicity: TPC tracks in |eta| < 0.5 (uncorrected).'}
SPECIES = {'Ks0': ('K0_S', 'K0S', '$K_s^{0}$'), 'Lambda': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR',
                                                         '$\\Lambda+\\bar{\\Lambda}$'),
           'Xi': ('Xi- + Xibar+', 'XI- XIBAR+', '$\\Xi+\\bar{\\Xi}$'),
           'Omega': ('Omega- + Omegabar+', 'OMEGA- OMEGABAR+', '$\\Omega+\\bar{\\Omega}$'),
           'Phi': ('phi', 'PHI', '$\\phi$')}
CENT = {'Cen080': '0-80%', 'Cen010': '0-10%', 'Cen1040': '10-40%', 'Cen4080': '40-80%'}
OUT = 'data/flow_vn/star_flow_vn_strange_UU_193gev_ins1852040.yaml'


def bins_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    if not mids:
        raise ValueError('one point')
    edges = [max(0.0, pts[0] - (mids[0] - pts[0]))] + mids + [pts[-1] + (pts[-1] - mids[-1])]
    return [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]


def main():
    docs = []
    for path in sorted(glob.glob(os.path.join(RAW, 'Fig5_*.yaml')) + glob.glob(os.path.join(RAW, 'Fig6_*.yaml'))):
        hf = os.path.basename(path)
        m = re.fullmatch(r'Fig([56])_(\w+?)_v(\d)pt_(Cen\d+)\.yaml', hf)
        fig, sp, n, cen = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        species, prod, qual = SPECIES[sp]
        t = yaml.safe_load(open(path))
        dv = t['dependent_variables'][0]
        assert dv['header']['name'] == f'$v_{{{n}}}$' and dv['qualifiers'][0]['value'] == qual, (hf, dv)
        assert [e['label'] for e in dv['values'][0]['errors']] == ['stat', 'sys'], hf
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        assert pts == sorted(pts), hf
        if len(pts) < 2:
            print(f'  {hf}: one point, skipped')
            continue
        bins = bins_from_points(pts)
        cls = CENT[cen]
        d = {'observable_type': 'flow_vn', 'inspire_id': '1852040', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'v{n}(pT) of {species}, U+U 193 GeV, {cls} ({hf})',
             'collision_system': {'energy_gev': 193.0, 'reaction': f'U U --> {prod} X'},
             'centrality': {'classes': [cls], **REFMULT},
             'particle_id': {'species': species},
             'harmonic': n, 'method': 'event_plane', 'value_type': f'v{n}',
             'particles': 'AB', 'subevents': subevents(0.05, 1.0, {'min': 0.15, 'max': 2.0}), 'eta_gap': 0.1,
             'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
             'dimensionality': 'differential_pT'}
        d['notes'] = {'value': f'{CITE}, Fig. {fig}. v{n}(pT) of {species}, |y| < 1, U+U 193 GeV {cls}; eta-sub event '
                               'plane: the particle in one TPC half, the plane from the other (gap 0.1; plane '
                               'tracks pT < 2 GeV/c), computed as the two-subevent v_n. Bin edges midway between '
                               'the tabulated pT (edges not published). Errors: stat and sys.'}
        docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate STAR directed flow at forward rapidity with the EPD, Au+Au 19.6 and 27 GeV, INSPIRE 2808515.

STAR Collaboration, "Measurement of directed flow in Au+Au collisions at
sqrt(sNN) = 19.6 and 27 GeV with the STAR Event Plane Detector", PRC 111
(2025) 014906, arXiv:2406.18213.

From the paper:
- charged particles in the EPD, 2.1 < |eta| < 5.1 (16 rings; the number of
  minimum-ionizing particles per tile, corrected for the STAR material to
  primary particles), at each ring's mean eta;
- v1 relative to the first-order event plane of TPC tracks, 0.15 < pT < 2
  GeV/c, |eta| < 1.0 (19.6 GeV) or < 0.8 (27 GeV: one sector off), weights
  w = -eta (odd), resolution from the three subevents TPC, EPD east, EPD
  west;
- centrality from the TPC charged multiplicity in |eta| < 0.5; seven
  classes 0-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50-60% and 10-40%;
- v1 crosses zero near the beam rapidity (3.0, 3.4); |v1| at eta < y_beam
  decreases toward central collisions, which orders HEPData's unlabelled
  tables _01 ... _07 as 50-60% ... 0-5%;
- errors: stat and sys.

Records (public): v1(eta) per class and energy (Fig. 12, 14 records) and
in 10-40% (2).  Computed as rapidity-odd v1 with an odd charged reference
in the TPC window (signs +-1 instead of -eta: the same v1 when it
factorizes); sign anchored on charged hadrons, v1 < 0 at 0 < eta < 0.8
(the STAR convention; every identified slope there is negative at 19.6
and 27 GeV, STAR 1277069 and 1618747).  POI of all pT; eta bins midway
between the ring points on each side (outer edges mirrored).  Staged
(--staged OUT): v1 vs eta - y_beam (Fig. 13, 0-40%: the same data,
forward and backward averaged).

Run from the repository root:
  python scripts/curate_star_v1_epd.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins2808515/HEPData-ins2808515-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'STAR Collaboration, PRC 111 (2025) 014906, arXiv:2406.18213'
OUT = 'data/flow_vn/star_flow_vn_v1_epd_AuAu_19.6_27gev_ins2808515.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'charged multiplicity in the TPC, |eta| < 0.5.'}
ENERGY = {'19p6': 19.6, '27': 27.0}
TPC_ETA = {19.6: 1.0, 27.0: 0.8}
CLASSES = ['50-60%', '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']   # _01 ... _07


def side_bins(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    edges = [pts[0] - (mids[0] - pts[0])] + mids + [pts[-1] + (pts[-1] - mids[-1])]
    return [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]


def record(hf, energy, cls, pts, label):
    neg = [p for p in pts if p < 0]
    pos = [p for p in pts if p > 0]
    assert neg + pos == pts and len(neg) == len(pos), pts
    bins = side_bins(neg) + side_bins(pos)
    w = TPC_ETA[energy]
    return {'observable_type': 'flow_vn', 'inspire_id': '2808515', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins},
                                'pT': {'min': 0.0, 'max': 100.0}},
            'harmonic': 1, 'method': 'event_plane', 'value_type': 'v1', 'dimensionality': 'differential_eta',
            'flow_reference': {
                'odd': True, 'detector': 'TPC', 'eta': [{'min': -w, 'max': w}], 'pT': {'min': 0.15, 'max': 2.0},
                'sign_convention': {'species': 'charged', 'eta': {'min': 0.0, 'max': 0.8}, 'sign': 'negative'},
                'notes': f'First-order event plane of TPC tracks, |eta| < {w:g}, 0.15 < pT < 2 GeV/c, weighted -eta '
                         '(here signs +-1); resolution from TPC, EPD east and EPD west.'},
            'notes': {'value': f'{CITE}. Rapidity-odd v1(eta) of charged particles in the EPD (2.1 < |eta| < 5.1, '
                               'minimum-ionizing particles per tile corrected for the STAR material to primary '
                               f'particles), Au+Au {energy:g} GeV {cls}, at the rings\' mean eta (bins midway '
                               'between them). Sign: STAR\'s convention, anchored on charged hadrons, v1 < 0 at 0 < '
                               'eta < 0.8 (all identified slopes there negative at 19.6 and 27 GeV). Errors: stat and '
                               'sys.'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for tag, energy in ENERGY.items():
        files = [(f'v1_{tag}GeV_7centrality_0{k + 1}.yaml', cls) for k, cls in enumerate(CLASSES)]
        files.append((f'v1_{tag}GeV_10_40_01.yaml', '10-40%'))
        inner = []
        for hf, cls in files:
            t = yaml.safe_load(open(os.path.join(RAW, hf)))
            assert t['independent_variables'][0]['header']['name'] == '$\\eta$', hf
            dv = t['dependent_variables'][0]
            assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat', 'sys')}, hf
            pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
            vals = [float(v['value']) for v in dv['values']]
            if cls != '10-40%':
                inner.append(abs(vals[16]) + abs(vals[15]))
            public.append(record(hf, energy, cls, pts, f'v1(eta) of charged particles in the EPD, Au+Au {energy:g} '
                                                       f'GeV, {cls} ({hf})'))
        # |v1| at eta = +-2.13 (< y_beam) decreases toward central collisions (the paper): the class order,
        # up to one adjacent inversion within errors (27 GeV, 50-60 vs 40-50%)
        inversions = sum(b > a for a, b in zip(inner, inner[1:]))
        assert inversions <= 1 and inner[0] > 1.5 * inner[-1], (tag, inner)
        for hf in (f'v1_LF_{tag}GeV_0_40_01.yaml',) + tuple(f'v1_LF_{tag}GeV_7centrality_0{k + 1}.yaml'
                                                            for k in range(7)):
            staged.append({'observable_type': 'flow_vn', 'inspire_id': '2808515', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': 0, 'hepdata_table_label': f'v1 vs eta - y_beam ({hf})',
                           'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
                           'harmonic': 1, 'value_type': 'v1',
                           'notes': {'value': f'{CITE}. v1 vs eta - y_beam (forward and backward averaged).'},
                           'staging_reason': 'v1 vs eta - y_beam: the same data as the v1(eta) records, forward and '
                                             'backward averaged'})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {len(staged)} staged')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

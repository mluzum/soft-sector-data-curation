#!/usr/bin/env python3
"""Curate STAR charged and strange hadron v2 in Cu+Cu at 62.4 and 200 GeV, INSPIRE 843985.

B. I. Abelev et al. (STAR), "Charged and strange hadron elliptic flow in
Cu+Cu collisions at sqrt(sNN) = 62.4 and 200 GeV", PRC 81 (2010) 044902,
arXiv:1001.5052.

From the paper (Sec. II-III):
- particles of interest at |eta| < 1.0 (TPC): charged hadrons, and K0_S,
  phi, Lambda (Lambdabar), Xi- (Xibar+) from their decay topology;
- centrality from the raw TPC charged multiplicity, |eta| < 0.5; the 60%
  most central events in 10% classes; K0_S and Lambda also in 0-20 and
  20-60%; "we calculated v2 in the 10% wide bins, then combined them using
  particle yield as the weight" (every wide class here: 0-20, 20-60,
  0-60%);
- v2{TPC}: event plane of TPC tracks (|eta| < 1, 0.15 < pT < 2 GeV/c,
  pT-weighted), the particle's own track excluded; v2{FTPC}: the plane of
  the two FTPCs (2.5 < |eta| < 4.0, east and west subevents giving the
  resolution); the paper reports v2{FTPC} ("for simplicity v2 denotes
  v2{FTPC}"): Figs. 7-9;
- v2{AA-pp}: the p+p azimuthal correlations subtracted (a non-flow
  estimate);
- errors: statistical (Figs. 3-9).

Records (public): v2{FTPC}(pT) of charged hadrons per 10% class and in
0-60% at 200 GeV (Figs. 4, 3b) and 62.4 GeV (Figs. 7, 8), v2{TPC}(pT)
0-60% at 200 GeV (Fig. 3a), integrated v2{TPC} and v2{FTPC} vs centrality
(0.15 < pT < 4 GeV/c, |eta| < 1; Fig. 6), v2{FTPC}(pT) of K0_S, Lambda
(0-60, 0-20, 20-60%), Xi and phi (0-60%; Fig. 9) at 200 GeV.  Event
planes as a reference (event_plane + flow_reference: the scalar product,
the event-plane result at low resolution); pT bins midway between the
tabulated points; Lambda and Xi as particle + antiparticle (the paper
reconstructs both).  Staged (--staged OUT): v2{AA-pp} (p+p correlations
subtracted: not computed) and the azimuthal correlation sums of Fig. 2
(Cu+Cu and p+p).

Run from the repository root:
  python scripts/curate_star_v2_cucu.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins843985/HEPData-ins843985-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. I. Abelev et al. (STAR), PRC 81 (2010) 044902, arXiv:1001.5052'
OUT = 'data/flow_vn/star_flow_vn_charged_strange_v2_CuCu_62.4_200gev_ins843985.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'raw TPC charged multiplicity in |eta| < 0.5.'}
REF = {'FTPC': {'detector': 'FTPC', 'eta': [{'min': -4.0, 'max': -2.5}, {'min': 2.5, 'max': 4.0}],
                'pT': {'min': 0.15, 'max': 2.0},
                'notes': 'Event plane of the two FTPCs (2.5 < |eta| < 4.0; tracks 0.15 < pT < 2 GeV/c, pT-weighted), '
                         'resolution from the east and west subevents.'},
       'TPC': {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.15, 'max': 2.0},
               'notes': 'Event plane of TPC tracks (|eta| < 1, 0.15 < pT < 2 GeV/c, pT-weighted), the particle\'s '
                        'own track excluded.'}}
SPECIES = {'K0s': ('K0_S', 'K0S'), 'Lambda': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR'),
           'Xi': ('Xi- + Xibar+', 'XI- XIBAR+'), 'phi': ('phi', 'PHI'), None: ('charged', 'CHARGED')}
COMBINE = {'combined_from': 'analysis_classes', 'analysis_class_width': 10.0, 'combination_weight': 'yield'}


def bins_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    edges = [max(0.0, pts[0] - (mids[0] - pts[0]))] + mids + [pts[-1] + (pts[-1] - mids[-1])]
    return [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]


def width(cls):
    lo, hi = (float(v) for v in cls.rstrip('%').split('-'))
    return hi - lo


def base(hf, i, label, energy, classes, species, prod, method, kr, dim, note, errors='statistical'):
    cent = {'classes': classes, **REFMULT}
    if any(width(c) > 10 for c in classes):
        cent.update(COMBINE)
        note += ' Wide classes combined from 10% classes weighted by the particle yield.'
    return {'observable_type': 'flow_vn', 'inspire_id': '843985', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': energy, 'reaction': f'CU CU --> {prod} X'},
            'centrality': cent, 'particle_id': {'species': species}, 'kinematic_range': kr, 'harmonic': 2,
            'method': 'event_plane', 'flow_reference': dict(REF[method]), 'value_type': f'v2{{{method}}}',
            'dimensionality': dim,
            'notes': {'value': f'{CITE}. {note} Errors: {errors}.'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        energy = 62.4 if '62.4 GeV' in desc.split('Data tables')[0] else 200.0
        iv = t['independent_variables'][0]
        if hf == 'Fig2.yaml':
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'flow_vn', 'inspire_id': '843985', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"Fig. 2: {dv['header']['name']} ({hf})",
                               'collision_system': {'energy_gev': 200.0, 'reaction': 'CU CU --> CHARGED X'
                                                    if 'Cu+Cu' in dv['header']['name'] else 'P P --> CHARGED X'},
                               'harmonic': 2, 'value_type': dv['header']['name'],
                               'notes': {'value': f'{CITE}. Fig. 2: {dv["header"]["name"]} vs pT, 0-60% Cu+Cu and '
                                                  'p+p at 200 GeV (Eq. 10).'},
                               'staging_reason': 'summed azimuthal correlation of a particle with a flow vector '
                                                 '(non-flow study): not computed'})
            continue
        if iv['header']['name'] == 'Centrality':
            classes = [f"{int(v['low'])}-{int(v['high'])}%" for v in iv['values']]
            for i, dv in enumerate(t['dependent_variables']):
                name = dv['header']['name']
                m = re.fullmatch(r'v2\{(AA-pp, )?(TPC|FTPC)\}', name)
                assert m, (hf, name)
                kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.15, 'max': 4.0}}
                d = base(hf, i, f'Fig. 6: integrated {name} of charged hadrons vs centrality, Cu+Cu {energy:g} GeV '
                                f'({hf})', energy, classes, 'charged', 'CHARGED', m.group(2), kr, 'integrated',
                         f'Integrated {name} of charged hadrons (0.15 < pT < 4 GeV/c, |eta| < 1) vs centrality, Cu+Cu '
                         f'{energy:g} GeV.')
                if m.group(1):
                    d['staging_reason'] = 'v2{AA-pp}: Cu+Cu correlations minus p+p (a non-flow estimate), not computed'
                (staged if 'staging_reason' in d else public).append(d)
            continue
        assert iv['header']['name'] == 'Transverse Momentum [GeV/c]', hf
        pts = [float(v['value']) for v in iv['values']]
        for i, dv in enumerate(t['dependent_variables']):
            name = re.sub(r'\s+', ' ', dv['header']['name']).strip()
            m = re.fullmatch(r'v2\{(AA-pp, )?(TPC|FTPC)\}(?: (K0s|Lambda|Xi|phi))?(?: in (\d+-\d+%))?', name)
            assert m, (hf, name)
            mc = re.search(r'in (\d+-\d+%)', desc)
            cls = m.group(4) or mc.group(1)
            species, prod = SPECIES[m.group(3)]
            rows = [r for r, v in enumerate(dv['values']) if v['value'] not in ('-', '', None)]
            assert rows == list(range(len(rows))), (hf, i)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat',)}, (hf, i)
            p = pts[:len(rows)]
            assert p == sorted(p), (hf, p)
            bins = bins_from_points(p)
            kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            meth = f"v2{{{m.group(1) or ''}{m.group(2)}}}"
            label = f"{s['name']}: {meth} of {species} vs pT, Cu+Cu {energy:g} GeV, {cls} ({hf})"
            d = base(hf, i, label, energy, [cls], species, prod, m.group(2), kr, 'differential_pT',
                     f'{meth}(pT) of {species}, |eta| < 1, Cu+Cu {energy:g} GeV {cls}. pT bins midway between the '
                     'tabulated points.')
            if len(rows) < len(pts):
                d['table_rows'] = rows
            if m.group(1):
                d['staging_reason'] = 'v2{AA-pp}: Cu+Cu correlations minus p+p (a non-flow estimate), not computed'
            (staged if 'staging_reason' in d else public).append(d)
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

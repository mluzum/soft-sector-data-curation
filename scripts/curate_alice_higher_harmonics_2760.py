#!/usr/bin/env python3
"""Curate ALICE v2 ... v5 of charged particles in Pb-Pb at 2.76 TeV, INSPIRE 900651.

K. Aamodt et al. (ALICE), "Higher harmonic anisotropic flow measurements of
charged particles in Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", PRL 107,
032301 (2011), arXiv:1105.3865.

From the paper:
- charged particles |eta| < 0.8, 0.2 < pT < 5.0 GeV/c; VZERO centrality;
- v_n{2} from two-particle correlations of particles separated by |Delta
  eta| > 1.0 (Figs. 1a, 2) or > 0.2 and > 1.0 (Fig. 3: v_n(pT), the
  particle of interest against the reference), corrected for the residual
  non-flow estimated with HIJING; HEPData labels them v_n{SP};
- v3{4}; v3 relative to the ZDC spectator plane; v3{QC5}; v_n/epsilon_n;
- Fig. 3: (a) 30-40%, (b) 0-5%, (c) 0-2%;
- the centrality axis gives class centres (2.5, 7.5, 15, ...).

Records (public): v2, v3, v4{2, |Delta eta| > 1} vs centrality (Tables
1-3, 11-12), v3{4} (Table 4), v_n{2}(pT) for |Delta eta| > 0.2 and > 1.0
(Tables 13-36).  The pair gap is stated here as two subevents -0.8 < eta
< -gap/2 and gap/2 < eta < 0.8 (particles AB; equal for a boost-invariant
model, close otherwise).  Classes from the centres (0-5, 5-10, 10-20,
...; 1% classes in Tables 11-12; the v3{QC5} centres are displaced).
pT: the tabulated points, bin edges at the midpoints (edges not
published).  Staged (--staged OUT): v3{ZDC} (spectator plane), v3{QC5}
(five-particle correlator), v_n/epsilon_n (Tables 7-10: model
eccentricities).  The old records had classes 5-11.25, 11.25-20% (edges
at the midpoints of the centres) and the eccentricity ratios as v_n.

Run from the repository root:
    python scripts/curate_alice_higher_harmonics_2760.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins900651/HEPData-ins900651-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'K. Aamodt et al. (ALICE), PRL 107, 032301 (2011), arXiv:1105.3865'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
PT = {'min': 0.2, 'max': 5.0}
STD = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]
FIG3 = {**{n: '30-40%' for n in range(13, 21)}, **{n: '0-5%' for n in range(21, 29)}, **{n: '0-2%' for n in range(29, 37)}}
OLD = 'data/flow_vn/alice_flow_vn_higher_harmonic_anisotropic_flow_measurements_PbPb_2760gev.yaml'


def classes_of(centres):
    out = []
    for c in centres:
        if c < 5 and abs(c - round(c - 0.5) - 0.5) < 1e-6 and len(centres) == 5:      # 1% classes
            out.append(f'{c - 0.5:g}-{c + 0.5:g}%')
        else:
            out.append(next(f'{lo}-{hi}%' for lo, hi in STD if lo <= c < hi))
    return out


def subevents(gap):
    h = round(gap / 2, 3)
    return {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -h}, 'pT': dict(PT)}},
            'B': {'kinematic_range': {'eta': {'min': h, 'max': 0.8}, 'pT': dict(PT)}}}


def edges_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    return [round(x, 4) for x in [max(0.0, pts[0] - (mids[0] - pts[0]))] + mids + [pts[-1] + (pts[-1] - mids[-1])]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for n in range(1, 37):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        h = dv['header']['name']
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        xs = [float(iv['values'][j]['value']) for j in rows]
        harm = int(re.match(r'v(\d)', h).group(1))
        d = {'observable_type': 'flow_vn', 'inspire_id': '900651', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'{h} ({hf})',
             'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
             'particle_id': {'species': 'charged'}, 'harmonic': harm, 'value_type': h.split(' (')[0]}
        if iv['header']['name'] == 'CENTRALITY':
            d.update({'centrality': {'classes': classes_of(xs), **V0M},
                      'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT)},
                      'dimensionality': 'integrated'})
            what = f'v{harm} vs centrality, 0.2 < pT < 5 GeV/c'
        else:
            all_pts = [float(v['value']) for v in iv['values']]
            e = edges_from_points(all_pts)
            pairs = [[e[j], e[j + 1]] for j in rows]
            d.update({'centrality': {'classes': [FIG3[n]], **V0M},
                      'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8},
                                          'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                      'dimensionality': 'differential_pT'})
            what = f'v{harm}(pT), {FIG3[n]}, pT bin edges at the midpoints of the tabulated points'
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        m = re.search(r'Deltaeta=([\d.]+)', h)
        gap = float(m.group(1)) if m else 1.0
        if re.fullmatch(r'v\d\{SP(,Deltaeta=[\d.]+)?\}', h.split(' (')[0]):
            d.update({'method': 'cumulant', 'cumulant_order': 2, 'particles': 'AB', 'eta_gap': gap,
                      'subevents': subevents(gap),
                      'notes': {'value': f'{CITE}. {what}; v{harm}{{2}} of particles separated by |Delta eta| > '
                                         f'{gap:g} (HEPData: {{SP}}), corrected for residual non-flow (HIJING); the '
                                         f'pair gap stated as subevents -0.8 < eta < -{gap / 2:g}, {gap / 2:g} < eta < '
                                         '0.8. Errors as tabulated.'}})
            public.append(d)
        elif h.startswith('v3{4}'):
            d.update({'method': 'cumulant', 'cumulant_order': 4,
                      'notes': {'value': f'{CITE}. v3{{4}} vs centrality, |eta| < 0.8, 0.2 < pT < 5 GeV/c.'}})
            public.append(d)
        else:
            d.update({'method': 'event_plane' if 'ZDC' in h else 'cumulant',
                      'notes': {'value': f'{CITE}. {h}.'},
                      'staging_reason': ('relative to the ZDC spectator plane' if 'ZDC' in h else
                                         'five-particle correlator v3{QC5}' if 'QC5' in h else
                                         'v_n divided by a model eccentricity (Glauber / CGC)')})
            if 'ZDC' in h:
                d['flow_reference'] = {'detector': 'ZDC', 'plane': 'spectator'}
            staged.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '900651' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/alice_flow_vn_v2_v5_PbPb_2760gev_ins900651.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter((d['harmonic'], d['dimensionality']) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

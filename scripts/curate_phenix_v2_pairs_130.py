#!/usr/bin/env python3
"""Curate PHENIX v2 from two-particle azimuthal correlations in Au+Au at 130 GeV, INSPIRE 585347.

K. Adcox et al. (PHENIX), "Flow measurements via two-particle azimuthal
correlations in Au+Au collisions at sqrt(sNN) = 130 GeV", PRL 89, 212301
(2002), nucl-ex/0204005 (PPG004).

From the paper:
- charged hadrons in the two central arms, |eta| < 0.35; centrality from
  cuts in BBC vs ZDC response (percentiles of 6.8 b);
- v2 from Fourier fits of C(Delta phi) = same-event / mixed-event pair
  distributions, with a Gaussian term for the small-angle background;
  pairs closer than 1 cm in the drift chamber removed; no eta gap;
- fixed-pT method: both particles in the same pT range, v2 = sqrt(v2^2);
  assorted-pT method: one particle in the pT range, the other outside it;
- the analysis of Figs. 1-2 is restricted to 0.3 < pT <= 2.5 GeV/c;
- errors: stat and sys (~5%).

Records (public): v2 vs centrality (Fig. 2a: fixed-pT, 0.4-0.6, 0.6-1.0,
1.0-2.5 GeV/c, v2{2} of the window; Fig. 2b: assorted-pT, the particle at
1.0-2.5 GeV/c and the reference 0.3-1.0 GeV/c, the rest of the analysis
range), computed as v2{2} within |eta| < 0.35 (the pair cut and the
Gaussian term are not modelled).  Staged (--staged OUT): v2 vs pT (Fig. 3:
points at the mean pT without bin edges; fixed-pT needs one record per bin,
and the assorted reference, outside each bin, is two windows).  Not
curated: correlation functions (Fig. 1), A2 = v2/epsilon (Fig. 4, model
eccentricity).  The 2 old records read A2 as v2.

Run from the repository root:
    python scripts/curate_phenix_v2_pairs_130.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins585347/HEPData-ins585347-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'K. Adcox et al. (PHENIX), PRL 89, 212301 (2002), nucl-ex/0204005'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge (with the ZDC energy in the paper).'}
ETA = {'min': -0.35, 'max': 0.35}
OLD = 'data/flow_vn/phenix_flow_vn_flow_measurements_via_two-particle_azimuthal_AuAu_200gev.yaml'
ASSORTED_REF = {'min': 0.3, 'max': 1.0}
BLANK = ('', '-', '--', 'none')


def base(hf, i, h, label):
    return {'observable_type': 'flow_vn', 'inspire_id': '585347', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf})', 'collision_system': {'energy_gev': 130.0, 'reaction': 'AU AU --> X'},
            'particle_id': {'species': 'charged'}, 'value_type': h, 'method': 'cumulant', 'cumulant_order': 2,
            'harmonic': 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf in ('ppg004fig2a.yaml', 'ppg004fig2b.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            m = re.fullmatch(r'\[([FA])\] ([\d.]+) < \$p_T\$ < ([\d.]+) \(GeV/\$c\$\)', h)
            kind, pt = m.group(1), {'min': float(m.group(2)), 'max': float(m.group(3))}
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
            win = f"{pt['min']:g} < pT < {pt['max']:g} GeV/c"
            d = base(hf, i, h, f"v2 vs centrality, {'fixed' if kind == 'F' else 'assorted'}-pT, {win}")
            d.update({'centrality': {'classes': classes, **BBC}, 'dimensionality': 'integrated',
                      'kinematic_range': {'eta': dict(ETA), 'pT': pt}})
            if kind == 'F':
                how = f'fixed-pT pairs (both particles at {win}), v2{{2}} of the window'
            else:
                d['flow_reference'] = {'pT': dict(ASSORTED_REF),
                                       'notes': 'Assorted-pT pairs: the other particle outside the POI range, here the '
                                                'rest of the 0.3 < pT <= 2.5 GeV/c analysis range.'}
                how = (f'assorted-pT pairs (one particle at {win}, the other at 0.3 < pT < 1.0 GeV/c), '
                       'v2 = c2(POI, ref) / v2{2}(ref)')
            d['notes'] = {'value': f'{CITE}, Fig. 2. v2 of charged hadrons vs centrality (BBC/ZDC), |eta| < 0.35, {how}; '
                                   'Fourier fit of the pair correlation function with a small-angle Gaussian, no eta '
                                   'gap (the 1 cm pair cut and the Gaussian term are not modelled). Errors: stat and '
                                   'sys.'}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    for hf in ('ppg004fig3a.yaml', 'ppg004fig3b.yaml', 'ppg004fig3c.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            m = re.fullmatch(r'\[([FA])\] (\d+) < Centrality \(%\) < (\d+)', h)
            kind, cls = m.group(1), f'{int(m.group(2))}-{int(m.group(3))}%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            x = [float(iv['values'][j]['value']) for j in rows]
            edges = [0.3] + [round((a + b) / 2, 4) for a, b in zip(x, x[1:])] + [round(x[-1] + (x[-1] - x[-2]) / 2, 4)]
            d = base(hf, i, h, f"v2(pT), {cls}, {'fixed' if kind == 'F' else 'assorted'}-pT")
            d.update({'centrality': {'classes': [cls], **BBC}, 'dimensionality': 'differential_pT',
                      'kinematic_range': {'eta': dict(ETA), 'pT': {'min': edges[0], 'max': edges[-1],
                                                                   'bin_ranges': [[a, b] for a, b in zip(edges, edges[1:])]}},
                      'notes': {'value': f'{CITE}, Fig. 3. v2(pT) of charged hadrons, {cls}, |eta| < 0.35, '
                                         f"{'fixed' if kind == 'F' else 'assorted'}-pT pairs; points at the mean pT "
                                         '(bin edges here: midpoints between points, not from the paper).'},
                      'staging_reason': ('fixed-pT pairs: both particles in each pT bin (one record per bin), and the '
                                         'bin edges are not given' if kind == 'F' else
                                         'assorted-pT pairs: the reference is outside each pT bin (two windows), and '
                                         'the bin edges are not given')})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            staged.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '585347' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_v2_pairs_AuAu_130gev_ins585347.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

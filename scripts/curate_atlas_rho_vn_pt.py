#!/usr/bin/env python3
"""Curate ATLAS rho(v_n^2, [pT]) in Pb+Pb at 5.02 TeV and Xe+Xe at 5.44 TeV, INSPIRE 2075412.

ATLAS Collaboration, "Correlations between flow and transverse momentum in
Xe+Xe and Pb+Pb collisions at the LHC with the ATLAS detector: a probe of
the heavy-ion initial state and nuclear deformation", PRC 107 (2023)
054910, arXiv:2205.00039.

From the paper (Sec. 4-5, Table 1):
- rho_n = cov_n / sqrt(var(v_n^2) c_k) (Eq. 4); cov_n over distinct
  triplets, triplet-weighted; c_k from all pairs of the full window;
  var(v_n^2) = c_n{4} (standard, full window) + c_n{2}^2 (two subevents);
- charged particles in |eta| < 2.5 (alternative |eta| < 1), pT 0.5-5 GeV
  (also 0.5-2, and 0.3-2 in Xe+Xe), corrected for efficiency;
- cov_n methods: standard (|eta| < 2.5), two-subevent (a: -2.5 < eta <
  -0.75, c: 0.75 < eta < 2.5; the pT particle from a or c), three-subevent
  (b: |eta| < 0.5); alternative windows 0.35 < |eta| < 1 and |eta| < 0.3;
  final results: three-subevent for n = 2, combined (the mean of two- and
  three-subevent) for n = 3, 4;
- every term is computed in narrow Sigma E_T bins (|Delta Sigma E_T| <
  0.002 TeV) and combined over broader ones (multiplicity_baseline, with
  the FCal multiplicity 3.2 < |eta| < 4.9 for Sigma E_T); the points are
  plotted at the mean centrality of each broad bin (FCal Sigma E_T
  percentiles); the bin edges are not tabulated, so the classes here run
  midway between neighbouring points (the outermost mirror their
  neighbour);
- errors: stat and syst.

Records (public): Sigma E_T-based centrality, Figs. 1-4 -- rho_n for each
method (Fig. 1), the final rho_n (Fig. 2), other pT ranges (Fig. 3), the
|eta| < 1 windows and cov_n (Fig. 4); tables repeating values already
taken are skipped.  Not curated: the Sigma E_T axis versions (Fig. 2
bottom), N_ch^rec-based centrality (Figs. 5-7: a different class
definition), the Trento comparisons (Figs. 8-9) and the auxiliary figures
(components, ratios, models).

Run from the repository root:  python scripts/curate_atlas_rho_vn_pt.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2075412/HEPData-ins2075412-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'ATLAS Collaboration, PRC 107 (2023) 054910, arXiv:2205.00039'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
SYSTEMS = {'Pb+Pb 5.02': (5020.0, 'PB PB --> CHARGED X'), 'Xe+Xe 5.44': (5440.0, 'XE XE --> CHARGED X')}
# subevents per full window (Table 1)
WINDOWS = {2.5: {'A': (-2.5, -0.75), 'B': (-0.5, 0.5), 'C': (0.75, 2.5)},
           1.0: {'A': (-1.0, -0.35), 'B': (-0.3, 0.3), 'C': (0.35, 1.0)}}
METHODS = {'Standard': 'standard', 'Two_subevent': 'two_subevent', 'Three_subevent': 'three_subevent',
           'Combined_subevent': 'combined_subevent'}
TABLES = list(range(1, 25)) + list(range(31, 58))      # Figs. 1-4, centrality axis
DESC = re.compile(r'\$(\\rho|Cov)_\{(\d)\}\$ (\w+) method, for (Pb\+Pb 5\.02|Xe\+Xe 5\.44) TeV, \$\|\\eta\|\$<'
                  r'([\d.]+), ([\d.]+)< \$p_\{T\}\$ <([\d.]+) GeV vs \$\\Sigma E_\{T\}\$ based Centrality')
OUT = 'data/flow_pT_correlations/atlas_flow_pT_correlations_rho_PbPb_XeXe_5tev_ins2075412.yaml'


def classes_from_points(points):
    """Classes midway between neighbouring mean centralities (descending)."""
    mids = [0.5 * (a + b) for a, b in zip(points[:-1], points[1:])]
    hi0 = points[0] + (points[0] - mids[0])
    lo_last = max(0.0, points[-1] - (mids[-1] - points[-1]))
    edges = [hi0] + mids + [lo_last]
    out = []
    for p, hi, lo in zip(points, edges[:-1], edges[1:]):
        assert lo <= p < hi, (lo, p, hi)
        out.append(f'{round(lo, 4):g}-{round(hi, 4):g}%')
    return out


def main():
    subs = {s['data_file']: s for s in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
            if s and 'data_file' in s}
    docs, seen = [], {}
    for n in TABLES:
        hf = f'table_{n}.yaml'
        desc = ' '.join(str(subs[hf]['description']).split())
        m = DESC.match(desc)
        assert m, desc
        quant, harm, meth, system, eta, plo, phi = m.groups()
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        dv = t['dependent_variables'][0]
        vals = tuple(v['value'] for v in dv['values'])
        assert t['independent_variables'][0]['header']['name'] == 'Centrality ($\\%$)', hf
        assert pts == sorted(pts, reverse=True), hf
        assert [e['label'] for e in dv['values'][0]['errors']] == ['stat. uncertainty', 'syst. uncertainty'], hf
        key = (tuple(pts), vals)
        if key in seen:
            print(f'  {hf}: the values of {seen[key]} (skipped)')
            continue
        seen[key] = hf
        eta, plo, phi, harm = float(eta), float(plo), float(phi), int(harm)
        energy, reaction = SYSTEMS[system]
        kr = {'eta': {'min': -eta, 'max': eta}, 'pT': {'min': plo, 'max': phi}}
        sub = {s: {'kinematic_range': {'eta': {'min': lo, 'max': hi}, 'pT': {'min': plo, 'max': phi}}}
               for s, (lo, hi) in WINDOWS[eta].items()}
        vt = 'rho_n' if quant == '\\rho' else 'cov_n'
        label = f'{"rho" if vt == "rho_n" else "cov"}_{harm}, {meth.replace("_", "-").lower()}, {system} TeV, ' \
                f'|eta| < {eta:g}, {plo:g} < pT < {phi:g} GeV'
        d = {'observable_type': 'flow_pT_correlations', 'inspire_id': '2075412', 'version': VERSION,
             'hepdata_file': hf, 'dv_index': 0, 'hepdata_table_label': f'{label} ({hf})',
             'collision_system': {'energy_gev': energy, 'reaction': reaction},
             'centrality': {'classes': classes_from_points(pts), **FCAL,
                            'notes': 'Sigma E_T-based classes; edges midway between the tabulated mean '
                                     'centralities of the Sigma E_T bins.'},
             'particle_id': {'species': 'charged'}, 'kinematic_range': kr, 'particles': 'ABC', 'subevents': sub,
             'observable': 'rho_vn2_pT' if vt == 'rho_n' else 'cov_vn2_pT', 'value_type': vt,
             'correlation_type': 'rho_vn2_pT', 'harmonic': harm, 'covariance_method': METHODS[meth],
             'variance_terms': {'c4': 'standard', 'c2': 'two_subevent'}, 'ck_window': 'full',
             'multiplicity_baseline': {'observable': 'Nch', 'eta': [dict(e) for e in FCAL['eta']],
                                       'pT': {'min': 0.0, 'max': 100.0}}}
        d['notes'] = {'value': f'{CITE}, {subs[hf]["location"]}. {label}{" (GeV)" if vt == "cov_n" else ""}; c_k '
                               'from the full window, var(v_n^2) = c_n{4} (full) + c_n{2}^2 (two subevents); '
                               'computed in narrow Sigma E_T bins and combined (multiplicity_baseline: FCal '
                               'multiplicity). Errors: stat and syst.'}
        docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()

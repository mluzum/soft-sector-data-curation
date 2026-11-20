#!/usr/bin/env python3
"""Curate CMS <pT> versus N_ch in ultra-central PbPb at 5.02 TeV, INSPIRE 2747107.

CMS Collaboration, "Extracting the speed of sound in quark-gluon plasma with
ultrarelativistic lead-lead collisions at the LHC", Rep. Prog. Phys. 87
(2024) 077801, arXiv:2401.06896.

From the paper (Secs. 3-4):
- the 10% of events with the largest HF transverse energy (2.9 < |eta| <
  5.2), in 50 GeV intervals of E_T^HF from 3400 to 5200 GeV;
- per interval, <pT> and N_ch of charged particles in |eta| < 0.5 from the
  corrected spectra, extrapolated to the full pT range (Hagedorn fits over
  0.4 < pT < 4.5 GeV), both over their values in the 0-5% class of E_T^HF
  (<pT>^0, N_ch^0); Fig. 3 gives <pT>/<pT>^0 at each interval's N_ch/N_ch^0.

Model counterpart: classes of the HF estimator labelled on the N_ch/N_ch^0
scale (centrality.abscissa_bins: the estimator values where the conditional
mean of N_ch in |eta| < 0.5 over its 0-5% mean equals the label edges;
analysis.weights.AbscissaBinWeights), edges midway between the tabulated
N_ch/N_ch^0 (the outermost mirror their neighbour); <pT> of charged
particles in |eta| < 0.5 over all pT; the HF E_T modelled by the charged
count in 2.9 < |eta| < 5.2.  The ratio record (centrality_ratio to the HF
0-5% class) has sides without values (the results are published only
normalized).  Not curated: the N_ch distribution (Fig. 2) and c_s^2 (Fig.
4, a fit).

Run from the repository root:  python scripts/curate_cms_meanpt_ucc_pbpb_5020.py
"""
import glob
import os

import numpy as np
import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

RAW = glob.glob('hepdata_raw/records/ins2747107/HEPData-ins2747107-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'CMS Collaboration, Rep. Prog. Phys. 87 (2024) 077801, arXiv:2401.06896'
OUT_RATIO = 'data/observable_ratios/cms_observable_ratios_meanpT_ucc_PbPb_5020gev_ins2747107.yaml'
OUT_SIDE = 'data/mean_pT/cms_mean_pT_ucc_sides_PbPb_5020gev_ins2747107.yaml'
SYSTEM = {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'}
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
KR = {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}


def fmt(x):
    return np.format_float_positional(float(x), precision=5, unique=False, fractional=False, trim='-')


def main():
    t = yaml.load(open(os.path.join(RAW, 'figure_3.yaml')), Loader=Loader)
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    assert iv['header']['name'] == '$N_{\\mathrm{ch}}/N_{\\mathrm{ch}}^0$', iv['header']
    quals = {q['name']: str(q['value']) for q in dv.get('qualifiers', [])}
    assert quals.get('eta range') == '$|\\eta| < 0.5$' and quals.get('pT range') == '$p_\\textrm{T} > 0$', quals
    xs = np.array([0.5 * (v['low'] + v['high']) if 'low' in v else v['value'] for v in iv['values']])
    assert np.all(np.diff(xs) > 0)
    mids = 0.5 * (xs[:-1] + xs[1:])
    edges = np.concatenate([[xs[0] - (mids[0] - xs[0])], mids, [xs[-1] + (xs[-1] - mids[-1])]])
    classes = [f'{fmt(lo)}-{fmt(hi)}' for lo, hi in zip(edges[:-1], edges[1:])]
    cent = {'classes': classes, **HF,
            'abscissa_bins': {'eta': [dict(KR['eta'])], 'pT': dict(KR['pT']), 'reference_class': '0-5%'},
            'notes': '50 GeV intervals of E_T^HF (3400-5200 GeV, the top 10%), labelled by N_ch/N_ch^0 (|eta| < 0.5, '
                     'all pT); classes midway between the tabulated points; the model estimator the charged count in '
                     '2.9 < |eta| < 5.2.'}
    num = {'observable_type': 'mean_pT', 'inspire_id': '2747107', 'version': VERSION, 'experimental_values': False,
           'hepdata_table_label': '<pT> of charged particles, |eta| < 0.5, all pT, in E_T^HF intervals (a side of '
                                  'figure_3; only the ratio to 0-5% is published)',
           'collision_system': dict(SYSTEM), 'centrality': cent, 'particle_id': {'species': 'charged'},
           'kinematic_range': {k: dict(v) for k, v in KR.items()}, 'observable': 'mean_pT', 'value_type': 'mean_pT',
           'dimensionality': 'integrated',
           'notes': {'value': f'{CITE}, Sec. 4: <pT> from the corrected |eta| < 0.5 spectra extrapolated to all pT.'}}
    den = yaml.safe_load(yaml.safe_dump(num))
    den['centrality'] = {'classes': ['0-5%'], **HF}
    den['hepdata_table_label'] = den['hepdata_table_label'].replace('in E_T^HF intervals (a side',
                                                                    'in the 0-5% class of E_T^HF (the 0-5% side')
    ratio = {'observable_type': 'observable_ratios', 'inspire_id': '2747107', 'version': VERSION,
             'hepdata_file': 'figure_3.yaml', 'dv_index': 0,
             'hepdata_table_label': '<pT>/<pT>^0 vs N_ch/N_ch^0, |eta| < 0.5 (Figure 3)',
             'collision_system': dict(SYSTEM), 'centrality': cent, 'particle_id': {'species': 'charged'},
             'kinematic_range': {k: dict(v) for k, v in KR.items()}, 'observable': 'mean_pT',
             'value_type': dv['header']['name'], 'ratio_type': 'centrality_ratio',
             'numerator': 'E_T^HF interval', 'denominator': '0-5%', 'numerator_system': 'PbPb',
             'denominator_system': 'PbPb', 'numerator_record': f'{OUT_SIDE}#0', 'denominator_record': f'{OUT_SIDE}#1',
             'denominator_class': '0-5%', 'dimensionality': 'integrated',
             'notes': {'value': f'{CITE}, Fig. 3: <pT>/<pT>^0 of charged particles in |eta| < 0.5 (all pT) in 50 GeV '
                                'intervals of E_T^HF, at N_ch/N_ch^0 of each interval; ^0 the 0-5% class. Errors: '
                                'stat (tabulated as 0) and sys.'}}
    for out, docs in ((OUT_SIDE, [num, den]), (OUT_RATIO, [ratio])):
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()

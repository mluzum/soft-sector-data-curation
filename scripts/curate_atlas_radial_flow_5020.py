#!/usr/bin/env python3
"""Curate ATLAS radial-flow fluctuations v0, v0(pT) in Pb+Pb at 5.02 TeV, INSPIRE 2907010.

G. Aad et al. (ATLAS), "Evidence for the collective nature of radial flow
in Pb+Pb collisions with the ATLAS detector", PRL 136 (2026) 032301,
arXiv:2503.24125.

From the paper:
- two subevents, -2.5 < eta_a < -eta_gap/2 and eta_gap/2 < eta_b < 2.5,
  eta_gap = 0, 1 (default), 2, 3; tracks weighted for efficiency and fakes;
- per subevent [pT] = sum w pT / sum w in the reference range pT^ref
  (0.5-2, 0.5-5, 1-5 GeV) and n(pT) = sum_{pT bin} w / sum w over the
  measurement range 0.5 < pT < 10 GeV;
- v0^2 = <d[pT]_a d[pT]_b> / <[pT]>^2, normalized covariance
  <dn(pT) d[pT]> / (<n(pT)> <[pT]>) = v0(pT) v0 (Eq. 1) with the a <-> b
  symmetrized covariance; computed in narrow N_ch ranges (|eta| < 2.5,
  0.5 < pT < 5 GeV) and combined into centrality classes;
- centrality from the FCal Sigma E_T (3.2 < |eta| < 4.9), 0-70%;
- errors: statistical (bootstrap) and systematic.

Records (observable_type radial_flow, multiplicity_baseline N_ch):
- v0 vs centrality, eta_gap = 1, three pT^ref (Fig. 2a; classes from the
  tabulated centres 2.5, 7.5, 15, ... -> 0-5, 5-10, 10-20, ...%);
- normalized covariance (x 10^3, value_scale) in 0-5% and 50-60%, eta_gap
  = 1, three pT^ref (Fig. 2c);
- v0(pT) for eta_gap = 0, 1, 2, 3 in eight classes, three pT^ref
  (auxiliary Figs. 2-5);
- v0(pT)/v0 for pT^ref = 0.5-2 GeV, eta_gap = 1, eight classes (Fig. 4b).
pT points as tabulated (mean pT, 0.1 GeV windows).  Not curated: Figs. 2d,
3, 4a, 5a (the same values as the auxiliary figures and Fig. 4b), the
N_ch-dependence and centrality-estimator studies, the sum-rule closures,
the zero-crossing points, <[pT]>, v0 sqrt(N_ch) and v0(pT)/v0 vs pT/<pT>.

Run from the repository root:  python scripts/curate_atlas_radial_flow_5020.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins2907010/HEPData-ins2907010-v1-yaml'
CITE = 'G. Aad et al. (ATLAS), PRL 136 (2026) 032301, arXiv:2503.24125'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
NCH = {'observable': 'Nch', 'eta': [{'min': -2.5, 'max': 2.5}], 'pT': {'min': 0.5, 'max': 5.0}}
CENTRES = {2.5: '0-5', 7.5: '5-10', 15.0: '10-20', 25.0: '20-30', 35.0: '30-40', 45.0: '40-50', 55.0: '50-60',
           65.0: '60-70', 75.0: '70-80'}
WINDOW = 0.1


def ref_pt(header):
    m = re.search(r'p\^\{ref\}_\{T\}\$ = ([0-9.]+)-([0-9.]+) GeV', header)
    return float(m.group(1)), float(m.group(2))


def record(hf, i, quantity, classes, gap, ref, pts, label, what, scale=None):
    d = {'observable_type': 'radial_flow', 'inspire_id': '2907010', 'version': 1, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': label,
         'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
         'centrality': {'classes': classes, **FCAL},
         'particle_id': {'species': 'charged'},
         'quantity': quantity, 'particles': 'AB',
         'subevents': {'A': {'kinematic_range': {'eta': {'min': -2.5, 'max': -gap / 2 if gap else 0.0}}},
                       'B': {'kinematic_range': {'eta': {'min': gap / 2, 'max': 2.5}}}},
         'eta_gap': gap,
         'reference_pT': {'min': ref[0], 'max': ref[1]},
         'kinematic_range': {'eta': {'min': -2.5, 'max': 2.5}, 'pT': {'min': 0.5, 'max': 10.0}},
         'multiplicity_baseline': dict(NCH),
         'value_type': label.split(' | ')[-1],
         'dimensionality': 'integrated' if pts is None else 'differential_pT'}
    if pts is not None:
        d['kinematic_range']['pT'].update(points=pts, point_window=WINDOW)
    if scale:
        d['value_scale'] = scale
    d['notes'] = {'value': (f'{CITE}. {what}; subevents -2.5 < eta_a < -{gap / 2:g}, {gap / 2:g} < eta_b < 2.5, '
                            f'[pT] in {ref[0]:g} < pT < {ref[1]:g} GeV, n(pT) normalized in 0.5 < pT < 10 GeV'
                            + ('' if pts is None else f', pT points (mean pT, {WINDOW} GeV windows)')
                            + '; deviations at fixed N_ch (|eta| < 2.5, 0.5 < pT < 5 GeV; the paper measures in '
                              'narrow N_ch ranges). Errors: statistical and systematic.')}
    return d


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        desc = ' '.join(x['description'].split())
        aux = re.match(r'Data from Auxiliary,? Figure [2-5], \$v_\{0\}\(p_\{T\}\)\$ For \$\\eta_\{gap\}\$ = ([0-3]), and '
                       r'([0-9]+-[0-9]+)% Centrality$', desc)
        f4b = re.match(r'Data from Figure 4, panel b, .*\$\\eta_\{gap\}\$=1, ([0-9]+-[0-9]+)% Centrality$', desc)
        f2c = re.match(r'Data from Figure 2, panel c, .* in ([0-9]+-[0-9]+)% Centrality$', desc)
        if not (aux or f4b or f2c or hf == 'figure_2a.yaml'):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            assert rows == list(range(len(iv['values']))), (hf, i)
            label = f'{desc} | {h}'
            if hf == 'figure_2a.yaml':
                classes = [f"{CENTRES[float(v['value'])]}%" for v in iv['values']]
                docs.append(record(hf, i, 'v0', classes, 1.0, ref_pt(h), None, label, 'v0 vs centrality (Fig. 2a)'))
                continue
            pts = [float(v['value']) for v in iv['values']]
            if aux:
                docs.append(record(hf, i, 'v0_pT', [f'{aux.group(2)}%'], float(aux.group(1)), ref_pt(h), pts, label,
                                   f'v0(pT), {aux.group(2)}%, eta_gap = {aux.group(1)} (auxiliary figures)'))
            elif f4b:
                assert len(t['dependent_variables']) == 1 and 'v_{0}(p_{T})/v_{0}' in h, hf
                docs.append(record(hf, i, 'v0_pT_over_v0', [f'{f4b.group(1)}%'], 1.0, (0.5, 2.0), pts, label,
                                   f'v0(pT)/v0, {f4b.group(1)}% (Fig. 4b)'))
            else:
                assert '\\times 10^{3}' in h, hf
                docs.append(record(hf, i, 'normalized_covariance', [f'{f2c.group(1)}%'], 1.0, ref_pt(h), pts, label,
                                   f'normalized covariance <dn(pT) d[pT]>/(<n(pT)><[pT]>), tabulated x 10^3, '
                                   f'{f2c.group(1)}% (Fig. 2c)', scale=1000.0))
    out = 'data/radial_flow/atlas_radial_flow_v0_PbPb_5020gev_ins2907010.yaml'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['quantity'], d['eta_gap']) for d in docs)))


if __name__ == '__main__':
    main()

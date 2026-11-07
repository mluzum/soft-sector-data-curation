#!/usr/bin/env python3
"""Curate STAR's flow and [pT] correlations in U+U and Au+Au (imaging nuclear shape), INSPIRE 2937844.

STAR Collaboration, "Imaging nuclear shape through anisotropic and radial
flow in high-energy heavy-ion collisions", Rep. Prog. Phys. 88 (2025)
108601, arXiv:2506.17785.

From the paper (Secs. II A-C, Eqs. 5-8, Appendix):
- U+U at 193 GeV, Au+Au at 200 GeV; charged tracks |eta| < 1, four pT
  ranges 0.2-2, 0.2-3, 0.5-2, 0.5-3 GeV/c; efficiency weights per track;
- centrality from N_ch^rec, reconstructed tracks |eta| < 0.5, pT > 0.15
  GeV/c, with a Glauber fit; the observables averaged over the events of
  a fixed N_ch^rec bin (deviations at fixed multiplicity);
- the tables' centrality values are the mean centralities of nominal
  classes (0-0.2, 0.2-0.5, ..., 55-60%): the edges are rebuilt from 0% as
  e_{k+1} = 2 p_k - e_k, snapped to multiples of 0.25% (or 0.1, 0.2%);
  every point lies within 10% of its class width of the class centre;
- the appendix figures (Figs. 30-37) use the two-subevent method: <v_n^2>
  and <(dpT)^2> from pairs i in -1 < eta < -0.1, j in 0.1 < eta < 1;
  <v_n^2 dpT> with k from either subevent; rho_n^pcc = <v_n^2 dpT> /
  sqrt(var(v_n^2) <(dpT)^2>), var = c_n{2}^2 (two-subevent) + c_n{4}
  (standard, |eta| < 1) (Eqs. 5, 8); rho_n = <v_n^2 dpT> / (<v_n^2>
  sqrt(<(dpT)^2>)) (Eq. 6); dpT from <[pT]>, the event average.

Records (public), each pT range, U+U and Au+Au (Figs. 30, 32, 34, 36):
- flow_vn <v_n^2> (n = 2, 3, 4): cumulant_combination square of the
  two-subevent v_n{2};
- mean_pT_fluctuations <(dpT)^2> (GeV^2): particles AB, averaging events,
  multiplicity_baseline N_ch^rec;
- flow_pT_correlations <v_n^2 dpT> (cov_n, x 10^6), rho_n (variance_terms
  c4 none), rho_2^pcc: covariance_method two_subevent, ck_window
  two_subevent, multiplicity_baseline N_ch^rec;
- mean_pT <pT>;
- observable_ratios: Fig. 38, U+U / Au+Au of each, linked (system_ratio).
Staged (--staged OUT): the same observables vs N_ch (Figs. 31, 33, 35, 37:
efficiency-corrected multiplicity, no N_ch^rec edges); Fig. 22
(rho_2^{pcc,sub}, rho_2^{sub}: differences between the two systems, Eq.
11); Fig. 6 (<v_n^2> vs the event's [pT] in 0-0.5%); Figs. 2, 3
(multiplicity distributions and mappings).

Run from the repository root:  python scripts/curate_star_imaging_nuclear_shape.py --staged OUT.yaml
"""
import argparse
import copy
import glob
import os
import re

import yaml

IID = '2937844'
RAW = glob.glob(f'hepdata_raw/records/ins{IID}/HEPData-ins{IID}-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'STAR Collaboration, Rep. Prog. Phys. 88 (2025) 108601, arXiv:2506.17785'
OUT = {'flow_vn': f'data/flow_vn/star_flow_vn_vn2_UU_AuAu_ins{IID}.yaml',
       'mean_pT_fluctuations': f'data/mean_pT_fluctuations/star_mean_pT_fluctuations_UU_AuAu_ins{IID}.yaml',
       'flow_pT_correlations': f'data/flow_pT_correlations/star_flow_pT_correlations_UU_AuAu_ins{IID}.yaml',
       'mean_pT': f'data/mean_pT/star_mean_pT_UU_AuAu_ins{IID}.yaml',
       'observable_ratios': f'data/observable_ratios/star_observable_ratios_UU_over_AuAu_ins{IID}.yaml'}
SYSTEMS = [('Au+Au', 200.0, 'AU AU --> CHARGED X', 'Au+Au 200 GeV'),
           ('U+U', 193.0, 'U U --> CHARGED X', 'U+U 193 GeV')]
CENT = {'estimator': {'name': 'Nch_rec', 'observable': 'Nch', 'detector': 'TPC'},
        'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.15, 'max': 100.0},
        'notes': 'N_ch^rec, reconstructed TPC tracks |eta| < 0.5, pT > 0.15 GeV/c (Glauber-matched percentiles); '
                 'nominal classes rebuilt from the tabulated mean centralities.'}
BASELINE = {'observable': 'Nch', 'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.15, 'max': 100.0}}
FIGS = {30: (0.2, 3.0), 32: (0.5, 3.0), 34: (0.2, 2.0), 36: (0.5, 2.0)}
SNAP = sorted(set([round(0.25 * k, 4) for k in range(401)] + [0.1, 0.2]))
# (row, panel) -> observable of Figs. 30-38
PANELS = {('upper', 'a'): 'v2sq', ('upper', 'b'): 'v3sq', ('upper', 'c'): 'v4sq', ('upper', 'd'): 'dpt2',
          ('middle', 'a'): 'cov2', ('middle', 'b'): 'cov3', ('middle', 'c'): 'cov4', ('middle', 'd'): 'meanpt',
          ('lower', 'a'): 'rho2', ('lower', 'b'): 'rho3', ('lower', 'c'): 'rho4', ('lower', 'd'): 'rho2pcc'}
WHAT = {'v2sq': '<v2^2>', 'v3sq': '<v3^2>', 'v4sq': '<v4^2>', 'dpt2': '<(dpT)^2>', 'cov2': '<v2^2 dpT>',
        'cov3': '<v3^2 dpT>', 'cov4': '<v4^2 dpT>', 'meanpt': '<pT>', 'rho2': 'rho_2', 'rho3': 'rho_3',
        'rho4': 'rho_4', 'rho2pcc': 'rho_2^pcc'}


def nominal_classes(points):
    """The nominal classes whose mean centralities are `points` (descending)."""
    pts = list(reversed(points))
    edges = [0.0]
    for p in pts:
        edges.append(min(SNAP, key=lambda c: abs(c - (2 * p - edges[-1]))))
    for p, lo, hi in zip(pts, edges[:-1], edges[1:]):
        assert lo < p < hi and abs(p - 0.5 * (lo + hi)) <= 0.1 * (hi - lo) + 0.01, (p, lo, hi)
    return [f'{lo:g}-{hi:g}%' for lo, hi in reversed(list(zip(edges[:-1], edges[1:])))]


def subevents(pt, labels=('A', 'B')):
    s = {'A': (-1.0, -0.1), 'B': (-0.3, 0.3), 'C': (0.1, 1.0)}
    if labels == ('A', 'B'):
        s = {'A': (-1.0, -0.1), 'B': (0.1, 1.0)}
    return {k: {'kinematic_range': {'eta': {'min': s[k][0], 'max': s[k][1]}, 'pT': {'min': pt[0], 'max': pt[1]}}}
            for k in labels}


def fields(obs, pt):
    """Observable type and method fields of one observable."""
    two = {'particles': 'AB', 'subevents': subevents(pt)}
    rho = {'observable': 'rho_vn2_pT', 'correlation_type': 'rho_vn2_pT', 'particles': 'ABC',
           'subevents': subevents(pt, ('A', 'B', 'C')), 'covariance_method': 'two_subevent',
           'ck_window': 'two_subevent', 'multiplicity_baseline': copy.deepcopy(BASELINE)}
    if obs.endswith('sq'):
        n = int(obs[1])
        return 'flow_vn', {'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2,
                           'cumulant_combination': {'kind': 'square', 'orders': [2], 'two_particle': two}}
    if obs == 'dpt2':
        return 'mean_pT_fluctuations', {'observable': 'pT_correlator', **two, 'averaging': 'events',
                                        'multiplicity_baseline': copy.deepcopy(BASELINE)}
    if obs.startswith('cov'):
        return 'flow_pT_correlations', {**rho, 'harmonic': int(obs[3]),
                                        'variance_terms': {'c4': 'standard', 'c2': 'two_subevent'}}
    if obs == 'meanpt':
        return 'mean_pT', {'observable': 'mean_pT'}
    n = int(obs[3])
    vt = {'c4': 'standard' if obs == 'rho2pcc' else 'none', 'c2': 'two_subevent'}
    return 'flow_pT_correlations', {**rho, 'harmonic': n, 'variance_terms': vt}


VALUE_TYPE = {'dpt2': '<dpt1_dpt2>', 'cov2': 'cov_n', 'cov3': 'cov_n', 'cov4': 'cov_n', 'meanpt': 'mean_pT',
              'rho2': 'rho_n', 'rho3': 'rho_n', 'rho4': 'rho_n', 'rho2pcc': 'rho_n'}
NOTE = {'v2sq': 'two-subevent <v_n^2> = c_n{2} (pairs across -1 < eta < -0.1 and 0.1 < eta < 1)',
        'dpt2': 'two-subevent <(dpT)^2> = <(pT_i - <pT>)(pT_j - <pT>)>, i and j across -1 < eta < -0.1 and 0.1 '
                '< eta < 1, event averages, at fixed N_ch^rec',
        'cov2': '<v_n^2 dpT>, two-subevent method (i, j across the subevents, k from either), at fixed N_ch^rec',
        'meanpt': '<pT> = <[pT]> of the events',
        'rho2': 'rho_n = <v_n^2 dpT> / (<v_n^2> sqrt(<(dpT)^2>)) (Eq. 6), two-subevent terms',
        'rho2pcc': 'rho_2^pcc = <v_2^2 dpT> / sqrt(var(v_2^2) <(dpT)^2>) (Eq. 5), var = c_2{2}^2 (two-subevent) '
                   '+ c_2{4} (|eta| < 1) (Eq. 8)'}


FAMILY = {'v2sq': 'v2sq', 'v3sq': 'v2sq', 'v4sq': 'v2sq', 'dpt2': 'dpt2', 'cov2': 'cov2', 'cov3': 'cov2',
          'cov4': 'cov2', 'meanpt': 'meanpt', 'rho2': 'rho2', 'rho3': 'rho2', 'rho4': 'rho2', 'rho2pcc': 'rho2pcc'}


def note(obs):
    return NOTE[FAMILY[obs]]


def panel_of(name):
    m = re.search(r'(upper|middle|lower) row ([abcd])', name)
    return (m.group(1), m.group(2)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    out = {k: [] for k in OUT}
    where = {}
    staged = []
    for x in sub:
        name = ' '.join(x['name'].split())
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        desc = ' '.join(x['description'].split())
        fig = int(re.match(r'Figure (\d+)', name).group(1))
        iv = t['independent_variables'][0]
        if fig in FIGS and panel_of(name):
            pt = FIGS[fig]
            assert f'{pt[0]:g}<p_{{T}}<{pt[1]:g} GeV/c' in desc, (name, desc)
            obs = PANELS[panel_of(name)]
            otype, f = fields(obs, pt)
            classes = nominal_classes([v['value'] for v in iv['values']])
            for j, (label, e, reaction, sysname) in enumerate(SYSTEMS):
                dv = t['dependent_variables'][j]
                assert f'({label})' in dv['header']['name'], (name, dv['header']['name'])
                assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), name
                d = {'observable_type': otype, 'inspire_id': IID, 'version': VERSION, 'hepdata_file': hf,
                     'dv_index': j, 'hepdata_table_label': f'{name}: {desc} (column {j}, {label})',
                     'collision_system': {'energy_gev': e, 'reaction': reaction},
                     'centrality': {'classes': classes, **copy.deepcopy(CENT)},
                     'particle_id': {'species': 'charged'},
                     'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': pt[0], 'max': pt[1]}}}
                d.update(copy.deepcopy(f))
                d.update(dimensionality='integrated',
                         value_type=VALUE_TYPE.get(obs, dv['header']['name']))
                if obs.startswith('cov'):
                    d['value_scale'] = 1.0e6
                d['notes'] = {'value': f'{CITE}, Fig. {fig}. {WHAT[obs]}: {note(obs)}; charged particles |eta| < 1, '
                                       f'{pt[0]:g} < pT < {pt[1]:g} GeV/c, {sysname}. Errors: stat and syst.'}
                where[(obs, pt, j)] = (otype, len(out[otype]))
                out[otype].append(d)
            continue
        if fig == 38:
            continue
        # staged: vs N_ch, the cross-system differences, conditional flow, distributions
        reason = {31: 'vs N_ch', 33: 'vs N_ch', 35: 'vs N_ch', 37: 'vs N_ch', 22: 'fig22', 6: 'fig6'}.get(fig, 'dist')
        why = {'vs N_ch': 'the same observables vs the efficiency-corrected N_ch (Figs. 31-37): no N_ch^rec bin '
                          'edges; the centrality view (Figs. 30-36) is public',
               'fig22': 'rho_2^{pcc,sub}, rho_2^{sub} (Eq. 11): built from differences of U+U and Au+Au '
                        'observables, a cross-system combination not computed',
               'fig6': '<v_n^2> vs the event [pT] (dpT/<[pT]> bins) in 0-0.5%: conditional on the event mean pT, '
                       'not computed',
               'dist': 'multiplicity distributions and centrality / N_ch mappings (detector level)'}[reason]
        for j, dv in enumerate(t['dependent_variables']):
            staged.append({'observable_type': 'flow_pT_correlations' if fig in (22, 6) else 'charged_multiplicity',
                           'inspire_id': IID, 'version': VERSION, 'hepdata_file': hf, 'dv_index': j,
                           'hepdata_table_label': f'{name}: {desc}', 'value_type': dv['header']['name'],
                           'collision_system': {'energy_gev': 193.0 if '(U+U)' in dv['header']['name'] else 200.0,
                                                'reaction': 'U U --> CHARGED X' if '(U+U)' in dv['header']['name']
                                                else 'AU AU --> CHARGED X'},
                           'staging_reason': why, 'notes': {'value': f'{CITE}, {name}.'}})
    # Fig. 38: U+U / Au+Au, one record per pT column
    files = OUT
    ratios = []
    for x in sub:
        name = ' '.join(x['name'].split())
        if not name.startswith('Figure 38'):
            continue
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        obs = PANELS[panel_of(name.replace(' c0', ' c').replace(' c1', ' c'))]
        classes = nominal_classes([v['value'] for v in t['independent_variables'][0]['values']])
        for j, dv in enumerate(t['dependent_variables']):
            m = re.search(r'\$(0\.\d)<p_\{T\}<(\d) GeV/c\$', dv['header']['name'])
            pt = (float(m.group(1)), float(m.group(2)))
            num, den = where[(obs, pt, 1)], where[(obs, pt, 0)]
            side = out[num[0]][num[1]]
            r = {'observable_type': 'observable_ratios', 'inspire_id': IID, 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': j, 'hepdata_table_label': f"{name}: {' '.join(x['description'].split())} (column {j})",
                 'collision_system': {'energy_gev': 193.0, 'reaction': 'U U --> CHARGED X, AU AU --> CHARGED X'},
                 'centrality': {'classes': classes, **copy.deepcopy(CENT)}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': copy.deepcopy(side['kinematic_range']),
                 'observable': side.get('observable') or WHAT[obs], 'value_type': dv['header']['name'],
                 'ratio_type': 'system_ratio', 'numerator': SYSTEMS[1][3], 'denominator': SYSTEMS[0][3],
                 'numerator_system': SYSTEMS[1][3], 'denominator_system': SYSTEMS[0][3],
                 'numerator_record': f'{files[num[0]]}#{num[1]}', 'denominator_record': f'{files[den[0]]}#{den[1]}',
                 'notes': {'value': f'{CITE}, Fig. 38. Ratio U+U (193 GeV) / Au+Au (200 GeV) of {WHAT[obs]} '
                                    f'({note(obs)}), {pt[0]:g} < pT < {pt[1]:g} GeV/c. Errors: stat and syst.'}}
            if 'harmonic' in side:
                r['harmonic'] = side['harmonic']
            ratios.append(r)
    out['observable_ratios'] = ratios
    for otype, docs in out.items():
        with open(OUT[otype], 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{OUT[otype]}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

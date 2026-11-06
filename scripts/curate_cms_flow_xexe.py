#!/usr/bin/env python3
"""Curate CMS flow harmonics in Xe+Xe at 5.44 TeV (and Pb+Pb at 5.02 TeV), INSPIRE 1716441.

A. M. Sirunyan et al. (CMS), "Charged-particle angular correlations in
XeXe collisions at sqrt(sNN) = 5.44 TeV", PRC 100 (2019) 044902,
arXiv:1901.07997.

From the paper:
- centrality from the total energy in the forward calorimeters, 3 <
  |eta| < 5; 11 classes 0-5, ..., 50-60, 60-70% (in Xe+Xe a centrality
  bias up to 5% at 50-60% and 10% at 60-70%, from HYDJET);
- tracks |eta| < 2.4, 0.3 < pT < 10 GeV/c;
- v_n{2}: two-particle correlations, |Delta eta| > 2, associated particles
  1.0 < pT < 3.0 GeV/c, v_n(pT) = V_n(pT, assoc) / sqrt(V_n(assoc, assoc))
  (Xe+Xe and Pb+Pb);
- v_n{SP}: particles of interest at -0.8 < eta < 0 (0 < eta < 0.8) with the
  HF Q-vector at 3 < eta < 5 (-5 < eta < -3), E_T-weighted; resolution
  from the opposite HF and the tracker (|eta| < 0.5);
- v_n{4, 6, 8}: Q-cumulants in |eta| < 2.4 (reference 0.3 < pT < 3.0
  GeV/c);
- integrated values are spectrum-weighted over 0.3 < pT < 3.0 GeV/c;
  errors stat and sys.

Records (public): v_n(pT) from two-particle correlations (Xe+Xe v2, v3, v4
and Pb+Pb v2, v3, v4: the pair gap, reference 1-3 GeV/c), the scalar
product (v2, v3, v4; the opposite HF), and four-, six- and eight-particle
cumulants (v2{4}, v3{4}, v2{6}, v2{8}; the reference 0.3 < pT < 3.0 GeV/c);
integrated v_n{2} (2PC), v2{4}, v2{6}, v2{8}, v3{4} vs class (Xe+Xe) and
v_n{2} (Pb+Pb); the ratios v2{4}/v2{2}, v2{6}/v2{4}, v3{4}/v3{2}
(cumulant combinations, the v{2} term the gapped 2PC); and the Xe+Xe /
Pb+Pb ratios of the 2PC v_n (system ratios, linked).  Nothing is staged
since v2{6}(pT) and v2{8}(pT) are computed (2026-11-06); --staged writes an
empty collection.

Run from the repository root:  python scripts/curate_cms_flow_xexe.py --staged OUT.yaml
"""
import argparse
import copy
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1716441/HEPData-ins1716441-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. M. Sirunyan et al. (CMS), PRC 100 (2019) 044902, arXiv:1901.07997'
OUT = 'data/flow_vn/cms_flow_vn_XeXe_5440gev_PbPb_5020gev_ins1716441.yaml'
OUT_R = 'data/observable_ratios/cms_observable_ratios_flow_XeXe_over_PbPb_ins1716441.yaml'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.0, 'max': -3.0}, {'min': 3.0, 'max': 5.0}],
      'notes': 'total energy in the forward calorimeters, 3 < |eta| < 5.'}
SYSTEMS = {'XE XE --> Charged X': (5440.0, 'XE XE --> CHARGED X', 'Xe+Xe'),
           'PB PB --> Charged X': (5020.0, 'PB PB --> CHARGED X', 'Pb+Pb')}
BIAS = (' In Xe+Xe the centrality determination may be biased (HYDJET: up to 5% at 50-60%, 10% at 60-70%).')
REF_2PC = {'eta': [{'min': -2.4, 'max': 2.4}], 'pT': {'min': 1.0, 'max': 3.0},
           'notes': 'Two-particle correlations, |Delta eta| > 2 (eta_gap, applied on 3D containers), associated '
                    'particles 1.0 < pT < 3.0 GeV/c in |eta| < 2.4.'}
REF_SP = {'detector': 'HF', 'mirror': True, 'eta': [{'min': 3.0, 'max': 5.0}], 'pT': {'min': 0.0, 'max': 100.0},
          'notes': 'HF on the side opposite the particle, 3 < |eta| < 5 (E_T-weighted in the paper; unit weights '
                   'here). Reference pT: all particles.'}
REF_CUM = {'eta': [{'min': -2.4, 'max': 2.4}], 'pT': {'min': 0.3, 'max': 3.0},
           'notes': 'Q-cumulant reference particles, |eta| < 2.4, 0.3 < pT < 3.0 GeV/c.'}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def quals(dv):
    return {k['name']: k['value'] for k in dv.get('qualifiers', [])}


def filled(dv):
    return [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]


def head(hf, i, label, system, classes):
    energy, reaction, _ = SYSTEMS[system]
    return {'observable_type': 'flow_vn', 'inspire_id': '1716441', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf} column {i})',
            'collision_system': {'energy_gev': energy, 'reaction': reaction},
            'centrality': {'classes': classes, **HF}, 'particle_id': {'species': 'charged'}}


def method_fields(kind, n):
    """kind: 2pc | sp | c4 | c6 | c8."""
    if kind == '2pc':
        return {'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2, 'eta_gap': 2.0,
                'flow_reference': copy.deepcopy(REF_2PC)}, (-2.4, 2.4)
    if kind == 'sp':
        return {'harmonic': n, 'method': 'scalar_product', 'flow_reference': copy.deepcopy(REF_SP)}, (-0.8, 0.8)
    k = int(kind[1])
    out = {'harmonic': n, 'method': 'cumulant', 'cumulant_order': k}
    return out, (-2.4, 2.4)


DESC = {'2pc': 'two-particle correlations, |Delta eta| > 2, associated 1.0 < pT < 3.0 GeV/c',
        'sp': 'scalar product with the opposite HF (3 < |eta| < 5)',
        'c4': 'four-particle Q-cumulant (reference 0.3 < pT < 3.0 GeV/c)',
        'c6': 'six-particle Q-cumulant (reference 0.3 < pT < 3.0 GeV/c)',
        'c8': 'eight-particle Q-cumulant (reference 0.3 < pT < 3.0 GeV/c)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, ratios, staged = [], [], []
    index = {}                              # (system, kind, n, class or 'int') -> doc index
    diff = {'table_1a.yaml': ('2pc', 2), 'table_1b.yaml': ('sp', 2), 'table_1c.yaml': ('c4', 2),
            'table_1d.yaml': ('c6', 2), 'table_1e.yaml': ('c8', 2), 'table_2a.yaml': ('2pc', 3),
            'table_2b.yaml': ('sp', 3), 'table_2c.yaml': ('c4', 3), 'table_3a.yaml': ('2pc', 4),
            'table_3b.yaml': ('sp', 4), 'table_6.yaml': ('2pc', 2), 'table_7.yaml': ('2pc', 3),
            'table_8.yaml': ('2pc', 4)}
    for hf, (kind, n) in diff.items():
        t = load(hf)
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            system = q['RE']
            name = SYSTEMS[system][2]
            cls = q['centrality']
            assert re.fullmatch(r'\d+-\d+%', cls), q
            rows = filled(dv)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'sys')}, (hf, i)
            rb = [bins[r] for r in rows]
            fields, (elo, ehi) = method_fields(kind, n)
            if kind in ('c4', 'c6', 'c8'):
                fields['flow_reference'] = copy.deepcopy(REF_CUM)
            d = head(hf, i, f'v{n}(pT), {DESC[kind].split(",")[0]}, {name}, {cls}', system, [cls])
            d['kinematic_range'] = {'eta': {'min': elo, 'max': ehi},
                                    'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}}
            d.update(fields)
            d.update(value_type=f'v_{n}', dimensionality='differential_pT')
            if rows != list(range(len(bins))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. v{n}(pT) of charged particles, |eta| < {ehi:g}, {DESC[kind]}, {name} '
                                   f'{SYSTEMS[system][0] / 1000:g} TeV {cls}.{BIAS if name == "Xe+Xe" else ""} '
                                   'Errors: stat and sys.'}
            index[(name, kind, n, cls)] = len(public)
            public.append(d)
    integ = {'table_4a.yaml': 2, 'table_4b.yaml': 3, 'table_4c.yaml': 4, 'table_10a.yaml': None}
    for hf, n0 in integ.items():
        t = load(hf)
        classes = [f"{v['low']:g}-{v['high']:g}%" for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            system = q['RE']
            name = SYSTEMS[system][2]
            assert q['pT range'] == '0.3 - 3.0', q
            m = re.search(r'(\d)-particle correlations (?:in )?v(\d)', dv['header']['name'])
            k, n = int(m.group(1)), int(m.group(2))
            assert n0 is None or n == n0, (hf, i)
            kind = '2pc' if k == 2 else f'c{k}'
            rows = filled(dv)
            fields, _ = method_fields(kind, n)
            d = head(hf, i, f'v{n}{{{k}}} vs centrality, {name}, 0.3 < pT < 3.0 GeV/c', system,
                     [classes[r] for r in rows])
            d['kinematic_range'] = {'eta': {'min': -2.4, 'max': 2.4}, 'pT': {'min': 0.3, 'max': 3.0}}
            d.update(fields)
            d.update(value_type=f'v_{n}', dimensionality='integrated')
            if rows != list(range(len(classes))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. v{n} of charged particles, |eta| < 2.4, spectrum-weighted over 0.3 < pT '
                                   f'< 3.0 GeV/c, {DESC[kind]}, {name} {SYSTEMS[system][0] / 1000:g} TeV vs '
                                   f'centrality.{BIAS if name == "Xe+Xe" else ""} Errors: stat and sys (rows without '
                                   'errors as tabulated).'}
            index[(name, kind, n, 'int')] = len(public)
            public.append(d)
    combos = {'table_5a.yaml': (2, [4, 2]), 'table_5b.yaml': (2, [6, 4]), 'table_5c.yaml': (3, [4, 2])}
    for hf, (n, orders) in combos.items():
        t = load(hf)
        classes = [f"{v['low']:g}-{v['high']:g}%" for v in t['independent_variables'][0]['values']]
        dv = t['dependent_variables'][0]
        rows = filled(dv)
        assert len(t['dependent_variables']) == 1
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat',)}
        d = head(hf, 0, f'v{n}{{{orders[0]}}}/v{n}{{{orders[1]}}} vs centrality, Xe+Xe', 'XE XE --> Charged X',
                 [classes[r] for r in rows])
        if rows != list(range(len(classes))):
            d['table_rows'] = rows
        d['kinematic_range'] = {'eta': {'min': -2.4, 'max': 2.4}, 'pT': {'min': 0.3, 'max': 3.0}}
        comb = {'kind': 'ratio', 'orders': orders}
        if 2 in orders:
            comb['two_particle'] = {'eta_gap': 2.0, 'flow_reference': copy.deepcopy(REF_2PC)}
        d.update(harmonic=n, method='cumulant', cumulant_order=max(orders), cumulant_combination=comb,
                 value_type=f'v{n}{{{orders[0]}}}/v{n}{{{orders[1]}}}', dimensionality='integrated')
        d['notes'] = {'value': f'{CITE}, Fig. 5. v{n}{{{orders[0]}}}/v{n}{{{orders[1]}}} of charged particles, |eta| < '
                               '2.4, 0.3 < pT < 3.0 GeV/c, Xe+Xe 5.44 TeV vs centrality'
                               + ('; v{2} from two-particle correlations, |Delta eta| > 2, associated 1-3 GeV/c'
                                  if 2 in orders else '') + f'.{BIAS} Errors: stat.'}
        public.append(d)
    for hf in ('table_9a.yaml', 'table_9b.yaml', 'table_9c.yaml', 'table_10b.yaml'):
        t = load(hf)
        integrated = hf == 'table_10b.yaml'
        ivs = t['independent_variables'][0]['values']
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            assert q['SQRT(S)/NUCLEON'] == '5.44 / 5.02', q
            n = int(re.search(r'v(\d)', dv['header']['name']).group(1))
            rows = filled(dv)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'sys')}, (hf, i)
            if integrated:
                classes = [f"{ivs[r]['low']:g}-{ivs[r]['high']:g}%" for r in rows]
                num, den = index[('Xe+Xe', '2pc', n, 'int')], index[('Pb+Pb', '2pc', n, 'int')]
                kr = {'eta': {'min': -2.4, 'max': 2.4}, 'pT': {'min': 0.3, 'max': 3.0}}
                what = f'v{n}{{2}} (0.3 < pT < 3.0 GeV/c)'
            else:
                cls = q['centrality']
                classes = [cls]
                num, den = index[('Xe+Xe', '2pc', n, cls)], index[('Pb+Pb', '2pc', n, cls)]
                rb = [[float(ivs[r]['low']), float(ivs[r]['high'])] for r in rows]
                kr = {'eta': {'min': -2.4, 'max': 2.4}, 'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}}
                what = f'v{n}{{2}}(pT), {cls}'
            d = head(hf, i, f'Xe+Xe/Pb+Pb {what}', 'XE XE --> Charged X', classes)
            d['observable_type'] = 'observable_ratios'
            d['collision_system'] = {'energy_gev': 5440.0, 'reaction': 'XE XE --> CHARGED X, PB PB --> CHARGED X'}
            d['kinematic_range'] = kr
            d.update(observable=f'v{n}', harmonic=n, value_type=f'v{n} Xe+Xe / Pb+Pb', ratio_type='system_ratio',
                     numerator='Xe+Xe 5.44 TeV', denominator='Pb+Pb 5.02 TeV',
                     numerator_system='Xe+Xe 5.44 TeV', denominator_system='Pb+Pb 5.02 TeV',
                     numerator_record=f'{OUT}#{num}', denominator_record=f'{OUT}#{den}')
            if not integrated:
                d['dimensionality'] = 'differential_pT'
            if rows != list(range(len(ivs))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}, Figs. 9, 10. Ratio of the two-particle-correlation {what} in Xe+Xe at '
                                   '5.44 TeV to Pb+Pb at 5.02 TeV, charged particles, |eta| < 2.4, |Delta eta| > 2. '
                                   'Errors: stat and sys.'}
            ratios.append(d)
    for out, docs in ((OUT, public), (OUT_R, ratios)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

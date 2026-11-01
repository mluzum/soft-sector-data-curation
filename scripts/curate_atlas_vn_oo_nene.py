#!/usr/bin/env python3
"""Curate ATLAS v_n in O+O and Ne+Ne at 5.36 TeV, INSPIRE 2967110.

G. Aad et al. (ATLAS), "Measurement of the azimuthal anisotropy of charged
particles in sqrt(sNN) = 5.36 TeV 16O+16O and 20Ne+20Ne collisions with
the ATLAS detector", PRC 113 (2026) 045205, arXiv:2509.05171.

From the paper (Sec. 4):
- tracks |eta| < 2.5; centrality from the FCal Sigma E_T (3.2 < |eta| < 4.9);
- two-particle correlations, |Delta eta| > 2: v_n(pT_b) = v_nn(pT_a, pT_b)
  / sqrt(v_nn(pT_a, pT_a)), reference 0.5 < pT_a < 5 GeV (v_n^2PC); the
  template fit removes the non-flow shape of > 80% peripheral events
  (v_n^sub, the primary measurement);
- four-particle cumulants, two-subevent method (-2.5 < eta < 0 and 0 < eta
  < 2.5), v_n{4} = (-c_n{4})^(1/4);
- errors: statistical and systematic.

Records:
- v_n^2PC(pT) and v_n^sub(pT), n = 2-4, O+O and Ne+Ne, 0-5% ... 60-70%
  (auxiliary Figs. 1-2; Fig. 5 repeats three of their classes): cumulant
  order 2, reference 0.5-5 GeV (flow_reference.pT), pair gap as eta_gap
  (not computed); v_n^sub compares best with a model without non-flow;
- v_n^sub{2} (0.5 < pT < 5 GeV) and v_n{4} (two subevents) vs centrality
  (Fig. 7a; 1% classes to 5%, then 5%, from the tabulated positions -- the
  Ne+Ne points are displaced only in the figure; the v2{4} point at 68 and
  the v3{4} point at 21 match no class and are left out with table_rows);
- template-fit v2, v3 vs centrality in 5% classes (Fig. 10; O+O and Ne+Ne
  columns).
Not curated: v_n vs N_ch (Figs. 6, 11: no centrality classes), the
Ne+Ne/O+O ratios, the Pb+Pb and Xe+Xe columns of Fig. 10 (other
publications), multiplicity distributions.

Run from the repository root:  python scripts/curate_atlas_vn_oo_nene.py
"""
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2967110/HEPData-ins2967110-v*-yaml')[0]
VERSION = int(re.search(r'-v(\d+)-yaml', RAW).group(1))
CITE = 'G. Aad et al. (ATLAS), PRC 113 (2026) 045205, arXiv:2509.05171'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
SYSTEMS = {'O+O': 'O O --> CHARGED X', 'Ne+Ne': 'NE NE --> CHARGED X'}
REF_2PC = {'pT': {'min': 0.5, 'max': 5.0},
           'notes': 'Two-particle correlations, |Delta eta| > 2, reference 0.5 < pT_a < 5 GeV (no pair gap in the '
                    'computation).'}
SUBEVENTS = {'A': {'kinematic_range': {'eta': {'min': -2.5, 'max': 0.0}, 'pT': {'min': 0.5, 'max': 5.0}}},
             'B': {'kinematic_range': {'eta': {'min': 0.0, 'max': 2.5}, 'pT': {'min': 0.5, 'max': 5.0}}}}


def classes_of(values):
    """Centrality classes of a table axis: the tabulated edges when they are
    classes (Fig. 10: 0-5, 5-10, ...); else from the plotting positions
    (Fig. 7a: bins of width 2.6 around the class centres) -- 1% classes
    where neighbouring centres are 1 apart, otherwise 5% (x2.5, x7.5) or
    10% (x5) classes, None where no class fits."""
    lo = [float(v['low']) for v in values]
    hi = [float(v['high']) for v in values]
    if all(a >= 0 and abs((b - a) - 5) < 1e-9 for a, b in zip(lo, hi)):
        return [f'{a:g}-{b:g}%' for a, b in zip(lo, hi)]
    cs = [0.5 * (a + b) for a, b in zip(lo, hi)]
    out = []
    for k, c in enumerate(cs):
        near = [abs(c - cs[j]) for j in (k - 1, k + 1) if 0 <= j < len(cs)]
        if c < 5 and abs(c % 1 - 0.5) < 1e-9 and any(abs(d - 1) < 1e-9 for d in near):
            out.append(f'{c - 0.5:g}-{c + 0.5:g}%')
        elif abs(c % 5 - 2.5) < 1e-9:
            out.append(f'{c - 2.5:g}-{c + 2.5:g}%')
        elif abs(c % 10 - 5) < 1e-9:
            out.append(f'{c - 5:g}-{c + 5:g}%')
        else:
            out.append(None)
    return out


def base(hf, i, system, classes, label):
    return {'observable_type': 'flow_vn', 'inspire_id': '2967110', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': 5360.0, 'reaction': SYSTEMS[system]},
            'centrality': {'classes': classes, **FCAL}, 'particle_id': {'species': 'charged'}}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        desc = ' '.join(x['description'].split())
        aux = hf.startswith(('figure_1', 'figure_2')) and 'auxiliary' in hf
        if not (aux or hf.startswith(('figure_7a', 'figure_10'))):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            label = f'{desc[:140]} | {h}'
            if aux:
                m = re.fullmatch(r'\$v_\{(\d)\}\$ (2PC|sub)', h)
                n, kind, system, cls = int(m.group(1)), m.group(2), q['System'], q['Centrality Class']
                vals = [iv['values'][j] for j in rows]
                pairs = [[float(v['low']), float(v['high'])] for v in vals]
                if rows == list(range(rows[0], rows[-1] + 1)):
                    pt = {'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [pairs[0][0]] + [p[1] for p in pairs]}
                else:                                  # blank high-pT bins: the rows with values
                    pt = {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}
                d = base(hf, i, system, [cls], label)
                d.update({'kinematic_range': {'eta': {'min': -2.5, 'max': 2.5}, 'pT': pt},
                          'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2, 'eta_gap': 2.0,
                          'value_type': h, 'dimensionality': 'differential_pT', 'flow_reference': dict(REF_2PC),
                          'notes': {'value': f'{CITE}. v{n}(pT_b) in {system}, {cls}, '
                                             + ('two-particle correlations' if kind == '2PC' else
                                                'two-particle correlations with the template-fit non-flow '
                                                'subtraction')
                                             + ', |Delta eta| > 2, reference 0.5 < pT_a < 5 GeV. Errors: statistical '
                                               'and systematic.'}})
                docs.append(d)
                continue
            system = re.search(r'\((O\+O|Ne\+Ne|Pb\+Pb|Xe\+Xe)\)', h).group(1)
            if system not in SYSTEMS:
                continue
            all_classes = classes_of(iv['values'])
            keep = [j for j in rows if all_classes[j] is not None]
            classes = [all_classes[j] for j in keep]
            cents = all_classes
            m = re.match(r'\$v_\{(\d)\}', h)
            n = int(m.group(1))
            d = base(hf, i, system, classes, label)
            d['kinematic_range'] = {'eta': {'min': -2.5, 'max': 2.5}, 'pT': {'min': 0.5, 'max': 5.0}}
            d.update({'harmonic': n, 'value_type': h, 'dimensionality': 'integrated'})
            if '{4}' in h:
                d.update({'method': 'cumulant', 'cumulant_order': 4, 'particles': 'AABB', 'subevents': SUBEVENTS})
                how = 'four-particle cumulant, two subevents (-2.5 < eta < 0, 0 < eta < 2.5)'
            else:
                d.update({'method': 'cumulant', 'cumulant_order': 2, 'eta_gap': 2.0, 'flow_reference': dict(REF_2PC)})
                how = 'two-particle correlations with the template-fit non-flow subtraction, |Delta eta| > 2'
            if keep != list(range(len(cents))):
                d['table_rows'] = keep
            d['notes'] = {'value': (f'{CITE}. v{n} vs centrality in {system}, 0.5 < pT < 5 GeV, |eta| < 2.5; {how}; '
                                    'classes from the tabulated positions (1% to 5%, then 5%)'
                                    + ('; positions matching no class left out' if len(keep) < len(rows) else '')
                                    + '. Errors: statistical and systematic.')}
            docs.append(d)
    old = 'data/flow_vn/atlas_flow_vn_azimuthal_anisotropy_of_charged_particles_NeNe_5020gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/atlas_flow_vn_OO_NeNe_5360gev_ins2967110.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(
        (d['collision_system']['reaction'][:5], d['dimensionality'], d.get('cumulant_order')) for d in docs)))


if __name__ == '__main__':
    main()

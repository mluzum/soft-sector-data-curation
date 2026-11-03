#!/usr/bin/env python3
"""Curate ALICE v2, v3 in O-O and Ne-Ne at 5.36 TeV, INSPIRE 2967353.

ALICE Collaboration, "Evidence of nuclear geometry-driven anisotropic flow
in OO and Ne-Ne collisions at sqrt(sNN) = 5.36 TeV", arXiv:2509.06428.

From the paper:
- centrality from the multiplicity in the FT0C, -3.3 < eta < -2.1 (one
  side); events with more than one vertex rejected;
- charged particles |eta| < 0.8, 0.2 < pT < 3 GeV/c (ITS + TPC tracks);
- generic-framework cumulants (acceptance and efficiency corrected):
  v_n{2} with |Delta eta| > 1.4, i.e. the two subevents -0.8 < eta < -0.7
  and 0.7 < eta < 0.8; v2{4} with the standard (single-acceptance) method;
- errors: statistical and systematic.

Records (one per table column, O-O and Ne-Ne):
- v2{2, |Delta eta| > 1.4}, v3{2, |Delta eta| > 1.4} vs centrality (Fig. 1;
  1% classes to 5%, then 5%), particles AB with the two subevents;
- v2{4} vs centrality (Fig. 1; 0-5%, 5-10%, then 10% classes);
- v2{4}/v2{2} and v3{2}/v2{2} vs centrality (Supplemental Figs. A.1, A.2;
  cumulant_combination ratios, the v{2} terms with the gap; v3/v2 through
  harmonics [3, 2]);
- the Ne-Ne / O-O ratios of all five (Fig. 3, Figs. A.1, A.2): system
  ratios (observable_ratios) linking the two systems' records; the sides
  are computed at the ratio tables' classes.
The paper does not say that wider classes are combined from narrower ones:
none is declared.

Run from the repository root:  python scripts/curate_alice_vn_oo_nene_5360.py
"""
import copy
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2967353/HEPData-ins2967353-v*-yaml')[0]
VERSION = int(re.search(r'-v(\d+)-yaml', RAW).group(1))
CITE = 'ALICE Collaboration, arXiv:2509.06428'
OUT = 'data/flow_vn/alice_flow_vn_OO_NeNe_5360gev_ins2967353.yaml'
OUT_RATIOS = 'data/observable_ratios/alice_observable_ratios_vn_NeNe_over_OO_5360gev_ins2967353.yaml'
FT0C = {'estimator': {'name': 'FT0C', 'observable': 'amplitude', 'detector': 'FT0C'},
        'eta': [{'min': -3.3, 'max': -2.1}]}
SYSTEMS = {'OO': ('O+O', 'O O --> CHARGED X'), 'Ne$-$Ne': ('Ne+Ne', 'NE NE --> CHARGED X')}
PT = {'min': 0.2, 'max': 3.0}
GAP = {'particles': 'AB', 'eta_gap': 1.4,
       'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.7}, 'pT': dict(PT)}},
                     'B': {'kinematic_range': {'eta': {'min': 0.7, 'max': 0.8}, 'pT': dict(PT)}}}}
GAP_TXT = 'v_n{2, |Delta eta| > 1.4}: subevents -0.8 < eta < -0.7 and 0.7 < eta < 0.8'
# table -> (harmonic, cumulant order, combination, description, figure)
TABLES = {
    'Table_v22.yaml': (2, 2, None, 'v2{2, |Delta eta| > 1.4}', 'Fig. 1'),
    'Table_v32.yaml': (3, 2, None, 'v3{2, |Delta eta| > 1.4}', 'Fig. 1'),
    'Table_v24.yaml': (2, 4, None, 'v2{4} (standard method)', 'Fig. 1'),
    'Table_v24overv22.yaml': (2, 4, {'kind': 'ratio', 'orders': [4, 2]},
                              'v2{4}/v2{2, |Delta eta| > 1.4}', 'Fig. A.2'),
    'Table_v32overv22.yaml': (3, 2, {'kind': 'ratio', 'orders': [2, 2], 'harmonics': [3, 2]},
                              'v3{2, |Delta eta| > 1.4}/v2{2, |Delta eta| > 1.4}', 'Fig. A.1'),
}
RATIOS = {'Table_NeNeoverOO_v22.yaml': 'Table_v22.yaml', 'Table_NeNeoverOO_v32.yaml': 'Table_v32.yaml',
          'Table_NeNeoverOO_v24.yaml': 'Table_v24.yaml',
          'Table_NeNeoverOO_v24overv22.yaml': 'Table_v24overv22.yaml',
          'Table_NeNeoverOO_v32overv22.yaml': 'Table_v32overv22.yaml'}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def classes(t):
    return [f"{float(v['low']):g}-{float(v['high']):g}%" for v in t['independent_variables'][0]['values']]


def quals(dv):
    return {q['name']: str(q['value']) for q in dv.get('qualifiers', [])}


def main():
    public, index = [], {}
    for hf, (n, order, comb, what, fig) in TABLES.items():
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            assert q['$p_\\mathrm{T}$'] == '0.2-3.0' and q['$\\eta$'] == '-0.8 - 0.8', (hf, q)
            label, reaction = SYSTEMS[q['RE']]
            d = {'observable_type': 'flow_vn', 'inspire_id': '2967353', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{what} vs centrality, {label} ({hf})',
                 'collision_system': {'energy_gev': 5360.0, 'reaction': reaction},
                 'particle_id': {'species': 'charged'}, 'value_type': dv['header']['name'],
                 'centrality': {'classes': classes(t), **copy.deepcopy(FT0C)},
                 'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT)},
                 'harmonic': n, 'method': 'cumulant', 'cumulant_order': order, 'dimensionality': 'integrated'}
            if comb is not None:
                d['cumulant_combination'] = dict(copy.deepcopy(comb), two_particle=copy.deepcopy(GAP))
            elif order == 2:
                d.update(copy.deepcopy(GAP))
            d['notes'] = {'value': f'{CITE}, {fig}. {what} vs centrality, charged particles |eta| < 0.8, '
                                   f'0.2 < pT < 3 GeV/c, {label} 5.36 TeV; centrality from FT0C (-3.3 < eta < -2.1). '
                                   + (f'{GAP_TXT}. ' if (order == 2 or comb) and '1.4' in what else '')
                                   + 'Errors: stat and sys.'}
            index[(hf, label)] = len(public)
            public.append(d)
    ratios = []
    for hf, side in RATIOS.items():
        t = load(hf)
        dv = t['dependent_variables'][0]
        q = quals(dv)
        assert q['RE'] == 'Ne$-$Ne, OO', q
        n, order, comb, what, fig = TABLES[side]
        ratios.append({
            'observable_type': 'observable_ratios', 'inspire_id': '2967353', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': f'{what} Ne+Ne / O+O vs centrality ({hf})',
            'collision_system': {'energy_gev': 5360.0, 'reaction': 'Ne+Ne and O+O'},
            'centrality': {'classes': classes(t), **copy.deepcopy(FT0C)}, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT)},
            'observable': f'v{n}', 'value_type': dv['header']['name'], 'ratio_type': 'system_ratio',
            'numerator': 'Ne+Ne', 'denominator': 'O+O', 'numerator_system': 'Ne+Ne', 'denominator_system': 'O+O',
            'numerator_record': f"{OUT}#{index[(side, 'Ne+Ne')]}", 'denominator_record': f"{OUT}#{index[(side, 'O+O')]}",
            'notes': {'value': f'{CITE}, {"Fig. 3" if fig == "Fig. 1" else fig}. {what} in Ne+Ne over O+O at 5.36 TeV '
                               'vs centrality (the sides at these classes). Errors: stat and sys.'}})
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    with open(OUT_RATIOS, 'w') as fh:
        yaml.safe_dump_all(ratios, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {OUT_RATIOS}: {len(ratios)} records')


if __name__ == '__main__':
    main()

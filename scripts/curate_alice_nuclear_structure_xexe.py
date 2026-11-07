#!/usr/bin/env python3
"""Curate ALICE flow observables for nuclear structure in Xe-Xe and Pb-Pb, INSPIRE 2825785.

S. Acharya et al. (ALICE), "Exploring nuclear structure with multiparticle
azimuthal correlations at the LHC", PLB 869 (2025) 139855,
arXiv:2409.04343.

From the paper (Secs. 2-3):
- charged tracks |eta| < 0.8, 0.2 < pT < 3.0 GeV/c; V0M centrality, 0-60%;
  Xe-Xe at 5.44 TeV, Pb-Pb at 5.02 TeV;
- v_n{2} and v_n{4} from the generic framework; a gap |Delta eta| > 1.0
  for the second-harmonic two-particle correlations (subevents -0.8 < eta
  < -0.5 and 0.5 < eta < 0.8), |Delta eta| > 0.8 for n >= 3 and for the
  multiparticle correlations (subevents -0.8 < eta < -0.4 and 0.4 < eta <
  0.8), none for v2{4};
- Eq. 6: v{2}^2 = <v>^2 + sigma^2, v{4}^2 = <v>^2 - sigma^2 (the tables
  reproduce <v2>, sigma_v2 from the v2{2, |Delta eta| > 1.0} and v2{4}
  columns); Eqs. 9-11: v_{4,22} = Re<V4 V2*^2> / sqrt(<|V2|^4>), rho_{4,22}
  = v_{4,22} / v4{2}, chi_{4,22} = v_{4,22} / sqrt(<|V2|^4>), v4^L =
  sqrt(v4{2}^2 - v_{4,22}^2); NSC(3,2) [67].

Records (public):
- flow_vn: v2{2, |Delta eta| > 1.0} (2% classes), v2{4}, <v2> and sigma_v2
  (cumulant_combination mean_flow, flow_fluctuation with the gapped v{2}),
  v3{2}, v4{2} (|Delta eta| > 0.8), each Xe-Xe and Pb-Pb;
- flow_harmonic_correlations: v_{4,22}, rho_{4,22}, v4^L, chi_{4,22}
  (mode_coupling, two subevents) and NSC(3,2) (symmetric_cumulant, two
  subevents: 3, 2 in one and -3, -2 in the other, "|Delta eta| > 0.8 was
  also applied" to the multiparticle correlations), each Xe-Xe and Pb-Pb;
- observable_ratios: every Xe-Xe / Pb-Pb ratio table (system_ratio), linked
  to the two columns of its observable.
Nothing is staged.

Run from the repository root:  python scripts/curate_alice_nuclear_structure_xexe.py
"""
import glob
import os

import yaml

IID = '2825785'
RAW = glob.glob(f'hepdata_raw/records/ins{IID}/HEPData-ins{IID}-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PLB 869 (2025) 139855, arXiv:2409.04343'
OUT_V = f'data/flow_vn/alice_flow_vn_nuclear_structure_XeXe_PbPb_ins{IID}.yaml'
OUT_C = f'data/flow_harmonic_correlations/alice_flow_harmonic_correlations_nuclear_structure_XeXe_PbPb_ins{IID}.yaml'
OUT_R = f'data/observable_ratios/alice_observable_ratios_XeXe_over_PbPb_ins{IID}.yaml'
PT = {'min': 0.2, 'max': 3.0}
KR = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT)}
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SYSTEMS = [('Xe-Xe', 5440.0, 'XE XE --> CHARGED X', 'Xe+Xe 5.44 TeV'),
           ('Pb-Pb', 5020.0, 'PB PB --> CHARGED X', 'Pb+Pb 5.02 TeV')]


def subevents(gap):
    """|Delta eta| > gap in |eta| < 0.8: the two halves beyond +-gap/2."""
    h = gap / 2
    return {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -h}, 'pT': dict(PT)}},
            'B': {'kinematic_range': {'eta': {'min': h, 'max': 0.8}, 'pT': dict(PT)}}}


GAP10 = {'particles': 'AB', 'subevents': subevents(1.0)}
GAP08 = {'particles': 'AB', 'subevents': subevents(0.8)}
# table -> (type, fields, description); the ratio table follows from the name
TABLES = {
    'Table1a': ('flow_vn', dict(harmonic=2, method='cumulant', cumulant_order=2, **GAP10),
                'v2{2, |Delta eta| > 1.0}, two subevents -0.8 < eta < -0.5 and 0.5 < eta < 0.8'),
    'Table1b': ('flow_vn', dict(harmonic=2, method='cumulant', cumulant_order=4), 'v2{4}, |eta| < 0.8, no gap'),
    'Table2a': ('flow_vn', dict(harmonic=2, method='cumulant', cumulant_order=4,
                                cumulant_combination={'kind': 'mean_flow', 'orders': [2, 4], 'two_particle': GAP10}),
                '<v2> = sqrt((v2{2}^2 + v2{4}^2)/2) (Eq. 6), v2{2} with |Delta eta| > 1.0'),
    'Table2b': ('flow_vn', dict(harmonic=2, method='cumulant', cumulant_order=4,
                                cumulant_combination={'kind': 'flow_fluctuation', 'orders': [2, 4],
                                                      'two_particle': GAP10}),
                'sigma_v2 = sqrt((v2{2}^2 - v2{4}^2)/2) (Eq. 6), v2{2} with |Delta eta| > 1.0'),
    'Table3a': ('flow_vn', dict(harmonic=3, method='cumulant', cumulant_order=2, **GAP08),
                'v3{2, |Delta eta| > 0.8}, subevents -0.8 < eta < -0.4 and 0.4 < eta < 0.8'),
    'Table3b': ('flow_vn', dict(harmonic=4, method='cumulant', cumulant_order=2, **GAP08),
                'v4{2, |Delta eta| > 0.8}, subevents -0.8 < eta < -0.4 and 0.4 < eta < 0.8'),
    'Table4a': ('flow_harmonic_correlations',
                dict(observable='v4_22', correlation_type='mode_coupling', harmonics=[4, 2, 2],
                     quantity='v_mode_coupled', **GAP08),
                'v_{4,22} = Re<V4 V2*^2> / sqrt(<|V2|^4>) (Eq. 9)'),
    'Table4b': ('flow_harmonic_correlations',
                dict(observable='rho_4_22', correlation_type='mode_coupling', harmonics=[4, 2, 2], quantity='rho',
                     **GAP08),
                'rho_{4,22} = v_{4,22} / v4{2} (Eq. 10)'),
    'TableA1a': ('flow_harmonic_correlations',
                 dict(observable='v4_L', correlation_type='mode_coupling', harmonics=[4, 2, 2], quantity='v_linear',
                      **GAP08),
                 'v4^L = sqrt(v4{2}^2 - v_{4,22}^2)'),
    'TableA1b': ('flow_harmonic_correlations',
                 dict(observable='chi_4_22', correlation_type='mode_coupling', harmonics=[4, 2, 2], quantity='chi',
                      **GAP08),
                 'chi_{4,22} = v_{4,22} / sqrt(<|V2|^4>) (Eq. 11)'),
    'TableA2Top': ('flow_harmonic_correlations',
                   dict(observable='NSC_3_2', correlation_type='symmetric_cumulant', harmonics=[3, 2],
                        normalized=True, **GAP08),
                   'NSC(3,2) = (<<3,2|-3,-2>> - <<3|-3>><<2|-2>>) / (<<3|-3>><<2|-2>>), the multiparticle '
                   'correlation with |Delta eta| > 0.8: 3, 2 in one subevent and -3, -2 in the other (curated '
                   'reading of Sec. 2)'),
}
RATIO = {'Table1a': 'Table1c', 'Table1b': 'Table1d', 'Table2a': 'Table2c', 'Table2b': 'Table2d',
         'Table3a': 'Table3c', 'Table3b': 'Table3d', 'Table4a': 'Table4c', 'Table4b': 'Table4d',
         'TableA1a': 'TableA1c', 'TableA1b': 'TableA1d', 'TableA2Top': 'TableA2Bottom'}


def load(name):
    t = yaml.safe_load(open(os.path.join(RAW, f'{name}.yaml')))
    iv = t['independent_variables'][0]
    classes = [f"{float(b['low']):g}-{float(b['high']):g}%" for b in iv['values']]
    for dv in t['dependent_variables']:
        assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), name
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        assert q['$p_\\mathrm{T}$'] == '0.2-3.0' and q['$\\eta$'] == '-0.8 - 0.8', (name, q)
    return t, classes


def main():
    sub = {x['data_file'][:-5]: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    out = {'flow_vn': [], 'flow_harmonic_correlations': []}
    where = {}
    for name, (otype, fields, what) in TABLES.items():
        t, classes = load(name)
        for j, (label, e, reaction, sysname) in enumerate(SYSTEMS):
            dv = t['dependent_variables'][j]
            assert label.replace('-', '$-$') in dv['header']['name'], (name, j, dv['header']['name'])
            d = {'observable_type': otype, 'inspire_id': IID, 'version': VERSION, 'hepdata_file': f'{name}.yaml',
                 'dv_index': j,
                 'hepdata_table_label': f"{sub[name]['name']}: {' '.join(sub[name]['description'].split())} "
                                        f"(column {j}, {label})",
                 'collision_system': {'energy_gev': e, 'reaction': reaction},
                 'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {k: dict(v) for k, v in KR.items()}}
            d.update(yaml.safe_load(yaml.safe_dump(fields)))
            d.update(dimensionality='integrated', value_type=dv['header']['name'],
                     notes={'value': f'{CITE}. {what}; charged particles |eta| < 0.8, 0.2 < pT < 3.0 GeV/c, '
                                     f'{sysname}, V0M centrality. Errors: stat and sys.'})
            where[(name, j)] = (otype, len(out[otype]))
            out[otype].append(d)
    files = {'flow_vn': OUT_V, 'flow_harmonic_correlations': OUT_C}
    ratios = []
    for name, rname in RATIO.items():
        t, classes = load(rname)
        _, side_classes = load(name)
        assert classes == side_classes, (name, rname)
        dv = t['dependent_variables'][0]
        num, den = where[(name, 0)], where[(name, 1)]
        side = out[num[0]][num[1]]
        r = {'observable_type': 'observable_ratios', 'inspire_id': IID, 'version': VERSION,
             'hepdata_file': f'{rname}.yaml', 'dv_index': 0,
             'hepdata_table_label': f"{sub[rname]['name']}: {' '.join(sub[rname]['description'].split())}",
             'collision_system': {'energy_gev': 5440.0, 'reaction': 'XE XE --> CHARGED X, PB PB --> CHARGED X'},
             'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'},
             'kinematic_range': {k: dict(v) for k, v in KR.items()},
             'observable': side.get('observable') or f"v{side['harmonic']}",
             'value_type': dv['header']['name'], 'ratio_type': 'system_ratio',
             'numerator': SYSTEMS[0][3], 'denominator': SYSTEMS[1][3],
             'numerator_system': SYSTEMS[0][3], 'denominator_system': SYSTEMS[1][3],
             'numerator_record': f'{files[num[0]]}#{num[1]}', 'denominator_record': f'{files[den[0]]}#{den[1]}',
             'notes': {'value': f'{CITE}. Ratio Xe-Xe (5.44 TeV) / Pb-Pb (5.02 TeV) of {TABLES[name][2]}. '
                                'Errors: stat and sys.'}}
        if 'harmonic' in side:
            r['harmonic'] = side['harmonic']
        ratios.append(r)
    for otype, docs in out.items():
        with open(files[otype], 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{files[otype]}: {len(docs)} records')
    with open(OUT_R, 'w') as fh:
        yaml.safe_dump_all(ratios, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT_R}: {len(ratios)} records')


if __name__ == '__main__':
    main()

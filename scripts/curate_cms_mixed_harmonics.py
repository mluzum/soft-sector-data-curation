#!/usr/bin/env python3
"""Curate CMS mixed higher-order flow harmonics and nonlinear response coefficients, INSPIRE 1759853.

A. M. Sirunyan et al. (CMS), "Mixed higher-order anisotropic flow and
nonlinear response coefficients of charged particles in PbPb collisions at
sqrt(sNN) = 2.76 and 5.02 TeV", Eur. Phys. J. C 80 (2020) 534,
arXiv:1910.08789.

From the paper (Secs. 2-4, Figs. 1-6):
- charged particles |eta| < 0.8, 0.3 < pT < 8.0 GeV/c (bins), and 0.3 < pT
  < 3.0 GeV/c vs centrality (Figs. 4-6);
- centrality from the summed HF energy (HF: 3.0 < |eta| < 5.2), classes
  0-20%, 20-60% (vs pT) and 0-5, ..., 50-60% (vs centrality);
- the scalar-product method (Eqs. 5-14): Q_n of the tracks in each pT
  range, Q_nA and Q_nB from the two HF sides (E_T-weighted towers, at
  least 2.2 units of eta from the tracks: 3.0 < |eta|; the upper end is not
  stated, curated as the HF edge 5.2), self-correlations removed;
  v4{Psi22} = Re<Q4 Q2A* Q2B*> / sqrt(Re<Q2A Q2A Q2B* Q2B*>),
  v5{Psi23} = Re<Q5 Q2A* Q3B*> / sqrt(Re<Q2A Q3A Q2B* Q3B*>),
  v6{Psi222} = Re<Q6 Q2A* Q2B* Q2B*> / sqrt(Re<Q2A Q2A Q2A Q2B* Q2B* Q2B*>),
  v6{Psi33}, v7{Psi223} likewise: the first lower harmonic in A, the rest
  in B; chi = Re<Q_k Q_nA* Q_mB* ...> / Re<Q_n Q_m ... Q_nA* Q_mB* ...>, the
  tracker Q_n Q_m (of the same pT range) in the denominator;
- the overall v_n{Psi_n} (Fig. 2, 5.02 TeV), "measured using the
  scalar-product method", whose formula the paper does not give.

Records (public):
- mode_coupling (analysis/mode_coupling.py, the split layout:
  flow_reference.subevents A, B the two HF sides, both orientations
  pooled): v_k{Psi_nm} as quantity v_mode_coupled, chi as quantity chi with
  chi_denominator poi; vs pT (Tables 1-20, 41-60) and vs centrality (Tables
  61-80), 2.76 and 5.02 TeV.  v7{Psi223} 0-20% at 5.02 TeV from Table 35
  (Table 5 with more digits).
- flow_vn: v_n{Psi_n}(pT), n = 4-7, 0-20% and 20-60% at 5.02 TeV (Tables
  21-23, 25-28, 30), curated as CMS's scalar product with the opposite HF
  (PRC 100 (2019) 044902; the paper does not define it here).
Not curated (duplicates): Tables 24, 29 (= 23, 28: v6{Psi6} drawn twice),
31-34, 36-40 (= 1-4, 6-10), 81-90 (= 71-80).  Nothing is staged.

Run from the repository root:  python scripts/curate_cms_mixed_harmonics.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1759853/HEPData-ins1759853-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. M. Sirunyan et al. (CMS), EPJC 80 (2020) 534, arXiv:1910.08789'
OUT_M = 'data/flow_harmonic_correlations/cms_flow_harmonic_correlations_mixed_harmonics_PbPb_ins1759853.yaml'
OUT_V = 'data/flow_vn/cms_flow_vn_vn_sp_higher_harmonics_PbPb_5020gev_ins1759853.yaml'
HF_ETA = (3.0, 5.2)
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -HF_ETA[1], 'max': -HF_ETA[0]}, {'min': HF_ETA[0], 'max': HF_ETA[1]}],
      'notes': 'summed energy in the forward calorimeters (HF), 3.0 < |eta| < 5.2.'}
SPLIT = {'detector': 'HF',
         'subevents': {'A': {'eta': [{'min': HF_ETA[0], 'max': HF_ETA[1]}], 'pT': {'min': 0.0, 'max': 100.0}},
                       'B': {'eta': [{'min': -HF_ETA[1], 'max': -HF_ETA[0]}], 'pT': {'min': 0.0, 'max': 100.0}}},
         'notes': 'Q_nA, Q_nB from the two HF sides, 3.0 < |eta| < 5.2 (E_T-weighted towers in the paper; charged '
                  'particles with unit weights here): the first lower harmonic in A, the rest in B (Eqs. 5-14), '
                  'both orientations pooled. Reference pT: all particles.'}
REF_SP = {'detector': 'HF', 'mirror': True, 'eta': [{'min': HF_ETA[0], 'max': HF_ETA[1]}],
          'pT': {'min': 0.0, 'max': 100.0},
          'notes': 'HF on the side opposite the particle, 3.0 < |eta| < 5.2 (E_T-weighted in the paper; unit weights '
                   'here). Reference pT: all particles.'}
SKIP = {5, 24, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40} | set(range(81, 91))
ENERGY = {'5.02': 5020.0, '2.76': 2760.0}


def harmonics_of(name):
    """'$v_5\\{\\Psi_{23}\\}$' -> ('v', [5, 2, 3]); '$\\chi_{7223}$' -> ('chi', [7, 2, 2, 3]);
    '$v_4\\{\\Psi_{4}\\}$' -> ('vn', [4])."""
    m = re.fullmatch(r'\$v_(\d)\\\{\\Psi_\{(\d+)\}\\\}\$', name)
    if m:
        k, low = int(m.group(1)), [int(c) for c in m.group(2)]
        return ('vn', [k]) if low == [k] else ('v', [k] + low)
    m = re.fullmatch(r'\$\\chi_\{(\d+)\}\$', name)
    return 'chi', [int(c) for c in m.group(1)]


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    modes, vn = [], []
    for x in sub:
        t = int(x['name'].split()[-1])
        if t in SKIP:
            continue
        hf = x['data_file']
        tab = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = tab['independent_variables'][0]
        dv = tab['dependent_variables'][0]
        name = dv['header']['name']
        energy = str(x['keywords'][1]['values'][0])
        assert x['keywords'][1]['name'] == 'cmenergies' and energy in ENERGY, (hf, x['keywords'])
        assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
        kind, harms = harmonics_of(name)
        assert harms[0] == sum(harms[1:]) or kind == 'vn', name
        desc = ' '.join(x['description'].split())
        d = {'inspire_id': '1759853', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f'{x["name"]}: {desc}',
             'collision_system': {'energy_gev': ENERGY[energy], 'reaction': 'PB PB --> CHARGED X'}}
        if iv['header']['name'] == 'PT':
            cls = re.search(r'in the (\d+)-(\d+)% centrality range', desc)
            classes = [f'{cls.group(1)}-{cls.group(2)}%']
            bins = [[float(b['low']), float(b['high'])] for b in iv['values']]
            kr = {'eta': {'min': -0.8, 'max': 0.8},
                  'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            dim, where = 'differential_pT', f'vs pT, {classes[0]}'
        else:
            assert iv['header']['name'] == 'Centrality', hf
            classes = [f"{float(b['low']):g}-{float(b['high']):g}%" for b in iv['values']]
            kr = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.3, 'max': 3.0}}
            dim, where = 'integrated', 'vs centrality, 0.3 < pT < 3.0 GeV/c'
        d.update(centrality={'classes': classes, **HF}, particle_id={'species': 'charged'}, kinematic_range=kr)
        tev = f'{energy} TeV'
        if kind == 'vn':
            n = harms[0]
            d = {'observable_type': 'flow_vn', **d, 'harmonic': n, 'method': 'scalar_product',
                 'flow_reference': dict(REF_SP), 'value_type': name, 'dimensionality': dim,
                 'notes': {'value': f'{CITE}. v{n}{{Psi_{n}}} {where}, charged particles |eta| < 0.8, {tev}, '
                                    'from "the scalar-product method" (Fig. 2), whose formula the paper does not '
                                    'give: curated as CMS\'s scalar product with the HF on the opposite side '
                                    '(PRC 100 (2019) 044902). Errors: statistical and systematic.'}}
            vn.append(d)
            continue
        low = ''.join(map(str, harms[1:]))
        q = 'v_mode_coupled' if kind == 'v' else 'chi'
        d = {'observable_type': 'flow_harmonic_correlations', **d,
             'observable': f'v{harms[0]}_Psi{low}' if kind == 'v' else f'chi{harms[0]}{low}',
             'correlation_type': 'mode_coupling', 'harmonics': harms, 'quantity': q}
        if kind == 'chi':
            d['chi_denominator'] = 'poi'
        nums = ' '.join([f'Q{harms[1]}A*'] + [f'Q{h}B*' for h in harms[2:]])
        if kind == 'v':
            what = (f'v{harms[0]}{{Psi_{low}}} = Re<Q{harms[0]} {nums}> / sqrt(Re<'
                    + ' '.join([f'Q{h}A' for h in harms[1:]] + [f'Q{h}B*' for h in harms[1:]]) + '>)')
        else:
            what = (f'chi_{harms[0]}{low} = Re<Q{harms[0]} {nums}> / Re<'
                    + ' '.join([f'Q{h}' for h in harms[1:]]) + f' {nums}>, the tracker Q of the same pT range')
        d.update(flow_reference=dict(SPLIT), dimensionality=dim, value_type=name,
                 notes={'value': f'{CITE}. {what} (Eqs. 5-14), {where}, charged particles |eta| < 0.8, {tev}; '
                                 'Q_nA, Q_nB from the two HF sides (3.0 < |eta| < 5.2).'
                                 + (' Table 35 (Table 5 with more digits).' if t == 35 else '')
                                 + ' Errors: statistical and systematic.'})
        modes.append(d)
    for out, docs in ((OUT_M, modes), (OUT_V, vn)):
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()

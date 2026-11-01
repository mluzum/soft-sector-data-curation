#!/usr/bin/env python3
"""Curate ALICE pi, K, p spectra, yields and mean pT in Pb-Pb at 2.76 TeV, INSPIRE 1222333.

B. Abelev et al. (ALICE), "Centrality dependence of pi, K, p production in
Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", PRC 88, 044910 (2013),
arXiv:1303.0737.

From the paper and tables:
- primary pi+-, K+-, p, pbar (weak-decay feed-down subtracted) at |y| <
  0.5, invariant yields 1/N_ev 1/(2 pi pT) d2N/(dpT dy) in ten V0M classes
  0-5% ... 80-90%;
- dN/dy and <pT> of each species, extrapolated to the full pT range with
  blast-wave fits (Tables 31, 32);
- errors: statistical and systematic.

Records: the 60 spectra (Tables 1-30), dN/dy of the six species vs
centrality (Table 31) and their <pT> (Table 32, mean_pT records), all at
|y| < 0.5; dN/dy and <pT> with the full pT range (0-100 GeV/c: the
extrapolated yields).  Not curated: the dNch/deta column of Table 31 (from
ALICE's multiplicity paper), the particle ratios (Tables 33-57: no ratio
computation).

Run from the repository root:  python scripts/curate_alice_pikp_pbpb_2760.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1222333/HEPData-ins1222333-v1-yaml'
CITE = 'B. Abelev et al. (ALICE), PRC 88, 044910 (2013), arXiv:1303.0737'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SPECIES = {'PI+': 'pi+', 'PI-': 'pi-', 'K+': 'K+', 'K-': 'K-', 'P': 'p', 'PBAR': 'pbar'}
FULL_PT = {'min': 0.0, 'max': 100.0}


def species_of(dv):
    re_ = {q['name']: q.get('value') for q in dv.get('qualifiers', [])}['RE']
    return SPECIES.get(re.fullmatch(r'PB PB --> (\S+) X', re_).group(1))


def base(otype, hf, i, sp, classes, label):
    return {'observable_type': otype, 'inspire_id': '1222333', 'version': 1, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': label,
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> X'},
            'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': sp}}


def main():
    docs = {'spectra': [], 'yields': [], 'mean_pt': []}
    for n in range(1, 33):
        hf = f'Table_{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
            sp = species_of(dv)
            if sp is None:                        # the dNch/deta column of Table 31
                continue
            assert q.get('YRAP') == '-0.5-0.5', (hf, q)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            label = f"{dv['header']['name']} of {sp} ({hf})"
            if n <= 30:
                cls = '%g-%g%%' % tuple(float(x) for x in q['CENTRALITY'].split('-'))
                vals = [iv['values'][j] for j in rows]
                assert rows == list(range(rows[0], rows[-1] + 1)), (hf, i)
                edges = [float(vals[0]['low'])] + [float(v['high']) for v in vals]
                d = base('particle_spectra', hf, i, sp, [cls], label)
                d.update({'kinematic_range': {'y': {'min': -0.5, 'max': 0.5},
                                              'pT': {'min': edges[0], 'max': edges[-1], 'bins': edges}},
                          'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                          'notes': {'value': f'{CITE}. 1/N_ev 1/(2 pi pT) d2N/(dpT dy) of primary {sp}, |y| < 0.5, '
                                             f'{cls}. Errors: statistical and systematic.'}})
                docs['spectra'].append(d)
                continue
            classes = ['%g-%g%%' % (float(iv['values'][j]['low']), float(iv['values'][j]['high'])) for j in rows]
            kr = {'y': {'min': -0.5, 'max': 0.5}, 'pT': dict(FULL_PT)}
            if n == 31:
                d = base('particle_yields', hf, i, sp, classes, label)
                d.update({'kinematic_range': kr, 'observable': 'dN_dy', 'value_type': 'dN_dy',
                          'notes': {'value': f'{CITE}. dN/dy of primary {sp}, |y| < 0.5, extrapolated to the full pT '
                                             'range (blast-wave fits). Errors: combined.'}})
                docs['yields'].append(d)
            else:
                d = base('mean_pT', hf, i, sp, classes, label)
                d.update({'kinematic_range': kr, 'observable': 'mean_pT', 'value_type': 'mean_pT',
                          'notes': {'value': f'{CITE}. <pT> of primary {sp}, |y| < 0.5, extrapolated to the full pT '
                                             'range (blast-wave fits). Errors: as tabulated.'}})
                docs['mean_pt'].append(d)
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
    outs = {'spectra': 'data/particle_spectra/alice_particle_spectra_centrality_dependence_of_pi_k_PbPb_2760gev.yaml',
            'yields': 'data/particle_yields/alice_particle_yields_centrality_dependence_of_pi_k_PbPb_2760gev.yaml',
            'mean_pt': 'data/mean_pT/alice_mean_pT_centrality_dependence_of_pi_k_p_PbPb_2760gev_ins1222333.yaml'}
    for k, out in outs.items():
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs[k], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs[k])} records;', dict(collections.Counter(d['particle_id']['species'] for d in docs[k])))


if __name__ == '__main__':
    main()

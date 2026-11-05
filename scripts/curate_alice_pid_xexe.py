#!/usr/bin/env python3
"""Curate ALICE pi, K, p and phi spectra in Xe-Xe at 5.44 TeV, INSPIRE 1840099.

S. Acharya et al. (ALICE), "Production of pions, kaons, (anti-)protons and
phi mesons in Xe-Xe collisions at sqrt(sNN) = 5.44 TeV", EPJC 81 (2021)
584, arXiv:2101.03100.

From the paper and the tables:
- primary pi+ + pi-, K+ + K-, p + pbar and phi at |y| < 0.5,
  (1/N_ev) d2N/(dpT dy) per event in pT bins (value_type dN_dpT_dy);
- centrality from the V0 amplitude (V0M): pi, K, p in 0-5, 5-10, 10-20
  ... 60-70, 70-90%; phi in 0-10, 10-30, 30-50, 50-70, 70-90% (Sec. 2);
- errors: stat and syst.

Records (public): the pT spectra (Fig. 1).  Not curated: K/pi, p/pi,
phi/p vs pT and the integrated K/pi, p/pi, phi/pi (Figs. 7-9: ratios of the
curated spectra and of the dN/dy of Table 4, which HEPData does not hold).

Run from the repository root:  python scripts/curate_alice_pid_xexe.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1840099/HEPData-ins1840099-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), EPJC 81 (2021) 584, arXiv:2101.03100'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SPECIES = {'XE XE --> PI+ X, XE XE --> PI- X': ('pi+ + pi-', 'XE XE --> PI+ PI- X'),
           'XE XE --> K+ X, XE XE --> K- X': ('K+ + K-', 'XE XE --> K+ K- X'),
           'XE XE --> P X, XE XE --> PBAR X': ('p + pbar', 'XE XE --> P PBAR X'),
           'XE XE --> PHI X': ('phi', 'XE XE --> PHI X')}
OUT = 'data/particle_spectra/alice_particle_spectra_pi_K_p_phi_XeXe_5440gev_ins1840099.yaml'


def cls_of(s):
    lo, hi = (float(x) for x in s.split('-'))
    return f'{lo:g}-{hi:g}%'


def main():
    docs = []
    for n in range(1, 33):
        hf = f'table_{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = t['dependent_variables'][0]
        q = {x['name']: str(x['value']) for x in dv['qualifiers']}
        assert dv['header']['name'] == '(1/Nev)*D2(N)/DPT/DYRAP', hf
        assert q['SQRT(S)/NUCLEON'] == '5440.0' and q['YRAP'] == '-0.5-0.5', q
        assert [e['label'] for e in dv['values'][0]['errors']] == ['stat.', 'syst.'], hf
        species, reaction = SPECIES[q['RE']]
        cls = cls_of(q['CENTRALITY'])
        edges = [(float(v['low']), float(v['high'])) for v in t['independent_variables'][0]['values']]
        assert all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(edges[:-1], edges[1:])), hf
        d = {'observable_type': 'particle_spectra', 'inspire_id': '1840099', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'{species} d2N/(dpT dy), Xe-Xe 5.44 TeV, {cls} ({hf})',
             'collision_system': {'energy_gev': 5440.0, 'reaction': reaction},
             'centrality': {'classes': [cls], **V0M},
             'particle_id': {'species': species},
             'kinematic_range': {'y': {'min': -0.5, 'max': 0.5},
                                 'pT': {'min': edges[0][0], 'max': edges[-1][1],
                                        'bins': [edges[0][0]] + [e[1] for e in edges]}},
             'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy'}
        d['notes'] = {'value': f'{CITE}, Fig. 1. (1/N_ev) d2N/(dpT dy) of primary {species}, |y| < 0.5, Xe-Xe 5.44 TeV '
                               f'{cls} (V0M). Errors: stat and syst.'}
        docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()

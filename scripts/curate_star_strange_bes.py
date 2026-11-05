#!/usr/bin/env python3
"""Curate STAR strange hadron production in Au+Au at 7.7-39 GeV, INSPIRE 1738953.

J. Adam et al. (STAR), "Strange hadron production in Au+Au collisions at
sqrt(sNN) = 7.7, 11.5, 19.6, 27, and 39 GeV", PRC 102 (2020) 034909,
arXiv:1906.03732.

From the paper (Sec. II-III):
- K0_S, Lambda, Lambdabar, Xi-, Xibar+, Omega-, Omegabar+, phi at
  mid-rapidity |y| < 0.5;
- centrality from the uncorrected charged multiplicity (RefMult, |eta| <
  0.5, as STAR PRC 96 044904), classes 0-5, 5-10, 10-20, 20-30, 30-40,
  40-60, 60-80% (Table II gives their <N_part>); Omega and phi in merged
  classes (identified here by their <N_part>: 0-10, 10-20, 20-40, 40-60,
  60-80%; Omega 10-60% at 11.5 and 0-60% at 7.7 GeV);
- Lambda (Lambdabar) corrected for the weak-decay feed-down from Xi and Xi0
  (and Omegabar+ for Lambdabar); Lambda from Sigma0 decays count as prompt;
- spectra d2N/(2 pi pT dpT dy) per event in pT bins; dN/dy and <pT>
  include the extrapolation to unmeasured pT (Levy, blast-wave fits);
- errors: stat and sys.

Records (public): pT spectra of K0_S, Lambda, Lambdabar, Xi-, Xibar+ per
energy and class (Figs. 5-9), dN/dy and <pT> of all eight species vs class
per energy (Fig. 17 and the online <pT> tables).  Not curated: <m_T> - m0
(Figs. 13-15, the same spectra), antiparticle/particle and Lambdabar/K0_S
ratios (Figs. 20-22, 29: ratios of the curated yields and spectra), R_CP
(Fig. 28) and the fitted mu_B/T, mu_S/T (Fig. 26).

Run from the repository root:  python scripts/curate_star_strange_bes.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1738953/HEPData-ins1738953-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (STAR), PRC 102 (2020) 034909, arXiv:1906.03732'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'uncorrected charged multiplicity in |eta| < 0.5 (STAR BES-I, PRC 96 044904).'}
# HEPData name -> (species, reaction product)
SPECIES = {'K0S': ('K0_S', 'K0S'), 'Lambda': ('Lambda', 'LAMBDA'), 'AntiLambda': ('Lambdabar', 'LAMBDABAR'),
           'Xi-': ('Xi-', 'XI-'), 'AntiXi+': ('Xibar+', 'XIBAR+'), 'Omega-': ('Omega-', 'OMEGA-'),
           'AntiOmega+': ('Omegabar+', 'OMEGABAR+'), 'Phi': ('phi', 'PHI')}
ENERGIES = {'7.7': 7.7, '11.5': 11.5, '19.6': 19.6, '27': 27.0, '39': 39.0}
CLASSES = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-60%', '60-80%']
# Table II <N_part>
NPART = {7.7: [337.4, 290.4, 226.2, 160.2, 109.9, 58.4, 20.2], 11.5: [338.2, 290.6, 226.0, 159.6, 110.0, 58.5, 20.1],
         19.6: [338.0, 289.2, 224.9, 158.1, 108.0, 57.7, 19.9], 27.0: [343.3, 299.3, 233.6, 165.5, 114.0, 61.2, 20.5],
         39.0: [341.7, 293.9, 229.8, 162.4, 111.4, 59.2, 20.0]}
MERGED = {'0-10%': ['0-5%', '5-10%'], '20-40%': ['20-30%', '30-40%'], '10-60%': ['10-20%', '20-30%', '30-40%', '40-60%'],
          '0-60%': ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-60%']}
OUT_S = 'data/particle_spectra/star_particle_spectra_strange_AuAu_7.7_39gev_ins1738953.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_strange_AuAu_7.7_39gev_ins1738953.yaml'
OUT_M = 'data/mean_pT/star_mean_pT_strange_AuAu_7.7_39gev_ins1738953.yaml'
FEED = {'Lambda': ' Corrected for the Xi and Xi0 weak-decay feed-down (Sigma0 decays count as prompt).',
        'Lambdabar': ' Corrected for the Xibar and Xibar0 (and Omegabar+) weak-decay feed-down (Sigma0bar decays '
                     'count as prompt).'}


def width(c):
    lo, hi = (float(x) for x in c.rstrip('%').split('-'))
    return hi - lo


def npart_of(energy, cls):
    single = dict(zip(CLASSES, NPART[energy]))
    if cls in single:
        return single[cls]
    parts = MERGED[cls]
    return sum(single[p] * width(p) for p in parts) / sum(width(p) for p in parts)


def classes_of(energy, npart):
    """Classes (single or merged) whose <N_part> match the table's."""
    cands = CLASSES + list(MERGED)
    out = []
    for n in npart:
        best = min(cands, key=lambda c: abs(npart_of(energy, c) - n))
        assert abs(npart_of(energy, best) - n) < 2.5, (energy, n, best, npart_of(energy, best))
        out.append(best)
    return out


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def base(otype, hf, energy, sp, label):
    species, prod = SPECIES[sp]
    return {'observable_type': otype, 'inspire_id': '1738953', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': f'{label} ({hf})',
            'collision_system': {'energy_gev': energy, 'reaction': f'AU AU --> {prod} X'},
            'particle_id': {'species': species}}


def main():
    subs = [s for s in yaml.load_all(open(os.path.join(RAW, 'submission.yaml')), Loader=yaml.CSafeLoader)
            if s and 'data_file' in s]
    spectra, yields, means = [], [], []
    for s in subs:
        hf, name = s['data_file'], s['name']
        m = re.fullmatch(r'(\S+) pT spectrum, Au\+Au ([\d.]+) GeV, (\d+-\d+%)', name)
        if m:
            sp, energy, cls = m.group(1), ENERGIES[m.group(2)], m.group(3)
            t = load(hf)
            dv = t['dependent_variables'][0]
            assert dv['header']['name'] == 'd^2N/(2*pi*Nevt*pT*dy*dpT)', hf
            assert dv['qualifiers'][0]['value'] == cls and cls in CLASSES, (hf, dv['qualifiers'])
            assert [e['label'] for e in dv['values'][0]['errors']] == ['stat', 'sys'], hf
            edges = [(float(v['low']), float(v['high'])) for v in t['independent_variables'][0]['values']]
            assert all(a[1] == b[0] for a, b in zip(edges[:-1], edges[1:])), hf
            d = base('particle_spectra', hf, energy, sp, f'{SPECIES[sp][0]} pT spectrum, Au+Au {energy:g} GeV, {cls}')
            d['centrality'] = {'classes': [cls], **REFMULT}
            d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5},
                                    'pT': {'min': edges[0][0], 'max': edges[-1][1],
                                           'bins': [edges[0][0]] + [e[1] for e in edges]}}
            d.update(observable='invariant_yield', value_type='invariant_yield')
            d['notes'] = {'value': f'{CITE}. d2N/(2 pi pT dpT dy) of {SPECIES[sp][0]}, |y| < 0.5, Au+Au {energy:g} GeV '
                                   f'{cls}.{FEED.get(SPECIES[sp][0], "")} Errors: stat and sys.'}
            spectra.append(d)
            continue
        m = re.fullmatch(r'(Table of )?(\S+) (dN/dy|<pT>), Au\+Au ([\d.]+) GeV', name)
        if not m:
            continue
        sp, what, energy = m.group(2), m.group(3), ENERGIES[m.group(4)]
        t = load(hf)
        dv = t['dependent_variables'][0]
        assert dv['qualifiers'][0]['value'] == sp, (hf, dv['qualifiers'])
        assert [e['label'] for e in dv['values'][0]['errors']] == ['stat', 'sys'], hf
        npart = [float(v['value']) for v in t['independent_variables'][0]['values']]
        classes = classes_of(energy, npart)
        species = SPECIES[sp][0]
        if what == 'dN/dy':
            assert dv['header']['name'] == 'dN/dy', hf
            d = base('particle_yields', hf, energy, sp, f'{species} dN/dy vs centrality, Au+Au {energy:g} GeV')
            d.update(observable='dN_dy', value_type='dN_dy')
            quantity = 'dN/dy'
        else:
            assert dv['header']['name'] == '<pT>' and dv['header'].get('units') == 'GeV/c', hf
            d = base('mean_pT', hf, energy, sp, f'{species} <pT> vs centrality, Au+Au {energy:g} GeV')
            d.update(observable='mean_pT', value_type='mean_pT')
            quantity = '<pT>'
        d['centrality'] = {'classes': classes, **REFMULT}
        d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
        d['notes'] = {'value': f'{CITE}, {s.get("location")}. {quantity} of {species}, |y| < 0.5, extrapolated to all pT '
                               f'(Levy / blast-wave fits), Au+Au {energy:g} GeV; classes {", ".join(classes)} (HEPData: '
                               f'their <N_part>, Table II).{FEED.get(species, "")} Errors: stat and sys.'}
        (yields if what == 'dN/dy' else means).append(d)
    for out, docs in ((OUT_S, spectra), (OUT_Y, yields), (OUT_M, means)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()

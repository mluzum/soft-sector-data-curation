#!/usr/bin/env python3
"""Curate STAR phi and Omega spectra at 7.7-39 GeV, INSPIRE 1378002.

L. Adamczyk et al. (STAR), "Probing parton dynamics of QCD matter with
Omega and phi production", PRC 93, 021903(R) (2016), arXiv:1506.07605.

From the paper:
- mid-rapidity (|y| < 0.5) invariant yields d2N/(2 pi N_evt pT dy dpT) of
  phi (Fig. 1) and Omega-, Omegabar+ (Fig. 2) in Au+Au at 7.7, 11.5,
  19.6, 27 and 39 GeV; centrality from the TPC multiplicity in |eta| < 0.5;
- phi: 0-10, 10-20, 20-30, 30-40, 40-60, 60-80%;
- Omega: 0-60% (7.7 GeV), 0-10 and 10-60% (11.5 GeV), 0-10, 10-20, 20-40,
  40-60% (19.6-39 GeV) and 60-80% (39 GeV);
- the figures scale classes by powers of 10 for display; the tables do not;
- errors: statistical and systematic.

HEPData labels need fixing (checked against the figures): the 30 phi tables
all read "7.7 GeV" and centralities "0-10, 0-20, ..., 0-80": they are five
energies (Tables 1-6 at 7.7, 7-12 at 11.5, 13-18 at 19.6, 19-24 at 27,
25-30 at 39 GeV; the pT reach grows with energy) of six classes each; the
Omega centralities "0-X" mean "previous edge to X".  phi pT bins are
encoded in the axis strings ("0.4461 + 0.5 - 0.4": mean 0.4461, bin
0.4-0.5); Omega tables give mean pT only (0.2 GeV/c windows here).

Records: one spectrum per table, |y| < 0.5.  Not curated: the
N(Omega)/2N(phi) ratios and k_s(pT^s) (no particle-ratio computation;
k_s also needs the paper's phi interpolation).

Run from the repository root:  python scripts/curate_star_omega_phi_bes.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1378002/HEPData-ins1378002-v1-yaml'
CITE = 'L. Adamczyk et al. (STAR), PRC 93, 021903(R) (2016), arXiv:1506.07605'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
PHI_ENERGIES = [7.7, 11.5, 19.6, 27.0, 39.0]
PHI_CLASSES = ['0-10%', '10-20%', '20-30%', '30-40%', '40-60%', '60-80%']
SPECIES = {'AU AU --> PHI X': 'phi', 'AU AU --> OMEGA- X': 'Omega-', 'AU AU --> OMEGABAR+ X': 'Omegabar+'}
OMEGA_WINDOW = 0.2


def phi_axis(values):
    pts, pairs = [], []
    for v in values:
        m = re.fullmatch(r'\s*([0-9.]+) \+ ([0-9.]+) - ([0-9.]+)\s*', str(v['value']))
        pts.append(float(m.group(1)))
        pairs.append([float(m.group(3)), float(m.group(2))])
    assert all(lo <= p <= hi for p, (lo, hi) in zip(pts, pairs))
    if all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs, pairs[1:])):
        return {'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [pairs[0][0]] + [p[1] for p in pairs]}
    return {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    prev_edge = {}                       # (species, energy) -> upper edge of the previous Omega class
    for k, x in enumerate(sub):
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = t['dependent_variables'][0]
        if dv['header']['name'] != 'Invariant Yield':
            continue                     # ratios and k_s
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        sp = SPECIES[q['RE']]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(len(dv['values']))), hf
        if sp == 'phi':
            n = int(re.fullmatch(r'Table(\d+)\.yaml', hf).group(1)) - 1
            e, cls = PHI_ENERGIES[n // 6], PHI_CLASSES[n % 6]
            assert q['CENTRALITY'] == f'0.0-{cls.split("-")[1].rstrip("%")}.0', (hf, q, cls)
            pt = phi_axis(t['independent_variables'][0]['values'])
            how = 'pT bins from the axis strings'
        else:
            e = float(q['SQRT(S)/NUCLEON'])
            hi = float(q['CENTRALITY'].split('-')[1])
            lo = prev_edge.get((sp, e), 0.0)
            prev_edge[(sp, e)] = hi
            cls = f'{lo:g}-{hi:g}%'
            pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
            pt = {'min': round(pts[0] - OMEGA_WINDOW / 2, 4), 'max': round(pts[-1] + OMEGA_WINDOW / 2, 4),
                  'points': pts, 'point_window': OMEGA_WINDOW}
            how = f'mean-pT points ({OMEGA_WINDOW} GeV/c windows)'
        docs.append({
            'observable_type': 'particle_spectra', 'inspire_id': '1378002', 'version': 1, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': f'{sp} spectrum, {cls} Au+Au at {e:g} GeV ({x.get("location", "")})',
            'collision_system': {'energy_gev': e, 'reaction': 'AU AU --> X'},
            'centrality': {'classes': [cls], **REFMULT},
            'kinematic_range': {'y': {'min': -0.5, 'max': 0.5}, 'pT': pt},
            'particle_id': {'species': sp},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield',
            'notes': {'value': (f'{CITE}, {x.get("location", "")}. d2N/(2 pi N_evt pT dy dpT) of {sp}, |y| < 0.5, '
                                f'{cls} at {e:g} GeV; {how}. Energy and class from the figure (the HEPData labels '
                                'read 7.7 GeV and 0-X% for these tables). Errors: statistical and systematic.')},
        })
    out = 'data/particle_spectra/star_particle_spectra_probing_parton_dynamics_of_qcd_AuAu_multigev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['particle_id']['species'],
                                                                     d['collision_system']['energy_gev'])
                                                                    for d in docs)))
    print({(d['particle_id']['species'], d['collision_system']['energy_gev']): None for d in docs} and
          sorted({(d['particle_id']['species'], d['collision_system']['energy_gev'], d['centrality']['classes'][0])
                  for d in docs if d['particle_id']['species'] != 'phi'}))


if __name__ == '__main__':
    main()

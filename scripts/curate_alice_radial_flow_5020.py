#!/usr/bin/env python3
"""Curate ALICE v0(pT) in Pb-Pb at 5.02 TeV, INSPIRE 2908602.

S. Acharya et al. (ALICE), "Long-range transverse momentum correlations and
radial flow in Pb-Pb collisions at the LHC", PRL 136 (2026) 032302,
arXiv:2504.04796.

From the paper:
- v0(pT) = (<f_A(pT) [pT]_B> - <f_A(pT)><[pT]_B>) / (<f_A(pT)> sigma_[pT]),
  sigma_[pT]^2 = <[pT]_A [pT]_B> - <[pT]_A><[pT]_B> (Eqs. 1-2); f_A(pT) the
  fraction of the particles of window A in the pT bin, [pT] the event's
  mean pT in a window -- the same quantity as ATLAS's v0(pT) =
  normalized covariance / v0;
- windows A and B in the barrel (|eta| < 0.8) separated by Delta eta = 0.4;
  tracks 0.2 < pT < 10 GeV/c (no separate range for [pT] is given);
- centrality from the V0 amplitude (V0M);
- errors: statistical and systematic.

Records (observable_type radial_flow): v0(pT) and v0(pT)/v0 of charged
particles in 10-20, 30-40, 60-70% (Figs. 1, 3): windows -0.8 < eta < -0.2
and 0.2 < eta < 0.8; [pT] and the n(pT) normalization over 0.2-10 GeV/c;
deviations from the class mean (the paper names no multiplicity binning);
the covariance is a <-> b symmetrized here (one-sided in the paper, the
same expectation for symmetric windows).

Identified pi, K, p (Fig. 2): f_A of an identified species is a fraction
"relative to the total number of particles" in window A -- the spectrum
species differs from the [pT] reference, which the computation does not
provide yet.  With --identified OUT those nine records are written to OUT
(a staging collection) with a staging_reason.

Run from the repository root:
    python scripts/curate_alice_radial_flow_5020.py [--identified OUT.yaml]
"""
import argparse
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins2908602/HEPData-ins2908602-v1-yaml'
CITE = 'S. Acharya et al. (ALICE), PRL 136 (2026) 032302, arXiv:2504.04796'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SPECIES = {'pions': 'pi+ + pi-', 'kaons': 'K+ + K-', 'protons': 'p + pbar'}
STAGING = ('identified v0(pT): the spectrum species (pi, K, p) differs from the charged [pT] reference, which the '
           'radial_flow computation does not provide yet; f_A normalization ("relative to the total number of '
           'particles" in window A) to be read as all charged particles')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--identified', help='write the identified-hadron records (staged) to this file')
    args = ap.parse_args()
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    public, staged = [], []
    for x in sub:
        hf = x['data_file']
        desc = ' '.join(x['description'].split())
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert len(t['dependent_variables']) == 1, hf
        dv = t['dependent_variables'][0]
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        h = dv['header']['name']
        quantity = 'v0_pT_over_v0' if h == '$v_{0}(p_\\mathrm{T})/v_{0}$' else 'v0_pT'
        assert quantity == 'v0_pT_over_v0' or h == '$v_{0}(p_\\mathrm{T})$', (hf, h)
        cls = re.sub(r'\$-\$', '-', q['centrality'])
        sp = SPECIES.get(q.get('particle'), 'charged')
        iv = t['independent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(rows[0], rows[-1] + 1)), hf
        vals = [iv['values'][j] for j in rows]
        edges = [float(vals[0]['low'])] + [float(v['high']) for v in vals]
        assert all(abs(float(a['high']) - float(b['low'])) < 1e-9 for a, b in zip(vals, vals[1:])), hf
        d = {'observable_type': 'radial_flow', 'inspire_id': '2908602', 'version': 1, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'{desc[:200]} | {h}',
             'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> X'},
             'centrality': {'classes': [cls], **V0M},
             'particle_id': {'species': sp},
             'quantity': quantity, 'particles': 'AB',
             'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.2}}},
                           'B': {'kinematic_range': {'eta': {'min': 0.2, 'max': 0.8}}}},
             'eta_gap': 0.4,
             'reference_pT': {'min': 0.2, 'max': 10.0},
             'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 10.0, 'bins': edges}},
             'value_type': h, 'dimensionality': 'differential_pT',
             'notes': {'value': (f'{CITE}. {"v0(pT)/v0" if quantity != "v0_pT" else "v0(pT)"} of {sp}, {cls}; '
                                 'windows -0.8 < eta < -0.2 and 0.2 < eta < 0.8 (Delta eta = 0.4), [pT] and the '
                                 'n(pT) normalization over 0.2 < pT < 10 GeV/c (the track selection; no separate '
                                 '[pT] range is stated); deviations from the class mean; covariance a <-> b '
                                 'symmetrized (one-sided in the paper). Errors: statistical and systematic.')}}
        if sp == 'charged':
            public.append(d)
        else:
            d['staging_reason'] = STAGING
            staged.append(d)
    out = 'data/radial_flow/alice_radial_flow_v0_pT_PbPb_5020gev_ins2908602.yaml'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter((d['quantity'], d['centrality']['classes'][0])
                                                                      for d in public)))
    if args.identified:
        with open(args.identified, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{args.identified}: {len(staged)} identified-hadron records (staged)')


if __name__ == '__main__':
    main()

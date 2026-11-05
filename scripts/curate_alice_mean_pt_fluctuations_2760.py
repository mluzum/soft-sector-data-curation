#!/usr/bin/env python3
"""Curate ALICE event-by-event mean pT fluctuations in Pb-Pb at 2.76 TeV, INSPIRE 1307102.

B. Abelev et al. (ALICE), "Event-by-event mean pT fluctuations in pp and
Pb-Pb collisions at the LHC", EPJC 74 (2014) 3077, arXiv:1407.5530.

From the paper (Sec. 2):
- charged tracks in |eta| < 0.8, 0.15 < pT < 2 GeV/c, not corrected for
  efficiency: sqrt(C_m)/M(pT)_m cancels it to first order, and the
  difference to the generator level is a systematic (0-4% in Pb-Pb);
- C_m = <Delta pT_i Delta pT_j> over the pairs of all events of class m
  (Eq. 2), deviations from M(pT)_m, the mean pT of all tracks of all
  events of the class (Eq. 3): the pipeline's pT_fluct estimator;
- Pb-Pb: centrality from the VZERO amplitude, 5% classes; within a class
  M(pT) is the class mean (the effect of its spread across the class is
  in the systematics, from 5% vs 10% classes); at least 10 accepted
  tracks per event; Table 6 gives the 16 classes 0-5 ... 75-80% by their
  <N_part> (382.8 ... 12.49, ALICE's Glauber values), peripheral first;
- errors: stat and sys.

Records (public, 1): Table 6, sqrt(C_m)/M(pT)_m vs class.  Staged
(--staged OUT): Table 5 (Pb-Pb in bins of the accepted multiplicity N_acc,
not percentile classes; M(pT) from Delta N_acc = 1 bins) and the pp
tables 1-4 (N_acc bins at 0.9, 2.76 and 7 TeV, and the inclusive value vs
sqrt(s)).  Not curated: Tables 7-8 (the same points over power-law fits).
The 8 old staged records filed every table as a conserved-charge cumulant.

Run from the repository root:
    python scripts/curate_alice_mean_pt_fluctuations_2760.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1307102/HEPData-ins1307102-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. Abelev et al. (ALICE), EPJC 74 (2014) 3077, arXiv:1407.5530'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
NACC = {'estimator': {'name': 'Nacc', 'observable': 'Nch', 'detector': 'TPC'},
        'eta': [{'min': -0.8, 'max': 0.8}], 'pT': {'min': 0.15, 'max': 2.0},
        'notes': 'accepted tracks N_acc in the correlation window (not efficiency corrected).'}
KR = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.15, 'max': 2.0}}
CLASSES = [f'{lo}-{lo + 5}%' for lo in range(75, -1, -5)]      # Table 6 rows, peripheral first
OUT = 'data/mean_pT_fluctuations/alice_mean_pT_fluctuations_PbPb_2760gev_ins1307102.yaml'
ERRORS = 'Errors: stat and sys (sys includes the generator-level difference of the uncorrected tracks).'


def load(n):
    return yaml.safe_load(open(os.path.join(RAW, f'Table{n}.yaml')))


def base(n, energy, reaction, label):
    return {'observable_type': 'mean_pT_fluctuations', 'inspire_id': '1307102', 'version': VERSION,
            'hepdata_file': f'Table{n}.yaml', 'dv_index': 0, 'hepdata_table_label': f'{label} (Table{n}.yaml)',
            'collision_system': {'energy_gev': energy, 'reaction': reaction}}


def body(d, cent):
    d.update({'centrality': cent, 'particle_id': {'species': 'charged'}, 'kinematic_range': dict(KR),
              'observable': 'sqrt_dpt_dpt_over_mean_pT', 'value_type': 'sqrt_<dpt1_dpt2>_over_<pT>',
              'method': 'two_particle'})
    return d


def check(t, energy, reaction):
    dv = t['dependent_variables'][0]
    q = {x['name']: str(x['value']) for x in dv['qualifiers']}
    assert q['ABS(ETARAP)'] == '< 0.8' and q['PT'] == '0.15-2.0', q
    assert float(q.get('SQRT(S)/NUCLEON', q.get('SQRT(S)'))) == energy and q['RE'] == reaction, q
    assert dv['header']['name'] == '$\\sqrt{C_m}/M(p_{\\rm T})_m$', dv['header']
    return dv


def public():
    t = load(6)
    check(t, 2760.0, 'Pb Pb --> CHARGED X')
    npart = [float(str(v['value']).split('+-')[0]) for v in t['independent_variables'][0]['values']]
    assert len(npart) == len(CLASSES) and npart == sorted(npart) and npart[-1] == 382.8, npart
    d = body(base(6, 2760.0, 'PB PB --> CHARGED X', 'sqrt(C_m)/M(pT)_m vs centrality, Pb-Pb 2.76 TeV'),
             {'classes': list(CLASSES), **V0M})
    d['notes'] = {'value': f'{CITE}, Fig. 7 right. sqrt(C_m)/M(pT)_m of charged particles, |eta| < 0.8, '
                           '0.15 < pT < 2 GeV/c, Pb-Pb 2.76 TeV, VZERO 5% classes 75-80 ... 0-5% (HEPData: their '
                           '<N_part>, peripheral first); deviations from the class mean pT, at least 10 accepted '
                           f'tracks per event. {ERRORS}'}
    return [d]


def staged():
    docs = []
    t = load(5)
    check(t, 2760.0, 'Pb Pb --> CHARGED X')
    d = body(base(5, 2760.0, 'PB PB --> CHARGED X', 'sqrt(C_m)/M(pT)_m vs <dNch/deta>, Pb-Pb 2.76 TeV'), dict(NACC))
    d['staging_reason'] = ('bins of the accepted multiplicity N_acc (Delta N_acc = 10-100), not percentile '
                           'classes; <dNch/deta> assigned by interpolation')
    d['notes'] = {'value': f'{CITE}, Fig. 5. Pb-Pb in N_acc bins, M(pT)_m in Delta N_acc = 1 bins below 1000. '
                           f'{ERRORS}'}
    docs.append(d)
    for n, energy in ((1, 900.0), (2, 2760.0), (3, 7000.0)):
        t = load(n)
        check(t, energy, 'P P --> CHARGED X')
        d = body(base(n, energy, 'P P --> CHARGED X', f'sqrt(C_m)/M(pT)_m vs <dNch/deta>, pp {energy / 1000:g} TeV'),
                 dict(NACC))
        d['staging_reason'] = 'p+p in N_acc bins (Delta N_acc = 1), no centrality classes'
        d['notes'] = {'value': f'{CITE}, Fig. 1. pp in N_acc bins, <dNch/deta> unfolded. {ERRORS}'}
        docs.append(d)
    t = load(4)
    d = body(base(4, None, 'P P --> CHARGED X', 'inclusive sqrt(C)/M(pT) vs sqrt(s), pp'), {})
    d['collision_system'] = {'reaction': 'P P --> CHARGED X'}
    d['staging_reason'] = 'p+p minimum bias, one point per energy (0.9, 2.76, 7 TeV)'
    d['notes'] = {'value': f'{CITE}, Fig. 2. Inclusive (no multiplicity classes). {ERRORS}'}
    docs.append(d)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged', help='write the staged records to this file')
    a = ap.parse_args()
    docs = public()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')
    if a.staged:
        st = staged()
        with open(a.staged, 'w') as fh:
            yaml.safe_dump_all(st, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{a.staged}: {len(st)} staged records')


if __name__ == '__main__':
    main()

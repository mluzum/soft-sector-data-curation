#!/usr/bin/env python3
"""Curate PHENIX forward-rapidity charged-hadron v2 in Au+Au at 200 GeV, INSPIRE 2830400.

N. J. Abdulameer et al. (PHENIX), "Measurements at forward rapidity of
elliptic flow of charged hadrons and open-heavy-flavor muons in Au+Au
collisions at sqrt(sNN) = 200 GeV", PRC 112, 034902 (2025),
arXiv:2409.12715.

From the paper (Secs. II-IV):
- charged hadrons and heavy-flavor muons in the muon arms, 1.2 < |eta| <
  2.0, 1 < pT < 5 GeV/c; centrality 0-70% (BBC);
- the second-order event plane from the FVTX arm opposite to the muon arm
  measuring the particle (FVTX 1.2 < |eta| < 2.2), resolution from three
  subevents (both FVTX arms and the central arms);
- errors: stat and sys.

Records (public): charged-hadron v2(pT), 0-70% (Fig. 10b), as the two-
subevent correlation (particles AB): the particle in one arm, the
reference (charged, all pT) in the other; the reference window is taken
as the particle's, 1.2 < |eta| < 2.0 (the FVTX reaches 2.2).  Moved to
the ancillary collection: heavy-flavor muon and electron v2 (Fig. 11b, c:
out of scope).  Not curated: Fig. 11a (repeats Fig. 10b), the hadron- and
muon-rich sample v2 (Fig. 7: detector-level samples), the sample fractions
and simulations (Figs. 8, 9).  The 5 old records put everything at |eta| <
0.35 with no reference.

Run from the repository root:
    python scripts/curate_phenix_forward_v2_200.py [--ancillary OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins2830400/HEPData-ins2830400-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'N. J. Abdulameer et al. (PHENIX), PRC 112, 034902 (2025), arXiv:2409.12715'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
REF_PT = {'min': 0.0, 'max': 100.0}
SUB = {'particles': 'AB', 'eta_gap': 2.4,
       'subevents': {'A': {'kinematic_range': {'eta': {'min': -2.0, 'max': -1.2}, 'pT': dict(REF_PT)}},
                     'B': {'kinematic_range': {'eta': {'min': 1.2, 'max': 2.0}, 'pT': dict(REF_PT)}}}}
OLD = 'data/flow_vn/phenix_flow_vn_measurements_AuAu_200gev.yaml'


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ancillary', help='write the out-of-scope records here')
    anc_out = ap.parse_args().ancillary
    public, ancillary = [], []
    for hf, sp, what in (('fig10.2.yaml', 'charged', 'charged hadrons'), ('fig11.2.yaml', 'HF muons', 'HF -> mu'),
                         ('fig11.3.yaml', 'HF electrons', 'HF -> e')):
        t = load(hf)
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
        if 'low' in iv['values'][0]:
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        else:
            pairs = [[float(iv['values'][j]['value'])] * 2 for j in rows]
        d = {'observable_type': 'flow_vn' if sp == 'charged' else 'heavy_flavor', 'inspire_id': '2830400',
             'version': VERSION, 'hepdata_file': hf, 'dv_index': 0, 'hepdata_table_label': f'v2(pT) of {what}, 0-70% ({hf})',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
             'centrality': {'classes': ['0-70%'], **BBC}, 'particle_id': {'species': sp},
             'harmonic': 2, 'value_type': dv['header']['name'], 'dimensionality': 'differential_pT'}
        if sp == 'charged':
            d.update({'kinematic_range': {'eta': {'min': -2.0, 'max': 2.0},
                                          'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                      'method': 'event_plane', **SUB,
                      'notes': {'value': f'{CITE}, Fig. 10b. v2(pT) of charged hadrons at 1.2 < |eta| < 2.0, 0-70%, '
                                         'relative to the FVTX event plane of the opposite arm; computed as the '
                                         'two-subevent correlation (reference window 1.2 < |eta| < 2.0, all pT). '
                                         'Errors: stat and sys.'}})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
        else:
            d.update({'notes': {'value': f'{CITE}. v2(pT) of {what}, 0-70%.'},
                      'staging_reason': 'open heavy flavor (muons, electrons from charm and bottom decays): out of '
                                        'the soft-sector scope'})
            ancillary.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '2830400' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_forward_v2_AuAu_200gev_ins2830400.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if anc_out:
        with open(anc_out, 'w') as fh:
            yaml.safe_dump_all(ancillary, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{anc_out}: {len(ancillary)} records')


if __name__ == '__main__':
    main()

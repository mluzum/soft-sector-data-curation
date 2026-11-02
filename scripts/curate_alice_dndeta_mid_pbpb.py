#!/usr/bin/env python3
"""Curate ALICE dNch/deta at mid-rapidity vs centrality in Pb-Pb at 2.76 TeV (INSPIRE 880049) or 5.02 TeV (1410589).

- 880049: K. Aamodt et al. (ALICE), "Centrality dependence of the charged-
  particle multiplicity density at mid-rapidity in Pb-Pb collisions at
  sqrt(sNN) = 2.76 TeV", PRL 106, 032301 (2011), arXiv:1012.1657.
- 1410589: J. Adam et al. (ALICE), "Centrality dependence of the charged-
  particle multiplicity density at midrapidity in Pb-Pb collisions at
  sqrt(sNN) = 5.02 TeV", PRL 116, 222302 (2016), arXiv:1512.06104.

From the papers: dNch/deta of primary charged particles at |eta| < 0.5
(SPD tracklets, all pT); centrality from the V0 amplitude (V0M).

Records (public): dNch/deta at |eta| < 0.5 vs centrality (880049 Table 1:
nine classes; 1410589 Table 2: eleven classes).  Not curated: <N_part>
and dNch/deta per participant pair (Glauber).  The old records had pT
0.15-50 GeV/c (1410589: |eta| < 0.8) and value_type dN_dy.

Run from the repository root:
    python scripts/curate_alice_dndeta_mid_pbpb.py --paper 880049|1410589
"""
import argparse
import glob
import os

import yaml

V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
PAPERS = {
    '880049': {'cite': 'K. Aamodt et al. (ALICE), PRL 106, 032301 (2011), arXiv:1012.1657', 'energy': 2760.0,
               'table': 'Table1.yaml',
               'old': 'data/particle_yields/alice_particle_yields_centrality_dependence_of_the_charged-particle_PbPb_2760gev.yaml',
               'out': 'data/particle_yields/alice_particle_yields_dNch_deta_mid_PbPb_2760gev_ins880049.yaml'},
    '1410589': {'cite': 'J. Adam et al. (ALICE), PRL 116, 222302 (2016), arXiv:1512.06104', 'energy': 5020.0,
                'table': 'Table2.yaml',
                'old': 'data/particle_yields/alice_particle_yields_centrality_dependence_of_the_charged-particle_PbPb_5020gev.yaml',
                'out': 'data/particle_yields/alice_particle_yields_dNch_deta_mid_PbPb_5020gev_ins1410589.yaml'},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--paper', required=True, choices=sorted(PAPERS))
    iid = ap.parse_args().paper
    P = PAPERS[iid]
    raw = glob.glob(f'hepdata_raw/records/ins{iid}/HEPData-ins{iid}-v*-yaml')[0]
    version = int(raw.rsplit('-v', 1)[1].split('-')[0])
    t = yaml.safe_load(open(os.path.join(raw, P['table'])))
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
    classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
    labels = sorted({e.get('label', '') for v in dv['values'] for e in v.get('errors') or []})
    d = {'observable_type': 'particle_yields', 'inspire_id': iid, 'version': version, 'hepdata_file': P['table'],
         'dv_index': 0, 'hepdata_table_label': f"dNch/deta at |eta| < 0.5 vs centrality ({P['table']})",
         'collision_system': {'energy_gev': P['energy'], 'reaction': 'PB PB --> CHARGED X'},
         'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}},
         'observable': 'dNch_deta', 'value_type': 'dNch_deta',
         'notes': {'value': f"{P['cite']}. dNch/deta of primary charged particles at |eta| < 0.5 vs centrality (V0M), "
                            f"all pT. Errors: {'stat and sys' if 'stat' in labels else 'total'}."}}
    if rows != list(range(len(iv['values']))):
        d['table_rows'] = rows
    if os.path.exists(P['old']):
        assert all(str(x['inspire_id']) == iid for x in yaml.safe_load_all(open(P['old'])) if x), P['old']
        os.remove(P['old'])
    with open(P['out'], 'w') as fh:
        yaml.safe_dump_all([d], fh, sort_keys=False, default_flow_style=False, width=120)
    print(f"{P['out']}: 1 record, {len(classes)} classes")


if __name__ == '__main__':
    main()

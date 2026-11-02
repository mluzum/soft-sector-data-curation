#!/usr/bin/env python3
"""Curate CMS charged-hadron dN/deta in Xe-Xe 5.44, Pb-Pb 5.36, O-O 5.36 TeV (INSPIRE 1719334, 2825521, 3166912).

- 1719334: A. M. Sirunyan et al. (CMS), "Charged-particle pseudorapidity
  distributions in xenon-xenon collisions at sqrt(sNN) = 5.44 TeV", PLB
  799 (2019) 135049, arXiv:1902.03603.
- 2825521: A. Hayrapetyan et al. (CMS), "Measurement of the charged-
  hadron pseudorapidity distributions in lead-lead collisions at sqrt(sNN)
  = 5.36 TeV", PLB 861 (2025) 139279, arXiv:2409.00838.
- 3166912: CMS Collaboration, "Centrality dependence of charged-hadron
  pseudorapidity distributions in oxygen-oxygen collisions at sqrt(sNN) =
  5.36 TeV", arXiv:2606.02285 (2026).

From the papers:
- dNch/deta from pixel tracklets (all pT, extrapolated below the
  tracklet threshold), averaged and symmetrised in eta;
- centrality from the total transverse energy in the HF calorimeters
  (2.9 < |eta| < 5.2);
- errors: total (Xe-Xe), stat and sys (Pb-Pb dN/deta(eta)).

Records (public): dNch/deta(eta) per class (Xe-Xe figure2, Pb-Pb
Table_1: 0-80, 0-5, 50-55%; O-O Table_1 0-100% and Table_2 eight classes;
differential_eta, computed on 3D containers) and dNch/deta at |eta| < 0.5
vs centrality (figure4l / Table_2a / Table_3a).  Not curated: the same
values divided by 2A or N_part (Glauber), and the Xe-Xe charged-hadron
rapidity density dN/dy (figure3: a mass assumption for unidentified
hadrons).  The old records called dN/deta dN/dy over |eta| < 2.4.

Run from the repository root:
    python scripts/curate_cms_dndeta_xexe_pbpb.py --paper 1719334|2825521|3166912
"""
import argparse
import glob
import os
import re

import yaml

HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
PAPERS = {
    '1719334': {'cite': 'A. M. Sirunyan et al. (CMS), PLB 799 (2019) 135049, arXiv:1902.03603',
                'beams': 'XE XE', 'energy': 5440.0, 'eta_tables': ['figure2.yaml'], 'cent_table': 'figure4l.yaml',
                'old': 'data/particle_yields/cms_particle_yields_pseudorapidity_distributions_of_charged_hadrons_XeXe_5440gev.yaml',
                'out': 'data/particle_yields/cms_particle_yields_dNch_deta_XeXe_5440gev_ins1719334.yaml'},
    '2825521': {'cite': 'A. Hayrapetyan et al. (CMS), PLB 861 (2025) 139279, arXiv:2409.00838',
                'beams': 'PB PB', 'energy': 5360.0, 'eta_tables': ['Table_1.yaml'], 'cent_table': 'Table_2a.yaml',
                'old': 'data/particle_yields/cms_particle_yields_pseudorapidity_distributions_of_charged_hadrons_PbPb_5360gev.yaml',
                'out': 'data/particle_yields/cms_particle_yields_dNch_deta_PbPb_5360gev_ins2825521.yaml'},
    '3166912': {'cite': 'CMS Collaboration, arXiv:2606.02285 (2026)',
                'beams': 'O O', 'energy': 5360.0, 'eta_tables': ['Table_1.yaml', 'Table_2.yaml'], 'cent_table': 'Table_3a.yaml',
                'old': 'data/particle_yields/cms_particle_yields_centrality_dependence_of_charged-hadron_pseudorapidity_5360gev.yaml',
                'out': 'data/particle_yields/cms_particle_yields_dNch_deta_OO_5360gev_ins3166912.yaml'},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--paper', required=True, choices=sorted(PAPERS))
    iid = ap.parse_args().paper
    P = PAPERS[iid]
    raw = glob.glob(f'hepdata_raw/records/ins{iid}/HEPData-ins{iid}-v*-yaml')[0]
    version = int(raw.rsplit('-v', 1)[1].split('-')[0])
    system = {'energy_gev': P['energy'], 'reaction': f"{P['beams']} --> CHARGED X"}
    errors = lambda dv: ('total' if all(e.get('label') == 'total' for v in dv['values'] for e in v.get('errors') or [])
                         else 'stat and sys')
    public = []
    for eta_table in P['eta_tables']:
        public += eta_records(raw, version, P, eta_table, system, errors)
    public.append(cent_record(raw, version, iid, P, system, errors))
    if os.path.exists(P['old']):
        assert all(str(x['inspire_id']) == iid for x in yaml.safe_load_all(open(P['old'])) if x), P['old']
        os.remove(P['old'])
    with open(P['out'], 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f"{P['out']}: {len(public)} records")


def eta_records(raw, version, P, eta_table, system, errors):
    """dNch/deta(eta) records, one per class column of a table."""
    iid = P['out'].rsplit('ins', 1)[1].split('.')[0]
    out = []
    t = yaml.safe_load(open(os.path.join(raw, eta_table)))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        lo, hi = (float(x) for x in re.findall(r'[\d.]+', q['CENTRALITY']))
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'particle_yields', 'inspire_id': iid, 'version': version, 'hepdata_file': eta_table,
             'dv_index': i, 'hepdata_table_label': f"dNch/deta(eta), {cls} ({eta_table})",
             'collision_system': dict(system), 'centrality': {'classes': [cls], **HF},
             'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs},
                                 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'dimensionality': 'differential_eta',
             'notes': {'value': f"{P['cite']}. dNch/deta of charged hadrons vs eta, {cls}, pixel tracklets, averaged "
                                f'and symmetrised; HF centrality. Errors: {errors(dv)}.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        out.append(d)
    return out


def cent_record(raw, version, iid, P, system, errors):
    """dNch/deta at |eta| < 0.5 vs centrality."""
    t = yaml.safe_load(open(os.path.join(raw, P['cent_table'])))
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
    classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
    d = {'observable_type': 'particle_yields', 'inspire_id': iid, 'version': version, 'hepdata_file': P['cent_table'],
         'dv_index': 0, 'hepdata_table_label': f"dNch/deta at |eta| < 0.5 vs centrality ({P['cent_table']})",
         'collision_system': dict(system), 'centrality': {'classes': classes, **HF}, 'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}},
         'observable': 'dNch_deta', 'value_type': 'dNch_deta',
         'notes': {'value': f"{P['cite']}. dNch/deta of charged hadrons at |eta| < 0.5 vs centrality, HF "
                            f'centrality. Errors: {errors(dv)}.'}}
    if rows != list(range(len(iv['values']))):
        d['table_rows'] = rows
    return d


if __name__ == '__main__':
    main()

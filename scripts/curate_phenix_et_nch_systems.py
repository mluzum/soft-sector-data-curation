#!/usr/bin/env python3
"""Curate PHENIX dET/deta and dNch/deta at midrapidity across systems, INSPIRE 1394433.

A. Adare et al. (PHENIX), "Transverse energy production and charged-particle
multiplicity at midrapidity in various systems from sqrt(sNN) = 7.7 to 200
GeV", PRC 93, 024901 (2016), arXiv:1509.06727.

From the paper:
- dET/deta (EMCal) and dNch/deta (drift and pad chambers) at midrapidity
  (central arms, |eta| < 0.35), per unit pseudorapidity;
- centrality from the BBC charge (3.0 < |eta| < 3.9); the Au+Au 200 and 130
  GeV data sets used the BBC with the ZDC (here the BBC as the estimator);
- systems: Au+Au 7.7-200 GeV, Cu+Cu 62.4 and 200, Cu+Au, d+Au, 3He+Au 200,
  U+U 193 GeV; each table also lists N_part, N_q and the ratios (Glauber).

Records (public): dET/deta and dNch/deta vs centrality for each system (the
dE_T/deta and dN_ch/deta columns).  Asymmetric systems: lab frame (equal
beam energies per nucleon), the lighter nucleus toward +eta.  Not curated:
N_part, N_q and the ratios to them (Glauber).  The 12 old records read the
first column of each ET table, N_part, as dET/deta.

Run from the repository root:  python scripts/curate_phenix_et_nch_systems.py
"""
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1394433/HEPData-ins1394433-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 93, 024901 (2016), arXiv:1509.06727'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
# table stem -> (reaction beams, energy, forward beam for asymmetric systems)
SYSTEMS = {'200': ('AU AU', 200.0, None), '130': ('AU AU', 130.0, None), '62': ('AU AU', 62.4, None),
           '39': ('AU AU', 39.0, None), '27': ('AU AU', 27.0, None), '19.6': ('AU AU', 19.6, None),
           '14.5': ('AU AU', 14.5, None), '7.7': ('AU AU', 7.7, None), 'CuCu200': ('CU CU', 200.0, None),
           'CuCu62': ('CU CU', 62.4, None), 'CuAu200': ('CU AU', 200.0, 'Cu'), 'dAu200': ('D AU', 200.0, 'd'),
           'HeAu200': ('HE3 AU', 200.0, '3He'), 'UU193': ('U U', 193.0, None)}
OLD_GLOB = 'data/transverse_energy/phenix_transverse_energy_*'


def tables():
    for f in sorted(os.listdir(RAW)):
        m = re.fullmatch(r'(?:ET|Nch)([\d.]+)\.yaml|([A-Za-z]+\d+)(ET|Nch)\.yaml', f)
        if not m:
            continue
        stem = m.group(1) or m.group(2)
        kind = 'ET' if 'ET' in f else 'Nch'
        yield f, SYSTEMS[stem], kind


def main():
    out = collections.defaultdict(list)
    for hf, (beams, energy, fwd), kind in tables():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        name = '$dE_{T}/d\\eta$' if kind == 'ET' else '$dN_{ch}/d\\eta$'
        i = [dv['header']['name'] for dv in t['dependent_variables']].index(name)
        dv = t['dependent_variables'][i]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
        cs = {'energy_gev': energy, 'reaction': f'{beams} --> CHARGED X'}
        if fwd:
            cs.update({'frame': 'lab', 'rapidity_shift': 0.0, 'forward_beam': fwd})
        otype = 'transverse_energy' if kind == 'ET' else 'particle_yields'
        vt = 'dET_deta' if kind == 'ET' else 'dNch_deta'
        d = {'observable_type': otype, 'inspire_id': '1394433', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
             'hepdata_table_label': f"{name.strip('$')} vs centrality, {beams.title()} {energy:g} GeV ({hf})",
             'collision_system': cs, 'centrality': {'classes': classes, **BBC},
             'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': vt, 'value_type': vt,
             'notes': {'value': f"{CITE}. {'dET/deta' if kind == 'ET' else 'dNch/deta'} at midrapidity per unit eta, "
                                f"{beams.title()} {energy:g} GeV, BBC centrality"
                                + (' (with the ZDC in the paper)' if beams == 'AU AU' and energy in (200.0, 130.0) else '')
                                + ('; lab frame, the lighter nucleus toward +eta' if fwd else '') + '. Errors: sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        out[otype].append(d)
    for f in glob.glob(OLD_GLOB):
        if all(str(x.get('inspire_id')) == '1394433' for x in yaml.safe_load_all(open(f)) if x):
            os.remove(f)
    for otype, docs in out.items():
        path = f'data/{otype}/phenix_{otype}_midrapidity_systems_ins1394433.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records')


if __name__ == '__main__':
    main()

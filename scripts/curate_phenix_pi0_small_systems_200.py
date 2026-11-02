#!/usr/bin/env python3
"""Curate PHENIX pi0 yields in p+Al, p+Au, d+Au and 3He+Au at 200 GeV, INSPIRE 1965617.

U. A. Acharya et al. (PHENIX), "Systematic study of nuclear effects in
p+Al, p+Au, d+Au, and 3He+Au collisions at sqrt(sNN) = 200 GeV using pi0
production", PRC 105, 064902 (2022), arXiv:2111.05756.

From the paper (Sec. II):
- pi0 -> gamma gamma in the EMCal, |eta| < 0.35;
- the projectile (p, d, 3He) travels north (+eta), the Au or Al nucleus
  south; centrality from the charge in the south BBC (-3.9 < eta < -3.0):
  0-20, 20-40, 40-60% and the rest of the minimum-bias sample (60-72%
  p+Al, 60-84% p+Au, 60-88% d+Au and 3He+Au), percentages of all inelastic
  collisions; 0-5% in some systems; 0-100% the total inelastic yield;
- errors: stat and sys.

Records (public): pi0 invariant yields per system and class (Fig. 6),
|y| < 0.35 used (y ~ eta above 1 GeV/c, as for the other PHENIX pi0
records); lab frame, the light nucleus toward +eta; pi0 by isospin, (pi+ +
pi-)/2, unless the container has a pi0 axis.  Staged (--staged OUT): the
p+p reference cross section (Fig. 5, p+p).  Not curated: R_xA, R_CP,
their averages and scaling exponents (Figs. 7-13: nuclear modification).
The 2 old records read the R_xA averages as dN/dy in 1% classes.

Run from the repository root:
    python scripts/curate_phenix_pi0_small_systems_200.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1965617/HEPData-ins1965617-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'U. A. Acharya et al. (PHENIX), PRC 105, 064902 (2022), arXiv:2111.05756'
BBCS = {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'}, 'eta': [{'min': -3.9, 'max': -3.0}],
        'notes': 'Charge in the south BBC (the Au/Al-going side).'}
SYSTEMS = {'p+Al': ('P AL', 'p'), 'p+Au': ('P AU', 'p'), 'd+Au': ('D AU', 'd'), '^3He+Au': ('HE3 AU', '3He')}
OLD = 'data/particle_yields/phenix_particle_yields_nuclear_effects_AuAu_200gev.yaml'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    t = yaml.safe_load(open(os.path.join(RAW, 'fig6.yaml')))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        lab = str(dv['qualifiers'][0]['value']).replace('$', '').replace('\\', '')
        m = re.fullmatch(r'(\S+)\s+(\d+)%?-(\d+)%', lab)
        sysname, lo, hi = m.group(1), m.group(2), m.group(3)
        beams, light = SYSTEMS[sysname]
        cls = f'{lo}-{hi}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', '--', 'none')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'particle_spectra', 'inspire_id': '1965617', 'version': VERSION, 'hepdata_file': 'fig6.yaml',
             'dv_index': i, 'hepdata_table_label': f'pi0 invariant yield, {sysname} {cls} (fig6.yaml)',
             'collision_system': {'energy_gev': 200.0, 'reaction': f'{beams} --> PI0 X', 'frame': 'lab',
                                  'rapidity_shift': 0.0, 'forward_beam': light},
             'centrality': {'classes': [cls], **BBCS}, 'particle_id': {'species': 'pi0'},
             'kinematic_range': {'y': {'min': -0.35, 'max': 0.35},
                                 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}, Fig. 6. pi0 invariant yield, {sysname.replace("^", "")} {cls} (south BBC '
                                'charge), EMCal |eta| < 0.35 (|y| < 0.35 used); lab frame, the light nucleus toward '
                                '+eta; pi0 computed by isospin. Errors: stat and sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        d['hepdata_corrections'] = {'pT': "HEPData qualifier '$p_T$' holds the system and class, not a pT range"}
        public.append(d)
    t = yaml.safe_load(open(os.path.join(RAW, 'fig5.yaml')))
    dv = t['dependent_variables'][0]
    staged.append({'observable_type': 'particle_spectra', 'inspire_id': '1965617', 'version': VERSION,
                   'hepdata_file': 'fig5.yaml', 'dv_index': 0, 'hepdata_table_label': 'pi0 cross section, p+p (fig5.yaml)',
                   'collision_system': {'energy_gev': 200.0, 'reaction': 'P P --> PI0 X'},
                   'particle_id': {'species': 'pi0'}, 'kinematic_range': {'y': {'min': -0.35, 'max': 0.35}, 'pT': {'min': 0.0, 'max': 100.0}},
                   'observable': 'invariant_cross_section', 'value_type': str(dv['header']['name']),
                   'notes': {'value': f'{CITE}, Fig. 5. pi0 invariant cross section in p+p.'},
                   'staging_reason': 'p+p cross section: no centrality class'})
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1965617' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/particle_spectra/phenix_particle_spectra_pi0_small_systems_200gev_ins1965617.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [(d['collision_system']['reaction'][:6], d['centrality']['classes'][0]) for d in public])
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate STAR directed flow of Lambda, Lambdabar, K+-, K0_S and phi in Au+Au at 7.7-200 GeV, INSPIRE 1618747.

L. Adamczyk et al. (STAR), "Beam-energy dependence of directed flow of
Lambda, Lambdabar, K+-, K0s and phi in Au+Au collisions", PRL 120 (2018)
062301, arXiv:1708.07132.

From the paper:
- rapidity-odd v1(y) relative to the first-order event plane of the BBCs
  (inner tiles 3.3 < |eta| < 5.0) at 7.7-39 GeV and of the ZDC-SMD
  (spectator neutrons, |eta| > 6.3) at 62.4 and 200 GeV;
- centrality from the charged multiplicity in |eta| < 0.5;
- TPC tracks (|eta| < 1.0), pT > 0.2 GeV/c; K+- at pT > 0.2 GeV/c and
  p < 1.6 GeV/c (dE/dx and TOF); Lambda, Lambdabar, K0_S at 0.2 < pT <
  5.0 GeV/c and phi at 0.15 < pT < 10.0 GeV/c from V0 topology and the
  invariant mass (their daughters in the TPC: no cut on the parent's eta
  here); feed-down to Lambda not discussed (inclusive);
- sign as in STAR 1277069 (nucleons positive near beam rapidity): anchored
  on pions here, v1 < 0 at 0 < y < 1 (scripts/curate_star_v1_ppi_bes.py);
- errors: statistical, and systematic for K+-, K0_S, Lambda, Lambdabar in
  10-40% at 7.7-39 and 200 GeV.

Records (public, 7.7-39 GeV, BBC plane, a charged odd reference in the BBC
acceptance, scalar product): v1(y) of the six species in 10-40% (Fig. 1)
and 40-80% (supplementary).  y bins contiguous from y = 0, centred on the
tabulated points (phi: 0.1, 0.3, ... edges at 11.5-62.4 GeV).  Staged
(--staged OUT): 62.4 and 200 GeV (ZDC-SMD spectator plane); the classes
HEPData labels "5%-10%" (supplementary only; STAR's companion analysis
1277069 uses 0-10%, unverified); the dv1/dy slopes (Figs. 2, 3: fits to
v1(y), net-particle and sum-rule combinations, not computed).

Run from the repository root:
  python scripts/curate_star_v1_strange_bes.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.curate_star_v1_ppi_bes import REFMULT, TPC, record, y_bins  # noqa: E402

RAW = glob.glob('hepdata_raw/records/ins1618747/HEPData-ins1618747-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'L. Adamczyk et al. (STAR), PRL 120 (2018) 062301, arXiv:1708.07132'
OUT = 'data/flow_vn/star_flow_vn_v1_strange_AuAu_7.7_39gev_ins1618747.yaml'
V0 = 'V0 topology and invariant mass (daughters in the TPC)'
# Fig1.<k>: species, product, kinematic_range (without y), cut note
SPECIES = {
    0: ('K+', 'K+', {**TPC, 'pT': {'min': 0.2, 'max': 1.6}, 'p': {'max': 1.6}},
        'TPC tracks |eta| < 1, pT > 0.2 GeV/c, p < 1.6 GeV/c'),
    1: ('K-', 'K-', {**TPC, 'pT': {'min': 0.2, 'max': 1.6}, 'p': {'max': 1.6}},
        'TPC tracks |eta| < 1, pT > 0.2 GeV/c, p < 1.6 GeV/c'),
    2: ('K0_S', 'K0S', {'pT': {'min': 0.2, 'max': 5.0}}, f'0.2 < pT < 5.0 GeV/c, {V0}'),
    3: ('Lambda', 'LAMBDA', {'pT': {'min': 0.2, 'max': 5.0}},
        f'0.2 < pT < 5.0 GeV/c, {V0}; feed-down not discussed (inclusive)'),
    4: ('Lambdabar', 'LAMBDABAR', {'pT': {'min': 0.2, 'max': 5.0}},
        f'0.2 < pT < 5.0 GeV/c, {V0}; feed-down not discussed (inclusive)'),
    5: ('phi', 'PHI', {'pT': {'min': 0.15, 'max': 10.0}}, '0.15 < pT < 10.0 GeV/c, K+ K- invariant mass')}
CLASS = {'5%–10%': '5-10%', '10%–40%': '10-40%', '40%–80%': '40-80%'}
UNVERIFIED = ('centrality class: HEPData labels it 5-10% (supplementary figure only; STAR\'s companion analysis '
              '1277069 uses 0-10%); unverified')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        m = re.fullmatch(r'Fig1\.(\d)\.[\d.]+_formatted\.yaml', hf)
        if not m:
            continue
        species, prod, kin, cut = SPECIES[int(m.group(1))]
        mc = re.search(r'in (\d+%–\d+%) central', desc)
        cls = CLASS[mc.group(1)]
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert re.fullmatch(r'rapidity \$y\$( +sysErr)?', t['independent_variables'][0]['header']['name']), hf
        centres = [float(v['value']) for v in t['independent_variables'][0]['values']]
        edges = y_bins(centres)
        for i, dv in enumerate(t['dependent_variables']):
            energy = float(re.fullmatch(r'\$v_1\$ at ([\d.]+) GeV', dv['header']['name']).group(1))
            # (energies from the column headers: HEPData's description of Fig1.1.1.0 lists 200 GeV, which
            # is in Fig1.1.1.1, described as 62.4 GeV only)
            assert f'{energy:g}' in desc or (hf, energy) == ('Fig1.1.1.1_formatted.yaml', 200.0), (hf, energy)
            labels = {tuple(e['label'] for e in v['errors']) for v in dv['values']}
            assert labels in ({('stat',)}, {('stat', 'sys')}), (hf, i, labels)
            errors = 'statistical and systematic' if ('stat', 'sys') in labels else 'statistical'

            label = (f"{'Fig. 1' if cls == '10-40%' else 'Supplementary'}: v1(y) of {species}, Au+Au {energy:g} GeV, "
                     f"{cls} ({s['name']}, {hf} column {i})")
            d = record('1618747', VERSION, CITE, hf, i, label, energy, cls, species, kin, prod, cut, edges, '',
                       errors)
            if cls == '5-10%' and 'staging_reason' not in d:
                d['staging_reason'] = UNVERIFIED
            (staged if 'staging_reason' in d else public).append(d)
    for s in sub:
        hf = s['data_file']
        if not hf.startswith(('Fig2', 'Fig3')):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            staged.append({'observable_type': 'flow_vn', 'inspire_id': '1618747', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{s['name']}: {name} vs beam energy, 10-40% ({hf} column {i})",
                           'collision_system': {'reaction': 'AU AU --> X'},
                           'centrality': {'classes': ['10-40%'], **REFMULT}, 'harmonic': 1,
                           'value_type': 'dv1/dy', 'dimensionality': 'integrated',
                           'notes': {'value': f'{CITE}. {name}: slope of a linear fit to v1(y) (|y| < 0.6 for phi, '
                                              '< 0.8 otherwise), net-particle slopes and constituent-quark '
                                              'combinations, one value per beam energy. Errors: stat. and sys.'},
                           'staging_reason': 'slope of a fit to v1(y) vs beam energy (and combinations of '
                                             'slopes): not computed (the v1(y) records are curated)'})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {len(staged)} staged')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

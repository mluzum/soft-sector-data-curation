#!/usr/bin/env python3
"""Curate STAR phi v2 and spectra in Au+Au at 200 GeV (2004 run), INSPIRE 746872.

B. I. Abelev et al. (STAR), "Partonic flow and phi-meson production in
Au+Au collisions at sqrt(sNN) = 200 GeV", PRL 99 (2007) 112301,
arXiv:nucl-ex/0703033.

From the paper:
- centrality from the charged hadron multiplicity in |eta| < 0.5;
  minimum-bias classes 0-10, ..., 70-80%; 0-5% (and 0-12%) from the
  central-triggered data set;
- Fig. 1: v2(pT) of phi (K+ K-) relative to the reaction plane (v2 vs
  minv method), 0-80, 10-40, 40-80, 0-5%; errors stat and syst; non-flow
  not subtracted.  The companion paper on the same data (PRC 79 064903,
  arXiv:0809.4737) gives the windows: phi at |y| < 1 for v2, and STAR's
  standard TPC event plane (pT-weighted up to 2 GeV/c, the kaon daughters
  excluded);
- Fig. 2: d2N/(2 pi pT dpT dy) at |y| < 0.5 in nine classes, statistical
  errors.

Records (public): v2(pT) (4; each class a block of the table's rows, pT
bins from HEPData's edges; |y| < 1 computed in |eta| < 1, the TPC plane
|eta| < 1, 0.15 < pT < 2 GeV/c, its pT weighting not modelled) and the
spectra (9).  Staged (--staged OUT): N(Omega)/N(phi) (Fig. 3: a ratio
with the Omega yields of another paper) and R_CP (Fig. 4).

Run from the repository root:  python scripts/curate_star_phi_v2_auau200.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins746872/HEPData-ins746872-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. I. Abelev et al. (STAR), PRL 99 (2007) 112301, arXiv:nucl-ex/0703033'
OUT_V = 'data/flow_vn/star_flow_vn_phi_v2_AuAu_200gev_ins746872.yaml'
OUT_S = 'data/particle_spectra/star_particle_spectra_phi_AuAu_200gev_ins746872.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': 'charged hadron multiplicity in |eta| < 0.5.'}
SYSTEM = {'energy_gev': 200.0, 'reaction': 'AU AU --> PHI X'}
TPC_EP = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.15, 'max': 2.0},
          'notes': "STAR's standard TPC event plane (PRC 79 064903): |eta| < 1, pT weights up to 2 GeV/c (not "
                   'modelled), the kaon daughters excluded.'}


def head(hf, i, label, otype, cls):
    return {'observable_type': otype, 'inspire_id': '746872', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf} column {i})', 'collision_system': dict(SYSTEM),
            'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': 'phi'}}


def filled(dv):
    return [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    flows, spectra, staged = [], [], []
    hf = 'figure_1.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    xs = t['independent_variables'][0]['values']
    for i, dv in enumerate(t['dependent_variables']):
        cls = re.fullmatch(r'\$\\phi\$ \((\d+-\d+%)\)', dv['qualifiers'][0]['value']).group(1)
        rows = filled(dv)
        assert rows == list(range(rows[0], rows[-1] + 1)), (hf, i, rows)
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'syst')}
        bins = [[float(xs[r]['low']), float(xs[r]['high'])] for r in rows]
        assert all(a[1] == b[0] for a, b in zip(bins, bins[1:])), bins
        d = head(hf, i, f'Fig. 1: phi v2(pT), {cls}', 'flow_vn', cls)
        d['kinematic_range'] = {'eta': {'min': -1.0, 'max': 1.0},
                                'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
        d.update(harmonic=2, method='event_plane', flow_reference=dict(TPC_EP), value_type='v_2',
                 dimensionality='differential_pT', table_rows=rows)
        trig = ' (central-triggered data set)' if cls == '0-5%' else ''
        d['notes'] = {'value': f'{CITE}, Fig. 1. v2(pT) of phi (K+ K-, v2 vs minv method), |y| < 1 (computed in '
                               f'|eta| < 1), Au+Au 200 GeV {cls}{trig}, relative to the TPC event plane; non-flow not '
                               'subtracted. pT bins from the table edges (points at the mean pT). Errors: stat and '
                               'syst.'}
        flows.append(d)
    hf = 'figure_2.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        cls = dv['qualifiers'][0]['value']
        assert re.fullmatch(r'\d+-\d+%', cls) and filled(dv) == list(range(len(bins))), (hf, i)
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat',)}
        d = head(hf, i, f'Fig. 2: phi invariant yield, {cls}', 'particle_spectra', cls)
        d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5},
                                'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
        d.update(observable='invariant_yield', value_type='invariant_yield')
        trig = ' (central-triggered data set)' if cls == '0-5%' else ''
        d['notes'] = {'value': f'{CITE}, Fig. 2. d2N/(2 pi pT dpT dy) of phi (K+ K-), |y| < 0.5, Au+Au 200 GeV '
                               f'{cls}{trig}. Errors: statistical (systematics 8% from the dE/dx cut and 4.5% from the background shape, not tabulated).'}
        spectra.append(d)
    for hf in ('figure_3.yaml', 'figure_4_top.yaml', 'figure_4_bottom.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            staged.append({'observable_type': 'particle_ratios' if hf == 'figure_3.yaml' else 'nuclear_modification',
                           'inspire_id': '746872', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']} {dv['qualifiers'][0]['value']} ({hf} "
                                                  f"column {i})",
                           'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}.'},
                           'staging_reason': ('N(Omega)/N(phi): a ratio with the Omega yields of another paper'
                                              if hf == 'figure_3.yaml' else 'R_CP: N_bin scaled')})
    for out, docs in ((OUT_V, flows), (OUT_S, spectra)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

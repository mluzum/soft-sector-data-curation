#!/usr/bin/env python3
"""Curate STAR K-, phi and Xi- production in Au+Au at 3 GeV (fixed target), INSPIRE 1897327.

M. S. Abdallah et al. (STAR), "Probing strangeness canonical ensemble with
K-, phi(1020) and Xi- production in Au+Au collisions at sqrt(sNN) = 3
GeV", PLB 831 (2022) 137152, arXiv:2108.00924.

From the paper:
- fixed-target Au+Au at sqrt(sNN) = 3 GeV (2018); TPC 0 < eta < 1.88, TOF
  0 < eta < 1.5 (lab, beam toward +eta); rapidities in the centre-of-mass
  frame with the beam toward +y, the target at y_cm = -1.05;
- centrality from the charged multiplicity in the TPC acceptance with a
  Glauber model; classes 0-10, 10-40, 40-60% (no results beyond 60%);
- K- from TPC dE/dx and TOF; phi -> K+ K-, Xi- -> Lambda pi-; spectra
  1/(2 pi mT) d2N/(dmT dy) vs mT - m0 per rapidity slice (K- 0.1 wide,
  phi and Xi- 0.2 wide), errors stat. and sys. added in quadrature
  (Fig. 2); dN/dy vs y, extrapolated in pT with an mT-exponential (Fig. 3).

Records (public): the spectra per class and rapidity slice, as 1/(2 pi pT)
d2N/(dpT dy) (mT dmT = pT dpT) in pT bins from the midpoints between each
slice's own mT points (outer edges mirrored; HEPData interleaves the Xi-
slices' two binnings in one column, empty elsewhere); dN/dy vs y per class.  The estimator window 0 < eta_lab <
1.88 is -1.046 < eta_cm < 0.834 (a symmetric system: no frame fields).
Staged (--staged OUT): the phi/K- and phi/Xi- ratios of 4pi yields (Fig. 4,
rapidity-extrapolated with Gaussians).

Run from the repository root:
  python scripts/curate_star_kphixi_3gev.py --staged OUT.yaml
"""
import argparse
import glob
import math
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.curate_star_pt_correlations_fxt import y_cm  # noqa: E402

RAW = glob.glob('hepdata_raw/records/ins1897327/HEPData-ins1897327-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'M. S. Abdallah et al. (STAR), PLB 831 (2022) 137152, arXiv:2108.00924'
OUT_S = 'data/particle_spectra/star_particle_spectra_k_phi_xi_AuAu_3gev_ins1897327.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_k_phi_xi_AuAu_3gev_ins1897327.yaml'
YC = y_cm(3.0)
FXT = {'estimator': {'name': 'FXTMult', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': round(-YC, 3), 'max': round(1.88 - YC, 3)}],
       'notes': f'charged multiplicity in the TPC, 0 < eta_lab < 1.88 (beam toward +eta); in the centre-of-mass frame '
                f'-{YC:.3f} < eta_cm < {1.88 - YC:.3f}.'}
SPECIES = {'01': ('K-', 'K-', 0.493677), '02': ('phi', 'PHI', 1.019461), '03': ('Xi-', 'XI-', 1.32171)}
CLASS = {'a': '0-10%', 'b': '10-40%', 'c': '40-60%'}
FRAME = 'y_cm with the beam toward +y (the target at y_cm = -1.05)'


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pt_of(dm, m):
    mt = dm + m
    return math.sqrt(max(mt * mt - m * m, 0.0))


def mt_edges(points):
    mids = [0.5 * (a + b) for a, b in zip(points[:-1], points[1:])]
    return [max(0.0, points[0] - (mids[0] - points[0]))] + mids + [points[-1] + (points[-1] - mids[-1])]


def head(species, prod, hf, i, label, otype, classes):
    return {'observable_type': otype, 'inspire_id': '1897327', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': 3.0, 'reaction': f'AU AU --> {prod} X'},
            'centrality': {'classes': classes, **FXT}, 'particle_id': {'species': species}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, staged = [], [], []
    for path in sorted(glob.glob(os.path.join(RAW, 'gpc310_fig2[abc]_data_0[123].yaml'))):
        hf = os.path.basename(path)
        c, s = re.fullmatch(r'gpc310_fig2([abc])_data_(0[123])\.yaml', hf).groups()
        species, prod, m = SPECIES[s]
        cls = CLASS[c]
        t = yaml.safe_load(open(path))
        assert t['independent_variables'][0]['header']['name'].startswith('$m_{T}-m_{0}'), hf
        dm = [num(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            q = {x['name']: x['value'] for x in dv['qualifiers']}
            y0, y1 = (float(v) for v in re.fullmatch(r'y\[(-?[\d.]+),(-?[\d.]+)\]', q['rapidity']).groups())
            rows = [r for r, v in enumerate(dv['values']) if num(v['value']) is not None]
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat.+sys.',)}, hf
            # each slice's own points (the Xi- slices alternate between two sets of mT bins)
            pts = [dm[r] for r in rows]
            assert pts == sorted(pts) and len(pts) >= 2, (hf, i, pts)
            edges = mt_edges(pts)
            bins = [[round(pt_of(edges[k], m), 5), round(pt_of(edges[k + 1], m), 5)] for k in range(len(pts))]
            d = head(species, prod, hf, i, f'{species} invariant yield vs mT - m0, {y0:g} < y_cm < {y1:g}, Au+Au 3 '
                     f'GeV, {cls} ({hf} column {i})', 'particle_spectra', [cls])
            d['kinematic_range'] = {'y': {'min': y0, 'max': y1},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            d.update(observable='invariant_yield', value_type='invariant_yield')
            if rows != list(range(len(dm))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}, Fig. 2. 1/(2 pi mT) d2N/(dmT dy) of {species}, {y0:g} < y_cm < {y1:g} '
                                   f'({FRAME}), Au+Au 3 GeV {cls}, as 1/(2 pi pT) d2N/(dpT dy): pT bins from the '
                                   'midpoints between the tabulated mT - m0 (outer edges mirrored). Errors: stat. and '
                                   'sys. in quadrature (one component).'}
            spectra.append(d)
    for s in ('01', '02', '03'):
        hf = f'gpc310_fig3_data_{s}.yaml'
        species, prod, m = SPECIES[s]
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert t['independent_variables'][0]['header']['name'].startswith('Particle Rapidity'), hf
        ys = [num(v['value']) for v in t['independent_variables'][0]['values']]
        step = round(ys[1] - ys[0], 6)
        assert all(abs(b - a - step) < 1e-6 for a, b in zip(ys, ys[1:])), ys
        for i, dv in enumerate(t['dependent_variables']):
            cls = re.fullmatch(r'(\d+)-(\d+) PCT', dv['qualifiers'][0]['value'])
            cls = f'{cls.group(1)}-{cls.group(2)}%'
            rows = [r for r, v in enumerate(dv['values']) if num(v['value']) is not None]
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'sys')}, hf
            bins = [[round(ys[r] - step / 2, 4), round(ys[r] + step / 2, 4)] for r in rows]
            d = head(species, prod, hf, i, f'{species} dN/dy vs y_cm, Au+Au 3 GeV, {cls} ({hf} column {i})',
                     'particle_yields', [cls])
            d['kinematic_range'] = {'y': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins},
                                    'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='dN_dy', value_type='dN_dy', dimensionality='differential_y')
            if rows != list(range(len(ys))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}, Fig. 3. dN/dy of {species} per rapidity slice of {step:g} ({FRAME}), '
                                   f'Au+Au 3 GeV {cls}; pT-integrated with an mT-exponential extrapolation. Errors: '
                                   'stat and sys.'}
            yields.append(d)
    hf = 'gpc310_fig4_data_01.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    for i, dv in enumerate(t['dependent_variables']):
        staged.append({'observable_type': 'particle_yields', 'inspire_id': '1897327', 'version': VERSION,
                       'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f"{dv['header']['name']} 0-10% ({hf})",
                       'collision_system': {'energy_gev': 3.0, 'reaction': 'AU AU --> X'},
                       'centrality': {'classes': ['0-10%'], **FXT}, 'value_type': dv['header']['name'],
                       'notes': {'value': f'{CITE}, Fig. 4. {dv["header"]["name"]}, 0-10%: ratio of 4pi yields.'},
                       'staging_reason': 'ratio of 4pi yields (rapidity-extrapolated with Gaussians): not computed'})
    for out, docs in ((OUT_S, spectra), (OUT_Y, yields)):
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

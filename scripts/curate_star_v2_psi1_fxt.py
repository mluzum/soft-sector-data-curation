#!/usr/bin/env python3
"""Curate STAR v2 of identified hadrons relative to the first-order plane at 3.2-4.5 GeV, INSPIRE 2907591.

STAR, "Observation of the onset of constituent quark number scaling in
heavy-ion collisions at RHIC", PRL (2025), doi:10.1103/2qhx-cp79,
arXiv:2504.02531.

From the paper:
- Au+Au fixed-target collisions at sqrt(sNN) = 3.2, 3.5, 3.9, 4.5 GeV; 10-40%
  centrality from the TPC charged multiplicity (Glauber fit);
- pi+-, K+-, K0S, p, Lambda at -0.5 < y < 0 (centre-of-mass rapidity);
- v2 relative to the first-order event plane, v2 = <cos 2(phi - Psi_1)> /
  R_21, R from three subevents (Eq. 1) at eta_lab [3.2, 6.1] (the plane),
  [2.8, 3.1] and [0, 1.0] (TPC), the lab frame with the beam toward +eta;
- pT-integrated v2 (Fig. 4a) over 0.2-1.6 GeV/c (pi), 0.4-1.6 (K, K0S),
  0.4-2.0 (p, Lambda); errors stat and sys;
- TPC tracks (0 < eta_lab < 2 in the FXT configuration): kinematic_range.eta,
  the POI acceptance besides the rapidity window.

Records (public): v2(pT) per species and energy, and the pT-integrated v2
per species and energy, flow_vn with flow_reference.harmonic 1
(analysis/mixed_plane.py: the scalar-product analog of the event-plane
result at low resolution; 3D containers).  The subevent windows are
converted to the centre-of-mass frame with the beam rapidity, eta_cm =
eta_lab - y_beam (exact for massless particles; protons near beam rapidity
are off by ~0.1).  The plane's subevent sits beyond beam rapidity, where the
EPD sees beam-rapidity nucleons and fragments that a model may lack.  pT
bins: midpoints between the tabulated centres.
Staged (--staged OUT): the 3.0 GeV points (from PLB 827 137003, whose
subevents are stated only as the EPD and the TPC), the NCQ-scaled tables
(the same data in scaled variables) and the Fig. 4b ratios of scaled v2 at
(mT - m0)/nq = 0.4 GeV/c^2 (interpolated).

Run from the repository root:  python scripts/curate_star_v2_psi1_fxt.py --staged OUT.yaml
"""
import argparse
import glob
import math
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2907591/HEPData-ins2907591-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'STAR, PRL (2025), doi:10.1103/2qhx-cp79, arXiv:2504.02531'
OUT = 'data/flow_vn/star_flow_vn_v2_psi1_fxt_AuAu_3.2_4.5gev_ins2907591.yaml'
M_N = 0.938
ENERGIES = (3.2, 3.5, 3.9, 4.5)
SPECIES = {'PiP': 'pi+', 'PiM': 'pi-', 'KP': 'K+', 'KM': 'K-', 'K0S': 'K0_S', 'Proton': 'p', 'Lambda': 'Lambda'}
INT_PT = {'pi+': (0.2, 1.6), 'pi-': (0.2, 1.6), 'K+': (0.4, 1.6), 'K-': (0.4, 1.6), 'K0_S': (0.4, 1.6),
          'p': (0.4, 2.0), 'Lambda': (0.4, 2.0)}
FIG4A = {'$\\pi^{+}$': 'pi+', '$\\pi^{-}$': 'pi-', '$K^{+}$': 'K+', '$K^{-}$': 'K-', '$K^{0}_{S}$': 'K0_S',
         '$p$': 'p', '$\\Lambda$': 'Lambda'}


def y_beam(e):
    return math.acosh(e / (2 * M_N))


def frame_fields(e):
    yb = y_beam(e)

    def w(lo, hi):
        return [{'min': round(lo - yb, 4), 'max': round(hi - yb, 4)}]
    fr = {'harmonic': 1, 'detector': 'EPD', 'eta': w(3.2, 6.1), 'pT': {'min': 0.0, 'max': 100.0},
          'resolution_subevents': {'B': {'eta': w(2.8, 3.1), 'pT': {'min': 0.0, 'max': 100.0}},
                                   'C': {'eta': w(0.0, 1.0), 'pT': {'min': 0.2, 'max': 2.0}}},
          'notes': (f'First-order plane from the EPD at 3.2 < eta_lab < 6.1, resolution from three subevents '
                    f'(eta_lab [3.2, 6.1], [2.8, 3.1], TPC [0, 1.0]); windows in the centre-of-mass frame, '
                    f'eta_cm = eta_lab - y_beam, y_beam = {yb:.4f}. TPC subevent pT 0.2-2 GeV/c assumed (not '
                    'stated); event-plane weights not modelled.')}
    cent = {'classes': ['10-40%'], 'estimator': {'name': 'FXTMult', 'observable': 'Nch', 'detector': 'TPC'},
            'eta': w(0.0, 2.0),
            'notes': f'TPC charged multiplicity, 0 < eta_lab < 2 (centre-of-mass: shifted by y_beam = {yb:.4f}).'}
    return fr, cent


def head(hf, i, label, e, species, kr, dim):
    fr, cent = frame_fields(e)
    yb = y_beam(e)
    # TPC tracks, 0 < eta_lab < 2 (centre-of-mass frame): the POI acceptance besides its rapidity window
    kr = {'eta': {'min': round(-yb, 4), 'max': round(2.0 - yb, 4)}, **kr}
    return {'observable_type': 'flow_vn', 'inspire_id': '2907591', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': e, 'reaction': 'AU AU --> X'},
            'centrality': cent, 'particle_id': {'species': species}, 'kinematic_range': kr,
            'harmonic': 2, 'method': 'event_plane', 'flow_reference': fr, 'dimensionality': dim,
            'value_type': 'v2'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    subs = {x['data_file']: ' '.join(x['description'].split())
            for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x}
    public, staged = [], []

    def stage(hf, reason):
        staged.append({'observable_type': 'flow_vn', 'inspire_id': '2907591', 'version': VERSION,
                       'hepdata_file': hf, 'dv_index': 0, 'hepdata_table_label': subs[hf],
                       'value_type': 'v2', 'notes': {'value': f'{CITE}. {subs[hf]}.'}, 'staging_reason': reason})

    for hf in subs:
        if hf.startswith('NCQ_'):
            stage(hf, 'NCQ-scaled v2 vs scaled kinetic energy: the v2(pT) tables in scaled variables')
            continue
        if hf == 'Fig4_b.yaml':
            stage(hf, 'ratios of NCQ-scaled v2 at (mT - m0)/nq = 0.4 GeV/c^2 (interpolated)')
            continue
        if hf == 'Fig4_a.yaml':
            t = yaml.safe_load(open(os.path.join(RAW, hf)))
            energies = [float(v['value']) for v in t['independent_variables'][0]['values']]
            for i, dv in enumerate(t['dependent_variables']):
                sp = FIG4A[dv['qualifiers'][0]['value']]
                for j, e in enumerate(energies):
                    v = dv['values'][j]
                    if e not in ENERGIES or str(v['value']).strip() in ('', '-'):
                        continue
                    assert [x['label'] for x in v['errors']] == ['stat', 'sys'], (hf, i, j)
                    lo, hi = INT_PT[sp]
                    d = head(hf, i, f'{subs[hf][:90]} ({sp}, {e:g} GeV)', e, sp,
                             {'y': {'min': -0.5, 'max': 0.0}, 'pT': {'min': lo, 'max': hi}}, 'integrated')
                    d['table_rows'] = [j]
                    d['notes'] = {'value': f'{CITE}. pT-integrated v2 of {sp}, {lo:g} < pT < {hi:g} GeV/c, '
                                           f'-0.5 < y < 0, relative to the first-order EPD plane, Au+Au {e:g} GeV '
                                           '10-40%. Errors: stat and sys.'}
                    public.append(d)
            continue
        m = re.fullmatch(r'v2pt_(\w+)_([\d.]+)\.yaml', hf)
        assert m, hf
        sp, e = SPECIES[m.group(1)], float(m.group(2))
        if e not in ENERGIES:
            stage(hf, '3.0 GeV points from PLB 827 137003 (INSPIRE 1897294), whose subevents are stated only as '
                      'the EPD (-5.3 < eta_lab < -2.6) and the TPC; R_21 converted from R_11')
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        dv = t['dependent_variables'][0]
        rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert {tuple(x['label'] for x in dv['values'][r]['errors']) for r in rows} == {('stat', 'sys')}, hf
        mids = [(a + b) / 2 for a, b in zip(pts, pts[1:])]
        edges = [round(2 * pts[0] - mids[0], 4)] + [round(x, 4) for x in mids] + [round(2 * pts[-1] - mids[-1], 4)]
        rb = [[edges[r], edges[r + 1]] for r in rows]
        d = head(hf, 0, f'{subs[hf]}', e, sp,
                 {'y': {'min': -0.5, 'max': 0.0}, 'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}},
                 'differential_pT')
        if rows != list(range(len(pts))):
            d['table_rows'] = rows
        d['notes'] = {'value': f'{CITE}. v2(pT) of {sp}, -0.5 < y < 0, relative to the first-order EPD plane, '
                               f'Au+Au {e:g} GeV 10-40%; pT bins from the midpoints between the tabulated '
                               'centres. Errors: stat and sys.'}
        public.append(d)
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

#!/usr/bin/env python3
"""Curate STAR charged-hadron v2 at 7.7-39 GeV, INSPIRE 1119620.

L. Adamczyk et al. (STAR), "Inclusive charged hadron elliptic flow in Au+Au
collisions at sqrt(sNN) = 7.7-39 GeV", PRC 86, 054908 (2012),
arXiv:1206.5528.

From the paper (Secs. II-IV):
- charged tracks |eta| < 1, pT > 0.2 GeV/c; centrality from the raw
  multiplicity in |eta| < 0.5 (RefMult); event-plane tracks 0.2 < pT < 2;
- v2{EP}: the TPC event plane of all tracks (the particle's own track
  removed), resolution from random subevents; v2{EtaSub}: the plane of the
  opposite eta hemisphere; v2{2}, v2{4}: Q-cumulants; v2{FTPC}: the plane of
  the FTPCs (2.5 < |eta| < 4.0); v2{BBC}: relative to the first-order BBC
  plane;
- integrated v2 over 0.2 < pT < 2.0 GeV/c (Fig. 4);
- Fig. 8 (v2(eta), 10-40%) and Figs. 9-10 show 62.4 and 200 GeV data from
  earlier STAR papers.

Records (public): v2 vs centrality (Fig. 4), v2(pT) at 20-30% (Fig. 5),
v2{EtaSub}(pT) at 10-20, 30-40, 50-60% (Fig. 6-7), v2{4}(pT) at 10-20,
20-30, 30-40% (Figs. 9-10), v2{EP}(eta) at 10-40% (Fig. 8), each at 7.7,
11.5, 19.6, 27, 39 GeV: EP with the TPC reference (the record window,
self-pairs removed), EtaSub as two subevents (particles AB), {2} and {4}
with reference particles 0.2 < pT < 2, FTPC with the FTPC reference.  pT
points with bin edges at the midpoints.  Staged (--staged OUT): v2{BBC}
(first-order plane).  Not curated: the generating-function cumulants of
Fig. 3 (repeat the Q-cumulants), the 62.4 and 200 GeV comparison data
(Figs. 8a, 9-10b, c: earlier papers; their HEPData energies are swapped).
The 8 old records took the first column of each table only (39 GeV),
with an inferred reference.

Run from the repository root:
    python scripts/curate_star_bes_charged_v2.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1119620/HEPData-ins1119620-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'L. Adamczyk et al. (STAR), PRC 86, 054908 (2012), arXiv:1206.5528'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
EP_PT = {'min': 0.2, 'max': 2.0}
TPC_REF = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': dict(EP_PT),
           'notes': 'TPC event plane of all tracks, the particle of interest removed.'}
FTPC_REF = {'detector': 'FTPC', 'eta': [{'min': -4.0, 'max': -2.5}, {'min': 2.5, 'max': 4.0}], 'pT': dict(EP_PT),
            'notes': 'Event plane of the two FTPCs, 2.5 < |eta| < 4.0.'}
HALVES = {'A': {'kinematic_range': {'eta': {'min': -1.0, 'max': 0.0}, 'pT': dict(EP_PT)}},
          'B': {'kinematic_range': {'eta': {'min': 0.0, 'max': 1.0}, 'pT': dict(EP_PT)}}}
OLD = 'data/flow_vn/star_flow_vn_inclusive_charged_hadron_elliptic_flow_200gev.yaml'


def edges_from_points(pts, floor=None):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    lo = pts[0] - (mids[0] - pts[0])
    if floor is not None:
        lo = max(floor, lo)
    return [round(x, 4) for x in [lo] + mids + [pts[-1] + (pts[-1] - mids[-1])]]


def method_fields(m):
    if m == 'EP':
        return {'method': 'event_plane', 'flow_reference': dict(TPC_REF)}, 'TPC event plane'
    if m == 'FTPC':
        return {'method': 'event_plane', 'flow_reference': dict(FTPC_REF)}, 'FTPC event plane'
    if m == 'EtaSub':
        return ({'method': 'event_plane', 'particles': 'AB', 'eta_gap': 0.0,
                 'subevents': {k: {'kinematic_range': {kk: dict(vv) for kk, vv in v['kinematic_range'].items()}}
                               for k, v in HALVES.items()}},
                'eta-sub event plane (the opposite hemisphere)')
    if m in ('2', '4'):
        return ({'method': 'cumulant', 'cumulant_order': int(m), 'flow_reference': {'pT': dict(EP_PT)}},
                f'Q-cumulant v2{{{m}}}, reference particles 0.2 < pT < 2 GeV/c')
    raise KeyError(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    plan = {'Fig4.yaml': 'cent', 'Fig5.yaml': 'pt', 'Fig67.yaml': 'pt', 'Fig910.yaml': 'pt', 'Fig8.yaml': 'eta'}
    for hf, kind in plan.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            label = str(dv['qualifiers'][0]['value'])
            energy = float(re.search(r'([\d.]+) GeV', label).group(1))
            m = re.search(r'\{(EP|EtaSub|2|4|FTPC|BBC)\}', h)
            meth = m.group(1) if m else 'EP'                        # Fig. 8: v2{EP}(eta)
            cm = re.search(r'\((\d+-\d+%)\)', h)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            labels = sorted({e.get('label', '?') for v in dv['values'] for e in v.get('errors', [])})
            d = {'observable_type': 'flow_vn', 'inspire_id': '1119620', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{h}, {energy:g} GeV ({hf})',
                 'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
                 'particle_id': {'species': 'charged'}, 'harmonic': 2, 'value_type': h.strip('$')}
            if kind == 'cent':
                classes = [f"{iv['values'][j]['value']}%" for j in rows]
                d.update({'centrality': {'classes': classes, **REFMULT},
                          'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 2.0}},
                          'dimensionality': 'integrated'})
                what = 'v2 vs centrality, 0.2 < pT < 2 GeV/c, |eta| < 1'
            else:
                cls = cm.group(1) if cm else ('20-30%' if hf == 'Fig5.yaml' else '10-40%')
                pts = [float(v['value']) for v in iv['values']]
                e = edges_from_points(pts, floor=0.0 if kind == 'pt' else None)
                pairs = [[e[j], e[j + 1]] for j in rows]
                if kind == 'pt':
                    kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
                    dim, what = 'differential_pT', f'v2(pT), {cls}, |eta| < 1, midpoint pT edges'
                else:
                    kr = {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}, 'pT': {'min': 0.2, 'max': 2.0}}
                    dim, what = 'differential_eta', f'v2(eta), {cls}, 0.2 < pT < 2 GeV/c'
                d.update({'centrality': {'classes': [cls], **REFMULT}, 'kinematic_range': kr, 'dimensionality': dim})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            err = 'statistical and systematic' if 'sys' in labels else 'statistical (systematic in the paper)'
            if meth == 'BBC':
                d.update({'method': 'event_plane', 'notes': {'value': f'{CITE}. {h}, {energy:g} GeV.'},
                          'staging_reason': 'relative to the first-order BBC event plane (mixed harmonic)'})
                staged.append(d)
                continue
            f, mdesc = method_fields(meth)
            d.update(f)
            d['notes'] = {'value': f'{CITE}. {what}, {energy:g} GeV, {mdesc}'
                                   + ('; computed with the scalar-product estimator' if meth in ('EP', 'FTPC') else
                                      '; computed as the two-subevent v2' if meth == 'EtaSub' else '')
                                   + f'. Errors: {err}.'}
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1119620' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_charged_v2_AuAu_7_39gev_ins1119620.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['hepdata_file'], d.get('cumulant_order') or d.get('particles') or (d.get('flow_reference') or {}).get('detector'))
        for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

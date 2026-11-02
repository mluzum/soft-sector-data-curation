#!/usr/bin/env python3
"""Curate STAR directed flow in Au+Au at 62.4 GeV, INSPIRE 695404 (staged only).

J. Adams et al. (STAR), "Directed flow in Au+Au collisions at sqrt(sNN) =
62 GeV", PRC 73, 034903 (2006), nucl-ex/0510053.

From the paper:
- charged particles in the TPC (|eta| < 1.3) and FTPCs (2.5 < |eta| <
  4.0); centrality from the raw TPC multiplicity, |eta| < 0.5;
- three methods: the three-particle cumulant v1{3} (<<cos(phi_a + phi_b -
  2 phi_c)>> = v1 v1 v2), the mixed-harmonic event plane v1{EP1,EP2} (FTPC
  first-order planes with the TPC second-order plane) and the
  spectator-neutron plane v1{ZDC-SMD};
- v1 positive for nucleons in the projectile fragmentation region;
- identified pi and p v1(y) in 40-70% (ZDC-SMD plane);
- values in percent; errors: stat.

Every record is staged (--staged OUT): v1(eta) in 10-70% (Fig. 1a-c) and
per class (Fig. 3), v1(y) of charged, pi, p (Fig. 2), v1{3}(pT) in the TPC
(10-70%) and FTPCs (four classes; Fig. 4).  Mixed-harmonic v1 and the
spectator plane are not computed; eta and pT are central values (windows
of +-0.1 around each eta, pT edges at the midpoints).  Not curated: model
curves (Fig. 1d-f), v1 vs impact parameter (Fig. 5), v1 vs eta - y_beam
(Fig. 6: the same 62.4 GeV points shifted, and other energies' data).  The
old public record read the impact parameter of Fig. 5 as v2.

Run from the repository root:  python scripts/curate_star_v1_auau62.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins695404/HEPData-ins695404-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRC 73, 034903 (2006), nucl-ex/0510053'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
OLD = 'data/flow_vn/star_flow_vn_directed_flow_200gev.yaml'
REASON = {'3': 'three-particle mixed-harmonic cumulant v1{3} (v1 v1 v2): not computed',
          'EP1,EP2': 'mixed-harmonic event plane v1{EP1,EP2} (first-order FTPC planes with the second-order TPC plane): '
                     'not computed',
          'ZDC-SMD': 'relative to the spectator-neutron plane (ZDC-SMD): models have no spectators'}
FIG2 = {'Fig2a.yaml': ('charged', (0.15, 2.0)), 'Fig2b.yaml': ('pi+ + pi-', (0.15, 0.75)),
        'Fig2c.yaml': ('p', (0.15, 1.0))}
BLANK = ('', '-', '--', 'none')


def method_of(h):
    m = re.search(r'v_1\$?\{([^}]+)\}', h.replace('$', ''))
    return m.group(1).strip()


def base(hf, i, h, meth, classes, species, kr, dim, note):
    d = {'observable_type': 'flow_vn', 'inspire_id': '695404', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': f'{note.split(".")[0]} ({hf})',
         'collision_system': {'energy_gev': 62.4, 'reaction': 'AU AU --> X'},
         'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': species},
         'kinematic_range': kr, 'harmonic': 1, 'value_type': h, 'value_scale': 100.0, 'dimensionality': dim,
         'notes': {'value': f'{CITE}. {note} Sign: positive for nucleons in the projectile fragmentation region. '
                            'Values in percent. Errors: stat.'},
         'staging_reason': REASON[meth]}
    if meth == '3':
        d.update({'method': 'cumulant', 'cumulant_order': 3})
    elif meth == 'ZDC-SMD':
        d.update({'method': 'event_plane', 'flow_reference': {'detector': 'ZDC-SMD', 'plane': 'spectator'}})
    else:
        d.update({'method': 'event_plane', 'flow_reference': {'detector': 'FTPC', 'eta': [
            {'min': -4.0, 'max': -2.5}, {'min': 2.5, 'max': 4.0}], 'pT': {'min': 0.15, 'max': 2.0},
            'notes': 'First-order FTPC planes (random subevents) with the second-order TPC plane.'}})
    return d


def eta_windows(x):
    return [[round(v - 0.1, 3), round(v + 0.1, 3)] for v in x]


def pt_edges(x):
    mids = [round((a + b) / 2, 4) for a, b in zip(x, x[1:])]
    e = [max(0.0, round(x[0] - (mids[0] - x[0]), 4))] + mids + [round(x[-1] + (x[-1] - mids[-1]), 4)]
    return [[a, b] for a, b in zip(e, e[1:])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    staged = []
    tables = (['Fig1a.yaml', 'Fig1b.yaml', 'Fig1c.yaml', 'Fig305.yaml', 'Fig3510.yaml', 'Fig3semi.yaml', 'Fig3per.yaml']
              + list(FIG2) + ['Fig4upper.yaml', 'Fig4lower.yaml'])
    for hf in tables:
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        x = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            if 'v_1' not in h:
                continue                                    # the column repeating the x values
            meth = method_of(h) if '{' in h else 'ZDC-SMD'
            q = str(dv['qualifiers'][0]['value']) if dv.get('qualifiers') else ''
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            if not rows:
                continue
            if hf.startswith('Fig4'):
                m = re.search(r'(\d+)-(\d+) ?%', q)
                cls = f'{m.group(1)}-{m.group(2)}%'
                bins = [pt_edges(x)[j] for j in rows]
                tpc = 'TPC' in q
                kr = {'eta': {'min': -1.3, 'max': 1.3} if tpc else {'min': 2.5, 'max': 4.0},
                      'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
                note = (f"v1{{3}}(pT) of charged particles, {cls}, {'TPC |eta| < 1.3' if tpc else 'FTPC 2.5 < |eta| < 4.0'}"
                        '. pT central values, edges at the midpoints'
                        + ('; the eta window is one side, the paper combines both FTPCs.' if not tpc else '.'))
                d = base(hf, i, h, meth, [cls], 'charged', kr, 'differential_pT', note)
            else:
                if hf in FIG2:
                    sp, (lo, hi) = FIG2[hf]
                    cls = '40-70%'
                else:
                    sp, (lo, hi) = 'charged', (0.15, 2.0)
                    m = re.search(r'(\d+) ?- ?(\d+) ?%', q)
                    cls = f'{m.group(1)}-{m.group(2)}%' if m else '10-70%'
                bins = [eta_windows(x)[j] for j in rows]
                var = 'y' if hf in FIG2 and sp != 'charged' else 'eta'
                kr = {var: {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}, 'pT': {'min': lo, 'max': hi}}
                note = (f'v1{{{meth}}}({var}) of {sp}, {cls}. {var} central values (windows of +-0.1); pT {lo:g}-{hi:g} '
                        'GeV/c' + (' (assumed for charged particles)' if hf not in FIG2 else '') + '.')
                d = base(hf, i, h, meth, [cls], sp, kr, 'differential_eta', note)
            if rows != list(range(len(x))):
                d['table_rows'] = rows
            staged.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '695404' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    print(f'{len(staged)} staged records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

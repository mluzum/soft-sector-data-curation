#!/usr/bin/env python3
"""Curate STAR charged and strange hadron v2 in Au+Au at 200 GeV, INSPIRE 777954.

B. I. Abelev et al. (STAR), "Centrality dependence of charged hadron and
strange hadron elliptic flow from sqrt(sNN) = 200 GeV Au+Au collisions",
PRC 77, 054901 (2008), arXiv:0801.3466.

From the paper (Secs. II-III):
- centrality from the charged tracks in the TPC, |eta| < 0.5; 0-10, 10-40,
  40-80% (and 10% classes for the integrated charged v2);
- charged tracks 0.15 < pT < 2.0 GeV/c, |eta| < 1.0 (LYZ product |eta| <
  1.3); the FTPCs at 2.5 < |eta| < 4.0;
- event plane (EP) from the TPC tracks, also for the FTPC particles;
  eta-subevent: particles in one hemisphere with the plane of the other,
  the hemispheres separated by |eta| < 0.075; four-particle cumulant; Lee-
  Yang zeros (LYZ, sum and product generating functions);
- K0_S, Lambda + Lambdabar, Xi- + Xibar+, Omega- + Omegabar+ from their
  decays (TPC); errors: stat (and syst for K0_S, Lambda in Fig. 7).

Records (public): charged v2 vs centrality (EP, v2{4}; Fig. 5), v2(pT) in
10-40% (EP, v2{4}) and 0-80% (EP) (Fig. 4), v2(eta) in 10-40% (EP, TPC and
FTPC; Fig. 3, 0.1-wide eta bins), K0_S and Lambda v2(pT) in 10-40% from
the EP and the eta-subevent method (Fig. 6), K0_S, Lambda, Xi, Omega
v2(pT) per class from the EP (Fig. 7).  EP computed with the scalar-
product estimator (TPC reference, |eta| < 1, 0.15 < pT < 2; the POI
removed); the eta-subevent as the two-subevent correlation (particles AB);
v2{4} with reference particles 0.15 < pT < 2.  The charged tables are in
percent, errors included (value_scale 100).  Strange hadrons in |eta| < 1
(no rapidity window stated).  pT points with edges at the midpoints.
Staged (--staged OUT): the LYZ results (no computation).  Not curated: the
pi and p comparison in Fig. 7a (other analyses), the Fig. 12d insets, Fig.
13 (v2 vs N_part) and Fig. 14 (energy comparison).  The 5 old records
filed LYZ as event plane and gave no reference.

Run from the repository root:
    python scripts/curate_star_v2_charged_strange_200.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins777954/HEPData-ins777954-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. I. Abelev et al. (STAR), PRC 77, 054901 (2008), arXiv:0801.3466'
TPC_CENT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.5, 'max': 0.5}]}
PT = {'min': 0.15, 'max': 2.0}
ETA = {'min': -1.0, 'max': 1.0}
TPC_REF = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': dict(PT),
           'notes': 'TPC event plane, |eta| < 1, 0.15 < pT < 2 GeV/c, the particle of interest removed.'}
HALVES = {'particles': 'AB', 'eta_gap': 0.15,
          'subevents': {'A': {'kinematic_range': {'eta': {'min': -1.0, 'max': -0.075}, 'pT': dict(PT)}},
                        'B': {'kinematic_range': {'eta': {'min': 0.075, 'max': 1.0}, 'pT': dict(PT)}}}}
SPECIES = {'K_{S}^{0}': 'K0_S', 'Lambda': 'Lambda + Lambdabar', 'Xi': 'Xi- + Xibar+', 'Omega': 'Omega- + Omegabar+'}
OLD = 'data/flow_vn/star_flow_vn_centrality_dependence_of_charged_hadron_200gev.yaml'


def species_of(text):
    for k, v in SPECIES.items():
        if k in text:
            return v
    return 'charged'


def method_of(text):
    t = text.replace(' ', '')
    if 'LYZ' in t:
        return 'LYZ'
    if 'eta$-sub' in t or 'eta-sub' in t:
        return 'etasub'
    if '4Cumulant' in t or '4p' in t:
        return '4'
    return 'EP'


def edges(pts, floor=None):
    mids = [round(0.5 * (a + b), 4) for a, b in zip(pts[:-1], pts[1:])]
    lo = pts[0] - (mids[0] - pts[0])
    return [round(lo if floor is None else max(floor, lo), 4)] + mids + [round(pts[-1] + (pts[-1] - mids[-1]), 4)]


def cls_from(text):
    m = re.search(r'(\d+)\s*-\s*(\d+)', text)
    return f'{m.group(1)}-{m.group(2)}%'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    tables = (['fig3.yaml', 'fig4_ep.yaml', 'fig4_4p.yaml', 'fig4_lyz.yaml', 'fig4_ep80.yaml', 'fig5_ep.yaml',
               'fig5_4p.yaml', 'fig5_lyz.yaml', 'fig5_prod.yaml']
              + sorted(f for f in os.listdir(RAW) if f.startswith('fig6_'))
              + sorted(f for f in os.listdir(RAW) if f.startswith('fig7_') and 'protonpion' not in f))
    for hf in tables:
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        ivname = iv['header']['name']
        desc = ' '.join(str(x) for x in [hf])
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            quals = ' '.join(str(q.get('value')) for q in dv.get('qualifiers', []))
            meth = method_of(h + ' ' + quals + ' ' + hf)
            sp = species_of(h)
            percent = '%' in h
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            if 'Centrality' in ivname:
                classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
                kr, dim, where = {'eta': dict(ETA), 'pT': dict(PT)}, 'integrated', 'vs centrality'
            elif 'eta' in ivname:
                pts = [float(iv['values'][j]['value']) for j in rows]
                pairs = [[round(p - 0.05, 3), round(p + 0.05, 3)] for p in pts]
                classes = [cls_from(quals)]
                kr = {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}, 'pT': dict(PT)}
                dim, where = 'differential_eta', f'vs eta, {classes[0]}'
            else:
                pts = [float(v['value']) for v in iv['values']]
                e = edges(pts, floor=0.0)
                pairs = [[e[j], e[j + 1]] for j in rows]
                classes = [cls_from(quals)] if re.search(r'\d+\s*-\s*\d+', quals) else ['10-40%']
                kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
                dim, where = 'differential_pT', f'vs pT, {classes[0]}'
            d = {'observable_type': 'flow_vn', 'inspire_id': '777954', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'v2 of {sp} {where}, {meth} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                 'centrality': {'classes': classes, **TPC_CENT}, 'particle_id': {'species': sp},
                 'kinematic_range': kr, 'harmonic': 2, 'value_type': h, 'dimensionality': dim}
            if percent:
                d['value_scale'] = 100.0
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if meth == 'EP':
                d.update({'method': 'event_plane', 'flow_reference': dict(TPC_REF)})
                how = 'TPC event plane, computed with the scalar-product estimator'
            elif meth == 'etasub':
                d.update({'method': 'event_plane', **{k: (dict(v) if isinstance(v, dict) else v) for k, v in HALVES.items()}})
                how = 'eta-subevent event plane (the opposite hemisphere), computed as the two-subevent correlation'
            elif meth == '4':
                d.update({'method': 'cumulant', 'cumulant_order': 4})
                if dim != 'integrated':
                    d['flow_reference'] = {'pT': dict(PT)}
                how = 'four-particle cumulant'
            else:
                d.update({'method': 'lyz', 'staging_reason': 'Lee-Yang zeros: no computation'})
                how = 'Lee-Yang zeros'
            d['notes'] = {'value': f'{CITE}. v2 of {sp} {where}, {how}'
                                   + ('; FTPC points use the TPC plane' if dim == 'differential_eta' else '')
                                   + ('; tabulated in percent' if percent else '')
                                   + ('. Errors: stat and syst.' if any(e.get('label') == 'syst' for v in dv['values']
                                                                       for e in v.get('errors') or []) else '. Errors: stat.')}
            (staged if meth == 'LYZ' else public).append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '777954' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_v2_charged_strange_AuAu_200gev_ins777954.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['particle_id']['species'], d['dimensionality'], d['method'], d.get('particles')) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

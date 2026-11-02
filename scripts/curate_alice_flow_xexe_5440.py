#!/usr/bin/env python3
"""Curate ALICE anisotropic flow in Xe-Xe at 5.44 TeV (and Pb-Pb 5.02 TeV), INSPIRE 1671792.

S. Acharya et al. (ALICE), "Anisotropic flow in Xe-Xe collisions at
sqrt(sNN) = 5.44 TeV", PLB 784 (2018) 82, arXiv:1805.01832.

From the paper (Sec. 2):
- charged particles |eta| < 0.8, 0.2 < pT < 3.0 GeV/c; V0M centrality
  (V0A 2.8 < eta < 5.1, V0C -3.7 < eta < -1.7);
- v_n{2, |Delta eta| > 2}: scalar product of the tracks with the V0A
  Q-vector (amplitude weights), normalized with the V0C and the TPC;
- v_n{2, |Delta eta| > 1} and v2{m >= 4}: generic framework, the gap
  between Q-vectors within the TPC; v2{4, 3 sub-event} with the sub-events
  -0.8 < eta <= -0.4, |eta| < 0.4, 0.4 <= eta < 0.8;
- errors: stat and sys.

Records (public): Xe-Xe v2, v3 {2, |Delta eta| > 2}, v2, v3, v4 {2,
|Delta eta| > 1}, v2{4}, v2{6}, v2{8} vs centrality (data1; 1% classes in
data3), v2(pT) and v3(pT) {2, |Delta eta| > 2} per class (data5, data6);
Pb-Pb 5.02 TeV v2, v3 {2, |Delta eta| > 2} (data2, data4).  The |Delta eta|
> 1 gap as subevents -0.8 < eta < -0.5 and 0.5 < eta < 0.8 (as for ALICE
900651, 1666817); the |Delta eta| > 2 records with the V0A reference
(scalar-product estimator).  Classes from the tabulated centres; points
with value and errors 0 (v2{6}, v2{8} where not measured) left out; pT
points with edges at the midpoints.  Staged (--staged OUT): v2{4, 3 sub-
event} (three subevents: no computation).  The 4 old records stated every
gap as two halves with no gap.

Run from the repository root:
    python scripts/curate_alice_flow_xexe_5440.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1671792/HEPData-ins1671792-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PLB 784 (2018) 82, arXiv:1805.01832'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
ETA = {'min': -0.8, 'max': 0.8}
PT = {'min': 0.2, 'max': 3.0}
V0A = {'detector': 'V0A', 'eta': [{'min': 2.8, 'max': 5.1}], 'pT': {'min': 0.0, 'max': 100.0},
       'notes': 'Q_n of the V0A (2.8 < eta < 5.1), channel amplitudes as weights; normalized with the V0C and the '
                'TPC (three subevents).'}
OLD_GLOB = 'data/flow_vn/alice_flow_vn_v2_2_etagap_*_data[1-4].yaml'


def subevents():
    return {'particles': 'AB', 'eta_gap': 1.0,
            'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.5}, 'pT': dict(PT)}},
                          'B': {'kinematic_range': {'eta': {'min': 0.5, 'max': 0.8}, 'pT': dict(PT)}}}}


def classes_from_centres(cs):
    """Classes from centres; widths from the spacing (0-5, 5-10, 10-20, ... or 1%)."""
    out, lo = [], 0.0
    for c in cs:
        hi = 2 * c - lo
        out.append(f'{lo:g}-{hi:g}%')
        lo = hi
    return out


def parse(h):
    """(harmonic, kind) from a header: kind gap1 | gap2 | 4 | 6 | 8 | 3sub."""
    n = int(re.search(r'v\$_\{(\d)\}', h).group(1))
    if '3sub' in h:
        return n, '3sub'
    m = re.search(r'Delta\\eta\| >(\d)', h)
    if m:
        return n, 'gap' + m.group(1)
    return n, re.search(r'\\\{(\d)\\\}', h).group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf in ('data1.yaml', 'data2.yaml', 'data3.yaml', 'data4.yaml', 'data5.yaml', 'data6.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        differential = 'p_' in iv['header']['name']
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            energy = float(q['SQRT(S)/NUCLEON'])
            beams = 'XE XE' if 'XE' in q['RE'] else 'PB PB'
            n, kind = parse(h)
            rows = [j for j, v in enumerate(dv['values'])
                    if str(v['value']).strip().lower() not in ('', '-', 'none')
                    and not (float(v['value']) == 0.0 and all(float(e.get('symerror', 0) or 0) == 0.0
                                                              for e in v.get('errors') or []))]
            if not rows:
                continue
            if differential:
                cls = re.search(r'\((\d+-\d+%)\)', h).group(1)
                pts = [float(v['value']) for v in iv['values']]
                mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
                e = [round(max(0.0, pts[0] - (mids[0] - pts[0])), 4)] + [round(m, 4) for m in mids] + \
                    [round(pts[-1] + (pts[-1] - mids[-1]), 4)]
                pairs = [[e[j], e[j + 1]] for j in rows]
                classes, dim, where = [cls], 'differential_pT', f'vs pT, {cls}'
                kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
            else:
                allc = classes_from_centres([float(v['value']) for v in iv['values']])
                classes = [allc[j] for j in rows]
                dim, where, kr = 'integrated', 'vs centrality, 0.2 < pT < 3 GeV/c', {'eta': dict(ETA), 'pT': dict(PT)}
            d = {'observable_type': 'flow_vn', 'inspire_id': '1671792', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{h} {where}, {beams.title()} {energy:g} GeV ({hf})',
                 'collision_system': {'energy_gev': energy, 'reaction': f'{beams} --> CHARGED X'},
                 'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': kr, 'harmonic': n, 'value_type': h, 'dimensionality': dim}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if kind == 'gap2':
                d.update({'method': 'scalar_product', 'flow_reference': dict(V0A)})
                how = 'scalar product with the V0A (|Delta eta| > 2), computed with the scalar-product estimator'
            elif kind == 'gap1':
                d.update({'method': 'cumulant', 'cumulant_order': 2, **subevents()})
                how = 'two-particle cumulant with |Delta eta| > 1, as subevents -0.8 < eta < -0.5, 0.5 < eta < 0.8'
            elif kind == '3sub':
                d.update({'method': 'cumulant', 'cumulant_order': 4,
                          'staging_reason': 'v2{4} from three subevents: no computation'})
                how = 'four-particle cumulant, three subevents'
            else:
                d.update({'method': 'cumulant', 'cumulant_order': int(kind)})
                how = f'{kind}-particle cumulant (no gap)'
            d['notes'] = {'value': f'{CITE}. v{n} of charged particles {where}, |eta| < 0.8, {beams.title()} '
                                   f'{energy:g} GeV, {how}'
                                   + ('; classes from the tabulated centres' if not differential else '')
                                   + '. Errors: stat and sys.'}
            (staged if kind == '3sub' else public).append(d)
    for f in glob.glob(OLD_GLOB):
        assert all(str(x['inspire_id']) == '1671792' for x in yaml.safe_load_all(open(f)) if x), f
        os.remove(f)
    out = 'data/flow_vn/alice_flow_vn_XeXe_5440gev_ins1671792.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['collision_system']['energy_gev'], d['dimensionality'], d['method'], d.get('cumulant_order')) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

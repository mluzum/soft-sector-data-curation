#!/usr/bin/env python3
"""Curate STAR v2 in Au+Au at 200 GeV (PRC 72, 014904), INSPIRE 660793.

J. Adams et al. (STAR), "Azimuthal anisotropy in Au+Au collisions at
sqrt(sNN) = 200 GeV", PRC 72, 014904 (2005), arXiv:nucl-ex/0409033.

From the paper:
- TPC tracks 0.15 < pT < 2.0 GeV/c, |eta| < 1.3 (Table I); FTPCs 2.4 <
  |eta| < 4.2; centrality from the TPC multiplicity in |eta| < 0.5;
  minimum bias = 0-80%;
- notation (Sec. III A): v_n is the standard method -- each particle
  correlated with the event plane of the full TPC event minus itself; v_n{N}
  the N-particle cumulant; the scalar product sums over 0.15 < pT < 2
  GeV/c, |eta| < 1.0 (Eq. 6); random-subevent and eta-subevent methods
  correlate each particle with the plane of the other subevent; the
  "modified" event plane drops the tracks within |Delta eta| < 0.5 of the
  leading particle;
- v2(eta) (Fig. 15): event plane at mid-rapidity (standard method);
- errors: statistical only.

Records (charged hadrons unless stated; POI |eta| < 1.3 in the TPC):
- standard v2(pT): Fig. 4 (minimum bias and nine classes), Fig. 5 (0-50%),
  Fig. 18 (six classes, to 9.5 GeV/c), and minimum-bias kaons (kinks,
  dE/dx; Fig. 9), pions and antiprotons (Fig. 11);
- v2{2}(pT) of pi, K, pbar in nine classes (Fig. 7), v2{4}(pT) of pi, K,
  pbar in 20-60% (Fig. 6);
- v2(eta) minimum bias (Fig. 15), v2{2}(eta), v2{4}(eta) in 20-70%
  (Fig. 16);
- integrated v2 (0.15 < pT < 2 GeV/c) vs centrality: standard, scalar
  product (|eta| < 1.0), eta subevents, random subevents, v2{2}, v2{4}
  (Fig. 29a), v2{6} (Fig. 30).
event_plane: the TPC reference |eta| < 1.3, 0.15 < pT < 2 GeV/c (the
particle's own pairs removed; eta subevents: the opposite hemisphere),
computed with the scalar-product estimator; cumulants: the 0.15-2 GeV/c
reference in the record window (TPC+FTPC for v2{2,4}(eta): the
pseudorapidity window between them is included here).  Tables in percent
without a '%' unit carry value_scale 100; v2{4} placeholders (0 +- 1e-5)
are left out with table_rows.  pT and eta points with windows of their
spacing (at most 0.1 GeV/c in pT).
Not curated: v1 (Fig. 2), RICH pi+K (Fig. 8), K0_S and Lambda (Figs. 9-10,
from other publications), the <uQ> correlators (Figs. 12-14), the FTPC
v2(pT) (Fig. 17, momentum resolution), the modified-event-plane and
p+p-subtracted v2 (Figs. 18, 20), v4 (Figs. 21-28), q-distribution v2,
g2, v2/n and the Blast Wave parameters.

Run from the repository root:  python scripts/curate_star_v2_auau200_2005.py
"""
import collections
import os
import re

import numpy as np
import yaml

RAW = 'hepdata_raw/records/ins660793/HEPData-ins660793-v1-yaml'
CITE = 'J. Adams et al. (STAR), PRC 72, 014904 (2005), arXiv:nucl-ex/0409033'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
TPC_ETA = (-1.3, 1.3)
TPC_REF = {'detector': 'TPC', 'eta': [{'min': -1.3, 'max': 1.3}], 'pT': {'min': 0.15, 'max': 2.0},
           'notes': 'Standard method: event plane of the full TPC event (|eta| < 1.3, 0.15 < pT < 2 GeV/c) minus '
                    'the particle of interest.'}
CUM_REF = {'pT': {'min': 0.15, 'max': 2.0}, 'notes': 'Reference particles 0.15 < pT < 2 GeV/c in the record window.'}
SPECIES = {'pions': 'pi+ + pi-', 'kaons': 'K+ + K-', 'anti-protons': 'pbar'}


def pt_axis(values, rows, var='pT', cap=0.1):
    vals = [values[j] for j in rows]
    if all(v.get('low') is not None and v.get('high') is not None for v in vals):
        pairs = [[float(v['low']), float(v['high'])] for v in vals]
        if all(abs(a[1] - b[0]) < 2e-3 for a, b in zip(pairs, pairs[1:])) and all(lo < hi for lo, hi in pairs):
            edges = [pairs[0][0]] + [p[1] for p in pairs]
            return {'min': edges[0], 'max': edges[-1], 'bins': edges}
    pts = [float(v['value']) for v in vals]
    w = round(float(min(min(np.diff(pts)) if len(pts) > 1 else cap, cap)), 4)
    return {'min': round(pts[0] - w / 2, 4), 'max': round(pts[-1] + w / 2, 4), 'points': pts, 'point_window': w}


def scale(dv, rows):
    """value_scale for a percent column whose unit does not say so."""
    vals = [abs(float(dv['values'][j]['value'])) for j in rows]
    units = str(dv['header'].get('units', ''))
    if '%' in units or max(vals) <= 0.5:
        return None
    assert max(vals) < 50, dv['header']
    return 100.0


def method(kind, eta_kr):
    if kind == 'standard':
        return {'method': 'event_plane', 'flow_reference': dict(TPC_REF)}, 'standard event plane (full TPC event)'
    if kind == 'etasubs':
        return ({'method': 'event_plane',
                 'flow_reference': {'detector': 'TPC', 'mirror': True, 'eta': [{'min': 0.0, 'max': 1.3}],
                                    'pT': {'min': 0.15, 'max': 2.0},
                                    'notes': 'eta subevents: the plane of the opposite TPC hemisphere.'}},
                'eta-subevent event plane')
    if kind == 'randomsubs':
        return ({'method': 'event_plane', 'flow_reference': dict(TPC_REF, notes='Random subevents: the plane of '
                                                                                 'the other half of the TPC event.')},
                'random-subevent event plane')
    if kind == 'sp':
        return {'method': 'scalar_product'}, 'scalar product (0.15 < pT < 2 GeV/c, |eta| < 1.0)'
    order = int(kind[1])
    return ({'method': 'cumulant', 'cumulant_order': order, 'flow_reference': dict(CUM_REF)},
            f'{order}-particle cumulant')


def record(hf, i, kind, sp, classes, kr, dim, label, what, vs=None, table_rows=None):
    fields, how = method(kind, kr.get('eta'))
    d = {'observable_type': 'flow_vn', 'inspire_id': '660793', 'version': 1, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': label,
         'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
         'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': sp},
         'kinematic_range': kr, 'harmonic': 2, 'value_type': label.split(' | ')[-1],
         'dimensionality': dim, **fields}
    if vs:
        d['value_scale'] = vs
    if table_rows is not None:
        d['table_rows'] = table_rows
    d['notes'] = {'value': (f'{CITE}. {what}; {how}' + ('; tabulated in percent' if vs else '')
                            + '. Statistical errors only.')}
    return d


def cls_of(text):
    m = re.search(r'([0-9.]+)\s*-\s*([0-9.]+)\s*%', text)
    return f'{float(m.group(1)):g}-{float(m.group(2)):g}%'


def main():
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    docs = []

    def table(hf):
        return yaml.safe_load(open(os.path.join(RAW, hf)))

    def rows_of(dv):
        # rows with a number (Table 4, minimum bias, row 9 reads '12 0': left out)
        out = []
        for j, v in enumerate(dv['values']):
            try:
                float(v['value'])
                out.append(j)
            except (TypeError, ValueError):
                pass
        return out

    def label(hf, dv):
        return f"{' '.join(sub[hf]['description'].split())[:150]} | {dv['header']['name'].strip()}"

    def pt_record(hf, i, kind, sp, cls, what, eta=TPC_ETA):
        t = table(hf)
        dv = t['dependent_variables'][i]
        r = rows_of(dv)
        kr = {'eta': {'min': eta[0], 'max': eta[1]}, 'pT': pt_axis(t['independent_variables'][0]['values'], r)}
        if r != list(range(r[0], r[-1] + 1)):
            assert 'points' in kr['pT'], (hf, i)
            what += f'; row(s) {sorted(set(range(r[0], r[-1] + 1)) - set(r))} unreadable in the table, left out'
        docs.append(record(hf, i, kind, sp, [cls], kr, 'differential_pT', label(hf, dv), what, scale(dv, r)))

    # charged v2(pT), standard method
    t = table('data4.yaml')
    for i, dv in enumerate(t['dependent_variables']):
        cls = '0-80%' if 'min. bias' in dv['header']['name'] else cls_of(dv['header']['name'])
        pt_record('data4.yaml', i, 'standard', 'charged', cls, f'charged-hadron v2(pT), {cls} (Fig. 4)')
    pt_record('data5.yaml', 0, 'standard', 'charged', '0-50%', 'charged-hadron v2(pT), 0-50% (Fig. 5)')
    for k in range(1, 7):
        hf = f'data18_{k}.yaml'
        cls = cls_of(sub[hf]['description'])
        pt_record(hf, 0, 'standard', 'charged', cls, f'charged-hadron v2(pT) to high pT, {cls} (Fig. 18)')
    # identified, minimum bias, standard method
    for hf, sp, what in (('data9_2.yaml', 'K+ + K-', 'kaons from kinks'), ('data9_3.yaml', 'K+ + K-', 'kaons from dE/dx'),
                         ('data11_1.yaml', 'pi+ + pi-', 'charged pions'), ('data11_2.yaml', 'pbar', 'antiprotons')):
        pt_record(hf, 0, 'standard', sp, '0-80%', f'v2(pT) of {what}, minimum bias ({sub[hf].get("location", "")})')
    # v2{2}(pT), v2{4}(pT) of identified particles
    for hf, sp in (('data7a.yaml', 'pi+ + pi-'), ('data7b.yaml', 'K+ + K-'), ('data7c.yaml', 'pbar')):
        for i, dv in enumerate(table(hf)['dependent_variables']):
            cls = cls_of(dv['header']['name'])
            pt_record(hf, i, 'c2', sp, cls, f'v2{{2}}(pT) of {sp}, {cls} (Fig. 7)')
    for hf, sp in (('data6_1.yaml', 'pi+ + pi-'), ('data6_2.yaml', 'K+ + K-'), ('data6_3.yaml', 'pbar')):
        pt_record(hf, 0, 'c4', sp, '20-60%', f'v2{{4}}(pT) of {sp}, 20-60% (Fig. 6)')
    # v2(eta)
    for hf, i, kind, cls, what in (('data15_27_1.yaml', 0, 'standard', '0-80%', 'v2(eta), minimum bias (Fig. 15)'),
                                   ('data16_1.yaml', 0, 'c2', '20-70%', 'v2{2}(eta), 20-70% (Fig. 16)'),
                                   ('data16_1.yaml', 1, 'c4', '20-70%', 'v2{4}(eta), 20-70% (Fig. 16)')):
        t = table(hf)
        dv = t['dependent_variables'][i]
        r = rows_of(dv)
        ax = pt_axis(t['independent_variables'][0]['values'], r, cap=0.2)
        kr = {'eta': ax, 'pT': {'min': 0.15, 'max': 2.0}}
        docs.append(record(hf, i, kind, 'charged', [cls], kr, 'differential_eta', label(hf, dv),
                           what + ', 0.15 < pT < 2 GeV/c, TPC and FTPCs', scale(dv, r)))
    # integrated v2 vs centrality
    for hf, cols in (('data29a.yaml', {0: 'standard', 1: 'sp', 2: 'etasubs', 3: 'randomsubs', 4: 'c2', 5: 'c4'}),
                     ('data30.yaml', {0: 'c6'})):
        t = table(hf)
        classes_all = ['%g-%g%%' % tuple(float(x) for x in re.split(r'\s*-\s*', str(v['value']).strip()))
                       for v in t['independent_variables'][0]['values']]
        for i, kind in cols.items():
            dv = t['dependent_variables'][i]
            r = [j for j in rows_of(dv) if not (float(dv['values'][j]['value']) == 0.0)]
            eta = (-1.0, 1.0) if kind == 'sp' else TPC_ETA
            kr = {'eta': {'min': eta[0], 'max': eta[1]}, 'pT': {'min': 0.15, 'max': 2.0}}
            docs.append(record(hf, i, kind, 'charged', [classes_all[j] for j in r], kr, 'integrated', label(hf, dv),
                               f'charged-hadron v2 integrated over 0.15 < pT < 2 GeV/c vs centrality '
                               f'({sub[hf].get("location", "")})', scale(dv, r),
                               r if r != list(range(len(classes_all))) else None))
    out = 'data/flow_vn/star_flow_vn_azimuthal_anisotropy_AuAu_200gev_660793.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;',
          dict(collections.Counter((d['method'], d.get('cumulant_order'), d['dimensionality']) for d in docs)))


if __name__ == '__main__':
    main()

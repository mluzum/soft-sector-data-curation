#!/usr/bin/env python3
"""Curate PHENIX v2 in Au+Au and Cu+Cu at 62.4 and 200 GeV (PPG124), INSPIRE 1332240.

A. Adare et al. (PHENIX), "Systematic study of azimuthal anisotropy in
Cu+Cu and Au+Au collisions at sqrt(sNN) = 62.4 and 200 GeV", PRC 92,
034913 (2015), arXiv:1412.1043.

From the paper:
- tracks in the central arms, |eta| < 0.35; pi (pi+ + pi-), K (K+ + K-),
  p (p + pbar) identified with the TOF and EMCal (Sec. II C);
- event plane: the two BBCs (3.0 < |eta| < 3.9) combined, every
  particle; centrality from the BBC charge;
- minimum bias: 0-92% (Au+Au 200), 0-83% (Au+Au 62.4), 0-88% (Cu+Cu);
- errors: stat. and sys.

System and energy of each table from its HEPData keywords (Fig. 7b
lists both systems; its description and Fig. 7 say Au+Au 62.4).

Records (public): charged-hadron v2(pT) (Figs. 6, 7, 8c), integrated
v2 vs centrality in 0.2-1, 1-2, 2-4 GeV/c (Fig. 11), identified
v2(pT) (Figs. 14, 16), with the scalar-product estimator and the BBC
reference.  Integrated v2 from Fig. 11 (top panels), which holds the
values of Figs. 9, 10 (checked) and more classes.  Not curated: Figs.
8a, 8b (Cu+Cu 200 GeV from PRL 98, 162301, a separate record); Fig. 12
(a comparison: Fig. 6 values with other high-pT bins and smaller sys.,
and Fig. 8a); Fig. 15 (repeats Fig. 14's values, checked; two bins have other edges
there, e.g. 2.0-2.3 and 2.3-3.0 for 2.0-2.4 and 2.4-3.0 GeV/c: Fig. 14's kept); v2/epsilon and
the n_q-scaled v2 (Figs. 11 bottom, 17-22).

Run from the repository root:  python scripts/curate_phenix_v2_cuau_auau_ppg124.py
"""
import collections
import glob
import os
import re

import numpy as np
import yaml

RAW = glob.glob('hepdata_raw/records/ins1332240/HEPData-ins1332240-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 92, 034913 (2015), arXiv:1412.1043'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
BBC_REF = {'detector': 'BBC', 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
           'pT': {'min': 0.0, 'max': 100.0},
           'notes': 'Combined (North-South) BBC event plane. Reference pT: all particles.'}
MB = {('AU AU', 200.0): '0-92%', ('AU AU', 62.4): '0-83%', ('CU CU', 200.0): '0-88%', ('CU CU', 62.4): '0-88%'}
SPECIES = {r'$\pi$': 'pi+ + pi-', '$K$': 'K+ + K-', '$p$': 'p + pbar', r'$\pi^+$': 'pi+', r'$\pi^-$': 'pi-',
           '$K^+$': 'K+', '$K^-$': 'K-', r'$\bar{p}$': 'pbar'}
FIGS = re.compile(r'ppg124fig(6|7|8c|11.1|14|15|16)')
DUPLICATE_FIGS = ('ppg124fig15',)


def system(x):
    kw = {k['name']: k['values'] for k in x.get('keywords', [])}
    beams = sorted({r.split('-->')[0].strip().upper() for r in kw['reactions']})
    if len(beams) > 1:                                   # Fig. 7b: Au+Au 62.4 (description, Fig. 7)
        assert 'Au+Au' in x['description'], x['data_file']
        beams = ['AU AU']
    return beams[0], float(kw['cmenergies'][0])


def pt_axis(iv, rows):
    pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
    if all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs[:-1], pairs[1:])):
        return {'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [p[0] for p in pairs] + [pairs[-1][1]]}
    return {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs, seen, dup = [], {}, collections.Counter()
    for x in sub:
        hf = x['data_file']
        if not FIGS.match(hf):
            continue
        beams, energy = system(x)
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        ivname = iv['header']['name']
        for i, dv in enumerate(t['dependent_variables']):
            if dv['header']['name'] != '$v_2$':
                continue                                 # N_part, v2/epsilon columns
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            labels = sorted({e['label'] for v in dv['values'] for e in v.get('errors', [])})
            assert labels == ['stat.', 'sys.'], (hf, labels)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            xs = [(iv['values'][j].get('low'), iv['values'][j].get('high')) for j in rows]
            vals = dict(zip(xs, (round(float(dv['values'][j]['value']), 6) for j in rows)))
            sp = SPECIES[q['Particle']] if 'Particle' in q else 'charged'
            if hf.startswith('ppg124fig16') and sp == 'p + pbar':
                sp = 'p'                                 # Fig. 16: particles and antiparticles apart
            d = {'observable_type': 'flow_vn', 'inspire_id': '1332240', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f"{' '.join(x['description'].split())[:120]} ({hf})",
                 'collision_system': {'energy_gev': energy, 'reaction': f'{beams} --> CHARGED X'}}
            if ivname.startswith('Centrality'):         # integrated v2 vs centrality, one pT window
                lo, hi = (float(v) for v in re.findall(r'[\d.]+', q['$p_T$'])[:2])
                classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
                d.update({'centrality': {'classes': classes, **BBC}, 'particle_id': {'species': sp},
                          'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': {'min': lo, 'max': hi}},
                          'dimensionality': 'integrated'})
                what = f'v2 of {sp} in {lo:g} < pT < {hi:g} GeV/c vs centrality'
                key = (beams, energy, sp, 'int', lo, hi)
            else:
                cls = q['Centrality']
                cls = MB[(beams, energy)] if cls == 'Minimum Bias' else cls.replace(' ', '')
                d.update({'centrality': {'classes': [cls], **BBC}, 'particle_id': {'species': sp},
                          'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': pt_axis(iv, rows)},
                          'dimensionality': 'differential_pT'})
                what = f'v2(pT) of {sp}, {cls}'
                key = (beams, energy, sp, cls)
            d.update({'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'flow_reference': dict(BBC_REF),
                      'notes': {'value': f'{CITE}. {what}, |eta| < 0.35, combined BBC event plane (3.0 < |eta| < '
                                         '3.9); computed with the scalar-product estimator. Errors: stat. and sys.'}})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if key in seen:
                # a repeat: the same values (Fig. 15 gives two bins of Fig.
                # 14 other edges, e.g. 2.0-2.3 for 2.0-2.4 GeV/c)
                assert list(seen[key].values()) == list(vals.values()), ('same selection, other values', hf, key)
                dup[hf] += 1
                continue
            assert not hf.startswith(DUPLICATE_FIGS), ('a repeat figure holds new data', hf, key)
            seen[key] = vals
            docs.append(d)
    old = 'data/flow_vn/phenix_flow_vn_azimuthal_anisotropy_CuCu_200gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/phenix_flow_vn_v2_AuAu_CuCu_62_200gev_ins1332240.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(
        (d['collision_system']['reaction'][:5], d['collision_system']['energy_gev'], d['particle_id']['species'])
        for d in docs)))
    print('repeats skipped:', dict(dup))


if __name__ == '__main__':
    main()

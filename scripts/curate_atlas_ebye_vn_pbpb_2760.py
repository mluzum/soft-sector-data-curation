#!/usr/bin/env python3
"""Curate ATLAS event-by-event flow harmonics in Pb-Pb at 2.76 TeV, INSPIRE 1233359.

G. Aad et al. (ATLAS), "Measurement of the distributions of event-by-event
flow harmonics in lead-lead collisions at sqrt(sNN) = 2.76 TeV with the
ATLAS detector at the LHC", JHEP 11 (2013) 183, arXiv:1305.2942.

From the paper:
- tracks |eta| < 2.5 with pT > 0.5, 0.5-1 and > 1 GeV/c, weighted by the
  inverse efficiency; centrality from the FCal sum E_T (3.2 < |eta| < 4.9):
  0-1, 1-2, 2-3, 3-4, 4-5, 5-10, ..., 65-70% (Table 1 gives <N_part> per
  class, the abscissa of the other tables);
- the v_n distributions unfolded for the detector response (from the two
  half-ID subevents); their mean, width, relative width; the cumulants
  v_n{2k}(calc) from the moments of the unfolded distributions;
- errors: stat and sys (the calc cumulants: total).

Records (public): v2{2}, v2{4}, v2{6}, v2{8} for the three pT ranges and v3{2,
4, 6, 8} for pT > 0.5 GeV/c (calc, Tables 36, 39, 44, 47), v3{2} for 0.5-1
and > 1 GeV/c and v4{2} for the three ranges (sqrt(<v_n^2>), Tables 22, 28,
18, 24, 30), computed as the cumulants of the acceptance (the unfolding
removes the finite-multiplicity resolution, as the cumulants'
self-correlation removal does); the classes from the <N_part> of Table 1.
v_n{6}/v_n{4} and v_n{8}/v_n{4} (Tables 38, 41, 46, 49) as combinations of
those cumulants from the same events (cumulant_combination).  Staged
(--staged OUT): <v_n>, sigma(v_n), sigma(v_n)/<v_n> (Tables 2-10: moments of
the |v_n| distribution) and the unfolded p(v_n) (Tables 52-201).  Not curated: Glauber values and eccentricities (Tables 1,
11-13), v_n(EP) of Ref. [PRC 86, 014907] and its ratios (Tables 14-31),
sqrt(<v2^2>) and sqrt(<v3^2>) at pT > 0.5 where the calc v{2} repeats them,
Bessel-Gaussian fit parameters and the F estimators (Tables 32-35, 37, 40,
42, 43, 45, 48, 50, 51).  The old record read the Bessel-Gaussian v2(RP)
as event-plane v2.

Run from the repository root:
    python scripts/curate_atlas_ebye_vn_pbpb_2760.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1233359/HEPData-ins1233359-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'G. Aad et al. (ATLAS), JHEP 11 (2013) 183, arXiv:1305.2942'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
ETA = {'min': -2.5, 'max': 2.5}
PT = {'> 0.5': (0.5, 100.0), '0.5-1.0': (0.5, 1.0), '> 1': (1.0, 100.0)}
OLD = 'data/flow_vn/atlas_flow_vn_distributions_of_event-by-event_flow_harmonics_PbPb_2760gev.yaml'
OUT = 'data/flow_vn/atlas_flow_vn_ebye_cumulants_PbPb_2760gev_ins1233359.yaml'
CALC = ('Table36.yaml', 'Table39.yaml', 'Table44.yaml', 'Table47.yaml')
SQRT = ('Table18.yaml', 'Table22.yaml', 'Table24.yaml', 'Table28.yaml', 'Table30.yaml')
MOMENTS = tuple(f'Table{k}.yaml' for k in range(2, 11))
RATIOS = {'Table38.yaml': 2, 'Table41.yaml': 3, 'Table46.yaml': 2, 'Table49.yaml': 2}
BLANK = ('', '-', '--', 'none')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def npart_classes():
    t = load('Table1.yaml')
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    out = {}
    for v, n in zip(iv['values'], dv['values']):
        cls = f"{float(v['low']):g}-{float(v['high']):g}%"
        if cls != '0-5%':
            out[float(n['value'])] = cls
    return out


def base(hf, i, h, n, order, classes, pt, note, rows, nrows):
    d = {'observable_type': 'flow_vn', 'inspire_id': '1233359', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': f'{note.split(".")[0]} ({hf})',
         'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
         'particle_id': {'species': 'charged'}, 'value_type': h, 'method': 'cumulant', 'cumulant_order': order,
         'harmonic': n, 'centrality': {'classes': classes, **FCAL}, 'dimensionality': 'integrated',
         'kinematic_range': {'eta': dict(ETA), 'pT': {'min': pt[0], 'max': pt[1]}},
         'notes': {'value': f'{CITE}. {note}'}}
    if rows != list(range(nrows)):
        d['table_rows'] = rows
    return d


def table_info(t, cmap):
    iv, dv0 = t['independent_variables'][0], t['dependent_variables'][0]
    q = {qq['name']: str(qq.get('value')) for qq in dv0.get('qualifiers') or []}
    npart = [float(v['value']) for v in iv['values']]
    return [cmap[x] for x in npart], PT[q['PT'].strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    cmap = npart_classes()
    public, staged = [], []
    for hf in CALC + SQRT + MOMENTS + tuple(RATIOS):
        t = load(hf)
        classes_all, pt = table_info(t, cmap)
        win = f"pT {'> %g' % pt[0] if pt[1] > 50 else '%g-%g' % pt} GeV/c"
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            classes = [classes_all[j] for j in rows]
            n = int(re.search(r'V(\d)', h).group(1))
            if hf in CALC:
                m = re.fullmatch(r'V\d\(calc\)\{(\d)\}', h)
                if not m:
                    continue                                              # V_n(RP): a Bessel-Gaussian fit parameter
                order = int(m.group(1))
                public.append(base(hf, i, h, n, order, classes, pt,
                                   f'v{n}{{{order}}} of charged particles vs centrality (FCal), |eta| < 2.5, {win}. From '
                                   'the moments of the unfolded event-by-event v_n distribution. Errors: total.',
                                   rows, len(classes_all)))
            elif hf in SQRT:
                if not h.startswith('SQRT'):
                    continue                                              # <v_n> and v_n(EP)
                public.append(base(hf, i, h, n, 2, classes, pt,
                                   f'v{n}{{2}} = sqrt(<v{n}^2>) of charged particles vs centrality (FCal), |eta| < 2.5, '
                                   f'{win}. From the unfolded event-by-event v_n distribution. Errors: stat and sys.',
                                   rows, len(classes_all)))
            elif hf in MOMENTS:
                d = base(hf, i, h, n, 2, classes, pt,
                         f'{h} of charged particles vs centrality, |eta| < 2.5, {win}. Moment of the unfolded '
                         'event-by-event |v_n| distribution. Errors: stat and sys.', rows, len(classes_all))
                d.update({'observable': h, 'staging_reason': 'moment of the event-by-event |v_n| distribution (mean, '
                                                             'width): not computed'})
                staged.append(d)
            else:
                a, b = (int(x) for x in re.findall(r'\{(\d)\}', h))
                d = base(hf, i, h, n, max(a, b), classes, pt,
                         f'{h} of charged particles vs centrality, |eta| < 2.5, {win}. From the moments of the '
                         'unfolded distribution; computed from the cumulants of the same events. Errors: total.',
                         rows, len(classes_all))
                d['cumulant_combination'] = {'kind': 'ratio', 'orders': [a, b]}
                public.append(d)
    for k in range(52, 202):
        hf = f'Table{k}.yaml'
        t = load(hf)
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
        cls, pt = f'{lo:g}-{hi:g}%', PT[q['PT'].strip()]
        n = int(iv['header']['name'][1])
        x = [float(v['value']) for v in iv['values']]
        d = base(hf, 0, dv['header']['name'], n, 2, [cls], pt,
                 f'Unfolded p(v{n}) of charged particles, {cls}, |eta| < 2.5, pT {q["PT"]} GeV/c, at the v{n} values '
                 'of v_points. Errors: stat and sys.', list(range(len(x))), len(x))
        d.update({'observable': f'p(v{n})', 'v_points': x,
                  'staging_reason': 'event-by-event v_n distribution (unfolded): not computed'})
        staged.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '1233359' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records;', dict(collections.Counter(
        (d['harmonic'], d['cumulant_order'], d['kinematic_range']['pT']['max']) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

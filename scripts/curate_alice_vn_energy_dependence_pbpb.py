#!/usr/bin/env python3
"""Curate ALICE v_n energy dependence and fluctuations in Pb-Pb, INSPIRE 1666817.

S. Acharya et al. (ALICE), "Energy dependence and fluctuations of
anisotropic flow in Pb-Pb collisions at sqrt(sNN) = 5.02 and 2.76 TeV",
JHEP 07 (2018) 103, arXiv:1804.02944.

From the paper (Sec. 2):
- charged tracks |eta| < 0.8, 0.2 < pT < 50 GeV/c; integrated results over
  0.2 < pT < 3 GeV/c; centrality from the V0 amplitude (V0A 2.8 < eta <
  5.1, V0C -3.6 < eta < -1.7);
- Q-cumulants; for pT-differential results the particle at pT is
  correlated with m - 1 particles of the full range 0.2 < pT < 50;
- v_n{2, |Delta eta| > 1}: two-particle cumulant between -0.8 < eta < -0.5
  and 0.5 < eta < 0.8; v_n{2, |Delta eta| > 2}: scalar product of the
  tracks (|eta| < 0.8) with the V0A Q-vector, normalized with the V0C and
  the TPC;
- "All observables are computed in small centrality bins (1%) and then
  integrated ... in wider centrality intervals using the charged particle
  yield in each 1% centrality bin as weight."

Records (public), Pb-Pb 2.76 and 5.02 TeV:
- vs centrality (0.2 < pT < 3): v2{2} (no gap), v_n{2, |Delta eta| > 1}
  (n = 2-6, subevents), v2{4}, v2{6}, v2{8};
- vs pT, 0-5 ... 60-70%: v_n{2, |Delta eta| > 1} (n = 2-6; the reference in
  the other subevent over 0.2 < pT < 50), v_n{2, |Delta eta| > 2} (n = 2-4,
  V0A scalar product), v2{4} (reference 0.2 < pT < 50).
Classes wider than 1% are combined from 1% classes weighted by the charged
yield (centrality.combined_from analysis_classes, analysis_class_width 1).
- vs centrality, combinations of cumulant orders from the same events
  (cumulant_combination): v2{6}/v2{4}, v2{8}/v2{4}, v2{8}/v2{6},
  (v2{4} - v2{6})/11, v2{6} - v2{8} and the skewness estimate gamma1 (Eq. 7,
  with v2{2, |Delta eta| > 1} as in the paper).
Not curated: the cumulants c2{m} (Fig. 10, the v2{m} in another form), the
ratios between energies (Figs. 2, 5), to 20-30% (Fig. 7) and v_n/v_m^(n/m)
(Fig. 8), all ratios of curated tables; the elliptic-power fit parameters
and the P(v2) they define (Figs. 16, 17).  The 16 old records had no
combination, the reference in the record window and v2{2} called v2{2}ab;
the 245 old staged records typed every table as 'fluctuations'.

Run from the repository root:
    python scripts/curate_alice_vn_energy_dependence_pbpb.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1666817/HEPData-ins1666817-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), JHEP 07 (2018) 103, arXiv:1804.02944'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.6, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
COMBINE = {'combined_from': 'analysis_classes', 'analysis_class_width': 1.0, 'combination_weight': 'charged_yield'}
PT_INT = {'min': 0.2, 'max': 3.0}
PT_REF = {'min': 0.2, 'max': 50.0}
V0A_REF = {'detector': 'V0A', 'eta': [{'min': 2.8, 'max': 5.1}], 'pT': {'min': 0.0, 'max': 100.0},
           'notes': 'Q_n of the V0A (2.8 < eta < 5.1), energy deposition as weights; normalized with the V0C and '
                    'the TPC (three subevents).'}
ENERGY = {'276TeV': 2760.0, '502TeV': 5020.0}
OLD = ['data/flow_vn/alice_flow_vn_energy_dependence_and_fluctuations_PbPb_2760_5020gev.yaml',
       'data/flow_vn/alice_flow_harmonic_correlations_fluctuations_of_anisotropic_flow_PbPb_multigev.yaml']
# combinations of cumulant orders (computed from the same events: cumulant_combination)
COMBINED = {'v26Overv24': ('v2{6}/v2{4}', {'kind': 'ratio', 'orders': [6, 4]}),
            'v28Overv24': ('v2{8}/v2{4}', {'kind': 'ratio', 'orders': [8, 4]}),
            'v28Overv26': ('v2{8}/v2{6}', {'kind': 'ratio', 'orders': [8, 6]}),
            'v24Minusv26': ('(v2{4} - v2{6})/11', {'kind': 'difference', 'orders': [4, 6], 'scale': 1.0 / 11.0}),
            'v26Minusv28': ('v2{6} - v2{8}', {'kind': 'difference', 'orders': [6, 8]}),
            'gamma1exp': ('gamma1^exp = -6 sqrt2 v2{4}^2 (v2{4} - v2{6}) / (v2{2, |Delta eta| > 1}^2 - v2{4}^2)^(3/2) '
                          '(Eq. 7)', {'kind': 'gamma1'})}


def load(f):
    return yaml.safe_load(open(os.path.join(RAW, f)))


def centrality(classes):
    c = {'classes': classes, **V0M}
    if any(float(x.rstrip('%').split('-')[1]) - float(x.rstrip('%').split('-')[0]) > 1 + 1e-9 for x in classes):
        c.update(COMBINE)
    return c


def base(hf, i, h, energy, label):
    return {'observable_type': 'flow_vn', 'inspire_id': '1666817', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf})',
            'collision_system': {'energy_gev': energy, 'reaction': 'PB PB --> CHARGED X'},
            'particle_id': {'species': 'charged'}, 'value_type': h}


def subevents(pt):
    return {'particles': 'AB', 'eta_gap': 1.0,
            'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.5}, 'pT': dict(pt)}},
                          'B': {'kinematic_range': {'eta': {'min': 0.5, 'max': 0.8}, 'pT': dict(pt)}}}}


def method(kind, n, differential):
    """Method fields and a description for v_n{2} (no gap), {2,|deta|>1}, {2,|deta|>2}, {4,6,8}."""
    if kind == 'gap1':
        return ({'method': 'cumulant', 'cumulant_order': 2, **subevents(PT_REF if differential else PT_INT)},
                f'v{n}{{2, |Delta eta| > 1}} (subevents -0.8 < eta < -0.5 and 0.5 < eta < 0.8'
                + ('; reference 0.2 < pT < 50 GeV/c in the other subevent)' if differential else ')'))
    if kind == 'gap2':
        return ({'method': 'scalar_product', 'flow_reference': dict(V0A_REF)},
                f'v{n}{{2, |Delta eta| > 2}} (scalar product with the V0A Q-vector, 2.8 < eta < 5.1)')
    order = int(kind)
    f = {'method': 'cumulant', 'cumulant_order': order}
    if differential:
        f['flow_reference'] = {'pT': dict(PT_REF)}
    return f, f'v{n}{{{order}}}' + (' (reference particles 0.2 < pT < 50 GeV/c)' if differential else '')


def parse(stem):
    """Table stem -> (kind, harmonic, energy key) or None."""
    m = re.fullmatch(r'v(\d)2Deta([12])_(276TeV|502TeV)(dif_cen\d+)?', stem)
    if m:
        return ('gap1' if m.group(2) == '1' else 'gap2'), int(m.group(1)), m.group(3), bool(m.group(4))
    m = re.fullmatch(r'v2([2468])_(276TeV|502TeV)(dif_cen\d+)?', stem)
    if m:
        kind = 'nogap' if m.group(1) == '2' else m.group(1)
        return kind, 2, m.group(2), bool(m.group(3))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf in sorted(os.listdir(RAW)):
        stem = hf[:-5]
        p = parse(stem)
        s = re.fullmatch(r'(' + '|'.join(COMBINED) + r')_(276TeV|502TeV)', stem)
        if not p and not s:
            continue
        t = load(hf)
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        h = dv['header']['name']
        if s:
            energy = ENERGY[s.group(2)]
            classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
            what, comb = COMBINED[s.group(1)]
            comb = dict(comb)
            if comb['kind'] == 'gamma1':          # the paper's v2{2, |Delta eta| > 1}
                comb['two_particle'] = subevents(PT_INT)
            d = base(hf, 0, h, energy, f'{what.split(" = ")[0]} vs centrality, {energy:g} GeV')
            cent = centrality(classes)
            d.update({'centrality': cent,
                      'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT_INT)},
                      'harmonic': 2, 'method': 'cumulant', 'cumulant_order': max(comb.get('orders') or [6]),
                      'cumulant_combination': comb, 'dimensionality': 'integrated',
                      'notes': {'value': f'{CITE}. {what} of charged particles vs centrality, |eta| < 0.8, 0.2 < pT < '
                                         f'3 GeV/c, {energy:g} GeV, from the same events'
                                         + ('; classes wider than 1% combined from 1% classes weighted by the charged '
                                            'yield' if 'combined_from' in cent else '')
                                         + '. Errors: stat and sys.'}})
            public.append(d)
            continue
        kind, n, ekey, differential = p
        energy = ENERGY[ekey]
        if differential:
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            cls = q['CENTRALITY'].rstrip('%') + '%'
            pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
            kr = {'eta': {'min': -0.8, 'max': 0.8},
                  'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
            classes, dim = [cls], 'differential_pT'
        else:
            classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
            kr = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': dict(PT_INT)}
            dim = 'integrated'
        if kind == 'nogap':
            f, desc = {'method': 'cumulant', 'cumulant_order': 2}, 'v2{2} (all pairs in |eta| < 0.8, no gap)'
        else:
            f, desc = method(kind, n, differential)
        what = (f"{desc} vs pT, {classes[0]}" if differential
                else f"{desc} vs centrality, 0.2 < pT < 3 GeV/c")
        d = base(hf, 0, h, energy, f'{what}, {energy:g} GeV')
        cent = centrality(classes)
        d.update({'centrality': cent, 'kinematic_range': kr, 'harmonic': n, **f, 'dimensionality': dim,
                  'notes': {'value': f'{CITE}. {what}, charged particles |eta| < 0.8, {energy:g} GeV'
                                     + ('; classes wider than 1% combined from 1% classes weighted by the charged '
                                        'yield' if 'combined_from' in cent else '')
                                     + ('; computed with the scalar-product estimator' if kind == 'gap2' else '')
                                     + '. Errors: stat and sys.'}})
        public.append(d)
    for f in OLD:
        if os.path.exists(f):
            assert all(str(x['inspire_id']) == '1666817' for x in yaml.safe_load_all(open(f)) if x), f
            os.remove(f)
    out = 'data/flow_vn/alice_flow_vn_energy_dependence_PbPb_2760_5020gev_ins1666817.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['collision_system']['energy_gev'], d['dimensionality'], d.get('cumulant_order') or 'SP',
         d.get('particles')) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

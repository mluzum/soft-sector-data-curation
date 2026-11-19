#!/usr/bin/env python3
"""Curate ALICE <pT> and spectra in ultra-central Pb-Pb at 5.02 TeV, INSPIRE 2933773.

ALICE Collaboration, "Study of <pT> and its higher moments, and extraction
of the speed of sound in Pb-Pb collisions with ALICE", JHEP 11 (2025) 076,
arXiv:2506.10394.

From the paper (Secs. 2-3, Table 1):
- nine centrality estimators (Table 1): N_ch of TPC tracks (0.15 <= pT <
  50 GeV) in |eta| <= 0.8 (I) or 0.5 <= |eta| < 0.8 (II); E_T, the summed
  transverse mass of those tracks with the pion mass (Sec. 2), in the same
  windows (III, IV); SPD tracklets (no pT selection, from ~0.03 GeV) in
  |eta| <= 0.8 (V), 0.5 <= |eta| < 0.8 (VI), 0.3 < |eta| < 0.6 (VII), 0.7
  <= |eta| < 1 (VIII); the V0 amplitude, -3.7 < eta < -1.7 and 2.8 < eta <
  5.1 (IX).  Fig. 8 adds N_ch in 0.5 <= |eta| <= 0.8 with 0.45 <= pT <= 50;
- <pT> and <dN_ch/deta> from the corrected spectra over 0 <= pT <= 10 GeV
  (extrapolated below 0.15), in |eta| <= 0.8 (I, III, V, IX) or |eta| <=
  0.3 (II, IV, VI-VIII, and Fig. 8), each over its value in the estimator's
  0-5% class (Sec. 3);
- primary charged particles (weak-decay daughters excluded).
Classes: narrow centrality intervals of each estimator, given by the
centrality percentiles of Fig. 1 (25 for the midrapidity estimators: 0-0.005,
0.005-0.01, 0.01-0.02, ..., 0.05-0.075, 0.075-0.1, 0.1-0.2, ..., 0.9-1, 1-1.5,
..., 4.5-5%; 18 for the V0: 0-0.1, ..., 0.9-1, 1-1.5, ..., 4.5-5%; the
points are the interval centres), in the order of Fig. 4's points
(descending multiplicity), which Fig. 2 labels confirm (0-0.005, 0.03-0.04,
0.1-0.2, 0.9-1%; V0 0-0.1, 0.4-0.5, 0.7-0.8%).

Records (public): <pT>/<pT>^{0-5%} per class for each estimator (Fig. 4a-i,
Fig. 8), and the normalized spectrum ratios (d2N/(<dN_ch/deta> deta dpT)
over the same in 0-5%, Fig. 2a-d: II, IV, VI, IX at four classes each):
centrality ratios to the 0-5% class (ratio_type centrality_ratio,
denominator_class 0-5%) whose sides hold no values (the results are
published only as ratios; experimental_values: false), the spectra each
over their 0-10 GeV yield (side_normalization).  The <dN_ch/deta> ratios
(the abscissae of Figs. 4, 7, 8) have no uncertainties and are not records.
Not curated: <N_part> (Fig. 1, a Glauber/ZDC quantity), the event-fraction
distributions (Fig. 3), c_s^2 (Fig. 6, a fit), and the higher moments of
[pT] (Fig. 7: their multiplicity intervals are not published -- the points
do not match the Fig. 1 intervals).

Run from the repository root:  python scripts/curate_alice_meanpt_ucc_pbpb_5020.py
"""
import glob
import os
import re

import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

RAW = glob.glob('hepdata_raw/records/ins2933773/HEPData-ins2933773-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'ALICE Collaboration, JHEP 11 (2025) 076, arXiv:2506.10394'
OUT_RATIO = 'data/observable_ratios/alice_observable_ratios_meanpT_spectra_ucc_PbPb_5020gev_ins2933773.yaml'
OUT_MPT = 'data/mean_pT/alice_mean_pT_ucc_sides_PbPb_5020gev_ins2933773.yaml'
OUT_SPEC = 'data/particle_spectra/alice_particle_spectra_ucc_sides_PbPb_5020gev_ins2933773.yaml'
SYSTEM = {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'}


def band(lo, hi):
    return [{'min': -hi, 'max': -lo}, {'min': lo, 'max': hi}]


# label: (estimator, eta windows, pT, sum, measurement |eta|), Table 1
ESTIMATORS = {
    'I': ({'name': 'TPC', 'observable': 'Nch', 'detector': 'TPC'}, [{'min': -0.8, 'max': 0.8}], (0.15, 50.0),
          'count', 0.8),
    'II': ({'name': 'TPC', 'observable': 'Nch', 'detector': 'TPC'}, band(0.5, 0.8), (0.15, 50.0), 'count', 0.3),
    'III': ({'name': 'TPC', 'observable': 'ET', 'detector': 'TPC'}, [{'min': -0.8, 'max': 0.8}], (0.15, 50.0),
            'transverse_mass', 0.8),
    'IV': ({'name': 'TPC', 'observable': 'ET', 'detector': 'TPC'}, band(0.5, 0.8), (0.15, 50.0),
           'transverse_mass', 0.3),
    'V': ({'name': 'SPD', 'observable': 'Nch', 'detector': 'SPD'}, [{'min': -0.8, 'max': 0.8}], None, 'count', 0.8),
    'VI': ({'name': 'SPD', 'observable': 'Nch', 'detector': 'SPD'}, band(0.5, 0.8), None, 'count', 0.3),
    'VII': ({'name': 'SPD', 'observable': 'Nch', 'detector': 'SPD'}, band(0.3, 0.6), None, 'count', 0.3),
    'VIII': ({'name': 'SPD', 'observable': 'Nch', 'detector': 'SPD'}, band(0.7, 1.0), None, 'count', 0.3),
    'IX': ({'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
           [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}], None, 'count', 0.8),
    'Fig8': ({'name': 'TPC', 'observable': 'Nch', 'detector': 'TPC'}, band(0.5, 0.8), (0.45, 50.0), 'count', 0.3),
}
EST_TEXT = {
    'I': 'N_ch of TPC tracks (0.15 <= pT < 50 GeV) in |eta| <= 0.8 (I)',
    'II': 'N_ch of TPC tracks (0.15 <= pT < 50 GeV) in 0.5 <= |eta| < 0.8 (II)',
    'III': 'E_T, the summed transverse mass of TPC tracks with the pion mass (0.15 <= pT < 50 GeV), in |eta| <= 0.8 (III)',
    'IV': 'E_T, the summed transverse mass of TPC tracks with the pion mass (0.15 <= pT < 50 GeV), in 0.5 <= |eta| < 0.8 (IV)',
    'V': 'SPD tracklets (no pT selection) in |eta| <= 0.8 (V)',
    'VI': 'SPD tracklets (no pT selection) in 0.5 <= |eta| < 0.8 (VI)',
    'VII': 'SPD tracklets (no pT selection) in 0.3 < |eta| < 0.6 (VII)',
    'VIII': 'SPD tracklets (no pT selection) in 0.7 <= |eta| < 1 (VIII)',
    'IX': 'the V0 amplitude, -3.7 < eta < -1.7 and 2.8 < eta < 5.1 (IX)',
    'Fig8': 'N_ch of TPC tracks with 0.45 <= pT <= 50 GeV in 0.5 <= |eta| <= 0.8',
}
FIG4 = {'figure_4a.yaml': 'I', 'figure_4b.yaml': 'II', 'figure_4c.yaml': 'III', 'figure_4d.yaml': 'IV',
        'figure_4e.yaml': 'V', 'figure_4f.yaml': 'VI', 'figure_4g.yaml': 'VII', 'figure_4h.yaml': 'VIII',
        'figure_4i.yaml': 'IX', 'figure_8.yaml': 'Fig8'}
FIG2 = {'figure_2a.yaml': 'II', 'figure_2b.yaml': 'IV', 'figure_2c.yaml': 'VI', 'figure_2d.yaml': 'IX'}
# the estimator text each table's description must contain
DESC_KEY = {'I': r'N_{\mathrm{ch}}$ $\in$ $|\eta| \leq 0.8', 'II': r'0.5 \leq |\eta| \leq 0.8',
            'III': r'E_{\mathrm{T}}$ $\in$ $|\eta| \leq 0.8', 'IV': r'E_{\mathrm{T}}',
            'V': r'N_{\mathrm{tracklets}}$ $\in$ $|\eta| \leq 0.8', 'VI': r'N_{\mathrm{tracklets}}',
            'VII': r'0.3 \leq |\eta| \leq 0.6', 'VIII': r'0.7 \leq |\eta| \leq 1', 'IX': r'-3.7<\eta<-1.7',
            'Fig8': r'0.45 \leq p_{\mathrm{T}}'}


def load(name):
    return yaml.load(open(os.path.join(RAW, name)), Loader=Loader)


def classes_from_centres(centres):
    """Contiguous intervals from 0 whose centres are the given points."""
    out, lo = [], 0.0
    for c in centres:
        hi = round(2 * c - lo, 6)
        assert hi > lo and abs(0.5 * (lo + hi) - c) < 1e-9, (lo, c)
        out.append(f'{lo:g}-{hi:g}%')
        lo = hi
    assert abs(lo - 5.0) < 1e-9, lo
    return out


def centrality(label, classes):
    est, eta, pt, kind, _ = ESTIMATORS[label]
    est = dict(est)
    if kind != 'count':
        est['sum'] = kind
    c = {'classes': list(classes), 'estimator': est, 'eta': [dict(e) for e in eta], 'percentiles': True}
    if pt is not None:
        c['pT'] = {'min': pt[0], 'max': pt[1]}
    c['notes'] = (f'Estimator {EST_TEXT[label]} (Table 1); narrow centrality intervals of Fig. 1 (interval centres), '
                  'each point over the estimator\'s 0-5% class.')
    return c


def main():
    t1a, t1b = load('figure_1a.yaml'), load('figure_1b.yaml')
    mid = classes_from_centres([v['value'] for v in t1a['independent_variables'][0]['values']])
    fwd = classes_from_centres([v['value'] for v in t1b['independent_variables'][0]['values']])
    subs = {s['data_file']: s for s in yaml.load_all(open(os.path.join(RAW, 'submission.yaml')), Loader=Loader)
            if s and 'data_file' in s}
    ratios, mpt_sides, spec_sides = [], [], []
    for hf, label in FIG4.items():
        desc = ' '.join(subs[hf]['description'].split())
        assert DESC_KEY[label].replace(" ", "") in desc.replace(" ", ""), (hf, desc)
        t = load(hf)
        dv = t['dependent_variables'][0]
        cls = fwd if label == 'IX' else mid
        assert len(dv['values']) == len(cls), (hf, len(dv['values']))
        x = [v['value'] for v in t['independent_variables'][0]['values']]
        assert x == sorted(x, reverse=True), hf
        etarap = {q['name']: q['value'] for q in dv.get('qualifiers', [])}['ETARAP']
        meas = ESTIMATORS[label][4]
        assert f'{meas}' in etarap, (hf, etarap, meas)
        kr = {'eta': {'min': -meas, 'max': meas}, 'pT': {'min': 0.0, 'max': 10.0}}
        side = {'observable_type': 'mean_pT', 'inspire_id': '2933773', 'version': VERSION,
                'experimental_values': False,
                'hepdata_table_label': f'<pT> of charged particles, |eta| <= {meas:g}, estimator {label} (a side of '
                                       f'{hf}; only the ratio to 0-5% is published)',
                'collision_system': dict(SYSTEM), 'centrality': centrality(label, cls),
                'particle_id': {'species': 'charged'}, 'kinematic_range': kr, 'observable': 'mean_pT',
                'value_type': 'mean_pT', 'dimensionality': 'integrated',
                'notes': {'value': f'{CITE}, Sec. 3: <pT> from the corrected spectra over 0 <= pT <= 10 GeV '
                                   f'(extrapolated below 0.15 GeV), primary charged particles in |eta| <= {meas:g}.'}}
        mpt_sides.append(side)
        ref = f'{OUT_MPT}#{len(mpt_sides) - 1}'
        den = yaml.safe_load(yaml.safe_dump(side))          # the reference class, its own side
        den['centrality']['classes'] = ['0-5%']
        den['hepdata_table_label'] = den['hepdata_table_label'].replace('(a side of', '(the 0-5% side of')
        mpt_sides.append(den)
        den_ref = f'{OUT_MPT}#{len(mpt_sides) - 1}'
        ratios.append({
            'observable_type': 'observable_ratios', 'inspire_id': '2933773', 'version': VERSION,
            'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': f'<pT>/<pT>^(0-5%), estimator {label} ({subs[hf]["name"]})',
            'collision_system': dict(SYSTEM), 'centrality': centrality(label, cls),
            'particle_id': {'species': 'charged'}, 'kinematic_range': kr,
            'observable': 'mean_pT', 'value_type': dv['header']['name'],
            'ratio_type': 'centrality_ratio', 'numerator': 'class', 'denominator': '0-5%',
            'numerator_system': 'Pb+Pb', 'denominator_system': 'Pb+Pb',
            'numerator_record': ref, 'denominator_record': den_ref, 'denominator_class': '0-5%',
            'dimensionality': 'integrated',
            'notes': {'value': f'{CITE}, {subs[hf]["name"]}. <pT>/<pT>^(0-5%) of charged particles in |eta| <= '
                               f'{meas:g} (0 <= pT <= 10 GeV), events classified by {EST_TEXT[label]}. The '
                               'abscissa, <dN_ch/deta>/<dN_ch/deta>^(0-5%), is tabulated without uncertainties. '
                               'Errors: stat and syst.'}})
    for hf, label in FIG2.items():
        desc = ' '.join(subs[hf]['description'].split())
        assert DESC_KEY[label].replace(" ", "") in desc.replace(" ", ""), (hf, desc)
        t = load(hf)
        bins = [(v['low'], v['high']) for v in t['independent_variables'][0]['values']]
        edges = [bins[0][0]] + [b[1] for b in bins]
        assert all(a[1] == b[0] for a, b in zip(bins[:-1], bins[1:])), hf
        cls_all = fwd if label == 'IX' else mid
        meas = ESTIMATORS[label][4]
        den_ref = None
        for k, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq['value']) for qq in dv.get('qualifiers', [])}
            cl = re.sub(r'[$ ]', '', q['Centrality (%)']) + '%'
            assert cl in cls_all, (hf, cl)
            assert f'{meas}' in q['ETARAP'], (hf, q['ETARAP'], meas)
            kr = {'eta': {'min': -meas, 'max': meas}, 'pT': {'min': edges[0], 'max': edges[-1], 'bins': edges}}
            side = {'observable_type': 'particle_spectra', 'inspire_id': '2933773', 'version': VERSION,
                    'experimental_values': False,
                    'hepdata_table_label': f'charged-particle spectrum, |eta| <= {meas:g}, estimator {label} (a side '
                                           f'of {hf}; only the ratio to 0-5% is published)',
                    'collision_system': dict(SYSTEM), 'centrality': centrality(label, [cl]),
                    'particle_id': {'species': 'charged'}, 'kinematic_range': kr, 'observable': 'invariant_yield',
                    'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
                    'notes': {'value': f'{CITE}: primary charged particles in |eta| <= {meas:g}; the ratio takes each '
                                       'side over its yield in 0 <= pT <= 10 GeV, <dN_ch/deta> (extrapolated below '
                                       '0.15 GeV in the paper).'}}
            spec_sides.append(side)
            ref = f'{OUT_SPEC}#{len(spec_sides) - 1}'
            if den_ref is None:                             # the 0-5% spectrum of this estimator
                den = yaml.safe_load(yaml.safe_dump(side))
                den['centrality']['classes'] = ['0-5%']
                den['hepdata_table_label'] = den['hepdata_table_label'].replace('(a side', '(the 0-5% side')
                spec_sides.append(den)
                den_ref = f'{OUT_SPEC}#{len(spec_sides) - 1}'
            ratios.append({
                'observable_type': 'observable_ratios', 'inspire_id': '2933773', 'version': VERSION,
                'hepdata_file': hf, 'dv_index': k,
                'hepdata_table_label': f'normalized spectrum {cl} / 0-5%, estimator {label} ({subs[hf]["name"]})',
                'collision_system': dict(SYSTEM), 'centrality': centrality(label, [cl]),
                'particle_id': {'species': 'charged'}, 'kinematic_range': kr,
                'observable': 'invariant_yield', 'value_type': dv['header']['name'],
                'ratio_type': 'centrality_ratio', 'numerator': cl, 'denominator': '0-5%',
                'numerator_system': 'Pb+Pb', 'denominator_system': 'Pb+Pb',
                'numerator_record': ref, 'denominator_record': den_ref, 'denominator_class': '0-5%',
                'side_normalization': {'pT': {'min': 0.0, 'max': 10.0}},
                'dimensionality': 'differential_pT',
                'notes': {'value': f'{CITE}, {subs[hf]["name"]}. d2N/(<dN_ch/deta> deta dpT) in {cl} over the same '
                                   f'in 0-5%, charged particles in |eta| <= {meas:g}, events classified by '
                                   f'{EST_TEXT[label]}; each spectrum over its yield in 0 <= pT <= 10 GeV '
                                   '(side_normalization). Errors: stat (the paper shows no systematic ones: they '
                                   'largely cancel).'}})
    for out, docs in ((OUT_MPT, mpt_sides), (OUT_SPEC, spec_sides), (OUT_RATIO, ratios)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()

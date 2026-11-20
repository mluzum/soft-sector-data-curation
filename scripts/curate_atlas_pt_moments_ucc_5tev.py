#!/usr/bin/env python3
"""Curate ATLAS [pT] moments in ultra-central Pb+Pb and Xe+Xe, INSPIRE 2806288.

ATLAS Collaboration, "Disentangling sources of momentum fluctuations in
Xe+Xe and Pb+Pb collisions with the ATLAS detector", PRL 133 (2024) 252301,
arXiv:2407.06413.

From the paper:
- charged particles 0.5 < pT < 5 GeV, |eta| < 2.5, efficiency- and
  fake-corrected; N_ch = sum of the track weights in the same window;
- per event [pT] and the central moments c_2, c_3 over distinct pairs and
  triplets, deviations from <[pT]>; averaged over events in unit N_ch^rec
  intervals (reconstructed tracks), then over wider intervals (Eq. 1 and
  the text before it); k2 = <c2>/<[pT]>^2, k3 = <c3>/<[pT]>^3, gamma =
  <c3>/<c2>^(3/2), Gamma = <c3><[pT]>/<c2>^2;
- Fig. 3 (and Suppl. Fig. 4b, gamma): each observable over its value at 5%
  centrality, against N_ch / N_ch^{5%}, N_ch^{5%} the N_ch at 5% centrality
  (FCal Sigma E_T; 4230 in Pb+Pb, 2425 in Xe+Xe).

Model counterpart: classes in units of the estimator count at the 5% point
of its own distribution (centrality.multiplicity_bins.normalized_to; the
N_ch^rec percentiles of ATLAS 1728935 match the Sigma E_T ones within 0.1
point there), the count Poisson(0.6 N) in the same window (ATLAS's
reconstructed tracks, efficiency 0.6 as for 1728935); the class edges
midway between the tabulated N_ch / N_ch^{5%} (their x uncertainties are
systematic, not bin edges; the outermost mirror their neighbour); the
reference class the interval of the finest binning around 1 (Fig. 3a:
+-0.01065 in Pb+Pb), for every observable (ATLAS normalizes k3, Gamma and
gamma, binned 2.5 times wider, to the value at 5% as well: their first
point, at 0.995, is 1.002).  The moments at fixed multiplicity within each
class (multiplicity_baseline: ATLAS's unit intervals); <[pT]> the pooled
mean (the class's events differ in N_ch by +-1%).

N_ch^{5%} as a percentile of the cross section: Suppl. Fig. 3 (the
distribution of N_ch / N_ch^{5%}, arbitrary normalization) and Aux. Fig.
2b (the FCal centrality of events at a given N_ch / N_ch^{5%}) fix it.
Where the distribution is flat in centrality (15-50%) the mean FCal
centrality at fixed N_ch is the N_ch percentile itself, and Aux. Fig. 2b
equals the distribution's cumulative times one constant there (0.4350 in
Pb+Pb, 0.4147 in Xe+Xe, within 0.1%): the fraction of the cross section per
unit of the histogram.  With it, 5.063% (Pb+Pb) and 5.099% (Xe+Xe) of the
cross section lie above N_ch / N_ch^{5%} = 1: the percentile the model's
reference count takes (normalized_to).  (In the tail Aux. Fig. 2b levels
off -- the regression of FCal on N_ch -- and is not the cumulative: 0.35%
at 1.24, where the distribution gives 0.16%.)

Records (public): centrality ratios to the reference class
(observable_ratios), sides without values (the results are published only
normalized); the distribution of N_ch / N_ch^{5%} (Suppl. Fig. 3) as the
fraction of the cross section per class (multiplicity_distribution,
event_fraction; value_scale the histogram's normalization above), the bins
within 0-80% of the cross section.  Not curated: the absolute N_ch
dependence (Figs. 1-2: the model's multiplicity scale differs, Fig. 3
carries the same information normalized), the [pT] distribution and
correlations (Fig. 1a, Suppl. Fig. 2), the FCal 0-1% events in N_ch slices
(Fig. 4: classes of two estimators), and the auxiliary figures.

Run from the repository root:  python scripts/curate_atlas_pt_moments_ucc_5tev.py
"""
import glob
import os

import numpy as np
import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

RAW = glob.glob('hepdata_raw/records/ins2806288/HEPData-ins2806288-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'ATLAS Collaboration, PRL 133 (2024) 252301, arXiv:2407.06413'
OUT_RATIO = 'data/observable_ratios/atlas_observable_ratios_pT_moments_ucc_PbPb_XeXe_ins2806288.yaml'
OUT_MOM = 'data/mean_pT_fluctuations/atlas_mean_pT_fluctuations_moments_sides_PbPb_XeXe_ins2806288.yaml'
OUT_MPT = 'data/mean_pT/atlas_mean_pT_ucc_sides_PbPb_XeXe_ins2806288.yaml'
OUT_FRAC = 'data/multiplicity_distribution/atlas_multiplicity_distribution_event_fraction_PbPb_XeXe_ins2806288.yaml'
SYSTEMS = {'pb': ({'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'}, 'Pb+Pb', 4230),
           'xe': ({'energy_gev': 5440.0, 'reaction': 'XE XE --> CHARGED X'}, 'Xe+Xe', 2425)}
WINDOW = {'eta': {'min': -2.5, 'max': 2.5}, 'pT': {'min': 0.5, 'max': 5.0}}
# table: (quantity: mean_pT or a pT_moment, label)
TABLES = {'figure_3a': ('mean_pT', '<[pT]>/<[pT]>^{5%}'), 'figure_3b': ('k2', 'k2/k2^{5%}'),
          'figure_3c': ('k3', 'k3/k3^{5%}'), 'figure_3d': ('Gamma', 'Gamma/Gamma^{5%}'),
          'suppl_figure_4_b': ('gamma', 'gamma/gamma^{5%}')}


def load(name):
    return yaml.load(open(os.path.join(RAW, name)), Loader=Loader)


def fmt(x):
    return np.format_float_positional(float(x), precision=5, unique=False, fractional=False, trim='-')


def classes(xs):
    """Classes midway between the points (the outermost mirror their neighbour)."""
    xs = np.asarray(xs, dtype=float)
    mids = 0.5 * (xs[:-1] + xs[1:])
    edges = np.concatenate([[xs[0] - (mids[0] - xs[0])], mids, [xs[-1] + (xs[-1] - mids[-1])]])
    for x, lo, hi in zip(xs, edges[:-1], edges[1:]):
        assert lo < x < hi, (lo, x, hi)
    return [f'{fmt(lo)}-{fmt(hi)}' for lo, hi in zip(edges[:-1], edges[1:])]


def n5_scale(sk):
    """(percent of the cross section above N_ch / N_ch^{5%} = 1, the fraction
    of the cross section per unit of Suppl. Fig. 3's histogram, its bin
    centres, values and width): Aux. Fig. 2b against the histogram's
    cumulative where the distribution is flat in centrality (15-50%)."""
    a, h = load(f'aux_figure_2_b_{sk}.yaml'), load(f'suppl_figure_3_{sk}.yaml')
    assert a['dependent_variables'][0]['header']['name'].startswith('Centrality'), a['dependent_variables'][0]
    assert h['independent_variables'][0]['header']['name'] == '$N_{ch}/N^{5\\%}_{ch}$', h['independent_variables'][0]
    xa = np.array([v['value'] for v in a['independent_variables'][0]['values']])
    ca = np.array([v['value'] for v in a['dependent_variables'][0]['values']])
    xh = np.array([v['value'] for v in h['independent_variables'][0]['values']])
    yh = np.array([v['value'] for v in h['dependent_variables'][0]['values']])
    w = float(np.median(np.diff(xh)))
    assert np.allclose(np.diff(xh), w, atol=2e-4), 'equal bins'
    cum = np.cumsum((yh * w)[::-1])[::-1]               # above each bin's lower edge
    c_at = np.interp(xh - w / 2, xa, ca)
    bulk = (c_at > 15) & (c_at < 50)
    r = c_at[bulk] / 100.0 / cum[bulk]
    assert bulk.sum() >= 10 and r.std() / r.mean() < 1e-3, (sk, r)
    c = float(r.mean())
    above = np.clip((xh + w / 2 - 1.0) / w, 0.0, 1.0)  # the share of each bin above 1
    return 100.0 * c * float(np.sum(yh * w * above)), c, xh, yh, w


N5_PERCENT = {}


def centrality(cls, sk):
    p = N5_PERCENT[sk]
    return {'classes': list(cls), 'estimator': {'name': 'ID', 'observable': 'Nch', 'detector': 'ID'},
            'eta': [dict(WINDOW['eta'])], 'pT': dict(WINDOW['pT']),
            'multiplicity_bins': {'efficiency': 0.6, 'normalized_to': {'percentile': round(p, 3)}},
            'notes': f'N_ch / N_ch^{{5%}}: charged tracks 0.5 < pT < 5 GeV in |eta| < 2.5 over their count at 5% '
                     f'centrality, which {p:.3f}% of the cross section exceeds (Suppl. Fig. 3 and Aux. Fig. 2b); '
                     'classes midway between the tabulated points; reconstructed tracks, efficiency 0.6 (see the '
                     'curation script).'}


def side(q, sysinfo, cls, what, sk):
    cs, name, n5 = sysinfo
    base = {'inspire_id': '2806288', 'version': VERSION, 'experimental_values': False,
            'hepdata_table_label': f'{what}, {name} (a side of a ratio; only the values over 5% centrality are '
                                   'published)',
            'collision_system': dict(cs), 'centrality': centrality(cls, sk), 'particle_id': {'species': 'charged'},
            'kinematic_range': {k: dict(v) for k, v in WINDOW.items()}, 'dimensionality': 'integrated'}
    if q == 'mean_pT':
        base.update(observable_type='mean_pT', observable='mean_pT', value_type='mean_pT',
                    notes={'value': f'{CITE}: <[pT]> of charged particles, 0.5 < pT < 5 GeV, |eta| < 2.5.'})
    else:
        base.update(observable_type='mean_pT_fluctuations', observable=q, value_type=q, pT_moment=q,
                    multiplicity_baseline={'observable': 'Nch', 'eta': [dict(WINDOW['eta'])], 'pT': dict(WINDOW['pT'])},
                    notes={'value': f'{CITE}, Eq. 1: {q} of [pT], charged particles 0.5 < pT < 5 GeV, |eta| < 2.5; '
                                    'deviations from <[pT]> at the event\'s N_ch (ATLAS: unit N_ch^rec intervals).'})
    return base


def main():
    subs = {s['data_file']: s for s in yaml.load_all(open(os.path.join(RAW, 'submission.yaml')), Loader=Loader)
            if s and 'data_file' in s}
    ratios, moms, mpts, fracs = [], [], [], []
    for sk in SYSTEMS:
        N5_PERCENT[sk] = n5_scale(sk)[0]
    for sk, sysinfo in SYSTEMS.items():
        x3a = [v['value'] for v in load(f'figure_3a_{sk}.yaml')['independent_variables'][0]['values']]
        half = 0.5 * float(np.median(np.diff(x3a)))
        ref = f'{fmt(1.0 - half)}-{fmt(1.0 + half)}'
        for stem, (q, label) in TABLES.items():
            hf = f'{stem}_{sk}.yaml'
            t = load(hf)
            iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
            assert iv['header']['name'].startswith('$N_{ch}/N^{5\\%}_{ch}$'), (hf, iv['header'])
            xs = [v['value'] for v in iv['values']]
            assert np.all(np.diff(xs) > 0), hf
            cls = classes(xs)
            assert [e['label'] for e in dv['values'][0]['errors']] == ['stat. uncertainty', 'syst. uncertainty'], hf
            what = 'the mean [pT]' if q == 'mean_pT' else f'{q} of [pT]'
            out = mpts if q == 'mean_pT' else moms
            path = OUT_MPT if q == 'mean_pT' else OUT_MOM
            out.append(side(q, sysinfo, cls, what, sk))
            num = f'{path}#{len(out) - 1}'
            out.append(side(q, sysinfo, [ref], what, sk))
            den = f'{path}#{len(out) - 1}'
            ratios.append({
                'observable_type': 'observable_ratios', 'inspire_id': '2806288', 'version': VERSION,
                'hepdata_file': hf, 'dv_index': 0,
                'hepdata_table_label': f'{label} vs N_ch/N_ch^5%, {sysinfo[1]} ({subs[hf]["description"][:60]})',
                'collision_system': dict(sysinfo[0]), 'centrality': centrality(cls, sk),
                'particle_id': {'species': 'charged'}, 'kinematic_range': {k: dict(v) for k, v in WINDOW.items()},
                'observable': q, 'value_type': dv['header']['name'],
                'ratio_type': 'centrality_ratio', 'numerator': 'N_ch/N_ch^5% class', 'denominator': '5% centrality',
                'numerator_system': sysinfo[1], 'denominator_system': sysinfo[1],
                'numerator_record': num, 'denominator_record': den, 'denominator_class': ref,
                'dimensionality': 'integrated',
                'notes': {'value': f'{CITE}, {hf.split(".")[0]}: {label} against N_ch / N_ch^5% (N_ch^5% = '
                                   f'{sysinfo[2]} in {sysinfo[1]}), charged particles 0.5 < pT < 5 GeV, |eta| < '
                                   '2.5; the reference the class around N_ch / N_ch^5% = 1. Errors: stat and syst.'}})
    for sk, sysinfo in SYSTEMS.items():
        # the distribution of N_ch / N_ch^{5%} (Suppl. Fig. 3): per bin the
        # fraction of the cross section, the bins within 0-80% of it
        p, c, xh, yh, w = n5_scale(sk)
        hf = f'suppl_figure_3_{sk}.yaml'
        cls = classes(xh)
        cum = c * np.cumsum((yh * w)[::-1])[::-1]      # the cross section above each bin's lower edge
        rows = [k for k in range(len(xh)) if cum[k] <= 0.80]
        assert rows == list(range(rows[0], len(xh))), rows
        print(f'{sysinfo[1]}: {p:.4f}% of the cross section above N_ch/N_ch^5% = 1; a unit of the histogram is '
              f'{c:.5f} of the cross section, its bins hold {cum[0]:.4f}; {len(rows)} of {len(xh)} bins within '
              '0-80%')
        fracs.append({
            'observable_type': 'multiplicity_distribution', 'inspire_id': '2806288', 'version': VERSION,
            'hepdata_file': hf, 'dv_index': 0, 'table_rows': rows,
            'hepdata_table_label': f'distribution of N_ch/N_ch^5%, {sysinfo[1]} ({subs[hf]["description"][:60]})',
            'collision_system': dict(sysinfo[0]), 'centrality': centrality([cls[k] for k in rows], sk),
            'particle_id': {'species': 'charged'}, 'kinematic_range': {k: dict(v) for k, v in WINDOW.items()},
            'observable': 'event_fraction', 'value_type': 'event_fraction', 'normalization': 'cross_section',
            'efficiency_corrected': True, 'dimensionality': 'integrated',
            'value_scale': float(f'{1.0 / (w * c):.6g}'),
            'notes': {'value': f'{CITE}, Suppl. Fig. 3: the distribution of N_ch / N_ch^5% (N_ch^5% = {sysinfo[2]} '
                               f'in {sysinfo[1]}), charged particles 0.5 < pT < 5 GeV, |eta| < 2.5, in arbitrary '
                               f'normalization; value_scale = 1 / (bin width x {c:.5f}) turns it into the fraction '
                               f'of the cross section per bin ({c:.5f}: the fraction per unit of the histogram, from '
                               'Aux. Fig. 2b where the distribution is flat in centrality; see the curation script). '
                               'The bins within 0-80% of the cross section. Errors: stat.'}})
    for out, docs in ((OUT_MPT, mpts), (OUT_MOM, moms), (OUT_RATIO, ratios), (OUT_FRAC, fracs)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Curate ATLAS multi-particle cumulants in Pb+Pb at 5.02 TeV, INSPIRE 1728935.

ATLAS Collaboration, "Fluctuations of anisotropic flow in Pb+Pb collisions
at sqrt(s_NN) = 5.02 TeV with the ATLAS detector", JHEP 01 (2020) 051,
arXiv:1904.04808.

From the paper:
- charged particles |eta| < 2.5, four pT ranges 0.5-5, 1-5, 1.5-5, 2-5 GeV
  (all particles of a multiplet in the range), corrected for efficiency
  (Secs. 3, 5);
- standard method: every particle in |eta| < 2.5; three-subevent method:
  a: -2.5 < eta < -2.5/3, b: |eta| < 2.5/3, c: 2.5/3 < eta < 2.5, each
  cumulant the mean of the three measurements with the subevents
  interchanged (Sec. 4.2); c_n{6} standard only;
- every two-particle cumulant (v_n{2}, and the normalizations) is
  c_n^{a|c}{2} between a and c (Eq. 7); nc_n{4} = c_n{4} / c_n{2}^2, nc_n{6}
  = c_n{6} / (4 c_n{2}^3), nsc_{n,m}{4} = sc / (c_n{2} c_m{2}), nac_2{3} =
  ac_2{3} / sqrt((2 c_2{2}^2 + c_2{4}) c_4{2}) (Eqs. 8-11), v_n{4}/v_n{2}
  and v_n{6}/v_n{4} from the same cumulants (Eq. 12);
- reference event classes (Sec. 5): the correlators are averaged in narrow
  intervals of FCal Sigma E_T (3.2 < |eta| < 4.9; 0.002 TeV) or of N_ch^rec
  (reconstructed tracks, 0.5 < pT < 5 GeV, |eta| < 2.5; unit intervals),
  the cumulants formed there and combined over wider classes with the
  numbers of events; results vs centrality use Sigma E_T (Sec. 7);
- Sigma E_T percentiles from a Glauber analysis of the Sigma E_T
  distribution (Sec. 3).

Classes:
- centrality tables: the points are mean centralities of classes of equal
  width (1, 2, 5 or 10%), [floor(x/w) w, +w);
- Sigma E_T and N_ch^rec tables: the points are the mean Sigma E_T
  (N_ch^rec) of intervals of a regular grid (0.05 TeV, 50 tracks, or
  wider): each point is checked to be the mean of its interval under the
  published distribution (Fig. 2, HEPData figure2_panel1/2), to a tenth
  of the width for most points (in the ultra-central tail to 1e-4 TeV and
  0.1 tracks; at low multiplicity the tables' means sit up to a quarter
  width higher, the multiplet weights rising across an interval), and the
  interval edges are converted to percentiles of that
  distribution: the fraction of the histogram above an edge (log-linear
  interpolation of the sampled density) times a scale, which the
  histograms need because their low-multiplicity peaks hold events beyond
  the centrality normalization.  The scales come from the paper's own
  tables: v_n{2} versus <Sigma E_T> (Fig. 13) matches v_n{2} versus
  centrality (Fig. 3) for 1.015 (v_2, every range from 2 to 60% and every
  pT range within 1.013-1.0155; v_3 1.014-1.025), and v_n{2} versus
  <N_ch^rec> for 1.126 (1.116-1.134; the two reference classes agree
  outside ultra-central collisions, Sec. 7.3.1).  Checks: Sigma E_T above
  the 4.1 TeV knee and the 4.21, 4.37, 4.54 TeV trigger thresholds then
  gives 1.88, 1.26, 0.52, 0.103% (the paper: 1.9, 1.3, 0.5, 0.1%); the
  N_ch^rec percentile of N equals the Sigma E_T percentile of <Sigma
  E_T>(N) (Fig. 1 middle) to 0.1 (absolute) from 3 to 53%, 2.68% at the
  N_ch^rec knee (the paper: 2.7%), and they part above the knee, where
  N_ch^rec-selected events have a lower <Sigma E_T>.  Intervals beyond
  80%, or (outside the top 5%, where the distributions fall like Gaussian
  tails) whose percentiles move by more than a tenth of their width with
  linear instead of log-linear interpolation, are left out (table_rows).
- N_ch^rec classes: percentiles of the model's charged multiplicity in
  |eta| < 2.5, 0.5 < pT < 5 GeV with efficiency 0.6 (Poisson), ATLAS's
  reconstructed count over the corrected one: <N_ch^rec> ~ 2800 in 0-5%
  (Fig. 2) against ~4600 charged particles there (dN/deta ~ 1940 at |eta|
  < 0.5, about 47% above 0.5 GeV, five units of eta).

Every cumulant record is computed at fixed reference multiplicity
(multiplicity_baseline: the FCal charged multiplicity for Sigma E_T
classes, the |eta| < 2.5, 0.5-5 GeV multiplicity for N_ch^rec classes),
the counterpart of the narrow intervals (analysis.multi_cumulants).  The
v_n{2} records (linear in the averages) need none.

Records: flow_vn (nc_n{4}, nc_n{6}, v_n{4}/v_n{2}, v_2{6}/v_2{4}, v_n{4},
v_2{6}, v_n{2}) and flow_harmonic_correlations (sc, nsc, ac, nac), standard
and three-subevent, versus centrality, <Sigma E_T> and <N_ch^rec>.
Already public: v_n{2} versus centrality (Fig. 3,
data/flow_vn/atlas_flow_vn_fluctuations_of_anisotropic_flow_PbPb_5020gev.yaml).
Not curated (workbench staging): the Sigma E_T and N_ch^rec distributions
(Figs. 1-2), c_1{4} and v_1{4} (Figs. 8-9: harmonic 1), the ratios
between the two reference classes (Fig. 14), and the N_ch^rec-class
results replotted against <Sigma E_T> (Figs. 16, 17 right, 18 right, 20).

Run from the repository root:  python scripts/curate_atlas_cumulants_pbpb_5020.py
[--staged FILE]: also rewrite a staged draft of this paper (records of
every table) to the tables not curated here.
"""
import glob
import os
import re

import sys

import numpy as np
import yaml

sys.path.insert(0, os.getcwd())
from analysis.dispatcher import centrality_classes_suspicious  # noqa: E402

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

RAW = glob.glob('hepdata_raw/records/ins1728935/HEPData-ins1728935-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'ATLAS Collaboration, JHEP 01 (2020) 051, arXiv:1904.04808'
FCAL_ETA = [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'}, 'eta': FCAL_ETA}
NCH_EFF = 0.6
NCH = {'estimator': {'name': 'Nch_rec', 'observable': 'Nch', 'detector': 'ID', 'efficiency': NCH_EFF},
       'eta': [{'min': -2.5, 'max': 2.5}], 'pT': {'min': 0.5, 'max': 5.0}}
BASELINE = {'Et': {'observable': 'Nch', 'eta': FCAL_ETA, 'pT': {'min': 0.0, 'max': 100.0}},
            'Nch': {'observable': 'Nch', 'eta': [{'min': -2.5, 'max': 2.5}], 'pT': {'min': 0.5, 'max': 5.0}}}
SUBS = {'A': (-2.5, -0.8333), 'B': (-0.8333, 0.8333), 'C': (0.8333, 2.5)}
MAX_CENT = 80.0
OUT_VN = 'data/flow_vn/atlas_flow_vn_multiparticle_cumulants_PbPb_5020gev_ins1728935.yaml'
OUT_CORR = 'data/flow_harmonic_correlations/atlas_flow_harmonic_correlations_sc_ac_PbPb_5020gev_ins1728935.yaml'
# tables not curated here (see the docstring), by file-name prefix
SKIP = ('figure1_', 'figure2_', 'figure3_', 'figure8_', 'figure9_', 'figure14_', 'figure16_', 'figure17_panel3',
        'figure18_panel2', 'figure20_')
DESC = re.compile(r'^(nsc|nac|nc|sc|ac|c|v)_(\d)(?:_(\d))?\{(\d)(?:, (Et|Nch))?\}'
                  r'(?: / v_(\d)\{(\d)(?:, (?:Et|Nch))?\})?, (standard|3-subevent), ([\d.]+)<pT<([\d.]+) GeV$')
AXIS = {'Centrality': None, 'Et': 'Et', 'NchRec': 'Nch'}


def load(name):
    return yaml.load(open(os.path.join(RAW, name)), Loader=Loader)


SCALE = {'Et': 1.015, 'Nch': 1.126}       # percentile scales (see the docstring)


class Distribution:
    """Percentile above x of a published distribution (density sampled at
    points): the fraction of its integral above x, log-linear (or linear)
    interpolation, times the scale."""

    def __init__(self, name, scale=1.0):
        self.scale = float(scale)
        t = load(name)
        x = np.array([v['value'] for v in t['independent_variables'][0]['values']], float)
        y = np.array([v['value'] for v in t['dependent_variables'][0]['values']], float)
        step = x[1] - x[0]
        self.fine = np.linspace(min(x[0] - step / 2, 0.0), x[-1] + step, 800001)
        top = x[y > 0][-1] + step / 2
        self.dens = {}
        for kind in ('log', 'lin'):
            if kind == 'log':
                d = np.exp(np.interp(self.fine, x, np.log(np.maximum(y, 1e-300))))
            else:
                d = np.interp(self.fine, x, y)
            d[self.fine > top] = 0.0
            self.dens[kind] = d
        self.cum = {k: np.cumsum(d[::-1])[::-1] for k, d in self.dens.items()}

    def above(self, x, kind='log'):
        # one normalization for both interpolations: the check compares the
        # local mapping only
        return 100.0 * self.scale * np.interp(x, self.fine, self.cum[kind]) / self.cum['log'][0]

    def mean(self, a, b):
        s = (self.fine >= a) & (self.fine < b)
        d = self.dens['log'][s]
        return float(np.sum(self.fine[s] * d) / np.sum(d))


def grid_width(xs, choices):
    sp = np.median(np.diff(xs))
    w = min(choices, key=lambda c: abs(c - sp))
    assert abs(sp - w) < 0.05 * w, (sp, w)
    return w


def centrality_classes(xs):
    w = grid_width(xs, (1.0, 2.0, 5.0, 10.0))
    out = []
    for x in xs:
        lo = np.floor(x / w + 1e-9) * w
        assert lo <= x < lo + w, (x, lo, w)
        out.append(f'{lo:g}-{lo + w:g}%')
    assert len(set(out)) == len(out), out
    return out, list(range(len(xs)))


def fmt(p):
    """Four significant digits, positional (class labels take no exponent)."""
    return np.format_float_positional(float(p), precision=4, unique=False, fractional=False, trim='-')


def interval_classes(xs, dist, choices):
    """Percentile classes of the grid intervals holding the points; rows kept."""
    w = grid_width(xs, choices)
    classes, rows, off = [], [], []
    for i, x in enumerate(xs):
        lo = np.floor(x / w + 1e-9) * w
        hi = lo + w
        p_hi, p_lo = dist.above(lo), dist.above(hi)
        if lo < 0 or p_hi > MAX_CENT:
            continue
        off.append(abs(dist.mean(lo, hi) - x) / w)
        # the low-multiplicity peak: drop classes whose edges depend on the
        # interpolation; the central tail falls like a Gaussian, where
        # log-linear holds (the 0.1% anchor at 4.54 TeV)
        shift = max(abs(dist.above(lo, 'lin') - p_hi), abs(dist.above(hi, 'lin') - p_lo))
        if p_hi > 5.0 and shift > 0.1 * (p_hi - p_lo):
            continue
        classes.append(f'{fmt(p_lo)}-{fmt(p_hi)}%')
        rows.append(i)
    assert len(set(classes)) == len(classes), classes
    # the grid: most points are their interval's mean (the multiplet weights
    # move it where the multiplicity changes much across an interval)
    assert np.mean(np.array(off) < 0.1) > 0.8 and max(off) < 0.25, off
    return classes, rows


def kr(eta, pt):
    return {'eta': {'min': eta[0], 'max': eta[1]}, 'pT': {'min': pt[0], 'max': pt[1]}}


def subevents(labels, pt):
    return {lab: {'kinematic_range': kr(SUBS[lab], pt)} for lab in labels}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='staged draft of this paper to reduce to the tables not curated here')
    args = ap.parse_args()
    subs = [s for s in yaml.load_all(open(os.path.join(RAW, 'submission.yaml')), Loader=Loader)
            if s and 'data_file' in s]
    dist = {'Et': Distribution('figure2_panel1.yaml', SCALE['Et']),
            'Nch': Distribution('figure2_panel2.yaml', SCALE['Nch'])}
    for e, p in ((4.1, 1.9), (4.21, 1.3), (4.37, 0.5), (4.54, 0.1)):
        got = dist['Et'].above(e)
        assert abs(got - p) < 0.06 * p, (e, got, p)
    t = load('figure1_panel3.yaml')
    xn = np.array([v['value'] for v in t['independent_variables'][0]['values']], float)
    et = np.array([v['value'] for v in t['dependent_variables'][0]['values']], float)
    for N in (300, 1000, 2000, 2800):
        a, b = dist['Nch'].above(N), dist['Et'].above(np.interp(N, xn, et))
        assert abs(a - b) < 0.2, (N, a, b)
    vn, corr, skipped, seen = [], [], [], {}
    for s in subs:
        hf = s['data_file']
        if hf.startswith(SKIP):
            skipped.append(hf)
            continue
        desc = ' '.join(str(s['description']).split())
        m = DESC.match(desc)
        assert m, (hf, desc)
        kind, n, n2, order, ref, rn, ro, meth, plo, phi = m.groups()
        n, order, pt = int(n), int(order), (float(plo), float(phi))
        method = 'three_subevent' if meth == '3-subevent' else 'standard'
        t = load(hf)
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        axis = iv['header']['name']
        assert AXIS[axis] == ref, (hf, axis, ref)
        xs = np.array([v['value'] for v in iv['values']], float)
        assert np.all(np.diff(xs) > 0), hf
        key = (tuple(xs), tuple(v['value'] for v in dv['values']))
        if key in seen:
            print(f'  {hf}: the values of {seen[key]} (skipped)')
            skipped.append(hf)
            continue
        seen[key] = hf
        if ref is None:
            classes, rows = centrality_classes(xs)
            cent = {'classes': classes, **FCAL,
                    'notes': 'FCal Sigma E_T classes of equal width; the table gives their mean centralities.'}
        elif ref == 'Et':
            classes, rows = interval_classes(xs, dist['Et'], (0.05, 0.1, 0.2, 0.25, 0.5))
            cent = {'classes': classes, **FCAL,
                    'notes': 'Sigma E_T intervals of the table (mean <Sigma E_T> per point) as percentiles of the '
                             'published Sigma E_T distribution (HEPData figure2_panel1).'}
        else:
            classes, rows = interval_classes(xs, dist['Nch'], (50.0, 100.0, 200.0, 250.0, 500.0))
            cent = {'classes': classes, **NCH,
                    'notes': 'N_ch^rec intervals of the table (mean <N_ch^rec> per point) as percentiles of the '
                             'published N_ch^rec distribution (HEPData figure2_panel2); reconstructed tracks, '
                             f'efficiency {NCH_EFF} (see the curation script).'}
        if not rows:
            print(f'  {hf}: no class within 0-{MAX_CENT:g}% (skipped)')
            skipped.append(hf)
            continue
        if centrality_classes_suspicious(classes):
            # non-integer classes that do not reach the central few percent
            # look like kinematic bins to the dispatcher's guard (v_2{6}/v_2{4}
            # at 2-5 GeV vs <Sigma E_T> starts at 5.1%); its centrality-axis
            # table (Fig. 7) is curated
            print(f'  {hf}: classes {classes[-1]}..{classes[0]} fail the percentile guard (skipped)')
            skipped.append(hf)
            continue
        assert [e['label'] for e in dv['values'][0]['errors']] == ['stat. uncertainty', 'syst. uncertainty'], hf
        d = {'inspire_id': '1728935', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f'{desc} ({s.get("location", hf)})',
             'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
             'centrality': cent}
        if len(rows) < len(xs):
            d['table_rows'] = rows
        d.update({'particle_id': {'species': 'charged'}, 'kinematic_range': kr((-2.5, 2.5), pt),
                  'dimensionality': 'integrated', 'value_type': dv['header']['name']})
        refclass = {'Et': 'narrow Sigma E_T intervals', 'Nch': 'unit N_ch^rec intervals',
                    None: 'narrow Sigma E_T intervals'}[ref]
        base = BASELINE['Nch' if ref == 'Nch' else 'Et']
        meth_note = ('the three-subevent method (a, b, c = A, B, C; the mean of the three interchanged '
                     'measurements)' if method == 'three_subevent' else 'the standard method (|eta| < 2.5)')
        if kind == 'v' and order == 2:
            assert method == 'three_subevent' and rn is None, hf
            d.update({'observable_type': 'flow_vn', 'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2,
                      'particles': 'AB', 'eta_gap': 1.67,
                      'subevents': {'A': {'kinematic_range': kr(SUBS['A'], pt)},
                                    'B': {'kinematic_range': kr(SUBS['C'], pt)}}})
            what = f'v_{n}{{2}} = sqrt(c_{n}^(a|c){{2}}) between -2.5 < eta < -2.5/3 and 2.5/3 < eta < 2.5 (Eq. 7)'
            vn.append(d)
        elif kind in ('nc', 'v'):
            if kind == 'nc':
                q, ratio = 'normalized', None
                what = (f'nc_{n}{{{order}}} = c_{n}{{{order}}} / '
                        + (f'c_{n}{{2}}^2' if order == 4 else f'(4 c_{n}{{2}}^3)') + ' (Eqs. 8-9)')
            elif rn is not None:
                q, ratio = 'ratio', int(ro)
                assert int(rn) == n, hf
                what = f'v_{n}{{{order}}} / v_{n}{{{ratio}}} (Eq. 12)'
            else:
                q, ratio = 'coefficient', None
                what = f'v_{n}{{{order}}}, the signed root of c_{n}{{{order}}} (Eq. 4)'
            d.update({'observable_type': 'flow_vn', 'harmonic': n, 'method': 'cumulant', 'cumulant_order': order,
                      'cumulant_method': method, 'cumulant_quantity': q})
            if ratio is not None:
                d['ratio_order'] = ratio
            if method == 'three_subevent':
                d.update(particles='AABC', subevents=subevents('ABC', pt))
            elif q == 'normalized' or ratio == 2:
                d['normalization_subevents'] = subevents('AC', pt)
            d['multiplicity_baseline'] = dict(base)
            vn.append(d)
        elif kind in ('sc', 'nsc', 'ac', 'nac'):
            normalized = kind.startswith('n')
            if kind.endswith('sc'):
                harms, ctype = [n, int(n2)], 'symmetric_cumulant'
                what = (f'{kind}_{{{n},{n2}}}{{4}}' + (f' = sc / (c_{n}{{2}} c_{n2}{{2}}) (Eq. 10)'
                                                         if normalized else ' (Eq. 6)'))
            else:
                assert n2 is None and order == 3, hf
                harms, ctype = [n], 'asymmetric_cumulant'
                what = (f'{kind}_{n}{{3}}' + (f' = ac / sqrt((2 c_{n}{{2}}^2 + c_{n}{{4}}) c_{2 * n}{{2}}) (Eq. 11)'
                                              if normalized else ' = <<e^{i(n phi1 + n phi2 - 2n phi3)}>> (Eq. 6)'))
            d.update({'observable_type': 'flow_harmonic_correlations', 'correlation_type': ctype,
                      'harmonics': harms, 'observable': f'{kind}_{"_".join(map(str, harms))}',
                      'cumulant_method': method, 'normalized': normalized})
            if method == 'three_subevent':
                d.update(particles='ABC' if ctype == 'asymmetric_cumulant' else 'AABC',
                         subevents=subevents('ABC', pt))
            elif normalized:
                d['normalization_subevents'] = subevents('AC', pt)
            d['multiplicity_baseline'] = dict(base)
            corr.append(d)
        else:
            raise AssertionError((hf, desc))
        cls = {None: 'FCal Sigma E_T centrality', 'Et': '<Sigma E_T> intervals',
               'Nch': '<N_ch^rec> intervals (reconstructed tracks)'}[ref]
        d['notes'] = {
            'value': f'{CITE}, {s.get("location", hf)}. {what}; {meth_note}; charged particles {pt[0]:g} < pT < '
                     f'{pt[1]:g} GeV; {cls}. Errors: stat and syst.',
            'reference_event_class': (f'ATLAS forms the cumulants in {refclass} and combines them with the numbers of '
                                      'events (Sec. 5)'
                                      + ('; computed at fixed multiplicity in the estimator acceptance '
                                         '(multiplicity_baseline).' if 'multiplicity_baseline' in d else
                                         '; a two-particle cumulant is linear in the averages.')),
        }
    for out, docs in ((OUT_VN, vn), (OUT_CORR, corr)):
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    # a staged draft keeps only the tables not curated here
    if args.staged:
        STAGED = args.staged
        keep = {hf for hf in skipped if not hf.startswith('figure3_')}     # Fig. 3 is public
        docs = [d for d in yaml.load_all(open(STAGED), Loader=Loader)
                if isinstance(d, dict) and d.get('hepdata_file') in keep]
        for d in docs:
            d['staging_reason'] = ('ATLAS 1728935 table not curated: distributions (Figs. 1-2), harmonic 1 (Figs. '
                                   '8-9), ratios between reference event classes (Fig. 14), N_ch^rec-class results '
                                   'against <Sigma E_T> (Figs. 16-20), classes the percentile guard rejects (Fig. 18 '
                                   'at 2-5 GeV), or the values of a curated table; see '
                                   'scripts/curate_atlas_cumulants_pbpb_5020.py')
        with open(STAGED, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{STAGED}: {len(docs)} records kept')


if __name__ == '__main__':
    main()

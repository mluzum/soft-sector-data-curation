#!/usr/bin/env python3
"""Curate ALICE symmetry-plane correlations with the Gaussian Estimator, INSPIRE 2628969 and 2825788.

- 2628969: S. Acharya et al. (ALICE), "Symmetry plane correlations in
  Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", EPJC 83 (2023) 576,
  arXiv:2302.01234.  Centrality from the SPD (curated as CL1, the outer
  layer's clusters, |eta| < 1.4: ALICE's SPD estimator), 0-50%.
- 2825788: S. Acharya et al. (ALICE), "Higher-order symmetry plane
  correlations in Pb-Pb collisions at sqrt(sNN) = 5.02 TeV", PRC 111 (2025)
  064913, arXiv:2409.04238.  Centrality from the V0 (V0M), 0-60%.

From the papers (Sec. 1-2, Eq. 4): charged tracks |eta| < 0.8, 0.2 < pT <
5.0 GeV/c; <cos(a_1 n_1 Psi_{n_1} + ...)>_GE = sqrt(pi/4) <v_{n_1}^{a_1} ...
cos(...)> / sqrt(<v_{n_1}^{2 a_1} ...>), numerator and denominator from
multi-particle correlators over all distinct particles of the window (Eq.
2), no eta gap.

Records (public): every SPC table as correlation_type plane_correlation,
estimator GE (key plane_corr_ge, analysis/plane_correlations.py): planes
[{harmonic n_i, coefficient a_i n_i / n_i with its sign}] parsed from the
table header.  Staged (--staged OUT): 2825788 Tables 26-33, sums and
products of two SPC (the paper's Fig. 4 tests of their relations), derived
from public records.

Run from the repository root:  python scripts/curate_alice_spc_ge.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

PAPERS = {
    '2628969': {'cite': 'S. Acharya et al. (ALICE), EPJC 83 (2023) 576, arXiv:2302.01234', 'energy': 2760.0,
                'out': 'data/flow_harmonic_correlations/alice_flow_harmonic_correlations_spc_ge_PbPb_2760gev_'
                       'ins2628969.yaml',
                'centrality': {'estimator': {'name': 'CL1', 'observable': 'Nch', 'detector': 'SPD'},
                               'eta': [{'min': -1.4, 'max': 1.4}],
                               'notes': 'the SPD (the paper: "estimated with the SPD"); curated as CL1, clusters in '
                                        'the outer SPD layer, |eta_lab| < 1.4.'}},
    '2825788': {'cite': 'S. Acharya et al. (ALICE), PRC 111 (2025) 064913, arXiv:2409.04238', 'energy': 5020.0,
                'out': 'data/flow_harmonic_correlations/alice_flow_harmonic_correlations_spc_ge_PbPb_5020gev_'
                       'ins2825788.yaml',
                'centrality': {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
                               'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}},
}


def planes_of(label):
    """'$\\langle\\cos[8(\\Psi_4-\\Psi_2)]\\rangle$' -> [(4, 2), (2, -4)];
    '... cos[2\\Psi_2+3\\Psi_3-5\\Psi_5] ...' -> [(2, 1), (3, 1), (5, -1)]."""
    body = re.search(r'\\cos\[(.*)\]', label).group(1).replace('{', '').replace('}', '').replace(' ', '')
    m = re.fullmatch(r'(\d+)\(\\Psi_(\d+)-\\Psi_(\d+)\)', body)
    if m:
        k, a, b = (int(x) for x in m.groups())
        terms = [(a, k), (b, -k)]
    else:
        terms = []
        for sign, k, n in re.findall(r'([+-]?)(\d*)\\Psi_(\d+)', body):
            terms.append((int(n), (-1 if sign == '-' else 1) * int(k or 1)))
        assert ''.join(re.findall(r'[+-]?\d*\\Psi_\d+', body)) == body, label
    out = []
    for n, k in terms:
        assert k % n == 0, (label, n, k)
        out.append((n, k // n))
    assert sum(n * c for n, c in out) == 0 and len({n for n, _ in out}) == len(out), (label, out)
    return out


def text(planes):
    s = ' '.join(f"{'+' if c > 0 else '-'} {abs(c) * n}Psi_{n}" for n, c in planes).lstrip('+ ')
    return f'<cos[{s}]>'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    staged = []
    for iid, P in PAPERS.items():
        raw = glob.glob(f'hepdata_raw/records/ins{iid}/HEPData-ins{iid}-v*-yaml')[0]
        version = int(raw.rsplit('-v', 1)[1].split('-')[0])
        sub = [x for x in yaml.safe_load_all(open(os.path.join(raw, 'submission.yaml'))) if x and 'data_file' in x]
        docs = []
        for x in sub:
            hf = x['data_file']
            t = yaml.safe_load(open(os.path.join(raw, hf)))
            assert len(t['dependent_variables']) == 1, hf
            dv, iv = t['dependent_variables'][0], t['independent_variables'][0]
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            assert q['ETARAP'] == '-0.8 TO 0.8' and q['PT'] == '0.2 TO 5.0', (hf, q)
            assert float(q['SQRT(S)/NUCLEON']) == P['energy'], (hf, q)
            assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
            name = dv['header']['name']
            classes = [f"{float(b['low']):g}-{float(b['high']):g}%" for b in iv['values']]
            d = {'observable_type': 'flow_harmonic_correlations', 'inspire_id': iid, 'version': version,
                 'hepdata_file': hf, 'dv_index': 0,
                 'hepdata_table_label': f"{x['name']}: {' '.join(x['description'].split())}",
                 'collision_system': {'energy_gev': P['energy'], 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': classes, **P['centrality']}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 5.0}}}
            if '+$' in name or '\\times' in name:
                d.update(observable=f"spc_combination_{x['name'].split()[-1]}",
                         correlation_type='multi_harmonic_correlation', harmonics=[], value_type=name,
                         dimensionality='integrated',
                         notes={'value': f"{P['cite']}. {' '.join(x['description'].split())} (Fig. 4), Gaussian "
                                         'Estimator.'},
                         staging_reason='a sum or product of two SPC (the paper tests their relations): derived '
                                        'from public records, not a measurement of its own')
                hs = sorted({int(n) for n in re.findall(r'\\Psi_\{?(\d+)', name)})
                d['harmonics'] = hs
                staged.append(d)
                continue
            planes = planes_of(name)
            d.update(observable='SPC_' + '_'.join(f'{n}{c:+d}' for n, c in planes),
                     correlation_type='plane_correlation', harmonics=[n for n, _ in planes],
                     planes=[{'harmonic': n, 'coefficient': c} for n, c in planes], estimator='GE',
                     dimensionality='integrated', value_type=name,
                     notes={'value': f"{P['cite']}. {text(planes)}_GE = sqrt(pi/4) <prod v^a cos(...)> / "
                                     f"sqrt(<prod v^2a>) (Eq. 4), {sum(abs(c) for _, c in planes)}- and "
                                     f"{2 * sum(abs(c) for _, c in planes)}-particle correlators of charged "
                                     'particles |eta| < 0.8, 0.2 < pT < 5.0 GeV/c (all distinct particles, no eta '
                                     'gap). Errors: stat and sys.'})
            docs.append(d)
        with open(P['out'], 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f"{P['out']}: {len(docs)} records;", [text([(p['harmonic'], p['coefficient']) for p in d['planes']])
                                                    for d in docs])
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()

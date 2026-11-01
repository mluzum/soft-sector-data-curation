#!/usr/bin/env python3
"""Curate PHOBOS charged-hadron pT spectra: Au+Au at 62.4 GeV and Cu+Cu at 62.4 and 200 GeV.

- B. B. Back et al. (PHOBOS), "Centrality dependence of charged hadron
  transverse momentum spectra in Au+Au collisions from sqrt(sNN) = 62.4 to
  200 GeV", PRL 94, 082304 (2005), arXiv:nucl-ex/0405003, INSPIRE 649714.
  Tables SpectraCentBins0-5.txt (hepdata_raw/phobos/AuAu_Spectra_62/),
  peripheral first (45-50% ... 0-6%, Table I; the low-pT yields rise
  monotonically through the files).
- B. Alver et al. (PHOBOS), "System size and centrality dependence of
  charged hadron transverse momentum spectra in Au+Au and Cu+Cu collisions
  at sqrt(sNN) = 62.4 and 200 GeV", PRL 96, 212301 (2006),
  arXiv:nucl-ex/0512016, INSPIRE 700249.  Tables
  20051205_Spectra_CuCu{62,200}_ALL.txt (hepdata_raw/phobos/CuCuSpectra_63_200/).

From the papers: invariant yield (2 pi pT)^-1 d2N/(deta dpT) of the AVERAGE
of positive and negative hadrons, (h+ + h-)/2, over 0.2 < eta < 1.4 (both
papers say pseudorapidity, unlike PLB 578, 297, which uses y_pi); corrected
for acceptance, efficiency, secondary and wrongly reconstructed particles,
binning and momentum resolution (values at the listed pT).  Centrality:
Au+Au from the Paddle counters (3.0 < |eta| < 4.5); Cu+Cu from the energy
deposited in the octagon (|eta| < 3.2).

Errors: the Cu+Cu tables give statistical and systematic errors.  The Au+Au
62.4 GeV tables give one error, statistical; the paper's systematic
uncertainties are "similar to those reported in the previous analysis"
(PLB 578, 297: ~10%) and not tabulated -- the records say so.

Bins: cells with each listed pT at the centre (as for PLB 578, 297).
Not curated: R_AA, R_PC and the ratios to scaled spectra (need a p+p or a
central reference).

Run from the repository root:  python scripts/curate_phobos_spectra_62_cucu.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.curate_phobos_spectra_200gev import edges  # noqa: E402
from scripts.value_tables import write_value_table  # noqa: E402

PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
OCTAGON = {'estimator': {'name': 'Octagon', 'observable': 'energy', 'detector': 'Octagon'},
           'eta': [{'min': -3.2, 'max': 3.2}]}
AUAU62 = 'B. B. Back et al. (PHOBOS), PRL 94, 082304 (2005), arXiv:nucl-ex/0405003'
CUCU = 'B. Alver et al. (PHOBOS), PRL 96, 212301 (2006), arXiv:nucl-ex/0512016'


def rows(path):
    return [[float(t) for t in ln.split()] for ln in open(path, encoding='latin-1').read().replace('\r', '').splitlines()
            if re.match(r'^\s*[0-9]\.[0-9]+\s', ln)]


def cucu_blocks(path):
    out, cur = {}, None
    for raw in open(path, encoding='latin-1'):
        ln = raw.replace('\r', '').strip()
        m = re.match(r'^([0-9]+-[0-9]+)%$', ln)
        if m:
            cur = out.setdefault(m.group(1), [])
        elif cur is not None and re.match(r'^[0-9]\.[0-9]+\s', ln):
            cur.append([float(t) for t in ln.split()])
    return out


def write(sets, raw_dir, table, cite, inspire, reaction, energy, cent, label, err_note, out):
    """sets: [(class, rows)], rows (pT, value, stat[, sys])."""
    pts = [r[0] for r in sets[0][1]]
    assert all([r[0] for r in rs] == pts for _, rs in sets)
    pt_edges = edges(pts)
    write_value_table(
        os.path.join(raw_dir, 'tables', table),
        f'{cite}: invariant yield of (h+ + h-)/2, 0.2 < eta < 1.4, {label} at {energy:g} GeV, by centrality; '
        f'values at the listed pT {pts}.',
        {'name': 'PT', 'units': 'GEV', 'bins': list(zip(pt_edges, pt_edges[1:]))},
        [{'name': '(2*PI*PT)**-1*D2N/DETA/DPT', 'units': 'GEV**-2',
          'qualifiers': {'CENTRALITY': f'{c}%', 'PARTICLE': '(h+ + h-)/2'},
          'values': [r[1] for r in rs],
          'errors': [[{'label': 'stat', 'sym': r[2]}] + ([{'label': 'sys', 'sym': r[3]}] if len(r) > 3 else [])
                     for r in rs]} for c, rs in sets])
    docs = []
    for i, (c, _) in enumerate(sets):
        docs.append({
            'observable_type': 'particle_spectra', 'inspire_id': inspire, 'version': 1,
            'hepdata_file': f'{os.path.basename(raw_dir)}/{table}', 'value_table': f'../../{raw_dir}/tables/{table}',
            'dv_index': i,
            'hepdata_table_label': f'charged-hadron invariant yield, {c}% {label} at {energy:g} GeV',
            'collision_system': {'energy_gev': energy, 'reaction': reaction},
            'centrality': {'classes': [f'{c}%'], **cent},
            'kinematic_range': {'eta': {'min': 0.2, 'max': 1.4},
                                'pT': {'min': pt_edges[0], 'max': pt_edges[-1], 'bins': pt_edges}},
            'particle_id': {'species': 'charged', 'charge_average': True},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield',
            'notes': {'value': (f'{cite}. (h+ + h-)/2 in 0.2 < eta < 1.4; corrected for secondaries, binning and '
                                f'momentum resolution: values at the listed pT; bins with each point at the centre. '
                                f'{err_note}')},
        })
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records; pT edges {pt_edges[:3]} ... {pt_edges[-3:]}')


def main():
    d62 = 'hepdata_raw/phobos/AuAu_Spectra_62'
    classes = ['45-50', '35-45', '25-35', '15-25', '6-15', '0-6']
    sets = [(c, rows(os.path.join(d62, f'SpectraCentBins{i}.txt'))) for i, c in enumerate(classes)]
    low = [rs[0][1] for _, rs in sets]
    assert low == sorted(low), low                     # peripheral first
    write(sets[::-1], d62, 'spectra_AuAu62.yaml', AUAU62, '649714', 'AU AU --> CHARGED X', 62.4, PADDLE, 'Au+Au',
          'Errors: statistical only; the systematic uncertainties (similar to PLB 578, 297, ~10%) are not tabulated.',
          'data/particle_spectra/phobos_particle_spectra_charged_AuAu_62.4gev_ins649714.yaml')
    dcu = 'hepdata_raw/phobos/CuCuSpectra_63_200'
    for energy, fn, want in ((200.0, '20051205_Spectra_CuCu200_ALL.txt', ['0-6', '6-15', '15-25', '25-35', '35-45', '45-50']),
                             (62.4, '20051205_Spectra_CuCu62_ALL.txt', ['0-6', '6-15', '15-25', '25-35', '35-40'])):
        b = cucu_blocks(os.path.join(dcu, fn))
        assert sorted(b) == sorted(want), sorted(b)
        write([(c, b[c]) for c in want], dcu, f'spectra_CuCu{energy:g}.yaml', CUCU, '700249', 'CU CU --> CHARGED X',
              energy, OCTAGON, 'Cu+Cu', 'Errors: statistical and systematic, as published.',
              f'data/particle_spectra/phobos_particle_spectra_charged_CuCu_{energy:g}gev_ins700249.yaml')


if __name__ == '__main__':
    main()

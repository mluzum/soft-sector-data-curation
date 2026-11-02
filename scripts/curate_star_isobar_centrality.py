#!/usr/bin/env python3
"""STAR isobar centrality as STAR defines it: INSPIRE 1914564 and 2822399.

M. Abdallah et al. (STAR), PRC 105, 014901 (2022), arXiv:2109.00131,
Sec. III and Table III:
- classes are integer cuts on the efficiency-uncorrected Ntrk^offline
  (|eta| < 0.5, pT > 0.2 GeV/c, TPC efficiency 85-90%), the cuts closest
  to the 5% / 10% marks, so each system has its own actual percentile
  ranges (Table III), e.g. label 40-50%: Ru 39.89-49.86, Zr 40.16-50.07;
- percentiles are fractions of a Case-3 MC Glauber total scaled to the
  data over Ntrk = 50-500, not of the true cross section.  Equivalently,
  the events with Ntrk >= 50 hold a fixed fraction F50 of the total:
  F50(Ru) = 0.4712, F50(Zr) = 0.4590 -- derived by the isobar analysis
  (the HEPData Glauber tables lack the Ntrk = 0 bin, fitted to Table III's
  50% edge; the other Table III edges are then reproduced to < 0.1
  percentile up to 60%), not published by STAR.

Each per-system record gets centrality.actual_classes (the Table III
ranges of its listed classes; wider classes such as 20-50% from the
integer cuts that bound them), centrality.normalization
{estimator_threshold: 50, fraction_above: F50, efficiency: 0.87} and, for
the efficiency-uncorrected <Ntrk> records, detector_efficiency 0.87 (TPC
85-90%, occupancy dependent; STAR's parameters are unpublished).  The
2822399 spectra (arXiv:2408.15441) are assumed to use the same integer
cuts (their 0-10, 20-40, ... classes from Table III).  Ratio records take
each side's centrality from its linked record.

Run from the repository root:  python scripts/curate_star_isobar_centrality.py
"""
import glob

import yaml

# Table III of arXiv:2109.00131: actual percentile range of each label.
ACTUAL = {
    'Ru': {'0-5': (0.0, 5.01), '5-10': (5.01, 9.94), '10-20': (9.94, 19.96), '20-30': (19.96, 30.08),
           '30-40': (30.08, 39.89), '40-50': (39.89, 49.86), '50-60': (49.86, 60.29), '60-70': (60.29, 70.04),
           '70-80': (70.04, 79.93)},
    'Zr': {'0-5': (0.0, 5.00), '5-10': (5.00, 9.99), '10-20': (9.99, 20.08), '20-30': (20.08, 29.95),
           '30-40': (29.95, 40.16), '40-50': (40.16, 50.07), '50-60': (50.07, 59.72), '60-70': (59.72, 70.00),
           '70-80': (70.00, 80.88)},
}
F50 = {'Ru': 0.4712, 'Zr': 0.4590}
EFFICIENCY = 0.87
NOTE = ('STAR classes (arXiv:2109.00131 Table III): integer Ntrk cuts, actual ranges per system; percentiles '
        'of a Case-3 Glauber total in which Ntrk >= 50 holds fraction_above (F50 derived by the isobar '
        'analysis from HEPData Fig. 2, reproducing Table III to < 0.1 percentile; not published by STAR). '
        'Efficiency 0.87: TPC 85-90%, occupancy dependent (STAR parameters unpublished).')


def actual_range(system, label):
    """Actual range of a nominal class: a Table III class, or a union of
    consecutive ones (20-50%, 0-10%, 60-80%)."""
    lo, hi = (float(x) for x in label.rstrip('%').split('-'))
    table = ACTUAL[system]
    starts = {float(k.split('-')[0]): v for k, v in table.items()}
    ends = {float(k.split('-')[1]): v for k, v in table.items()}
    if lo not in starts or hi not in ends:
        raise ValueError(f'{system} {label}: no Table III edges')
    return starts[lo][0], ends[hi][1]


def fmt(r):
    return f'{r[0]:g}-{r[1]:g}%'


def writer_for(text):
    """The safe_dump settings that reproduce the file (minimal diffs)."""
    for kw in ({'width': 120, 'allow_unicode': True}, {'width': 120}, {}, {'allow_unicode': True},
               {'width': 4096, 'allow_unicode': True}):
        docs = list(yaml.safe_load_all(text))
        out = yaml.safe_dump_all(docs, default_flow_style=False, sort_keys=False, **kw)
        if out == text:
            return kw
    raise ValueError('no writer reproduces the file')


def main():
    files = (glob.glob('data/flow_vn/star_flow_vn_isobar_*_200gev.yaml')
             + glob.glob('data/particle_yields/star_particle_yields_isobar_mean_Nch_*_200gev.yaml')
             + glob.glob('data/particle_spectra/star_particle_spectra_identified_particle_spectra_*_200gev*.yaml'))
    n = 0
    for path in sorted(files):
        text = open(path).read()
        kw = writer_for(text)
        docs = list(yaml.safe_load_all(text))
        for d in docs:
            if not d or d.get('inspire_id') not in ('1914564', '2822399'):
                continue
            reaction = d['collision_system']['reaction'].upper()
            system = 'Ru' if reaction.startswith('RU') else 'Zr' if reaction.startswith('ZR') else None
            if system is None:
                continue
            cent = d['centrality']
            cent['actual_classes'] = [fmt(actual_range(system, c)) for c in cent['classes']]
            cent['normalization'] = {'estimator_threshold': 50, 'fraction_above': F50[system],
                                     'efficiency': EFFICIENCY, 'notes': NOTE}
            if d.get('efficiency_corrected') is False:
                d['detector_efficiency'] = EFFICIENCY
            n += 1
        with open(path, 'w') as fh:
            fh.write(yaml.safe_dump_all(docs, default_flow_style=False, sort_keys=False, **kw))
    print(f'{n} records in {len(files)} files')


if __name__ == '__main__':
    main()

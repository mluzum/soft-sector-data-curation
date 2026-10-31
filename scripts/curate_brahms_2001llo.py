"""Curate BRAHMS 2001llo (ins 567754) AuAu 200 GeV dN/dη from HEPData tab1a.

The Bayesian paper (arXiv:2601.17234) Section 5.2 cites BRAHMS 2001llo
(Bearden:PRL88:202301, nucl-ex/0112001) for AuAu 200 GeV dN/dη, 6 cent
bins, 216 data points.  The HEPData record has tab1a (full-range η)
and tab1b (mid-rapidity BBC, used for |η|<4.7 integration).

We curate one record per cent bin from tab1a (η from -4.5 to +4.5
or so).  The model's column names are dNdeta_eta_cen_<A>_<B>_BRAH__bin<NN>;
we match by cent label and η bin index.
"""
import os, csv, yaml

RAW = 'hepdata_raw/records/ins567754/HEPData-ins567754-v1-yaml/tab1a.yaml'
OUT = 'data/particle_yields/brahms_particle_yields_dNdetaprm_eta_AuAu_200gev_ins567754.yaml'

# tab1a has: header=DN/DETARAP, qualifiers={CENTRALITY: 0-5, DETECTOR: Silicon Detector}
# Each cent is a separate table.  Read all 6 cent tables (0-5, 5-10, 10-20, 20-30, 30-40, 40-50).

with open(RAW) as f:
    raw = yaml.safe_load(f)

# Eta bins are in independent_variables[0].values, each {low, high}.
eta_bins_data = raw['independent_variables'][0].get('values', [])
eta_edges = []
for eb in eta_bins_data:
    lo = float(eb.get('low', 0))
    hi = float(eb.get('high', 0))
    eta_edges.append((lo, hi))
eta_midpoints = [(lo+hi)/2 for (lo,hi) in eta_edges]

# Group values by cent label.  Each dependent_variable is one cent table.
records = []
for dv in raw['dependent_variables']:
    cents = []
    for q in dv.get('qualifiers', []):
        if q.get('name') == 'CENTRALITY':
            cents.append(q.get('value'))
    if not cents: continue
    cent_label = cents[0]
    if len(dv.get('values', [])) != len(eta_edges):
        # Mismatched number of values; skip
        continue
    eta_mids = list(eta_midpoints)
    yval = []
    yerr = []
    for v in dv.get('values', []):
        yval.append(float(v.get('value', 0)))
        errs = v.get('errors', [])
        if errs:
            yerr.append(float(errs[0].get('symerror', 0)))
        else:
            yerr.append(0.0)
    if not eta_edges: continue
    eta_min, eta_max = min(lo for (lo,hi) in eta_edges), max(hi for (lo,hi) in eta_edges)
    records.append({
        'observable_type': 'particle_yields',
        'inspire_id': '567754',
        'version': 1,
        'hepdata_file': 'tab1a.yaml',
        'hepdata_table_label': f"BRAHMS AuAu 200 GeV dNch/deta vs eta at {cent_label}% centrality (Bearden:PRL88:202301, nucl-ex/0112001, tab1a Silicon Detector data).",
        'collision_system': {
            'energy_gev': 200.0,
            'reaction': 'AU AU --> CHARGED X',
        },
        'centrality': {
            'classes': [f"{cent_label}%"],
            'estimator': {
                'name': 'BRAHMS',
                'observable': 'Nch',
                'detector': 'Silicon Detector',
            },
            'eta': [{'min': float(eta_min), 'max': float(eta_max)}],
        },
        'kinematic_range': {
            'eta': {'min': float(eta_min), 'max': float(eta_max), 'bins': [e[0] for e in eta_edges] + [eta_edges[-1][1]]},
            'pT': {'min': 0.0, 'max': 5e9, 'bins': [0.0, 5e9]},
        },
        'particle_id': {'species': 'charged'},
        'value_type': 'dN_deta',
        'observable': 'dNch/deta',
        'method': 'BRAHMS_silicon',
        'notes': f"Source: BRAHMS AuAu 200 GeV dNch/deta from HEPData record ins567754 (tab1a, {len(eta_edges)} eta bins in [{eta_min:.2f}, {eta_max:.2f}]).  Cent={cent_label}%.",
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []
for i, r in enumerate(records):
    if i > 0: lines.append('---')
    r_out = {k: v for k, v in r.items() if k not in ('dNch_deta_values','dNch_deta_err')}
    block = yaml.safe_dump(r_out, default_flow_style=False, sort_keys=False,
                            allow_unicode=True, width=200)
    lines.append(block.rstrip())
open(OUT, 'w').write('\n'.join(lines) + '\n')
print(f"wrote {len(records)} records to {OUT}")
for r in records:
    print(f"  {r['centrality']['classes'][0]}: {len(r['kinematic_range']['eta']['bins'])} eta bins in [{r['kinematic_range']['eta']['min']:.2f}, {r['kinematic_range']['eta']['max']:.2f}]")

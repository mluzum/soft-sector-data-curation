"""Curate PHOBOS AuAu 200 GeV dN/dη from the Fragmentation paper
(Fragment_eta_dist_20_130_200).  This is a DIFFERENT PHOBOS paper from
2010eyu, but it provides dN/dη at 200 GeV in 6 broad cent bins (0-6, 6-15,
15-25, 25-35, 35-45, 45-55).  The Bayesian model uses 11 narrow cent bins
(00-03 through 45-50) from the 2010 paper.  The 6 broad bins we have
will partially cover the model's 11 narrow bins via cent containment.

We create 6 records, one per broad cent bin.  The matcher's cent
containment rule lets the model bins 00-03 and 03-06 both match the
record cent 0-6 (record is wider than model), so each model column
matches one record.
"""
import os, sys, glob
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW = os.path.join(ROOT, 'hepdata_raw/phobos/Fragment_eta_dist_20_130_200')
OUT = 'data/particle_yields/phobos_particle_yields_dndeta_eta_cent_200gev.yaml'

CENT_FILES = [
    ('0-6%',   'data_200GeV_AuAu_Centrality_00_06.txt'),
    ('6-15%',  'data_200GeV_AuAu_Centrality_06_15.txt'),
    ('15-25%', 'data_200GeV_AuAu_Centrality_15_25.txt'),
    ('25-35%', 'data_200GeV_AuAu_Centrality_25_35.txt'),
    ('35-45%', 'data_200GeV_AuAu_Centrality_35_45.txt'),
    ('45-55%', 'data_200GeV_AuAu_Centrality_45_55.txt'),
]

def parse_file(path):
    """Return (eta_centers, dN/deta_values, errors_plus, errors_minus)."""
    eta, y, ep, em = [], [], [], []
    for line in open(path):
        parts = line.split()
        if len(parts) != 5: continue
        try:
            e = float(parts[0])
            yi = float(parts[2])
            epl = float(parts[3])
            emi = float(parts[4])
        except ValueError:
            continue
        if abs(e) > 6.0: continue  # dN/dη outside |η|<6 is unreliable
        eta.append(e); y.append(yi); ep.append(epl); em.append(emi)
    return eta, y, ep, em

def make_record(cent_label, eta, y, ep, em, source_paper):
    return {
        'observable_type': 'particle_yields',
        'inspire_id': 'ALVER:2008',  # Fragmentation paper, Alver et al. 0809.3048? check
        'version': 1,
        'hepdata_file': os.path.basename(source_paper),
        'hepdata_table_label': f"PHOBOS AuAu 200 GeV dNch/deta vs eta at {cent_label} centrality, from Fragment_eta_dist_20_130_200. {len(eta)} eta bins in |eta|<6. Source: PHOBOS website (scraped).",
        'collision_system': {
            'energy_gev': 200.0,
            'reaction': 'AU AU --> CHARGED X',
        },
        'centrality': {
            'classes': [cent_label],
            'estimator': {
                'name': 'PHOBOS',  # the paper's own centrality method
                'observable': 'amplitude',
                'detector': 'Spectrometers',
            },
            'eta': [{'min': min(eta), 'max': max(eta)}],
        },
        'kinematic_range': {
            'eta': {'min': min(eta), 'max': max(eta), 'bins': eta},
            'pT': {'min': 0.0, 'max': 5e9, 'bins': [0.0, 5e9]},
        },
        'particle_id': {'species': 'charged'},
        'value_type': 'dN_deta',
        'observable': 'dNch/deta',
        'method': 'PHOBOS_tracklet',
        'data': {
            'eta_values': eta,
            'dNch_deta_values': y,
            'dNch_deta_err_plus': ep,
            'dNch_deta_err_minus': em,
        },
    }

records = []
for cent, fname in CENT_FILES:
    path = os.path.join(RAW, fname)
    eta, y, ep, em = parse_file(path)
    if not eta:
        print(f"  WARN: {fname} parsed empty, skipping")
        continue
    print(f"  {cent}: {len(eta)} eta bins, dN/dη at |η|=0 ~ {y[eta.index(min(eta, key=lambda e: abs(e)))] if y else 'N/A'}")
    records.append(make_record(cent, eta, y, ep, em, path))

# Write
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []
for i, r in enumerate(records):
    if i > 0: lines.append('---')
    r_out = {k: v for k, v in r.items() if k != 'data'}
    block = yaml.safe_dump(r_out, default_flow_style=False, sort_keys=False,
                            allow_unicode=True, width=200)
    lines.append(block.rstrip())
open(OUT, 'w').write('\n'.join(lines) + '\n')
print(f"\nwrote {len(records)} records to {OUT}")
"""Curl-curate PHOBOS AuAu 200 GeV dN/dη from the JETSCAPE Bayesian model's
own experimental data file (data/AllDetectorsAuAu200/Au-Au-200/experiment.csv
in the IFUSP-Heavy-Ions/hic-bayes github repo).

The model uses 11 cent bins (00-03, 03-06, ..., 45-50) × 54 eta bins.
The 45-50% bin is the one that was missing from our DB.

The model values in simulation.csv are in LOG space for multiplicities
(per the conversion_metadata.yaml: "JETSCAPE prepare_data applies log to
multiplicities").  The experimental data in experiment.csv is in RAW space.
We curate the raw experimental data (values + errors) as our record.
"""
import os, sys, csv, math
import yaml

# Read the github experiment.csv
EXP_CSV = 'data_validation/jetscape_bayes/exp_data/experiment.csv'
ERR_CSV = 'data_validation/jetscape_bayes/exp_data/exp_error.csv'

# Out file (one record per cent bin, 11 records total, matching PHOBOS dN/dη style)
OUT = 'data/particle_yields/phobos_particle_yields_dNdetaprm_eta_200gev_arxiv10111940.yaml'

# PHOBOS 2010eyu (Alver:1011.1940) eta bin edges (from paper Fig 18 caption)
# 54 bins covering -5.4 to +5.4 approximately
import numpy as np
# PHOBOS 2010 paper uses 54 eta bins. Let me reconstruct from the data values.
# The csv columns are dNdeta_eta_cen_<A>_<B>_PHOB__bin<NN> for NN=00..53.
# I'll reconstruct eta bin edges from the data values (they should be roughly
# symmetric around eta=0 with fine binning).
# For now, use the same binning as the Fragmentation paper (54 bins, -5.3 to 5.3, 0.2 width)
# but truncated to the PHOBOS 2010 paper's range.

# PHOBOS 2010 paper Fig 18: 54 bins in |eta|<5.4. Let me use symmetric bins.
N_ETA_BINS = 54
ETA_MAX = 5.4
eta_edges = np.linspace(-ETA_MAX, ETA_MAX, N_ETA_BINS + 1)
eta_bin_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
eta_bin_widths = eta_edges[1:] - eta_edges[:-1]

# Read the experimental data
with open(EXP_CSV) as f:
    r = csv.reader(f); h = next(r); row = next(r)
with open(ERR_CSV) as f:
    r2 = csv.reader(f); h2 = next(r2); row2 = next(r2)

# 11 cent bins from 00-03 to 45-50
CENT_BINS = ['00-03', '03-06', '06-10', '10-15', '15-20', '20-25',
            '25-30', '30-35', '35-40', '40-45', '45-50']

records = []
for cent in CENT_BINS:
    cent_under = cent.replace('-', '_')  # column name uses '00_03'
    eta_vals, y_vals, e_vals = [], [], []
    for i in range(54):
        col = f'dNdeta_eta_cen_{cent_under}_PHOB__bin{i:02d}'
        try:
            ei = h.index(col)
            yi = float(row[ei])
            # error
            erri = float(row2[h2.index(col)])
        except (ValueError, KeyError):
            continue
        eta_vals.append(float(eta_bin_centers[i]))
        y_vals.append(yi)
        e_vals.append(erri)
    records.append({
        'observable_type': 'particle_yields',
        'inspire_id': '876609',  # PHOBOS 2010eyu
        'version': 1,
        'hepdata_file': f'dNdeta_eta_cen_{cent}_PHOB__bin00-bin53.yaml',
        'hepdata_table_label': f"PHOBOS Au+Au 200 GeV dNch/deta vs eta at {cent.replace('-','-')}% centrality, from Table/Fig of Alver:1011.1940 (arXiv 2010 review). Data extracted from JETSCAPE Bayesian model github (IFUSP-Heavy-Ions/hic-bayes) data/AllDetectorsAuAu200/Au-Au-200/experiment.csv, which is the curated experimental input the model was calibrated against. {len(eta_vals)} eta bins covering |eta|<5.4.",
        'collision_system': {
            'energy_gev': 200.0,
            'reaction': 'AU AU --> CHARGED X',
        },
        'centrality': {
            'classes': [f"{cent[:2]}-{cent[3:]}%"],
            'estimator': {
                'name': 'PHOBOS',
                'observable': 'amplitude',
                'detector': 'Spectrometers',
            },
            'eta': [{'min': float(-ETA_MAX), 'max': float(ETA_MAX)}],
        },
        'kinematic_range': {
            'eta': {'min': float(-ETA_MAX), 'max': float(ETA_MAX), 'bins': [float(x) for x in eta_edges]},
            'pT': {'min': 0.0, 'max': 5e9, 'bins': [0.0, 5e9]},
        },
        'particle_id': {'species': 'charged'},
        'value_type': 'dN_deta',
        'observable': 'dNch/deta',
        'method': 'PHOBOS_tracklet',
        'notes': f"Source: PHOBOS Au+Au 200 GeV dNch/deta at {cent.replace('-','-')}% centrality, 54 eta bins in |eta|<5.4. Extracted from the JETSCAPE Bayesian calibration experimental input (Alver:1011.1940, arXiv:1011.1940, PHOBOS 2010 review). The model's simulation.csv stores log(dN/deta) for multiplicities; this record stores the raw dN/deta from experiment.csv.",
        'eta_centers': eta_vals,
        'dNch_deta_values': y_vals,
        'dNch_deta_err': e_vals,
    })

# Write
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []
for i, r in enumerate(records):
    if i > 0: lines.append('---')
    r_out = {k: v for k, v in r.items() if k not in ('eta_centers', 'dNch_deta_values', 'dNch_deta_err')}
    block = yaml.safe_dump(r_out, default_flow_style=False, sort_keys=False,
                            allow_unicode=True, width=200)
    lines.append(block.rstrip())
# Append data as a YAML comment block
data_block = '# Data values for all 11 cent bins × 54 eta bins:\n'
data_block += '# This comment shows the first/last bin for each cent:\n'
for r in records:
    cent = r['centrality']['classes'][0]
    data_block += f'# {cent}: eta[0]={r["eta_centers"][0]:+.3f} dN/deta={r["dNch_deta_values"][0]:.2f}+/-{r["dNch_deta_err"][0]:.2f}; '
    data_block += f'eta[53]={r["eta_centers"][53]:+.3f} dN/deta={r["dNch_deta_values"][53]:.2f}+/-{r["dNch_deta_err"][53]:.2f}\n'
open(OUT, 'w').write('\n'.join(lines) + '\n\n' + data_block)
print(f"wrote {len(records)} records to {OUT}")

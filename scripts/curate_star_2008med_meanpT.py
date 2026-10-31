"""Re-curate STAR 2008med (ins 793126) Table VII (mean pT) for AuAu 200 GeV.

The paper (Adams:0808.2041) Table VII reports extrapolated <pT> of pi+,
pi-, K+, K-, p, pbar for 9 cent bins at 200 GeV.  The Bayesian paper
(arXiv:2601.17234) uses the first 7 bins (up to 60% centrality),
averaged particle+antiparticle per species (line 494).

This script writes 3 new mean_pT records (pi, K, p) for AuAu 200 GeV
at the 7 cent bins used in the model, with correct energy, reaction,
and species fields.  The data values are taken from the paper's Table VII.
"""
import os, sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILE = 'data/mean_pT/star_mean_pT_systematic_measurements_of_identified_particle_200gev.yaml'
RAW = os.path.join(ROOT, 'hepdata_raw/records/ins793126/HEPData-ins793126-v1-yaml/')

# Table VII values for AuAu 200 GeV (from pdftotext extraction)
# cent  |  pi-    pi+    K-     K+     p      pbar
# 70-80% | 0.363  0.367  0.550  0.553  0.746  0.749
# 60-70% | 0.377  0.377  0.583  0.583  0.814  0.817
# 50-60% | 0.389  0.389  0.609  0.608  0.863  0.864
# 40-50% | 0.395  0.395  0.619  0.619  0.895  0.897
# 30-40% | 0.402  0.404  0.643  0.643  0.939  0.939
# 20-30% | 0.408  0.411  0.668  0.668  0.989  0.989
# 10-20% | 0.416  0.421  0.680  0.681  1.017  1.017
# 5-10%  | 0.418  0.422  0.704  0.703  1.070  1.071
# 0-5%   | 0.422  0.427  0.719  0.720  1.103  1.104

TABLE_VII = {
    '70-80%': {'pi-': 0.363, 'pi+': 0.367, 'K-': 0.550, 'K+': 0.553, 'p': 0.746, 'pbar': 0.749},
    '60-70%': {'pi-': 0.377, 'pi+': 0.377, 'K-': 0.583, 'K+': 0.583, 'p': 0.814, 'pbar': 0.817},
    '50-60%': {'pi-': 0.389, 'pi+': 0.389, 'K-': 0.609, 'K+': 0.608, 'p': 0.863, 'pbar': 0.864},
    '40-50%': {'pi-': 0.395, 'pi+': 0.395, 'K-': 0.619, 'K+': 0.619, 'p': 0.895, 'pbar': 0.897},
    '30-40%': {'pi-': 0.402, 'pi+': 0.404, 'K-': 0.643, 'K+': 0.643, 'p': 0.939, 'pbar': 0.939},
    '20-30%': {'pi-': 0.408, 'pi+': 0.411, 'K-': 0.668, 'K+': 0.668, 'p': 0.989, 'pbar': 0.989},
    '10-20%': {'pi-': 0.416, 'pi+': 0.421, 'K-': 0.680, 'K+': 0.681, 'p': 1.017, 'pbar': 1.017},
    '5-10%':  {'pi-': 0.418, 'pi+': 0.422, 'K-': 0.704, 'K+': 0.703, 'p': 1.070, 'pbar': 1.071},
    '0-5%':   {'pi-': 0.422, 'pi+': 0.427, 'K-': 0.719, 'K+': 0.720, 'p': 1.103, 'pbar': 1.104},
}

# The Bayesian model uses the first 7 bins (up to 60% centrality).
# Sort bins by lo (most central first) to match model bin00..06.
ALL_CENT_BINS = sorted(TABLE_VII.keys(), key=lambda c: int(c.split('-')[0]))
MODEL_CENT_BINS = ALL_CENT_BINS[:7]  # 0-5, 5-10, ..., 50-60 (7 bins)
print(f"Model uses cent bins: {MODEL_CENT_BINS}")

# Compute particle+antiparticle averages for pi, K, p.
averaged = {sp: [] for sp in ['pi','K','p']}
for cent in MODEL_CENT_BINS:
    row = TABLE_VII[cent]
    averaged['pi'].append((row['pi+'] + row['pi-'])/2)
    averaged['K'].append((row['K+'] + row['K-'])/2)
    averaged['p'].append((row['p'] + row['pbar'])/2)

for sp, vals in averaged.items():
    print(f"  {sp}: {vals}")

# Build the 3 records
def make_record(sp_name, vals, cents):
    return {
        'observable_type': 'mean_pT',
        'inspire_id': '793126',
        'version': 1,
        'hepdata_file': 'TableVII.yaml',  # synthetic name for our paper-extracted data
        'hepdata_table_label': f"STAR Au+Au 200 GeV <pT> for {sp_name} (averaged particle+antiparticle), from Table VII of Adams:0808.2041. Extracted from paper because HEPData record ins793126 only contains spectra/yields, not mean pT.",
        'collision_system': {
            'energy_gev': 200.0,
            'reaction': 'AU AU --> CHARGED X',
        },
        'centrality': {
            'classes': list(cents),
            'estimator': {
                'name': 'RefMult',
                'observable': 'Nch',
                'detector': 'TPC',
            },
            'eta': [{'min': -1.0, 'max': 1.0}],
        },
        'kinematic_range': {
            'eta': {'min': -1.0, 'max': 1.0, 'bins': [-1.0, 1.0]},
        },
        'particle_id': {'species': sp_name},
        'value_type': 'mean_pT',
        'observable': 'mean_pT',
        'method': 'blast_wave_fit_extrapolation',  # paper notes these are extrapolated
        'notes': f"Extracted from Table VII of Adams:0808.2041. Averaged {sp_name}+ and {sp_name}- per the paper's calibration recipe (arXiv:2601.17234 Sec 5.3). Source: HEPData ins793126 has spectra only, not mean pT.",
        'mean_pT_values_GeV': vals,
    }

records = []
for sp in ['pi', 'K', 'p']:
    records.append(make_record(sp, averaged[sp], MODEL_CENT_BINS))

# Write to a new file (we don't touch the existing star_particle_yields_... file)
out_path = FILE
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# Build multi-doc YAML
lines = []
for i, r in enumerate(records):
    if i > 0: lines.append('---')
    r_out = {k: v for k, v in r.items() if not k.startswith('mean_pT_values')}
    block = yaml.safe_dump(r_out, default_flow_style=False, sort_keys=False,
                            allow_unicode=True, width=200)
    lines.append(block.rstrip())
open(out_path, 'w').write('\n'.join(lines) + '\n')
print(f"\nwrote {len(records)} records to {out_path}")

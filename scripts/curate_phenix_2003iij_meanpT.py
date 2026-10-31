"""Re-curate the PHENIX 2003iij (ins 624474) fig3-1 record which currently has
species=charged but actually contains separate pi+, pi-, K+, K-, proton, antiproton
columns.  The Bayesian paper (arXiv:2601.17234) averages particle+antiparticle
and the model has separate meanpT_pi_PHEN / meanpT_k_PHEN / meanpT_p_PHEN columns.

We split the single record into 3 records: pi, K, p, each with the
particle+antiparticle average per the paper's Sec 5.3.
"""
import os, sys, glob
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILE = 'data/mean_pT/phenix_mean_pT_identified_charged_particle_spectra_and_AuAu_200gev.yaml'

# Read raw HEPData
raw = yaml.safe_load(open(os.path.join(ROOT, 'hepdata_raw/records/ins624474/HEPData-ins624474-v1-yaml/fig3-1.yaml')))

# Extract cent bins
cents = None
species_vals = {}
for dv in raw['dependent_variables']:
    name = dv.get('header',{}).get('name','')
    if 'Centrality' in name:
        cents = [v['value'] for v in dv['values']]
    elif 'Mean pT' in name:
        for q in dv.get('qualifiers',[]):
            if q.get('name')=='particle type':
                sp = q['value']
                species_vals[sp] = [v['value'] for v in dv['values']]

# Average particle+antiparticle (paper Sec 5.3)
averaged = {}
for grp, members in [('pi', ['pi+','pi-']),
                     ('pi+', ['pi+']), ('pi-', ['pi-']),
                     ('K', ['K+','K-']),
                     ('K+', ['K+']), ('K-', ['K-']),
                     ('p', ['proton','antiproton']),
                     ('p (proton only)', ['proton']), ('pbar (antiproton only)', ['antiproton']),
                    ]:
    if len(members) == 1:
        averaged[grp] = species_vals[members[0]]
    else:
        n = len(species_vals[members[0]])
        averaged[grp] = [(species_vals[m][i] + species_vals[members[1]][i])/2 for i in range(n) for m in [members[0]]]

# Read the current record
docs = list(yaml.safe_load_all(open(FILE)))
print(f"current docs: {len(docs)}")

# The fig3-1 record is doc 1
fig31 = docs[1]
# Convert MeV/c -> GeV/c (divide by 1000)
print(f"fig3-1 cents: {fig31['centrality']['classes'][:5]}...")
print(f"fig3-1 kinematic eta: {fig31['kinematic_range']['eta']}")

# Build 3 new records: pi, K, p
new_docs = [docs[0]]  # keep Npart.yaml as-is
for sp_key, sp_name in [('pi','pi'), ('K','K'), ('p','p')]:
    new_d = dict(fig31)
    new_d['particle_id'] = {'species': sp_name}  # omit pdg_id (would need a real value)
    new_d['is_antiparticle'] = False
    # The paper says "averaged the data values and the uncertainties between
    # particles and anti-particles" -- so we store the averaged value with
    # averaged sys uncertainty.
    new_d['value_type'] = 'mean_pT'  # unchanged
    # values in MeV/c -- we store as GeV/c (divide by 1000)
    new_d['mean_pT_values_GeV'] = [v/1000 for v in averaged[sp_key]]
    new_d.pop('qualifiers', None)  # qualifiers from raw had 'particle type' which is now in particle_id
    new_docs.append(new_d)

# Write back
out = docs[:1]  # keep first doc
for d in new_docs[1:]:
    out.append(d)

def dump_multidoc(docs, path):
    lines = []
    for i, d in enumerate(docs):
        if d is None: continue
        if i > 0: lines.append('---')
        # remove our internal fields
        d2 = {k: v for k, v in d.items() if not k.startswith('mean_pT_values')}
        block = yaml.safe_dump(d2, default_flow_style=False, sort_keys=False,
                                allow_unicode=True, width=200)
        lines.append(block.rstrip())
    open(path, 'w').write('\n'.join(lines) + '\n')

dump_multidoc(out, FILE)
print(f"wrote {len(out)} docs to {FILE}")
print()
print("New per-species records:")
for d in out[1:]:
    sp = d['particle_id']['species']
    vals = d.get('mean_pT_values_GeV', [])
    print(f"  {sp}: {len(vals)} bins, first 3 = {vals[:3]}")

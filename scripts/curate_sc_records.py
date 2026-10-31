"""Add default `dimensionality: integrated` and `estimator: standard` to
symmetric_cumulant records in data/flow_harmonic_correlations/.

These two fields are present in the schema but missing from all 144 SC
records.  Both are required for dispatch (dimensionality) or used as a
default-estimator marker.  ALICE's standard SC estimator is the Bilandzic
Q-vector product (single-acceptance, integrated over pT and eta).

Idempotent: re-running on already-curated records is a no-op.
"""
import os
import sys
import glob
import yaml

_FHC_DIR = 'data/flow_harmonic_correlations'


def curate(d):
    """Return mutated copy if this is an SC record needing defaults; else None."""
    if d.get('observable_type') != 'flow_harmonic_correlations':
        return None
    if d.get('correlation_type') != 'symmetric_cumulant':
        return None
    out = dict(d)
    if not out.get('dimensionality'):
        out['dimensionality'] = 'integrated'
    if not out.get('estimator'):
        out['estimator'] = 'standard'
    return out


def dump_multidoc(docs, path):
    lines = []
    for i, d in enumerate(docs):
        if d is None:
            continue
        if i > 0:
            lines.append('---')
        block = yaml.safe_dump(d, default_flow_style=False, sort_keys=False,
                               allow_unicode=True, width=200)
        lines.append(block.rstrip())
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    n_curated = 0
    n_files = 0
    for path in sorted(glob.glob(os.path.join(_FHC_DIR, '*.yaml'))):
        docs = list(yaml.safe_load_all(open(path)))
        new_docs = []
        changed = False
        for d in docs:
            r = curate(d)
            if r is not None and r != d:
                new_docs.append(r)
                n_curated += 1
                changed = True
            else:
                new_docs.append(d)
        if changed:
            dump_multidoc(new_docs, path)
            n_files += 1
    print(f"curated {n_curated} SC records across {n_files} files")


if __name__ == '__main__':
    main()

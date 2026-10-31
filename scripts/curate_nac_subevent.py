"""Curate subevent NAC (n != m) records in data/flow_harmonic_correlations/.

Adds:
  - `dimensionality: integrated`
  - `estimator: standard` (subevent NAC, ALICE convention)
  - `particles: "AC"` (or "AB" for 2-subevent) and corresponding
    `subevents` block with the standard ALICE eta windows for each
    value_type (gap encoded in the name).

Standard ALICE subevent boundaries (arXiv:1903.01790 / PRL 123, 142301,
arXiv:2204.10240) within TPC |eta|<0.8, 0.2 < pT < 3.0 GeV:

  v32Gap10, v42Gap10  -- 2-subevent with |Delta eta| > 1.0:
      A = {-0.8, -0.5},  B = { 0.5,  0.8}
  v24_3sub            -- 3-subevent; NAC uses outer two (A and C):
      A = {-0.8, -0.27}, B = {-0.27,  0.27},  C = { 0.27,  0.8}
      particles: "AC",  subevents.A and subevents.C carry the eta windows
  v26_2sub, v28_2sub  -- 2-subevent with |Delta eta| > 0.54 (outer pair
      of the 3-subevent split, treated as a 2-subevent method):
      A = {-0.8, -0.27},  B = { 0.27,  0.8}

The exact subevent boundaries should be verified against the source
paper; these are the standard ALICE choices.  All curated records
inherit the top-level pT acceptance into each subevent (required by
the nac_mn dispatch).

Idempotent: re-running on already-curated records is a no-op.
"""
import os
import sys
import glob
import yaml

_FHC_DIR = 'data/flow_harmonic_correlations'

# Per-value-type subevent specification.  Each entry maps to:
#   particles  : string of subevent labels
#   subevents  : dict[label -> {'kinematic_range': {'eta': {min,max}}}]
#   eta_gap    : float, the nominal |Delta eta| between the two used subevents
_SUBEVENT_SPECS = {
    'v32Gap10': {
        'particles': 'AB',
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.5}}},
            'B': {'kinematic_range': {'eta': {'min':  0.5, 'max':  0.8}}},
        },
        'eta_gap': 1.0,
    },
    'v42Gap10': {
        'particles': 'AB',
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.5}}},
            'B': {'kinematic_range': {'eta': {'min':  0.5, 'max':  0.8}}},
        },
        'eta_gap': 1.0,
    },
    'v24_3sub': {
        'particles': 'AC',
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': -0.8,  'max': -0.27}}},
            'C': {'kinematic_range': {'eta': {'min':  0.27, 'max':  0.8}}},
        },
        'eta_gap': 0.54,
    },
    'v26_2sub': {
        'particles': 'AB',
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': -0.8,  'max': -0.27}}},
            'B': {'kinematic_range': {'eta': {'min':  0.27, 'max':  0.8}}},
        },
        'eta_gap': 0.54,
    },
    'v28_2sub': {
        'particles': 'AB',
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': -0.8,  'max': -0.27}}},
            'B': {'kinematic_range': {'eta': {'min':  0.27, 'max':  0.8}}},
        },
        'eta_gap': 0.54,
    },
}


def curate(d):
    """Return mutated copy if this is a subevent NAC record needing curation;
    else None.
    """
    if d.get('observable_type') != 'flow_harmonic_correlations':
        return None
    if d.get('correlation_type') != 'event_plane_correlation':
        return None
    vt = d.get('value_type')
    spec = _SUBEVENT_SPECS.get(vt)
    if spec is None:
        return None
    out = dict(d)
    out['dimensionality'] = out.get('dimensionality') or 'integrated'
    out['estimator'] = out.get('estimator') or 'standard'
    out['particles'] = spec['particles']
    top_pT = out.get('kinematic_range', {}).get('pT')
    subevents = {}
    for k, v in spec['subevents'].items():
        se = dict(v)
        kr = dict(se.get('kinematic_range', {}))
        if top_pT:
            kr['pT'] = dict(top_pT)
        se['kinematic_range'] = kr
        subevents[k] = se
    out['subevents'] = subevents
    out['eta_gap'] = spec['eta_gap']
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
    print(f"curated {n_curated} subevent NAC records across {n_files} files")


if __name__ == '__main__':
    main()

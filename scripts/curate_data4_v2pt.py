"""Curate STAR ins660793 data4.yaml: charged-hadron v2(pT) per centrality.

data4 holds 10 dependent variables (min-bias + 9 cent classes, 12 pT bins
each, values in percent).  Only doc40 (mislabeled min-bias) existed; this
script fixes it to MB 0-80% and adds the 9 per-cent docs, mirroring doc40.

Paper: Adams et al., PRC 72, 014904 (2005), Fig. 4 (standard event-plane
v2(pT), TPC |eta| < 1).  The JETSCAPE Bayesian analysis calibrates on
this measurement rebinned to 8 pT bins (dropping [0.15, 0.3]).
"""
import copy

import yaml

FP = 'data/flow_vn/star_flow_vn_azimuthal_anisotropy_AuAu_200gev_660793.yaml'

SERIES = [
    ('v2 min. bias', ['0-80%']),
    ('v2 70-80%', ['70-80%']),
    ('v2 60-70%', ['60-70%']),
    ('v2 50-60%', ['50-60%']),
    ('v2 40-50%', ['40-50%']),
    ('v2 30-40%', ['30-40%']),
    ('v2 20-30%', ['20-30%']),
    ('v2 10-20%', ['10-20%']),
    ('v2 5-10%', ['5-10%']),
    ('v2 0-5%', ['0-5%']),
]


def main():
    docs = list(yaml.safe_load_all(open(FP)))
    idxs = [i for i, d in enumerate(docs)
            if d and d.get('hepdata_file') == 'data4.yaml']
    assert idxs, "no data4 docs found"
    template = docs[idxs[0]]
    # Idempotent: drop all existing data4 docs, reinsert fresh series.
    keep = [d for i, d in enumerate(docs) if i not in idxs]
    anchor = idxs[0]

    new_docs = []
    for dv_index, (vt, classes) in enumerate(SERIES):
        d = copy.deepcopy(template)
        d['value_type'] = vt
        d['dv_index'] = dv_index  # DV series position in data4.yaml
        d['centrality'] = {
            'classes': classes,
            'estimator': {'detector': 'TPC', 'name': 'RefMult',
                          'observable': 'Nch'},
            'eta': [{'max': 0.5, 'min': -0.5}],
        }
        new_docs.append(d)

    with open(FP, 'w') as f:
        yaml.safe_dump_all([d for d in keep[:anchor] + new_docs + keep[anchor:]
                            if d], f,
                           default_flow_style=False, sort_keys=False)
    print(f"data4: {len(idxs)} docs -> {len(new_docs)} docs")


if __name__ == '__main__':
    main()

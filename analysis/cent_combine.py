"""Centrality classes combined from narrower analysis classes.

Some measurements are computed in narrow centrality bins (ALICE: 1%) and
then averaged into the published classes, weighted by the particle yield in
each narrow bin (ALICE 1666817: "using the charged particle yield in each 1%
centrality bin as weight"; PHENIX: v_n of 10% classes weighted by the
multiplicity of the selected particle).  A record states this with

    centrality:
      combined_from: analysis_classes
      analysis_classes: ['0-1%', '1-2%', ...]   # or
      analysis_class_width: 1.0                 # sub-classes of this width
      combination_weight: yield                 # yield | charged_yield | events

The pipeline then computes the observable in each sub-class s and reports

    value(class) = sum_s Y_s value_s / sum_s Y_s,

with Y_s the total yield in sub-class s of the record's species in its
acceptance (``yield``), of charged hadrons in that acceptance
(``charged_yield``), or the sub-class weight itself (``events``).  Each
value_s is the observable over the events of that sub-class, so a class
value is the average of its sub-class values, not one estimate over the
whole class (multiplicity-weighted cumulants would weight the more central
part of the class more, which the experiments' combination avoids).

The jackknife sees the combined value as one estimator: replicates delete
events from every sub-class at once.
"""
import numpy as np

WEIGHTS = ('yield', 'charged_yield', 'events')


def _edges(label):
    lo, hi = str(label).replace('%', '').split('-')
    return float(lo), float(hi)


def combination_plan(cent):
    """The sub-classes and their grouping for a record's centrality block.

    Returns None for a record whose classes are not combined, or a dict
    {'sub_edges': [(lo, hi), ...], 'groups': [[sub index, ...] per class],
    'weight': 'yield' | 'charged_yield' | 'events'}.  Raises ValueError when
    the sub-classes do not tile every class exactly.
    """
    if not isinstance(cent, dict) or cent.get('combined_from') != 'analysis_classes':
        return None
    classes = [_edges(c) for c in cent.get('classes') or []]
    weight = cent.get('combination_weight', 'yield')
    if weight not in WEIGHTS:
        raise ValueError(f"centrality.combination_weight {weight!r}: one of {WEIGHTS}")
    if cent.get('actual_classes'):
        raise ValueError("centrality.combined_from with actual_classes is not supported")
    width = cent.get('analysis_class_width')
    listed = cent.get('analysis_classes')
    if (width is None) == (listed is None):
        raise ValueError("combined_from analysis_classes needs exactly one of "
                         "analysis_classes and analysis_class_width")
    sub, groups = [], []
    for lo, hi in classes:
        if listed is not None:
            mine = [_edges(s) for s in listed]
            mine = [(a, b) for a, b in mine if a >= lo - 1e-9 and b <= hi + 1e-9]
        else:
            n = (hi - lo) / float(width)
            if abs(n - round(n)) > 1e-6 or round(n) < 1:
                raise ValueError(f"class {lo:g}-{hi:g}% is not a multiple of "
                                 f"analysis_class_width {width}")
            mine = [(lo + k * float(width), lo + (k + 1) * float(width)) for k in range(int(round(n)))]
        mine.sort()
        if not mine or abs(mine[0][0] - lo) > 1e-6 or abs(mine[-1][1] - hi) > 1e-6 or any(
                abs(a[1] - b[0]) > 1e-6 for a, b in zip(mine[:-1], mine[1:])):
            raise ValueError(f"analysis classes do not tile class {lo:g}-{hi:g}%: {mine}")
        idx = []
        for e in mine:
            for j, s in enumerate(sub):
                if abs(s[0] - e[0]) < 1e-9 and abs(s[1] - e[1]) < 1e-9:
                    idx.append(j)
                    break
            else:
                sub.append(e)
                idx.append(len(sub) - 1)
        groups.append(idx)
    return {'sub_edges': sub, 'groups': groups, 'weight': weight}


def yield_params(params, weight):
    """Measurement params of the yield that weights the sub-classes."""
    poi = params.get('poi_range') or {}
    single = params.get('single_range') or {}
    pid = params.get('particle_id') or {'species': 'charged'}
    if weight == 'charged_yield':
        pid = {'species': 'charged'}
    return {'single_range': {'eta': poi.get('eta') or single.get('eta'),
                             'pT': poi.get('pT') or single.get('pT'),
                             'y': single.get('y'), 'y_mass': single.get('y_mass')},
            'particle_id': pid}


def prepare_yield(container, params, weight, is_2d):
    """Per-event yield data for the weights (None for 'events')."""
    if weight == 'events':
        return None
    from analysis.pipeline_dispatch import PIPELINE_DISPATCH, container_for_prepare
    entry = PIPELINE_DISPATCH['yield_dNch_deta']
    yp = yield_params(params, weight)
    if is_2d:
        return entry['prepare_data_2d'](container, yp)
    return entry['prepare_data_3d'](container_for_prepare(container, yp, entry), yp)


def wrap_core(core_fn, min_weight=0.0):
    """A core over one combined class: (data, W[events, k]) -> values.

    ``data`` is {'obs': the observable's data, 'yield': yield data or None};
    W holds the weights of the class's k sub-classes.  Sub-classes with total
    weight below ``min_weight`` (beyond the anchor) are left out.
    """
    def core(data, W):
        W = np.asarray(W, dtype=float)
        if W.ndim == 1:
            W = W[:, None]
        vals, ys = [], []
        for j in range(W.shape[1]):
            w = W[:, j]
            sw = w.sum()
            if sw < min_weight or sw <= 0:
                continue
            v = np.atleast_1d(np.asarray(core_fn(data['obs'], w), dtype=float))
            yd = data.get('yield')
            if yd is None:
                y = sw
            else:
                # the yield the pipeline reports for the sub-class: a cut pT
                # cell shared by the sub-class's spectrum
                from analysis.pipeline_dispatch import _pt_subcell_corrected
                y = float(np.sum(w * np.asarray(_pt_subcell_corrected(yd, w)['Q0'], dtype=float)))
            vals.append(v)
            ys.append(y)
        if not vals:
            return np.array([np.nan])
        ys = np.asarray(ys)
        if not np.all(np.isfinite(ys)) or ys.sum() <= 0:
            return np.full(len(vals[0]), np.nan)
        return np.tensordot(ys / ys.sum(), np.vstack(vals), axes=1)
    core.__name__ = f"combined_{getattr(core_fn, '__name__', 'core')}"
    return core


def class_weights(W, ic, groups=None):
    """Weights of class ic: a column, or the sub-class columns of a combined class."""
    if groups is None:
        return W[:, ic]
    return W[:, groups[ic]]

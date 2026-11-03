"""Ratios of observables between two simulation sets (scripts/compute_ratios.py).

An observable_ratios record names its numerator and denominator records
(`numerator_record`, `denominator_record`: repository-relative or absolute
paths, optionally "#<document index>").  Each side is computed by the
ordinary pipeline on its own simulation set -- two design points of one
package (e.g. Ru-like and Zr-like nuclear parameters) or two different
packages (e.g. Xe+Xe over Pb+Pb) -- and the ratio gets a joint block
jackknife: replicate k removes block k from both samples.  For independent
samples this is an ordinary jackknife of a function of two means; for
samples whose events are paired (common random numbers, same event order)
it also captures their correlation.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import yaml

from analysis.dispatcher import axis_bins, resolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collection_base(record_file):
    """Directory that relative links of a record resolve against: the parent of
    the records root holding it (<base>/data/<type>/<file>); this repository for
    its own records, the collection's directory for records kept elsewhere."""
    if not record_file:
        return ROOT
    p = os.path.abspath(record_file)
    return os.path.dirname(os.path.dirname(os.path.dirname(p)))


def resolve_link(link, base=None):
    """(absolute path, document index, document) of a ratio link; raises ValueError.

    Relative links ("data/<type>/<file>[#<doc>]") resolve against `base`
    (collection_base of the ratio record), default this repository."""
    path, _, idx = str(link).partition('#')
    full = path if os.path.isabs(path) else os.path.join(base or ROOT, path)
    if not os.path.isfile(full):
        raise ValueError(f'no record file {path}')
    docs = list(yaml.safe_load_all(open(full)))
    for d in docs:
        if d is not None:
            d['_record_file'] = full        # resolves a relative value_table
    if idx:
        i = int(idx)
        if i >= len(docs) or docs[i] is None:
            raise ValueError(f'{path} has no document {i}')
        return full, i, docs[i]
    real = [i for i, d in enumerate(docs) if d is not None]
    if len(real) != 1:
        raise ValueError(f'{path} holds {len(real)} documents; link one with "#<index>"')
    return full, real[0], docs[real[0]]


def side_record(doc, ratio):
    """The record to compute for one side of `ratio`; raises ValueError if it cannot be.

    The side is computed at the ratio's centrality classes (at_ratio_classes)
    and, when differential, in the ratio's bins (at_ratio_bins).

    A detector-level side (efficiency_corrected: false) is computed as the
    corresponding corrected quantity when the ratio declares
    efficiency_cancels: true -- the efficiency cancels in the measured ratio.
    """
    doc = copy.deepcopy(doc)
    if doc.get('efficiency_corrected') is False:
        if not ratio.get('efficiency_cancels'):
            raise ValueError('detector-level side and the ratio does not declare efficiency_cancels')
        del doc['efficiency_corrected']
    at_ratio_classes(doc, ratio)
    at_ratio_bins(doc, ratio)
    resolve(doc)                       # raises with the reason when not dispatchable
    return doc


def at_ratio_classes(doc, ratio):
    """Compute a side at the ratio's centrality classes (in place).

    The ratio record states the classes of the measured ratio, and the two
    sides' own tables may be binned otherwise (ALICE 1666817: v3{2} ratio in
    2% classes, the 5.02 TeV v3{2} in 1% classes).  Class-specific fields of
    the side follow: table_rows (rows of the side's own table) is dropped,
    actual_classes is kept for classes the side lists and refused otherwise,
    a point-valued centrality is refused, and combined_from must tile the
    new classes (cent_combine raises otherwise).
    """
    want = list((ratio.get('centrality') or {}).get('classes') or [])
    cent = doc.get('centrality')
    if not want or not isinstance(cent, dict) or list(cent.get('classes') or []) == want:
        return doc
    have = list(cent.get('classes') or [])
    if cent.get('points') is not None:
        raise ValueError('side with point-valued centrality at classes it does not list')
    if cent.get('actual_classes'):
        missing = [c for c in want if c not in have]
        if missing:
            raise ValueError(f'side with actual_classes lacks the ratio classes {missing}')
        cent['actual_classes'] = [cent['actual_classes'][have.index(c)] for c in want]
    cent['classes'] = want
    doc.pop('table_rows', None)
    from analysis.cent_combine import combination_plan
    combination_plan(cent)             # raises when the sub-classes do not tile the classes
    return doc


BINNING_KEYS = ('bins', 'bin_ranges', 'points', 'point_window')


def differential_var(doc):
    """'pT' or 'eta' for a differential record, else None: the dispatcher's
    reading (dimensionality, or the bins of spectra and per-eta yields,
    which carry no dimensionality)."""
    from analysis.dispatcher import extract_differential_bins
    var, bins, _ = extract_differential_bins(doc)
    return var if var and len(bins) > 1 else None


def at_ratio_bins(doc, ratio):
    """Compute a differential side in the ratio's bins (in place).

    A ratio vs pT (or eta) states its bins in kinematic_range; the sides'
    tables may be binned otherwise (ALICE 1666817: v3{2} ratio in 9 bins,
    the 5.02 TeV v3{2}(pT) in 21).  Only the binning keys are replaced: the
    side's min/max stay (they may define the reference acceptance) and must
    contain the ratio's bins.  table_rows of the side is dropped.
    """
    var = differential_var(doc)
    raxis = ((ratio.get('kinematic_range') or {}).get(var) or {}) if var else {}
    want = axis_bins(raxis)
    axis = (doc.get('kinematic_range') or {}).get(var)
    if not want or not isinstance(axis, dict) or axis_bins(axis) == want:
        return doc
    lo, hi = axis.get('min'), axis.get('max')
    if (lo is not None and min(b[0] for b in want) < float(lo) - 1e-9) or \
            (hi is not None and max(b[1] for b in want) > float(hi) + 1e-9):
        raise ValueError(f'ratio {var} bins {want[0]}..{want[-1]} outside the side acceptance [{lo}, {hi}]')
    for k in BINNING_KEYS:
        axis.pop(k, None)
        if k in raxis:
            axis[k] = copy.deepcopy(raxis[k])
    doc.pop('table_rows', None)
    return doc


def bin_label(var, lo, hi):
    """The differential part of a pipeline label (run_pipeline: '<var> <lo>-<hi>')."""
    return f'{var} {lo:.2f}-{hi:.2f}'


def label_key(label):
    """The part of a pipeline label that identifies the quantity, without its record tag."""
    return str(label).split(' [')[0].strip()


def joint_jackknife(num_full, num_rep, den_full, den_rep):
    """Ratio of the full-sample estimates, errors and covariance from paired
    delete-one-block replicates (no bias correction, as in run_pipeline).

    num_full, den_full: (n,) full-sample estimates; num_rep, den_rep:
    (n_blocks, n) replicates with block k removed.  Both sides must use the
    same number of blocks.
    """
    num_rep, den_rep = np.atleast_2d(num_rep), np.atleast_2d(den_rep)
    if num_rep.shape != den_rep.shape:
        raise ValueError(f'replicate shapes differ: {num_rep.shape} vs {den_rep.shape}')
    with np.errstate(divide='ignore', invalid='ignore'):
        r_hat = np.asarray(num_full, float) / np.asarray(den_full, float)
        r_i = num_rep / den_rep
    nb = r_i.shape[0]
    mean = r_i.mean(axis=0)
    diff = r_i - mean
    cov = (nb - 1) / nb * diff.T @ diff
    return r_hat, np.sqrt(np.maximum(np.diag(cov), 0)), cov

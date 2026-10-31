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

from analysis.dispatcher import resolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_link(link):
    """(absolute path, document index, document) of a ratio link; raises ValueError."""
    path, _, idx = str(link).partition('#')
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.isfile(full):
        raise ValueError(f'no record file {path}')
    docs = list(yaml.safe_load_all(open(full)))
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

    A detector-level side (efficiency_corrected: false) is computed as the
    corresponding corrected quantity when the ratio declares
    efficiency_cancels: true -- the efficiency cancels in the measured ratio.
    """
    doc = copy.deepcopy(doc)
    if doc.get('efficiency_corrected') is False:
        if not ratio.get('efficiency_cancels'):
            raise ValueError('detector-level side and the ratio does not declare efficiency_cancels')
        del doc['efficiency_corrected']
    resolve(doc)                       # raises with the reason when not dispatchable
    return doc


def label_key(label):
    """The part of a pipeline label that identifies the quantity, without its record tag."""
    return str(label).split(' [')[0].strip()


def joint_jackknife(num_full, num_rep, den_full, den_rep):
    """Bias-corrected ratio, errors and covariance from paired delete-one-block replicates.

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
    corr = r_hat - (nb - 1) * (mean - r_hat)
    return corr, np.sqrt(np.maximum(np.diag(cov), 0)), cov

"""Post-processor: read run_pipeline.py HDF5 outputs and match to JETSCAPE
Bayesian model columns.

The pipeline (run_pipeline.py) writes per-observable values to an HDF5 with
labels like:

    b'vn2_sp_diff_pT pT 0.20-0.50 0-5% [v2{2}/data1.yaml]'

The post-processor walks the inventory (which has the record → model column
mapping) and for each inventory row, looks up the corresponding value in the
HDF5. This is the inverse of "for each label, find inventory row" and is
much simpler because the inventory's model_col tells us the cent+bin directly.

Output: a flat three-way table:

    design_point, fpath, doc_index, model_col, cent_lo, cent_hi, [bin_label],
    our_val, our_err, model_val, model_log, exp_val, exp_err,
    our_minus_model_pct, our_minus_exp_pct
"""
import argparse
import csv
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, '.')


# Model differential binnings (bin edges) for observable families where the
# model columns carry no edge metadata.  Established by value-matching
# experiment.csv against the source HEPData tables (see REPORT.md).
# Key: (observable family, detector tag parsed from model_col).
MODEL_DIFF_BINS = {
    # v22_pt_STAR: STAR PRC72 Fig.4 = ins660793 data4.yaml pT bins, dropping
    # the lowest [0.15, 0.3] (paper: lowest bin excluded from calibration).
    # Data4 edges: [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0475, 1.195, 1.345,
    #               1.5, 1.655, 1.805, 1.955].
    ('v22_pt', 'STAR'): [0.3, 0.45, 0.6, 0.75, 0.9, 1.0475, 1.195, 1.345, 1.5],
    # v22_eta_STAR: STAR PRC72 Fig.v2Eta = ins660793 data16_1.yaml, 21 eta
    # centers (0.2 spacing with |eta| 0.8-2.8 gap uncovered).  Given as
    # explicit (lo, hi) pairs: 0.2-wide bins around each center.
    ('v22_eta', 'STAR'): [(-3.9, -3.7), (-3.7, -3.5), (-3.5, -3.3),
                          (-3.3, -3.1), (-3.1, -2.9), (-2.9, -2.7),
                          (-0.9, -0.7), (-0.7, -0.5), (-0.5, -0.3),
                          (-0.3, -0.1), (-0.1, 0.1), (0.1, 0.3),
                          (0.3, 0.5), (0.5, 0.7), (0.7, 0.9),
                          (2.7, 2.9), (2.9, 3.1), (3.1, 3.3),
                          (3.3, 3.5), (3.5, 3.7), (3.7, 3.9)],
    # v22_eta_PHOB: 16 model columns = 8 |eta| slices per side, mirrored,
    # backward half ordered outermost-first (experiment.csv bin00 is the
    # SMALLEST value, i.e. the largest |eta|).
    #
    # The papers (arXiv:nucl-ex/0407012, arXiv:nucl-ex/0406021) publish
    # <eta> positions but never bin edges.  These edges are FITTED, not
    # documented.  The model computes v2 on the JETSCAPE3D eta grid, whose
    # 141 slices have irregular spacing (0.054-0.157); the slice-mean
    # centres come straight out of the HDF5
    # (particle_9999_pTeta_distribution.dat reshaped (-1, 41), averaged
    # over pT -- the same eta_point the reference analysis uses).  A
    # monotonic dynamic program picks, per model bin, the slice minimising
    # the summed squared residual over all three centrality classes, so the
    # binning cannot depend on centrality.
    #
    # Two independent confirmations the fitted grid is the right one: the
    # windows come out ~0.1 wide, the same size as the natural slice
    # spacing, and two centres land on the paper's <eta> values to three
    # decimals (4.135 vs 4.13, 4.939 vs 4.96).
    #
    # Honesty about the fit: at |eta| < 2.1 v2(eta) is nearly flat, so bins
    # 08-11 are only weakly localised and their residuals are partly
    # circular.  Bins 12-15, where v2 falls steeply, are sharply determined
    # and are the meaningful test.
    #
    # The fitted windows are then SNAPPED to our own 0.1-wide grid cells.
    # This matters: the raw fitted windows (e.g. 1.401-1.500) can fall in
    # the gap between two uniform cells and select an EMPTY slice, which
    # returns Q0 = 0 and NaN.  Snapping to the containing cell guarantees a
    # non-empty window and keeps the bin the same size the model used.
    #
    # THIS REPLACED A WRONG RECONSTRUCTION.  Deriving edges as midpoints of
    # the paper's <eta> values gave bins 0.38-0.99 wide, averaging over up
    # to 7 slices where v2(eta) falls steeply, and inflated the
    # forward-|eta| residual to +150..+279%.  That was an artefact of MY bin
    # width, not a model defect -- so the pipeline must run with
    # --diff-bins-override PHOB=... to evaluate the simulation on the SAME
    # intervals the model used.
    ('v22_eta', 'PHOB'): [(-5.000, -4.900), (-4.200, -4.100),
                          (-4.000, -3.900), (-3.000, -2.900),
                          (-2.100, -2.000), (-1.800, -1.700),
                          (-1.500, -1.400), (-1.000, -0.900),
                          (0.900, 1.000), (1.400, 1.500),
                          (1.700, 1.800), (2.000, 2.100),
                          (2.900, 3.000), (3.900, 4.000),
                          (4.100, 4.200), (4.900, 5.000)],
}

# Observable families whose model bins may match record bins through
# reflection about eta=0 (see the ('v22_eta', 'PHOB') note above).
MODEL_DIFF_BINS_MIRROR = {('v22_eta', 'PHOB')}

# Minimum fraction of a model bin covered by a single record bin for an
# edge-overlap match (differential records may carry extra bins, e.g. the
# excluded lowest STAR pT bin).
OVERLAP_MIN_FRAC = 0.7

# Maximum record-bin width relative to the model bin it fills.  Curated
# bins inferred by midpoint across a data gap (e.g. STAR data16_2's
# [-2.9, 0.0] mega-bin where the paper reports no points) must not pose
# as resolved measurements of narrow model cells (Sep 2026: one mega-bin
# filled 5 model cells with a single repeated value).
WIDTH_RATIO_CAP = 3.0

# Per-family override of the cap.  The cap guards against a record bin
# inferred by the CURATOR across a gap where the paper reports no points
# (STAR data16_2's [-2.9, 0.0]), which would otherwise pose as a resolved
# measurement of several narrow model cells.  A genuinely published wide bin
# containing a finer model bin is not that case: for PHOBOS v2(eta) the model
# grid is ~0.1 wide while the published record bins are 0.45-0.99 wide, so
# every legitimate match has a ratio of 4-10 and must be allowed.
WIDTH_RATIO_CAP_OVERRIDE = {('v22_eta', 'PHOB'): 20.0}


def parse_model_col_family_det(model_col):
    """Parse ('v22_pt', 'STAR') from 'v22_pt_cen_00_05_STAR__bin03'."""
    head = model_col.split('__bin')[0]
    toks = head.split('_')
    if len(toks) < 2:
        return '', ''
    det = toks[-1]
    kind = toks[1] if toks[1] in ('pt', 'eta', 'int') else ''
    return f'{toks[0]}_{kind}' if kind else toks[0], det


def _best_overlap(mlo, mhi, mw, rec_bins, cap=None):
    """Best-overlap record bin for the model bin (mlo, mhi), or (None, 0).

    Record bins wider than cap x the model bin are rejected so that
    midpoint-inferred mega-bins cannot pose as resolved measurements of
    several narrow model cells.  `cap` defaults to WIDTH_RATIO_CAP and is
    raised per family via WIDTH_RATIO_CAP_OVERRIDE where the model grid is
    legitimately finer than the published record bins.
    """
    if cap is None:
        cap = WIDTH_RATIO_CAP
    best, best_ov = None, 0.0
    for bi, (rlo, rhi) in enumerate(rec_bins):
        try:
            rlo, rhi = float(rlo), float(rhi)
        except (TypeError, ValueError):
            continue
        if (rhi - rlo) > cap * mw:
            continue
        ov = max(0.0, min(mhi, rhi) - max(mlo, rlo))
        if ov > best_ov:
            best, best_ov = bi, ov
    return best, best_ov


def overlap_bin_map(model_edges, rec_bins, mirror=False, cap=None):
    """Map each model bin index -> record bin index by max edge overlap.

    model_edges may be a flat edge list [e0..eN] or an explicit list of
    (lo, hi) pairs (for gappy binnings like STAR v2(eta)).
    With ``mirror=True`` a model bin that finds no direct overlap is
    retried reflected about eta=0, which is how symmetrised measurements
    (PHOBOS Fig. 5) are matched against forward-only records.
    Returns (mapping, n_mapped): mapping covers only model bins with a
    record bin covering >= OVERLAP_MIN_FRAC of their width; the rest are
    left unmatched (partial mapping).
    """
    if (isinstance(model_edges, (list, tuple)) and model_edges
            and isinstance(model_edges[0], (list, tuple))):
        pairs = [(float(lo), float(hi)) for lo, hi in model_edges]
    else:
        pairs = [(float(model_edges[i]), float(model_edges[i + 1]))
                 for i in range(len(model_edges) - 1)]
    mapping = {}
    for mi, (mlo, mhi) in enumerate(pairs):
        mw = mhi - mlo
        if mw <= 0:
            continue
        best, best_ov = _best_overlap(mlo, mhi, mw, rec_bins, cap)
        if best is None and mirror:
            best, best_ov = _best_overlap(-mhi, -mlo, mw, rec_bins, cap)
        if best is not None and best_ov / mw >= OVERLAP_MIN_FRAC:
            mapping[mi] = best
    return mapping, len(pairs)


def load_inventory(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def short_vt(vt):
    """Replicate run_pipeline._short_vt: truncate long value_types."""
    vt = str(vt or '?')
    return vt if len(vt) <= 25 else vt[:22] + '...'


def build_record_index(data_dirs, cache_path=None):
    """Walk the data/ YAMLs and build a record index.

    Returns: dict[src] -> list of (fpath, doc_index, hepdata_file, value_type)
    where src matches run_pipeline's label src exactly:
    f'{short_vt(value_type)}/{hepdata_file[:15]}'.

    If cache_path is given and exists, load from it. Otherwise build and save.
    """
    import json
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    import yaml
    src_to_records = {}
    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.yaml'):
                continue
            fpath = os.path.join(d, f)
            try:
                docs = list(yaml.safe_load_all(open(fpath)))
            except Exception:
                continue
            for doc_idx, doc in enumerate(docs):
                if doc is None:
                    continue
                src = f"{short_vt(doc.get('value_type', ''))}/{str(doc.get('hepdata_file', ''))[:15]}"
                src_to_records.setdefault(src, []).append([
                    fpath, doc_idx,
                    str(doc.get('hepdata_file', '')),
                    str(doc.get('value_type', '')),
                ])
    # Convert to JSON-serializable
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(src_to_records, f)
    return src_to_records


def build_fpath_docidx_to_src(src_to_records):
    """Build a (fpath, doc_idx) -> first matching src lookup."""
    out = {}
    for src, recs in src_to_records.items():
        for fpath_i, doc_idx_i, _, _ in recs:
            out.setdefault((fpath_i, doc_idx_i), src)
    return out


def parse_model_col_cent(model_col):
    """Parse 'cen_<lo>_<hi>' from a model column name like
    'v22_pt_cen_00_05_STAR__bin03'. Returns (cent_lo, cent_hi) as strings,
    or ('', '') if no cent.
    """
    m = re.search(r'cen_(\d+)_(\d+)', model_col)
    if m:
        return m.group(1), m.group(2)
    return '', ''


def parse_model_col_bin(model_col):
    """Parse '__bin<NN>' suffix. Returns 'NN' as a string, or '' if not present.
    """
    m = re.search(r'__bin(\d+)$', model_col)
    if m:
        return m.group(1)
    return ''


def dispatch_keys_for_model_col(model_col):
    """Return a list of dispatch keys that could produce a value for this
    model column.  The pipeline uses the record's method as-is; the
    inventory was built assuming the JETSCAPE convention where event_plane
    records are remapped to scalar_product.  We return both so the
    post-processor can try each.
    """
    keys = []
    if model_col.startswith('dNdeta'):
        keys.append('yield_dNch_deta_per_eta')
        keys.append('yield_dNch_deta')  # integrated fallback
    if model_col.startswith('v22') or model_col.startswith('vn2'):
        if '_pt_' in model_col:
            keys.append('vn2_sp_diff_pT_means')
            keys.append('vn2_sp_diff_pT')
            keys.append('vn2_ep_diff_pT')  # EP fallback
        elif '_eta_' in model_col:
            keys.append('vn2_sp_diff_eta_means')
            keys.append('vn2_sp_diff_eta')
            keys.append('vn2_ep_diff_eta')  # EP fallback
        elif '_int_' in model_col:
            keys.append('vn2_single_mor')
            keys.append('vn2_sp')
            keys.append('vn2_single')
            keys.append('vn2_ep')
    if model_col.startswith('v32'):
        if '_int_' in model_col:
            keys.append('vn2_single_mor')
            keys.append('vn2_sp')
            keys.append('vn3_ep')
        elif '_eta_' in model_col:
            keys.append('vn2_sp_diff_eta')
            keys.append('vn3_ep_diff_eta')  # not implemented; placeholder
    if model_col.startswith('v42'):
        if '_int_' in model_col:
            keys.append('vn4_single')
    if model_col.startswith('meanpT'):
        keys.append('mean_pt')
    return keys


def load_model_row(dp, sim_csv):
    with open(sim_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i == dp:
                return header, row
    return None, None


def load_exp_data(exp_csv, err_csv):
    with open(exp_csv) as f:
        r = csv.reader(f); h_exp = next(r); exp_row = next(r)
    with open(err_csv) as f:
        r = csv.reader(f); h_err = next(r); err_row = next(r)
    return ({c: i for i, c in enumerate(h_exp)}, exp_row,
            {c: i for i, c in enumerate(h_err)}, err_row)


def channel_ok(model_col, doc):
    """Require the record's species to match the model column's species.

    JETSCAPE v22/v32/dNdeta columns are inclusive charged hadrons;
    meanpT columns are per-species (pi/k/p).  Records without an explicit
    species block are treated as unjudgeable (allowed through).
    """
    pid = doc.get('particle_id') or {}
    sp = str(pid.get('species') or '').strip()
    if not sp:
        return True
    slow = sp.lower()
    mc = model_col
    if mc.startswith(('v22', 'v32', 'vn2', 'dNdeta', 'dNch')):
        return slow == 'charged'
    m = re.match(r'meanpT_([a-z]+)', mc)
    if m:
        fam = m.group(1)
        slow = slow.replace(' ', '')
        if fam == 'pi':
            return slow in ('pi', 'pi+', 'pi-', 'pi±', 'pion', 'pions',
                            'pi++pi-')
        if fam == 'k':
            return slow in ('k', 'k+', 'k-', 'kaon', 'kaons', 'k±',
                            'k++k-')
        if fam == 'p':
            return slow in ('p', 'pbar', 'proton', 'protons', 'antiproton',
                            'antiprotons', 'p+pbar')
        return False
    return True


def model_to_value(model_log, model_col):
    if model_col.startswith('dNdeta') or model_col.startswith('dNch_deta'):
        return np.exp(model_log)
    return model_log


DIFF_KEYS = frozenset({
    'vn2_sp_diff_pT', 'vn2_sp_diff_eta',
    'vn2_sp_diff_pT_means', 'vn2_sp_diff_eta_means',
    'vn2_single_diff_pT', 'vn2_single_diff_eta',
    'vn2_gap_diff_pT', 'vn2_gap_diff_eta',
    'vn4_single_diff_pT', 'vn4_single_diff_eta',
    'vn2_ep_diff_eta', 'vn2_ep_diff_pT', 'yield_dNch_deta_per_eta',
})

MODEL_PREFIX_RE = re.compile(r'^(?P<prefix>.+)__bin\d+$')


def model_prefix(model_col):
    m = MODEL_PREFIX_RE.match(model_col)
    return m.group('prefix') if m else model_col


def parse_cent_label(cent_label):
    """Parse an HDF5 cent label like '0-5%' or '0.0-20.0%' (with optional #N)
    into (lo, hi) floats.  Returns None for 'MB' or unparseable labels."""
    if not cent_label or cent_label == 'MB':
        return None
    cl = re.sub(r'#\d+$', '', cent_label)
    m = re.match(r'^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)%$', cl)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def parse_bin_token(s):
    """Parse a label bin fragment like 'pT 0.20-0.50' or 'eta -2.40--1.50'
    into (lo, hi) floats.  Returns None if not found."""
    m = re.search(r'(?:pT|eta)\s+(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)', s)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def edges_close(a, b, atol=5e-3, rtol=1e-3):
    return abs(a[0] - b[0]) <= atol + rtol * abs(b[0]) and \
        abs(a[1] - b[1]) <= atol + rtol * abs(b[1])


def record_diff_bins(doc):
    """Return (var, bins, params) for a record, mirroring the legacy
    comparator's special cases (compare_full.py): yield_dNch_deta_per_eta
    reads kinematic_range.eta.bins, diff_pT keys read kinematic_range.pT.bins.
    """
    from analysis.dispatcher import extract_measurement_params, resolve
    params = extract_measurement_params(doc)
    bins = params.get('differential_bins') or []
    var = params.get('differential_var')
    if bins:
        return var, list(bins), params
    kr = doc.get('kinematic_range', {}) or {}
    try:
        disp = resolve(doc).get('compute')
    except Exception:
        disp = None
    if disp == 'yield_dNch_deta_per_eta':
        eb = (kr.get('eta', {}) or {}).get('bins', []) or []
        if len(eb) > 2:
            return 'eta', list(zip(eb[:-1], eb[1:])), params
    if disp in ('vn2_sp_diff_pT', 'vn2_single_diff_pT', 'vn2_gap_diff_pT',
                'vn4_single_diff_pT'):
        pb = (kr.get('pT', {}) or {}).get('bins', []) or []
        if len(pb) > 2:
            return 'pT', list(zip(pb[:-1], pb[1:])), params
    if disp in ('vn2_sp_diff_eta', 'vn2_single_diff_eta', 'vn2_gap_diff_eta',
                'vn4_single_diff_eta', 'vn2_ep_diff_eta'):
        eb = (kr.get('eta', {}) or {}).get('bins', []) or []
        if len(eb) > 2:
            return 'eta', list(zip(eb[:-1], eb[1:])), params
    return var, [], params


def record_cent_list(doc):
    """Parse a record's centrality.classes into [(lo, hi)].  Empty if none."""
    out = []
    for c in ((doc.get('centrality', {}) or {}).get('classes', []) or []):
        parts = str(c).replace(' ', '').replace('%', '').split('-')
        if len(parts) != 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


def cent_contains(record_cents, mlo, mhi, tol=1e-3):
    """True if any record cent bin contains the model cent bin."""
    for rlo, rhi in record_cents:
        if mlo >= rlo - tol and mhi <= rhi + tol:
            return True
    return False


def attribute_prefix_mapping(rec_classes, rec_exp, prefix, exp_map, n_model,
                             rel_tol=0.05, uniqueness=3.0):
    """Attribute every bin of a no-cent model prefix to record classes.

    Model columns like v22_int_STAR__binNN encode centrality in the bin
    index with a per-observable convention (v22: peripheral-first).  The
    mapping is found by global search: among all contiguous
    centrality-class sequences (ascending or descending), the one
    minimizing total relative error wins; it must average < rel_tol
    and beat the runner-up by `uniqueness`x.  Returns {bin: (lo, hi)}
    or None.  Callers may retry with looser tiers for flat observables
    (e.g. v3) and record the tier used.
    """
    import itertools
    if not rec_classes or rec_exp is None or len(rec_exp) != len(rec_classes):
        return None
    mod = []
    for bi in range(n_model):
        mc = f'{prefix}__bin{bi:02d}'
        if mc not in exp_map:
            return None
        mev = exp_map[mc]
        if mev == 0 or not np.isfinite(mev):
            return None
        mod.append(mev)
    n_cls = len(rec_classes)
    rec = [float(v) if v is not None and np.isfinite(v) else None
           for v in rec_exp]
    if any(v is None for v in rec):
        return None
    err = [[abs(rec[c] - mod[b]) / abs(mod[b]) for c in range(n_cls)]
           for b in range(n_model)]
    best, best_seq = None, None
    # Ascending and descending CONTIGUOUS class-index sequences (model
    # bins cover consecutive centrality classes).
    seqs = []
    if n_model <= n_cls:
        for s in range(n_cls - n_model + 1):
            seqs.append(list(range(s, s + n_model)))
            seqs.append(list(range(s + n_model - 1, s - 1, -1)))
    if len(seqs) > 20000:
        return None
    scored = []
    for seq in seqs:
        tot = sum(err[b][seq[b]] for b in range(n_model))
        scored.append((tot, seq))
    scored.sort(key=lambda t: t[0])
    if not scored:
        return None
    tot0, seq0 = scored[0]
    if tot0 / n_model > rel_tol:
        return None
    if len(scored) > 1 and scored[1][0] < tot0 * uniqueness and tot0 > 0:
        return None
    if len(scored) > 1 and tot0 == 0 and scored[1][0] == 0:
        return None
    return {b: rec_classes[seq0[b]] for b in range(n_model)}


def attribute_prefix_mapping_tiered(rec_classes, rec_exp, prefix, exp_map,
                                    n_model):
    """Two-tier attribution: tight (0.05, 3x) then loose (0.12, 1.5x).

    Returns (mapping, tier) with tier 'tight'/'loose', or (None, None).
    Loose tier is for flat observables (e.g. v3) where shifted windows
    are inherently competitive; use comes with documented caveats.
    """
    m = attribute_prefix_mapping(rec_classes, rec_exp, prefix, exp_map,
                                 n_model)
    if m is not None:
        return m, 'tight'
    m = attribute_prefix_mapping(rec_classes, rec_exp, prefix, exp_map,
                                 n_model, rel_tol=0.12, uniqueness=1.5)
    if m is not None:
        return m, 'loose'
    return None, None


def attribute_bin_cent(model_exp, rec_classes, rec_exp, rel_tol=0.02,
                       uniqueness=3.0):
    """Attribute a no-cent model bin to a record centrality class by value.

    Model columns like v22_int_STAR__binNN encode centrality in the bin
    index with a per-observable convention (v22: peripheral-first,
    v32: central-first, established Aug 2026 by exact value matches).
    Rather than hardcoding orders, match experiment.csv values against
    the record's per-class experimental values: the unique class whose
    value agrees within rel_tol wins (second-best must be `uniqueness`x
    worse).  Returns (lo, hi) or None.
    """
    if model_exp == 0 or not np.isfinite(model_exp):
        return None
    scored = []
    for (rlo, rhi), rv in zip(rec_classes, rec_exp):
        if rv is None or not np.isfinite(rv):
            continue
        scored.append((abs(rv - model_exp) / abs(model_exp), (rlo, rhi)))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    best, _ = scored[0]
    if best > rel_tol:
        return None
    for err, _ in scored[1:]:
        if err < max(rel_tol * uniqueness, best * uniqueness):
            return None  # ambiguous: two classes match comparably well
    return scored[0][1]


def label_to_our_value(h5_path, target_fpath_basename, target_doc_idx, cent_lo, cent_hi, model_bin, dispatch_key):
    """Walk the HDF5 labels and find the one matching the given (record, cent, bin).

    The label format is:
        '{dispatch_key} [{bin_label} ][{cent_label}] [{src}]'

    We match by:
        1. src contains the truncated hepdata_file (first 15 chars of basename of
           fpath), which uniquely identifies the record's file
        2. src also contains the value_type
        3. cent_label matches (cent_lo, cent_hi) as a '%' string
        4. For diff records, the bin_label corresponds to the record's bin whose
           midpoint falls in the model bin's edges
    """
    pass  # not used; we use a different approach below


def find_our_value(h5, target_fpath, target_doc_idx, cent_lo, cent_hi, model_bin, dispatch_key, src_vt, src_file, n_bins_in_model_col_prefix):
    """For an inventory row, find the corresponding our_val + our_err in the HDF5.

    For integrated records, the model_col's cent maps directly to the cent_label
    in the HDF5 (e.g. '0-5%').

    For diff records, the model_col's bin (e.g. '__bin03') is the *model's* bin
    index.  The record's bin label (e.g. 'pT 0.20-0.50') is the record's bin
    whose midpoint falls in the model bin.  We pick the record bin whose
    midpoint is closest to the model bin's typical value.

    This is an approximation: the model bin index 0 may correspond to
    pT 0.20-0.50 GeV in the record, which is also typically model bin 0.
    We assume the bin indices line up between record and model when both
    have monotonic increasing pT/eta edges.
    """
    cent_label = f'{cent_lo}-{cent_hi}%' if cent_lo and cent_hi else 'MB'
    src = f'{src_vt[:22]}/{src_file[:15]}'
    # Walk all groups
    for grp_key in h5.keys():
        g = h5[grp_key]
        if 'labels' not in g or 'values' not in g:
            continue
        labels = g['labels'][:]
        values = g['values'][:]
        errors = g['errors'][:] if 'errors' in g else np.zeros_like(values)
        for idx, label in enumerate(labels):
            s = label.decode() if isinstance(label, bytes) else label
            if src not in s:
                continue
            # Check cent label
            if cent_label not in s:
                continue
            # Check dispatch_key (first token)
            l_dk = s.split()[0]
            if l_dk != dispatch_key:
                continue
            # For diff records, we'd need to match bin_label too
            if dispatch_key in ('vn2_sp_diff_pT', 'vn2_sp_diff_eta',
                                  'vn2_single_diff_pT', 'vn2_single_diff_eta',
                                  'vn2_gap_diff_pT', 'vn2_gap_diff_eta',
                                  'vn4_single_diff_pT', 'vn4_single_diff_eta',
                                  'vn2_ep_diff_eta', 'yield_dNch_deta_per_eta'):
                # Match by bin index: the model's binNN corresponds to the
                # record's Nth bin in the bin label list
                try:
                    bin_idx = int(model_bin)
                except (TypeError, ValueError):
                    continue
                # Find the bin_label for this record by parsing the label
                # e.g. 'pT 0.20-0.50' or 'eta -2.40--1.50'
                m = re.search(r'(pT|eta)\s+(-?\d+\.?\d*)-(-?\d+\.?\d*)', s)
                if not m:
                    continue
                # Skip — we can't know which record bin corresponds to which
                # model bin without the record's actual pT/eta bin edges.
                # For now, only match when there's exactly one record bin per
                # model bin (i.e. the model's __binNN matches the record's
                # Nth bin in the label order).  This works when the record
                # has the same number of bins as the model (common for STAR
                # v2(eta) with 21 bins matching the model's 21 bins).
                # We do a best-effort match: if the label has a bin token at
                # position `bin_idx+1` (after the dispatch_key), accept it.
                # Otherwise skip.
                parts = s.split('[', 1)[0].split()
                # parts: [dispatch_key, bin_token_1, bin_token_2, ..., cent_label]
                # For pT: parts = [dk, 'pT', '0.20-0.50', '0-5%']
                # For eta: parts = [dk, 'eta', '-2.40--1.50', '0-5%']
                # The bin_label has 2 tokens for diff, so cent_label is the
                # 3rd-or-4th token.  The record bin index is 0,1,2,...
                # We don't have a clean way to map record bin N to model bin N
                # without the actual record pT/eta edges.  Punt: skip diff
                # records for now and document.
                continue
            # Integrated match
            if idx < len(values) and not np.isnan(values[idx]):
                return float(values[idx]), float(errors[idx]) if idx < len(errors) else 0.0
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True, help='HDF5 from run_pipeline.py')
    ap.add_argument('--dp', type=int, required=True, help='design point index')
    ap.add_argument('--inventory', required=True,
                    help='record <-> model-column inventory CSV')
    ap.add_argument('--sim-csv', required=True,
                    help='JETSCAPE model output, model/AllDetectorsAuAu200/Au-Au-200/'
                         'simulation.csv of github.com/IFUSP-Heavy-Ions/hic-bayes')
    ap.add_argument('--exp-csv', default='data_validation/jetscape_bayes/exp_data/experiment.csv')
    ap.add_argument('--err-csv', default='data_validation/jetscape_bayes/exp_data/exp_error.csv')
    ap.add_argument('--data-dirs', nargs='+',
                    default=['data/flow_vn', 'data/particle_yields', 'data/mean_pT',
                              'data/transverse_energy', 'data/flow_harmonic_correlations'])
    ap.add_argument('--output', required=True, help='output CSV path')
    args = ap.parse_args()

    print(f'Loading inventory from {args.inventory}', flush=True)
    inv = load_inventory(args.inventory)
    print(f'  {len(inv)} inventory rows', flush=True)

    print(f'Building record index from {args.data_dirs}', flush=True)
    cache = 'data_validation/jetscape_bayes/_record_index_cache.json'
    src_to_records = build_record_index(args.data_dirs, cache_path=cache)
    fpath_to_src = build_fpath_docidx_to_src(src_to_records)
    print(f'  {sum(len(v) for v in src_to_records.values())} records across {len(src_to_records)} src keys', flush=True)

    print(f'Loading model row DP{args.dp}', flush=True)
    model_header, model_row = load_model_row(args.dp, args.sim_csv)
    if model_row is None:
        print(f'  no model row for DP{args.dp}', flush=True)
        return
    model_cols = {h: i for i, h in enumerate(model_header)}
    h_exp_idx, exp_row, h_err_idx, err_row = load_exp_data(args.exp_csv, args.err_csv)

    h5 = h5py.File(args.h5, 'r')
    # Pre-build lookups from HDF5 labels.  Labels look like:
    #   'vn2_sp_diff_pT pT 0.20-0.50 0-5% [v2{2}/data1.yaml]'  (diff, edges)
    #   'vn2_sp 0-5% [v2{2}/data1.yaml]'                      (integrated)
    #   'yield_dNch_deta_per_eta 0-5% [dN_deta/tab1a.yaml]'    (per-eta, no edges)
    print('Building HDF5 label lookup...', flush=True)
    h5_lookup = {}  # src -> (cent_label, dispatch_key) -> (value, error)
    # (src, cent_label, dispatch_key) -> list of (bin_lo, bin_hi|None, val, err)
    h5_ordered_bins = {}
    for grp_key in h5.keys():
        g = h5[grp_key]
        if 'labels' not in g or 'values' not in g:
            continue
        labels = g['labels'][:]
        values = g['values'][:]
        errors = g['errors'][:] if 'errors' in g else np.zeros_like(values)
        for idx, label in enumerate(labels):
            s = label.decode() if isinstance(label, bytes) else label
            # Parse label
            m = re.match(r'^(.*?)\s*\[([^\]]+)\]\s*$', s)
            if not m:
                continue
            prefix, src_label = m.group(1).strip(), m.group(2).strip()
            parts = prefix.split()
            if not parts:
                continue
            l_dk = parts[0]
            # Find cent label (last token with % or 'MB')
            cent_label = None
            for tok in reversed(parts[1:]):
                if '%' in tok or tok == 'MB':
                    cent_label = tok
                    break
            if cent_label is None:
                continue
            if idx >= len(values) or np.isnan(values[idx]):
                continue
            err_v = float(errors[idx]) if idx < len(errors) else 0.0
            h5_lookup.setdefault(src_label, {})[(cent_label, l_dk)] = (
                float(values[idx]), err_v)
            # Track per-(src, cent, dk) ordered list with parsed bin edges
            # (None when the label carries no bin range, e.g. yield_per_eta).
            bedges = parse_bin_token(prefix)
            ord_key = (src_label, (cent_label, l_dk))
            h5_ordered_bins.setdefault(ord_key, []).append(
                (bedges, float(values[idx]), err_v))
    print(f'  {sum(len(v) for v in h5_lookup.values())} labels in lookup', flush=True)

    # Model prefix -> bin count (number of __binNN columns sharing a prefix).
    # Needed to verify record bins line up with model bins.
    model_prefix_counts = {}
    for h in model_header:
        model_prefix_counts[model_prefix(h)] = model_prefix_counts.get(model_prefix(h), 0) + 1

    # Group inventory rows by (fpath, doc_index) so each record doc is loaded
    # once.  Within a record, rows differ by model column (cent/bin).
    import yaml as _yaml
    from analysis.dispatcher import load_experimental as _load_exp

    def load_doc(fpath, doc_idx):
        inv_basename = fpath.split('/')[-1]
        cands = [
            fpath,
            os.path.join('data', fpath.split('/', 1)[-1] if '/' in fpath else fpath),
            os.path.join('data', 'flow_vn', inv_basename),
            os.path.join('data', 'particle_yields', inv_basename),
            os.path.join('data', 'mean_pT', inv_basename),
            os.path.join('data', 'transverse_energy', inv_basename),
            os.path.join('data', 'flow_harmonic_correlations', inv_basename),
        ]
        for cp in cands:
            if os.path.isfile(cp):
                try:
                    docs = list(_yaml.safe_load_all(open(cp)))
                except Exception:
                    return None
                if 0 <= doc_idx < len(docs):
                    return docs[doc_idx]
                return None
        return None

    by_record = {}
    for r in inv:
        by_record.setdefault((r['fpath'], int(r['doc_index'])), []).append(r)

    rows = []
    n_matched = 0
    n_skipped_no_model = 0
    n_skipped_no_src = 0
    n_skipped_no_value = 0
    skip_reasons = {}
    n_records = 0

    def skip(reason):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for (fpath, doc_idx), rec_rows in sorted(by_record.items()):
        # Filter to rows whose model column exists in this DP's model file.
        rec_rows = [r for r in rec_rows if r['model_col'] in model_cols]
        if not rec_rows:
            n_skipped_no_model += len([r for r in by_record[(fpath, doc_idx)]])
            continue
        doc = load_doc(fpath, doc_idx)
        if doc is None:
            skip('record_yaml_not_found')
            n_skipped_no_src += len(rec_rows)
            continue
        n_records += 1
        # Recover src exactly as run_pipeline builds it.
        src = f"{short_vt(doc.get('value_type', ''))}/{str(doc.get('hepdata_file', ''))[:15]}"
        src_dict = h5_lookup.get(src, {})
        if not src_dict and not any(k in h5_lookup for k in [src]):
            # Fall back to the prebuilt index (handles odd truncations).
            inv_basename = fpath.split('/')[-1]
            alt = fpath_to_src.get((fpath, doc_idx)) or fpath_to_src.get((inv_basename, doc_idx))
            if alt and alt in h5_lookup:
                src = alt
                src_dict = h5_lookup[src]
        rec_cents = record_cent_list(doc)
        var, rec_bins, params = record_diff_bins(doc)
        # Record experimental values for the shape cross-check (may be None).
        rec_exp = None
        try:
            ev, _ = _load_exp(params)
            if ev is not None:
                rec_exp = [float(v) for v in np.atleast_1d(ev)]
        except Exception:
            rec_exp = None

        for r in rec_rows:
            model_col = r['model_col']
            if not channel_ok(model_col, doc):
                skip('channel_mismatch')
                continue
            keys = dispatch_keys_for_model_col(model_col)
            if not keys:
                skip('no_dispatch_keys_for_model_col')
                continue
            mlo_s, mhi_s = parse_model_col_cent(model_col)
            mbin_s = parse_model_col_bin(model_col)
            try:
                mbin = int(mbin_s)
            except (TypeError, ValueError):
                skip('model_col_parse_failed')
                continue
            if mlo_s and mhi_s:
                try:
                    mlo, mhi = float(mlo_s), float(mhi_s)
                except (TypeError, ValueError):
                    skip('model_col_parse_failed')
                    continue
                tier_hit = None
            else:
                # No-cent model column (e.g. v22_int_STAR__binNN): centrality
                # comes from the inventory's value-attributed record_cent
                # (consensus locked in inventory.py).
                mlo = mhi = None
                tier_hit = None
                try:
                    _rlo = float(r.get('record_cent_lo', ''))
                    _rhi = float(r.get('record_cent_hi', ''))
                except (TypeError, ValueError):
                    _rlo = _rhi = None
                if _rlo is None:
                    skip('model_cent_unattributable')
                    continue
                mlo, mhi = _rlo, _rhi
                mlo_s, mhi_s = f'{mlo:g}', f'{mhi:g}'
                tier_hit = r.get('attrib_tier', '') or None
            is_diff = any(k in DIFF_KEYS for k in keys)
            if not is_diff:
                # Integrated: match (cent range, dispatch_key) numerically
                # (label strings vary: '00-03%' vs '0-3%').
                ov, oe, dk_hit = None, 0.0, None
                for dk in keys:
                    for (cl, ldk), entry in list(src_dict.items()):
                        if ldk != dk:
                            continue
                        rng = parse_cent_label(cl)
                        if rng is None:
                            if not (mlo_s == '' and mhi_s == ''):
                                continue
                        elif abs(rng[0] - mlo) > 1e-3 or abs(rng[1] - mhi) > 1e-3:
                            continue
                        ov, oe, dk_hit = entry[0], entry[1], dk
                        break
                    if ov is not None:
                        break
                if ov is None:
                    skip('integrated_no_hdf5_value')
                    n_skipped_no_value += 1
                    continue
            else:
                # --- Strict differential matching ---
                # 1. Record must have explicit cent classes containing model cent.
                if not rec_cents:
                    skip('record_cent_missing')
                    continue
                if not cent_contains(rec_cents, mlo, mhi):
                    skip('cent_mismatch')
                    continue
                # 2. Map model bins -> record bins.  Preferred: edge-overlap
                # mapping against curated MODEL_DIFF_BINS (handles records
                # carrying extra bins, e.g. STAR data4's excluded lowest pT
                # bin).  Fallback: legacy equal-count positional mapping.
                n_model_bins = model_prefix_counts.get(model_prefix(model_col), 0)
                fam, det = parse_model_col_family_det(model_col)
                model_edges = MODEL_DIFF_BINS.get((fam, det))
                if (isinstance(model_edges, (list, tuple)) and model_edges
                        and isinstance(model_edges[0], (list, tuple))):
                    n_model_edges = len(model_edges)
                elif model_edges:
                    n_model_edges = len(model_edges) - 1
                else:
                    n_model_edges = -1
                bin_map = None  # model_idx -> record_idx
                if ((fam, det) in MODEL_DIFF_BINS_MIRROR and rec_bins
                        and min(float(b[0]) for b in rec_bins) < -1e-6):
                    # Mirroring is only legitimate for a forward-only
                    # record (a symmetrised publication, e.g. PHOBOS Fig. 5).
                    # A two-sided record here is a different measurement
                    # method and must not be folded into this comparison.
                    skip('mirror_family_record_not_forward_only')
                    continue
                if (model_edges and n_model_edges == n_model_bins
                        and rec_bins):
                    bin_map, _ = overlap_bin_map(
                        model_edges, rec_bins,
                        mirror=(fam, det) in MODEL_DIFF_BINS_MIRROR,
                        cap=WIDTH_RATIO_CAP_OVERRIDE.get((fam, det)))
                if bin_map is None:
                    if not rec_bins or len(rec_bins) != n_model_bins:
                        skip(f'bin_count_mismatch_rec{len(rec_bins)}_vs_model{n_model_bins}')
                        continue
                    bin_map = {b: b for b in range(n_model_bins)}
                if len(bin_map) < 3:
                    # Too few model bins mappable: binnings incompatible
                    # (e.g. guessed model edges vs record edges).
                    skip('model_record_bin_overlap_low')
                    continue
                # 3. Experimental shape cross-check (scale-invariant): the
                # record's exp values across bins must track the model's exp
                # values up to a global scale (catches wrong-species matches
                # like Lambda v2 vs charged-hadron v2).  Skipped when either
                # side is unavailable or has < 3 mapped bins.
                if (rec_exp is not None and len(rec_exp) == len(rec_bins)
                        and len(bin_map) >= 3
                        and len(set(bin_map.values())) == len(bin_map)):
                    # Many-to-one skipped on purpose: the check compares
                    # record exp values against model exp values one bin at
                    # a time, which only means anything when each model bin
                    # has its own record bin.  When the model grid is finer
                    # than the published binning (PHOBOS v2(eta): 14 model
                    # bins onto 8 record bins) several model bins share a
                    # record value and the ratio is not constant by
                    # construction, so the test would reject valid matches.
                    mod_exp = []
                    ok = True
                    mapped = sorted(bin_map)
                    for bj in mapped:
                        mc = f'{model_prefix(model_col)}__bin{bj:02d}'
                        if mc in h_exp_idx:
                            try:
                                mod_exp.append(float(exp_row[h_exp_idx[mc]]))
                            except (TypeError, ValueError):
                                ok = False
                                break
                        else:
                            ok = False
                            break
                    if ok and len(mod_exp) == len(mapped):
                        rec_mapped = [rec_exp[bin_map[bj]] for bj in mapped]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            ratios = np.array(rec_mapped) / np.array(mod_exp)
                        finite = ratios[np.isfinite(ratios) & (np.array(mod_exp) != 0)]
                        if len(finite) == len(ratios) and np.mean(np.abs(finite)) > 0:
                            cv = float(np.std(finite) / np.mean(np.abs(finite)))
                            if cv > 0.5:
                                skip('exp_shape_mismatch')
                                continue
                # 4. Record bin for this model bin (overlap-mapped, or same
                # index in the legacy equal-count path).
                bi = bin_map.get(mbin)
                if bi is None or bi < 0 or bi >= len(rec_bins):
                    skip('record_bin_out_of_range')
                    continue
                # 5. Find the HDF5 value.  Candidate dispatch keys first;
                # then the record bin (positional order; edge-verified when
                # labels carry bin ranges).  Cent labels are matched
                # numerically ('00-03%' == '0-3%').
                # Intervals the HDF5 could carry for this model bin, best
                # first.  When the pipeline was run with a differential-bin
                # override (--diff-bins-override) it evaluated the MODEL's
                # intervals, so the label edges are the model's, not the
                # record's; without the override they coincide.  Trying both
                # keeps this correct either way, and stops several model
                # bins that share a record bin from reading the same value.
                want_list = []
                if model_edges and 0 <= mbin < len(model_edges):
                    me = model_edges[mbin]
                    if isinstance(me, (list, tuple)) and len(me) == 2:
                        want_list.append((float(me[0]), float(me[1])))
                want_list.append((float(rec_bins[bi][0]), float(rec_bins[bi][1])))
                ov, oe, dk_hit = None, 0.0, None
                for dk in keys:
                    for (s2, (cl, ldk)), ordered in list(h5_ordered_bins.items()):
                        if s2 != src or ldk != dk:
                            continue
                        rng = parse_cent_label(cl)
                        if rng is None:
                            continue
                        contained = any(
                            mlo >= rlo - 1e-3 and mhi <= rhi + 1e-3
                            for (rlo, rhi) in rec_cents)
                        if not contained:
                            continue
                        if abs(rng[0] - mlo) > 1e-3 or abs(rng[1] - mhi) > 1e-3:
                            # HDF5 cent bin must equal the model cent bin.
                            continue
                        if bi >= len(ordered):
                            continue
                        # Resolve by bin edges, not blindly by position.
                        # Several record docs can share one label key
                        # (same value_type + hepdata_file): the PHOBOS
                        # v2(eta) file holds three different figures -- the
                        # hit-based (14 bins), track-based (6) and Fig-5
                        # combined (8) series all keyed
                        # "[v2/phobos/final_as]".  Positional lookup then
                        # returns another figure's bins, so fall back to an
                        # edge search when the positional entry disagrees.
                        cand = None
                        bedges = ordered[bi][0]
                        if bedges is None or any(edges_close(bedges, we)
                                                 for we in want_list):
                            cand = ordered[bi]
                        else:
                            for entry in ordered:
                                if entry[0] is None:
                                    continue
                                if any(edges_close(entry[0], we)
                                       for we in want_list):
                                    cand = entry
                                    break
                        if cand is None:
                            continue
                        ov, oe, dk_hit = cand[1], cand[2], dk
                        break
                    if ov is not None:
                        break
                if ov is None:
                    skip('diff_no_hdf5_value')
                    n_skipped_no_value += 1
                    continue
            # Model value
            mv_log = float(model_row[model_cols[model_col]])
            mv = model_to_value(mv_log, model_col)
            if mv == 0:
                n_skipped_no_value += 1
                continue
            # Experimental value
            exp_v = float(exp_row[h_exp_idx[model_col]]) if model_col in h_exp_idx else float('nan')
            err_v = float(err_row[h_err_idx[model_col]]) if model_col in h_err_idx else float('nan')
            our_minus_model = (ov - mv) / mv * 100
            our_minus_exp = (ov - exp_v) / exp_v * 100 if exp_v and not np.isnan(exp_v) else float('nan')
            rows.append({
                'design_point': args.dp, 'fpath': fpath, 'doc_index': doc_idx,
                'model_col': model_col, 'compute': dk_hit or (keys[0] if keys else ''),
                'cent_lo': mlo_s, 'cent_hi': mhi_s,
                'match_tier': tier_hit or '',
                'eta_bin': r.get('eta_bin', ''), 'pT_bin': r.get('pT_bin', ''),
                'our_val': ov, 'our_err': oe,
                'model_val': mv, 'model_log': mv_log,
                'exp_val': exp_v, 'exp_err': err_v,
                'our_minus_model_pct': our_minus_model,
                'our_minus_exp_pct': our_minus_exp,
            })
            n_matched += 1

    h5.close()
    print(f'  matched: {n_matched} records-used={n_records}, '
          f'skipped (no value): {n_skipped_no_value}, '
          f'skipped (no model col): {n_skipped_no_model}', flush=True)
    print('  skip reasons:', flush=True)
    for reason, cnt in sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:15]:
        print(f'    {cnt:6d}  {reason}', flush=True)
    if not rows:
        print('  no rows produced', flush=True)
        return
    fieldnames = list(rows[0].keys())
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'  wrote {len(rows)} rows to {args.output}', flush=True)


if __name__ == '__main__':
    main()

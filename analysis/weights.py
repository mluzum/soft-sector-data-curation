import math
import warnings

import numpy as np
import scipy.stats


def _parse_anchor_str(s):
    """Parse a CLI anchor string into a dict or a bare float.

    Accepted forms:
      "0.8"                         -> 0.8  (bare coverage fraction)
      "bmax:12,sigtot:42"           -> {b_max:12, sigma_tot:42}
      "sigma_tot:42,b_max:12"       -> {sigma_tot:42, b_max:12}
      "f:0.8" / "n_ref:1000"        -> {f:..} / {n_ref:..}
      "sigma_eff:30,sigtot:42"      -> {sigma_eff:30, sigma_tot:42}
    """
    parts = str(s).split(',')
    if len(parts) == 1 and ':' not in parts[0]:
        return float(parts[0].strip())
    d = {}
    for part in parts:
        part = part.strip()
        if not part or ':' not in part:
            continue
        k, v = part.split(':', 1)
        d[k.strip().lower()] = float(v.strip())
    return d


def resolve_anchor(spec, n_sim):
    """Resolve an anchor specification to canonical (f, n_ref, anchor_cent).

    The canonical internal quantity is the coverage fraction ``f``: the
    simulation covers the central fraction ``f`` of the total cross section.
    Derived quantities: ``n_ref = n_sim / f`` (reference total event count)
    and ``anchor_cent = 100 * f`` (centrality % at the sim's peripheral edge).

    Accepted ``spec`` forms:
      * float ``v`` in (0, 1]            -> coverage fraction ``f = v``
      * float ``v > 1``                  -> reference total count ``n_ref = v``
      * dict / CLI string with keys:
          - ``f``           : coverage fraction
          - ``n_ref``       : reference total event count
          - ``sigma_tot`` + ``b_max``      : uniform-b sampling, f = pi*b_max^2/sigma_tot
          - ``sigma_tot`` + ``sigma_eff``  : f = sigma_eff/sigma_tot

    Parameters
    ----------
    spec : float | dict | str
    n_sim : int — number of events in the loaded simulation set (used as the
            percentile denominator when ``f`` is given).

    Returns
    -------
    dict with keys ``f``, ``n_ref``, ``anchor_cent``.
    """
    if isinstance(spec, str):
        spec = _parse_anchor_str(spec)

    f = None
    n_ref = None

    if isinstance(spec, (int, float)):
        v = float(spec)
        if v <= 0:
            raise ValueError(f"anchor must be positive, got {v}")
        if v <= 1.0:
            f = v
        else:
            n_ref = v
    elif isinstance(spec, dict):
        d = {k.lower(): float(v) for k, v in spec.items()}
        if 'f' in d:
            f = d['f']
        elif 'n_ref' in d or 'nref' in d:
            n_ref = d.get('n_ref', d.get('nref'))
        elif 'b_max' in d or 'bmax' in d:
            sig_tot = d.get('sigma_tot', d.get('sigtot'))
            b_max = d.get('b_max', d.get('bmax'))
            if sig_tot is None or b_max is None:
                raise ValueError("b_max anchor requires both b_max and sigma_tot")
            f = math.pi * b_max ** 2 / sig_tot
        elif 'sigma_eff' in d or 'sigeff' in d:
            sig_eff = d.get('sigma_eff', d.get('sigeff'))
            sig_tot = d.get('sigma_tot', d.get('sigtot'))
            if sig_eff is None or sig_tot is None:
                raise ValueError(
                    "sigma_eff anchor requires both sigma_eff and sigma_tot")
            f = sig_eff / sig_tot
        else:
            raise ValueError(f"unrecognized anchor spec keys: {list(spec.keys())}")
    else:
        raise ValueError(f"unsupported anchor spec type: {type(spec)!r}")

    # Normalise to (f, n_ref).
    if f is None:
        n_ref = float(n_ref)
        if n_ref <= 0:
            raise ValueError(f"n_ref must be positive, got {n_ref}")
        f = n_sim / n_ref
    else:
        f = float(f)
        if f <= 0:
            raise ValueError(f"coverage fraction f must be > 0, got {f}")
        n_ref = n_sim / f

    # The simulation cannot cover more than the full cross section.
    if f > 1.0:
        warnings.warn(
            f"anchor implies f={f:.4f} > 1 (sim exceeds full cross section); "
            f"clamping f=1, n_ref=n_sim={n_sim}", stacklevel=2)
        f = 1.0
        n_ref = float(n_sim)

    return {'f': f, 'n_ref': n_ref, 'anchor_cent': 100.0 * f}


def threshold_anchor(event_var, method, threshold, fraction_above, efficiency=1.0,
                     event_weights=None):
    """Coverage of the simulation when the experiment defines its percentiles
    by the fraction of ITS total that lies above an estimator threshold.

    STAR's isobar classes (arXiv:2109.00131, Sec. III) are fractions of a
    Glauber total normalized so that the events with Ntrk >= 50 hold
    F50 of it, not fractions of the true inelastic cross section.  The
    model reproduces that by requiring the same: with the model estimator
    count Ntrk = Poisson(efficiency x N) ('poisson') or efficiency x N
    ('simple'), and p = <P(Ntrk_i >= threshold)> over the simulated events,
    the coverage is f = fraction_above / p (n_ref = n_sim / f).  The class
    edges then follow from the usual anchored weights; no sigma_inel is
    needed.  f > 1 is legitimate here and is NOT clamped: a model whose
    multiplicity is low (fewer events above the threshold than the
    experiment's fraction) has n_ref < n_sim, its most peripheral events
    fall beyond 100%, and the listed classes stay defined -- exactly what
    the experiment's procedure does to such a model.  (Until 2026-11-02 f
    was clamped to 1, which replaced the normalization by n_ref = n_sim at
    low-multiplicity design points: <N> 40-50% off by up to 80%, zr96c2.)

    Returns dict f, n_ref, anchor_cent (= 100 f, may exceed 100), p_above.
    """
    x = float(efficiency) * np.asarray(event_var, dtype=float)
    if method == 'poisson':
        p_ev = scipy.stats.poisson.sf(math.ceil(threshold) - 1, x)       # P(N >= threshold)
    else:
        p_ev = (x >= threshold).astype(float)
    w = np.ones_like(x) if event_weights is None else np.asarray(event_weights, dtype=float)
    p = float(np.sum(w * p_ev) / np.sum(w))
    if p <= 0:
        raise ValueError(f'no simulated event reaches the estimator threshold {threshold}')
    f = float(fraction_above) / p
    return {'f': f, 'n_ref': len(x) / f, 'anchor_cent': 100.0 * f, 'p_above': p}


def cent_bins_partial(cent_edges, anchor_cent):
    """Flag centrality bins that straddle the anchor (partial coverage).

    A bin ``(lo, hi)`` is *partial* when ``lo < anchor_cent < hi``: part of the
    bin lies beyond the sim's covered cross section.  Such bins are kept but
    flagged (and experimental values masked) by run_pipeline / compute_flow_vn.

    Parameters
    ----------
    cent_edges : list[(lo, hi)] — centrality class percentile edges
    anchor_cent : float — upper percentile of the sim-covered cross section
                            (= 100*f)

    Returns
    -------
    (n_cent,) bool array
    """
    return np.array([(lo < anchor_cent < hi) for (lo, hi) in cent_edges],
                    dtype=bool)


def combine_weights(cent_w, event_weights):
    """Combine centrality-bin weights with an optional per-event reweighting.

    Both 2D and 3D pipelines funnel their per-event averages through this
    single function so the reweighting semantics stay identical across
    simulation formats.

    Args:
        cent_w : (n_events, n_cent) float — centrality-bin membership weight
            (from direct_cent_weights_percentile or poisson_cdf_cent_weights).
        event_weights : (n_events,) float or None — per-(hydro-)event
            importance/reweighting.  None (or all-ones) ⇒ equal weighting.

    Returns:
        (n_events, n_cent) float — cent_w * event_weights[:, None].
    """
    if event_weights is None:
        return np.asarray(cent_w, dtype=float)
    ew = np.asarray(event_weights, dtype=float)
    if cent_w is None:
        return ew[:, None]
    cent_w = np.asarray(cent_w, dtype=float)
    if ew.shape != (cent_w.shape[0],):
        raise ValueError(
            f"event_weights shape {ew.shape} != (n_events,) {cent_w.shape[0]}")
    return cent_w * ew[:, None]


def direct_cent_weights(event_var, cent_edges):
    event_var = np.asarray(event_var)
    n_events = len(event_var)
    weights = np.zeros((n_events, len(cent_edges) - 1), dtype=np.float64)
    for i in range(len(cent_edges) - 1):
        lo, hi = cent_edges[i], cent_edges[i + 1]
        if i == 0:
            weights[:, i] = (event_var <= hi).astype(np.float64)
        elif i == len(cent_edges) - 2:
            weights[:, i] = (event_var > lo).astype(np.float64)
        else:
            weights[:, i] = ((event_var > lo) & (event_var <= hi)).astype(np.float64)
    return weights


def cent_weights(method, event_var, cent_edges, n_ref=None, efficiency=1.0,
                 anchor_cent=100.0):
    """Centrality weights robust to class listing order.

    Curated ``centrality.classes`` follow HEPData table order, which is
    sometimes peripheral-first (e.g. STAR data29a/data30 list 70-80% ..
    0-5%).  Percentile slicing needs ascending edges, so this wrapper
    sorts a copy ascending, computes weights, and scatters columns back
    to LISTED order.  Column ``i`` always corresponds to
    ``cent_edges[i]``.

    Parameters
    ----------
    method : 'simple' | 'poisson'
    event_var : (n_events,) float — ranking/multiplicity variable
    cent_edges : list[(lo, hi)] — centrality bins in listed order
    n_ref : reference total for simple percentiles (default n_events)
    efficiency, anchor_cent : poisson options
    """
    order = np.argsort([float(lo) for lo, _ in cent_edges], kind='stable')
    inv = np.argsort(order)
    asc = [cent_edges[i] for i in order]
    n = len(np.asarray(event_var))
    if method == "simple":
        # A curated class list need NOT tile the percentile axis.  STAR's
        # eta-sub tables, for instance, publish only
        # ['10-20%', '30-40%', '50-60%'].  Treating those pairs as
        # consecutive EDGES would silently reinterpret them as
        # 10-20 / 20-40 / 40-60 and mislabel columns 2 and 3 -- every
        # non-contiguous class list was silently mislabelled.
        # Contiguous lists take the single-slice path (unchanged, and
        # cheaper); disjoint ones are sliced per class.
        contiguous = all(
            abs(float(asc[i][1]) - float(asc[i + 1][0])) < 1e-9
            for i in range(len(asc) - 1))
        if contiguous or len(asc) == 1:
            edges_pct = np.array([float(asc[0][0])]
                                 + [float(hi) for (_, hi) in asc])
            w_asc = direct_cent_weights_percentile(
                event_var, edges_pct,
                n_ref=n_ref if n_ref is not None else n)
        else:
            n_ref_eff = n_ref if n_ref is not None else n
            cols = []
            for lo, hi in asc:
                one = direct_cent_weights_percentile(
                    event_var, np.array([float(lo), float(hi)]),
                    n_ref=n_ref_eff)
                cols.append(one[:, 0])
            w_asc = np.column_stack(cols)
    elif method == "poisson":
        # Already per-pair: poisson_cdf_cent_weights maps each (cmin, cmax)
        # to its own percentile range, so disjoint classes are handled.
        w_asc = poisson_cdf_cent_weights(
            event_var, asc, efficiency=efficiency, anchor_cent=anchor_cent)
    else:
        raise ValueError(f"unknown cent method {method!r}")
    return w_asc[:, inv]


def direct_cent_weights_percentile(event_var, cent_pct_edges, n_ref=None):
    """One-hot centrality bins from a sorted event variable (percentile edges).

    The percentile axis is anchored to a reference total event count ``n_ref``.
    By default ``n_ref = len(event_var)`` (the full sim set == full cross
    section; today's behaviour).  If the simulation only covers a central
    fraction ``f`` of the total cross section (e.g. a finite impact-parameter
    cutoff), pass ``n_ref = n_sim / f > n``: events beyond the sim coverage
    then map to peripheral percentiles the sim does not contain, and those
    bins come out empty (later excluded by the pipeline).

    Parameters
    ----------
    event_var : (n_events,) float — quantity used for ranking (e.g. multiplicity)
    cent_pct_edges : (m+1,) float — percentile boundaries in [0, 100]
    n_ref : float or None — reference total event count (default = n_events)
    """
    sorted_idx = np.argsort(-event_var)
    n = len(event_var)
    if n_ref is None:
        n_ref = n
    n_ref = float(n_ref)
    weights = np.zeros((n, len(cent_pct_edges) - 1), dtype=np.float64)
    for i in range(len(cent_pct_edges) - 1):
        lo = int(cent_pct_edges[i] / 100 * n_ref)
        hi = int(cent_pct_edges[i + 1] / 100 * n_ref)
        lo = max(lo, 0)
        hi = min(hi, n)
        if hi > lo:
            weights[sorted_idx[lo:hi], i] = 1.0
    return weights


def poisson_cent_weights(event_var, cent_edges, sharpness=50.0):
    event_var = np.asarray(event_var)
    n_events = len(event_var)
    n_cent = len(cent_edges) - 1
    weights = np.ones((n_events, n_cent), dtype=np.float64)
    for i in range(n_cent):
        lo, hi = cent_edges[i], cent_edges[i + 1]
        if i == 0:
            weights[:, i] = 1.0 / (1.0 + np.exp(sharpness * (event_var - hi)))
        elif i == n_cent - 1:
            weights[:, i] = 1.0 / (1.0 + np.exp(sharpness * (lo - event_var)))
        else:
            w_hi = 1.0 / (1.0 + np.exp(sharpness * (event_var - hi)))
            w_lo = 1.0 / (1.0 + np.exp(sharpness * (lo - event_var)))
            weights[:, i] = w_hi * w_lo
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)


def normalize_weights(weights):
    row_sums = weights.sum(axis=1, keepdims=True)
    return weights / np.maximum(row_sums, 1e-30)


def _find_cdf_crossing(Q0, target, M_min=0, M_max=100000):
    """Binary search: smallest M such that mean Poisson CDF(Q0, M) ≥ target."""
    left, right = M_min, M_max
    while left < right:
        mid = (left + right) // 2
        cdf = np.mean(scipy.stats.poisson.cdf(mid, Q0))
        if cdf < target:
            left = mid + 1
        else:
            right = mid
    return left


def poisson_cdf_cent_weights(Q0ch, centbins, efficiency=1.0, anchor_cent=100.0):
    """Poisson-fluctuation centrality weights via CDF inversion.

    Each hydro event's smooth charged multiplicity Q0ch is interpreted as
    the mean of a Poisson-distributed detected count, scaled by
    *efficiency*.  An event's weight in a class is the probability that its
    detected count falls in that class.

    efficiency controls the detector resolution:
      = 1.0  -- pure Poisson counting statistics (default)
      -> inf -- deterministic (-> direct_cent_weights_percentile)
      -> 0   -- maximal smearing (all bins equally likely)

    Classes are percentiles of the TOTAL cross section.  The simulation
    covers its central fraction f = anchor_cent/100, so the total count
    distribution is F_tot(M) = (1 - f) + f * mean_i P(N_i <= M) (the
    uncovered events lie below every simulated count), and a class edge at
    c% is the count where F_tot = 1 - c/100.  Weights are not renormalized:
    probability mass outside the coverage, or in classes the record does
    not list, is simply not counted.  The summed weight of a covered class
    is then n_ref x width (n_ref = n_sim / f), as for the sharp cuts.

    Until November 2026 the inversion used the simulated events' CDF at
    1 - c/100 (an f-compressed percentile axis: 0-10% held 199 instead of
    223 of 2000 events at f = 0.897), and each event's row was renormalized
    over the record's classes, so a single-class record (spectra) gave every
    event weight 1 and the most peripheral class of a list collected the
    events beyond it (found by the isobar analysis, zr96c2 dp0001).

    Parameters
    ----------
    Q0ch: (n_events,) array -- smooth charged multiplicity per event
    centbins: list of (min_cent%, max_cent%) tuples -- true centrality
    efficiency: float -- effective Poisson mean = efficiency x Q0ch
    anchor_cent: float -- upper percentile of the sim-covered cross section
                (default 100 = full set)

    Returns: (n_events, n_cent_bins) float -- probabilities per class.
    """
    Q0ch = np.asarray(Q0ch)
    Q0ch_eff = efficiency * Q0ch
    nev = len(Q0ch)
    weights = np.zeros((nev, len(centbins)))
    maxQ0_eff = np.max(Q0ch_eff)
    max_search = int(maxQ0_eff + 4.0 * np.sqrt(maxQ0_eff)) + 1

    if anchor_cent <= 0:
        raise ValueError(f"anchor_cent must be > 0, got {anchor_cent}")
    f = anchor_cent / 100.0
    # f > 1 only from an experiment-defined normalization (threshold_anchor):
    # the simulation's most peripheral events then lie beyond 100%.
    if not f > 0.0:
        raise ValueError(f"anchor_cent={anchor_cent} must be positive")

    def sim_target(frac):
        """Total-cross-section CDF value -> the simulated events' CDF value."""
        return min(max((frac - (1.0 - f)) / f, 0.0), 1.0)

    def cdf(M):
        return scipy.stats.poisson.cdf(M, Q0ch_eff) if M >= 0 else np.zeros(nev)

    def below(target):
        """Per-event probability of a count below the edge where the mean
        CDF equals ``target``.  The integer count at the edge is split in
        proportion -- a count uniform within its unit, Ntrk + U(-1/2, 1/2)
        -- so a class covers exactly its percentile range.  (Until November
        2026 edges snapped to whole counts: at Ntrk ~ 50 a class could be off
        by a few tenths of a percentile.)"""
        if target <= 0.0:
            return np.zeros(nev)
        if target >= 1.0:
            return np.ones(nev)
        M = _find_cdf_crossing(Q0ch_eff, target, 0, max_search)
        c_hi, c_lo = cdf(M), cdf(M - 1)
        m_hi, m_lo = c_hi.mean(), c_lo.mean()
        phi = (target - m_lo) / (m_hi - m_lo) if m_hi > m_lo else 1.0
        return c_lo + min(max(phi, 0.0), 1.0) * (c_hi - c_lo)

    for ibin, (cmin, cmax) in enumerate(centbins):
        lo = sim_target(1.0 - cmax / 100.0)
        hi = sim_target(1.0 - cmin / 100.0)
        if hi <= lo:
            continue                     # beyond the coverage: no weight
        weights[:, ibin] = below(hi) - below(lo)
    return weights

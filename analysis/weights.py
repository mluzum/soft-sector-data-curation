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
                 anchor_cent=100.0, *, _edges=None, _hints=None):
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
    _edges, _hints : a sample's _PoissonEdges and a larger sample's
        crossings (CentralityWeights)
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
            event_var, asc, efficiency=efficiency, anchor_cent=anchor_cent,
            _edges=_edges, _hints=_hints)
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


def poisson_cdf_cent_weights(Q0ch, centbins, efficiency=1.0, anchor_cent=100.0, *,
                             _edges=None, _hints=None):
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
    if anchor_cent <= 0:
        raise ValueError(f"anchor_cent must be > 0, got {anchor_cent}")
    f = anchor_cent / 100.0
    # f > 1 only from an experiment-defined normalization (threshold_anchor):
    # the simulation's most peripheral events then lie beyond 100%.
    if not f > 0.0:
        raise ValueError(f"anchor_cent={anchor_cent} must be positive")
    pe = _edges if _edges is not None else _PoissonEdges(efficiency * np.asarray(Q0ch))
    weights = np.zeros((pe.nev, len(centbins)))

    def sim_target(frac):
        """Total-cross-section CDF value -> the simulated events' CDF value."""
        return min(max((frac - (1.0 - f)) / f, 0.0), 1.0)

    for ibin, (cmin, cmax) in enumerate(centbins):
        lo = sim_target(1.0 - cmax / 100.0)
        hi = sim_target(1.0 - cmin / 100.0)
        if hi <= lo:
            continue                     # beyond the coverage: no weight
        weights[:, ibin] = pe.below(hi, cmin, _hints) - pe.below(lo, cmax, _hints)
    return weights


class _PoissonEdges:
    """The per-event probabilities below the class edges of one sample
    (poisson_cdf_cent_weights): each mean CDF value and each edge computed
    once, whatever classes share it.  ``crossings`` records the count found
    for each edge percentile; a smaller sample (a jackknife replicate)
    starts its search there (``hints``) instead of bisecting [0, max]: the
    same count, since the mean CDF is monotone in the count."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        self.nev = len(self.x)
        m = np.max(self.x) if self.nev else 0.0
        self.max_search = int(m + 4.0 * np.sqrt(m)) + 1
        self.crossings = {}
        self._mean, self._below, self._arr = {}, {}, {}

    def cdf(self, M):
        if M < 0:
            return np.zeros(self.nev)
        c = self._arr.get(M)
        return c if c is not None else scipy.stats.poisson.cdf(M, self.x)

    def mean(self, M):
        v = self._mean.get(M)
        if v is None:
            c = self._arr[M] = scipy.stats.poisson.cdf(M, self.x)    # kept for this edge's below()
            v = self._mean[M] = np.mean(c)
        return v

    def _bisect(self, left, right, target):
        while left < right:
            mid = (left + right) // 2
            if self.mean(mid) < target:
                left = mid + 1
            else:
                right = mid
        return left

    def crossing(self, target, hint=None):
        """Smallest M in [0, max_search) with mean CDF >= target, else
        max_search (_find_cdf_crossing over [0, max_search])."""
        top = self.max_search
        if hint is None or top < 1:
            return self._bisect(0, top, target)
        h = min(max(int(hint), 0), top - 1)
        step = 1
        if self.mean(h) >= target:
            ok, b = h, h - 1
            while b >= 0 and self.mean(b) >= target:
                ok, step = b, 2 * step
                b = ok - step
            return self._bisect(max(b + 1, 0), ok, target)
        low, b = h, h + 1
        while b < top and self.mean(b) < target:
            low, step = b, 2 * step
            b = low + step
        return self._bisect(low + 1, min(b, top), target)

    def below(self, target, pct=None, hints=None):
        """Per-event probability of a count below the edge where the mean
        CDF equals ``target`` (``pct``: the edge's percentile, for the
        hints).  The integer count at the edge is split in proportion -- a
        count uniform within its unit, Ntrk + U(-1/2, 1/2) -- so a class
        covers exactly its percentile range.  (Until November 2026 edges
        snapped to whole counts: at Ntrk ~ 50 a class could be off by a few
        tenths of a percentile.)"""
        if target <= 0.0:
            return np.zeros(self.nev)
        if target >= 1.0:
            return np.ones(self.nev)
        b = self._below.get(target)
        if b is None:
            self._arr = {}
            M = self.crossing(target, (hints or {}).get(pct))
            if pct is not None:
                self.crossings[pct] = M
            c_hi, c_lo = self.cdf(M), self.cdf(M - 1)
            m_hi, m_lo = c_hi.mean(), c_lo.mean()
            phi = (target - m_lo) / (m_hi - m_lo) if m_hi > m_lo else 1.0
            b = self._below[target] = c_lo + min(max(phi, 0.0), 1.0) * (c_hi - c_lo)
            self._arr = {}
        return b


class CentralityWeights:
    """The centrality weights of one estimator -- its multiplicity per event,
    the method, the efficiency and a record's normalization -- for any class
    list, on all events or on the events a jackknife replicate keeps.

    The class edges are percentiles of the events themselves, and an
    experiment-defined normalization (threshold_anchor) is a fraction of
    them, so both are part of the estimator: a replicate rebuilds them from
    the events it keeps.  The anchor's coverage fraction f (b_max and
    sigma_tot, or a given f or n_ref) is external and stays.  Until November
    2026 the weights were fixed from the full sample, so the jackknife
    missed the error of the edges: <N> per class 2-4x too small an error,
    flow ~1.0-1.1x (found by the isobar analysis, zr96c2;
    tests/test_jackknife_centrality.py).

    ``weights(edges, mask)``: (n_kept, n_classes), the class weights times
    the event weights (``raw=True``: without them).  ``mask`` None, or one
    keeping every event, gives the full sample's (kept); another mask's are
    kept until the next mask, so the records of one replicate share them.
    The arrays are read-only: they are shared.
    """

    def __init__(self, method, event_var, anchor=None, efficiency=1.0, normalization=None,
                 event_weights=None):
        if method not in ('simple', 'poisson'):
            raise ValueError(f"unknown cent method {method!r}")
        self.method = method
        self.x = np.asarray(event_var, dtype=float)
        self.n = len(self.x)
        anchor = anchor or {'f': 1.0, 'n_ref': float(self.n)}
        self.f, self.n_ref = float(anchor['f']), float(anchor['n_ref'])
        self.efficiency = float(efficiency)
        self.normalization = None if not normalization else (
            float(normalization['estimator_threshold']), float(normalization['fraction_above']))
        self.event_weights = None if event_weights is None else np.asarray(event_weights, dtype=float)
        self._full = {}
        self._mask, self._rep = None, {}
        self._hints = {}

    def _state(self, mask):
        if mask is None:
            return self._full, None
        if mask is self._mask:
            return self._rep, mask
        if mask.all():
            return self._full, None
        self._mask, self._rep = mask, {}
        return self._rep, mask

    def anchor(self, mask=None):
        """f, n_ref, anchor_cent of the events kept (threshold_anchor's for
        a record with a normalization)."""
        memo, m = self._state(mask)
        if 'anchor' not in memo:
            x = self.x if m is None else self.x[m]
            if self.normalization is None:
                n_ref = self.n_ref if m is None else self.n_ref * (len(x) / self.n)
                memo['anchor'] = {'f': self.f, 'n_ref': n_ref, 'anchor_cent': 100.0 * self.f}
            else:
                ew = None if self.event_weights is None else (
                    self.event_weights if m is None else self.event_weights[m])
                thr, frac = self.normalization
                memo['anchor'] = threshold_anchor(x, self.method, thr, frac,
                                                  efficiency=self.efficiency, event_weights=ew)
        return memo['anchor']

    def weights(self, edges, mask=None, raw=False):
        memo, m = self._state(mask)
        key = tuple((float(lo), float(hi)) for lo, hi in edges)
        if ('raw', key) not in memo:
            x = self.x if m is None else self.x[m]
            a = self.anchor(mask)
            if self.method == 'simple':
                W = cent_weights('simple', x, list(key), n_ref=a['n_ref'])
            else:
                if 'edges' not in memo:
                    memo['edges'] = _PoissonEdges(self.efficiency * x)
                W = cent_weights('poisson', x, list(key), efficiency=self.efficiency,
                                 anchor_cent=a['anchor_cent'], _edges=memo['edges'],
                                 _hints=None if m is None else self._hints)
                if m is None:
                    self._hints.update(memo['edges'].crossings)
            W.setflags(write=False)
            memo[('raw', key)] = W
            ew = None if self.event_weights is None else (
                self.event_weights if m is None else self.event_weights[m])
            C = combine_weights(W, ew)
            if C is not W:
                C.setflags(write=False)
            memo[('comb', key)] = C
        return memo[('raw' if raw else 'comb', key)]

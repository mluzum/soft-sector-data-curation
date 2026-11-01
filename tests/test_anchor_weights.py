import math
import sys

import numpy as np

sys.path.insert(0, '.')
from analysis.weights import (
    resolve_anchor, direct_cent_weights_percentile, poisson_cdf_cent_weights,
    cent_bins_partial, cent_weights)


def test_resolve_fraction():
    a = resolve_anchor(0.8, n_sim=1000)
    assert abs(a['f'] - 0.8) < 1e-12
    assert abs(a['n_ref'] - 1250.0) < 1e-9
    assert abs(a['anchor_cent'] - 80.0) < 1e-9


def test_resolve_nref():
    a = resolve_anchor(1250.0, n_sim=1000)
    assert abs(a['f'] - 0.8) < 1e-12
    assert abs(a['n_ref'] - 1250.0) < 1e-9


def test_resolve_bmax_sigtot():
    b_max = 12.0
    sig_tot = math.pi * b_max ** 2 / 0.8  # chosen so f = 0.8
    a = resolve_anchor({'b_max': b_max, 'sigma_tot': sig_tot}, n_sim=1000)
    assert abs(a['f'] - 0.8) < 1e-9


def test_resolve_sigeff_sigtot():
    a = resolve_anchor({'sigma_eff': 40.0, 'sigma_tot': 50.0}, n_sim=1000)
    assert abs(a['f'] - 0.8) < 1e-9


def test_resolve_string():
    a = resolve_anchor("bmax:10,sigtot:500", n_sim=1000)
    assert 0 < a['f'] < 1.0


def test_resolve_default_is_full():
    a = resolve_anchor(1.0, n_sim=1000)
    assert abs(a['f'] - 1.0) < 1e-12
    assert abs(a['n_ref'] - 1000.0) < 1e-9


def test_direct_percentile_full_set():
    rng = np.random.RandomState(0)
    ev = rng.rand(100) + 1.0
    w = direct_cent_weights_percentile(ev, [0, 50, 100])
    assert w[:, 0].sum() == 50
    assert w[:, 1].sum() == 50


def test_direct_percentile_anchor_shifts():
    rng = np.random.RandomState(1)
    ev = rng.rand(100) + 1.0
    w_full = direct_cent_weights_percentile(ev, [0, 50, 100], n_ref=100)
    # anchor f=0.5 -> n_ref=200: the whole sim is the central 50%.
    w_anchor = direct_cent_weights_percentile(ev, [0, 50, 100], n_ref=200)
    assert w_full[:, 0].sum() == 50
    assert w_anchor[:, 0].sum() == 100
    assert w_anchor[:, 1].sum() == 0


def test_direct_percentile_peripheral_empty():
    ev = np.arange(100)[::-1].astype(float)  # 100..1, central-first
    # anchor f=0.8 -> n_ref=125; bins beyond 80% are empty
    w = direct_cent_weights_percentile(ev, [0, 80, 90, 100], n_ref=125)
    assert w[:, 0].sum() == 100   # the entire sim == central 0-80%
    assert w[:, 1].sum() == 0
    assert w[:, 2].sum() == 0


def test_poisson_anchor_empty_beyond():
    Q0 = np.full(200, 500.0)  # all high multiplicity (central)
    centbins = [(0, 5), (90, 95), (95, 100)]
    w_full = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=100.0)
    w_anch = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=80.0)
    # with anchor 80, the 95-100% bin is beyond coverage -> ~0 weight
    assert w_full[:, 2].sum() > 0
    assert w_anch[:, 2].sum() < 1e-6


def test_poisson_class_totals_anchored():
    """Classes are percentiles of the total cross section: with coverage f the
    summed weight of a covered class is n_ref x width (n_ref = n_sim / f), the
    same whatever other classes the record lists (no per-event
    renormalization; a single-class spectra record used to give every event
    weight 1, and the last class of a list collected the events beyond it)."""
    rng = np.random.RandomState(3)
    Q0 = rng.gamma(2.0, 150.0, size=4000)
    f = 0.897
    n_ref = len(Q0) / f
    centbins = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (90, 100)]
    w = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=100 * f)
    for i, (a, b) in enumerate(centbins[:5]):
        want = n_ref * (b - a) / 100
        assert abs(w[:, i].sum() - want) < 0.02 * want, (centbins[i], w[:, i].sum(), want)
        one = poisson_cdf_cent_weights(Q0, [centbins[i]], efficiency=1.0, anchor_cent=100 * f)
        assert np.allclose(one[:, 0], w[:, i], rtol=0, atol=1e-12)
    assert w[:, 5].sum() < 1e-9          # beyond the coverage
def test_poisson_deterministic_limit_is_sharp_cuts():
    """With negligible counting fluctuations the Poisson weights reduce to the
    anchored sharp percentile cuts (direct_cent_weights_percentile, n_ref = n/f)."""
    rng = np.random.RandomState(7)
    Q0 = rng.gamma(2.0, 150.0, size=3000)
    for A in (60.0, 80.0, 100.0):
        f = A / 100.0
        edges = [0, 10, 30, 50]
        wp = poisson_cdf_cent_weights(Q0, list(zip(edges, edges[1:])), efficiency=1e4, anchor_cent=A)
        wd = direct_cent_weights_percentile(Q0, np.array(edges, float), n_ref=len(Q0) / f)
        assert np.allclose(wp.sum(axis=0), wd.sum(axis=0), rtol=0.01), (A, wp.sum(0), wd.sum(0))
        for i in range(3):
            assert abs(np.average(Q0, weights=wp[:, i]) - np.average(Q0, weights=wd[:, i])) < 0.01 * Q0.mean()
def test_direct_anchor_scaled_full_invariant():
    # Same exact identity for the direct (rank/percentile) method: n_ref = n_sim/f
    # is equivalent to scaling bin edges by 1/f on the full axis, so the two
    # anchor implementations agree.
    rng = np.random.RandomState(8)
    ev = rng.rand(1000) + 1.0
    n = len(ev)
    for A in (60.0, 80.0):
        f = A / 100.0
        w_anch = direct_cent_weights_percentile(ev, [0, 40, 80, 100], n_ref=n / f)
        w_full = direct_cent_weights_percentile(ev, [0, 40 / f, 80 / f, 100 / f], n_ref=n)
        # bins (0,40) and (40,80) are fully covered -> identical assignment
        assert np.array_equal(w_anch[:, 0], w_full[:, 0])
        assert np.array_equal(w_anch[:, 1], w_full[:, 1])
        # bin (80,100) is beyond coverage -> empty
        assert w_anch[:, 2].sum() == 0


def test_poisson_anchor_is_scaled_edges():
    """Coverage f is the same as scaling the class edges by 1/f on the
    simulated events' own axis (the identity the direct method obeys)."""
    rng = np.random.RandomState(42)
    Q0 = rng.gamma(50.0, 5.0, size=2000)
    for A in (60.0, 80.0):
        f = A / 100.0
        centbins = [(0, 5), (5, 10), (10, 20), (40, 50)]
        w_anch = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=A)
        w_scal = poisson_cdf_cent_weights(Q0, [(a / f, b / f) for a, b in centbins], efficiency=1.0)
        assert np.allclose(w_anch, w_scal, rtol=1e-12, atol=1e-12)
def test_direct_anchor_labels_true_centrality():
    # For the direct (rank/percentile) method with finite coverage, the
    # cent bin labels remain TRUE centrality: a 0-10% bin at anchor 50
    # holds the events whose rank falls in [0%, 10%] of the *total*
    # distribution (= the top 10% of n_ref=2000 events); events with
    # rank >= 50% of n_ref are not in the sim and the bin beyond coverage
    # is empty.
    rng = np.random.RandomState(11)
    ev = rng.rand(1000) + 1.0
    n = len(ev)
    # anchor f=0.5: n_ref=2000
    # 0-10% bin: ranks [0, 200) in full = sorted_idx[0:200] in sim = 200 events
    # 10-20%: ranks [200, 400) in sim = 200 events
    # 20-50%: ranks [400, 1000) in sim = 600 events
    # 50-100%: beyond coverage = 0
    w = direct_cent_weights_percentile(ev, [0, 10, 20, 50, 100], n_ref=2000)
    assert w[:, 0].sum() == 200
    assert w[:, 1].sum() == 200
    assert w[:, 2].sum() == 600
    assert w[:, 3].sum() == 0


def test_cent_bins_partial():
    edges = [(0, 20), (20, 60), (60, 100)]
    # anchor at 80%: only the 60-100% bin straddles it
    part = cent_bins_partial(edges, 80.0)
    assert list(part) == [False, False, True]
    # anchor at 100% (full set): no bin straddles
    assert not cent_bins_partial(edges, 100.0).any()
    # anchor at 50%: the 20-60% bin straddles (20<50<60)
    assert list(cent_bins_partial(edges, 50.0)) == [False, True, False]


def test_resolve_clamp_gt_one():
    # b_max/sigma_tot that would imply f>1 must clamp to f=1
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        a = resolve_anchor({'b_max': 100.0, 'sigma_tot': 10.0}, n_sim=1000)
        assert abs(a['f'] - 1.0) < 1e-12
        assert any(issubclass(x.category, UserWarning) for x in w)


def test_cent_weights_listed_order_robust():
    # Curated classes follow HEPData table order, which is sometimes
    # peripheral-first (STAR data29a/data30).  cent_weights must return
    # columns in LISTED order regardless.
    from analysis.weights import cent_weights
    rng = np.random.RandomState(0)
    ev = rng.rand(1000) + 1.0
    asc = [(0, 25), (25, 50), (50, 75), (75, 100)]
    desc = list(reversed(asc))
    for method in ('simple', 'poisson'):
        w_asc = cent_weights(method, ev, asc)
        w_desc = cent_weights(method, ev, desc)
        assert w_asc.shape == w_desc.shape == (1000, 4)
        for i in range(4):
            assert np.array_equal(w_desc[:, i], w_asc[:, 3 - i])
        # every event in exactly one bin (simple) / classes tiling 0-100%
        # hold all of each event's probability (poisson)
        if method == 'simple':
            assert (w_desc.sum(axis=1) == 1).all()
        else:
            assert np.allclose(w_desc.sum(axis=1), 1.0, atol=1e-6)


def test_cent_weights_disjoint_classes():
    """A curated class list need not tile the percentile axis.

    STAR's eta-sub tables publish only ['10-20%', '30-40%', '50-60%'].
    Reading those pairs as consecutive EDGES reinterpreted them as
    10-20 / 20-40 / 40-60, so columns 2 and 3 were silently mislabelled --
    every non-contiguous class list in the database was affected.
    """
    rng = np.random.default_rng(3)
    v = rng.gamma(3.0, 100.0, 20000)

    def mean_mult(w, i):
        return np.average(v, weights=w[:, i])

    disjoint = [(10, 20), (30, 40), (50, 60)]
    w = cent_weights('simple', v, disjoint)
    m_dis = [mean_mult(w, i) for i in range(3)]

    # Each class must select its own percentile slice: ~10% of the events,
    # with a mean multiplicity that falls as the class label rises.
    for i in range(3):
        n_ev = w[:, i].sum()
        assert abs(n_ev - 2000) < 60, f'bin{i} took {n_ev} events, want ~2000'
    assert m_dis[0] > m_dis[1] > m_dis[2], m_dis

    # The shared labels must agree with the contiguous case: 10-20% there is
    # bin0, and 30-40% is reached by extending that list.
    contig = [(10, 20), (20, 30), (30, 40)]
    wc = cent_weights('simple', v, contig)
    assert abs(mean_mult(wc, 0) - m_dis[0]) < 1.0, '10-20% disagrees'
    assert abs(mean_mult(wc, 2) - m_dis[1]) < 1.0, '30-40% disagrees'

    # The Poisson path was already per-pair; the two must now agree.
    wp = cent_weights('poisson', v, disjoint, efficiency=50.0)
    for i in range(3):
        assert abs(mean_mult(wp, i) - m_dis[i]) < 0.05 * m_dis[i], (
            f'poisson/simple disagree on bin{i}')

    # Peripheral-first listing must still scatter back to LISTED order.
    rev = [(50, 60), (30, 40), (10, 20)]
    wr = cent_weights('simple', v, rev)
    for i, (lo, hi) in enumerate(rev):
        assert abs(mean_mult(wr, i) - m_dis[[(10, 20), (30, 40),
                                             (50, 60)].index((lo, hi))]) < 1.0

    # A single class is unchanged.
    w1 = cent_weights('simple', v, [(10, 20)])
    assert abs(mean_mult(w1, 0) - m_dis[0]) < 1.0


if __name__ == '__main__':
    tests = [fn for fn in sorted(dir()) if fn.startswith('test_')]
    n_pass = n_fail = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"  PASS  {name}\n")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}\n")
            n_fail += 1
    print(f"{n_pass}/{n_pass + n_fail} passed")
    sys.exit(1 if n_fail else 0)

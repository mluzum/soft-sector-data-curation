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


def test_poisson_anchor_total_rescaled():
    # f=0.8 -> anchor_cent=80.  Cent labels are TRUE centrality: a bin
    # (cmin, cmax) refers to the [cmin, cmax] percentile range of the total
    # cross section.  The Poisson weighting computes the probability of
    # landing in that *full* percentile range; below the sim's coverage
    # (lower_cov = 0.2) the CDF is clipped to lower_cov.  Per-event rows
    # are renormalized to sum to 1 across the covered bins, so the total
    # weight of covered bins per event is 1 and the excluded bins are 0.
    rng = np.random.RandomState(3)
    Q0 = rng.gamma(30.0, 10.0, size=500)
    centbins = [(0, 20), (20, 50), (50, 80), (80, 100)]
    w_anch = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=80.0)
    # peripheral 80-100% bin excluded (zero weight)
    assert w_anch[:, 3].sum() < 1e-6
    # covered bins renormalized to 1 per event (or 0 for events with no
    # covered mass)
    row_sums = w_anch.sum(axis=1)
    nz = row_sums > 1e-12
    assert np.allclose(row_sums[nz], 1.0, atol=1e-9)
    # The covered-bin shape (NOT the old 1/f-scaled edges) matches the
    # full-anchor result restricted to the same percentile range:
    w_full = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=100.0)
    # rows of the full-anchor result restricted to the 0-80% range, then
    # renormalized to sum to 1 (per event)
    w_full_covered = w_full[:, :3]
    fs = w_full_covered.sum(axis=1, keepdims=True)
    fs = np.maximum(fs, 1e-30)
    w_full_cov_norm = w_full_covered / np.where(fs > 1e-15, fs, 1.0)
    # the renormalized anchored weights for the covered bins must equal the
    # renormalized full-anchor restricted weights
    assert np.allclose(w_anch[:, :3], w_full_cov_norm, rtol=1e-9, atol=1e-12)


def test_poisson_anchor_scaled_full_invariant():
    # True-centrality (NOT 1/f-scaled) interpretation: bin labels are the
    # percentile range of the total cross section.  With anchor A, any bin
    # above A is excluded; bins within coverage are computed on the
    # full-percentile range and renormalized per event.
    rng = np.random.RandomState(7)
    Q0 = rng.gamma(30.0, 10.0, size=500)
    for A in (60.0, 80.0):
        # bins: one fully inside coverage, one straddling, one beyond
        centbins = [(0, A / 2), (A / 2 - 5, A / 2 + 5), (A, 100)]
        w_anch = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=A)
        w_full = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=100.0)
        # bin beyond coverage (max > A) -> ~0 weight
        assert w_anch[:, 2].sum() < 1e-9
        # the fully-inside bin (0, A/2) matches the full-anchor restricted
        # + renormalized weight (here the row-sums are 1 because the row is
        # inside coverage)
        w_full_in = w_full[:, :2]
        fs = w_full_in.sum(axis=1, keepdims=True)
        fs = np.maximum(fs, 1e-30)
        w_full_in_norm = np.where(fs > 1e-15, w_full_in / fs, 0.0)
        assert np.allclose(w_anch[:, :2], w_full_in_norm, rtol=1e-9, atol=1e-12)
        # the straddling bin (A/2-5, A/2+5) is centered on A/2, fully
        # inside coverage for A>50.  Both anchored and full-anchor
        # include this bin, so w_anch[i, 1] = w_full[i, 1] / p(coverage)
        # and w_full[i, 1] = p(bin).  The renormalized anchored weight
        # is the bin probability divided by the coverage probability,
        # so w_anch[i, 1] >= w_full[i, 1] for any event with non-zero
        # excluded-bin mass.  For an event fully inside coverage
        # (all 3 bin probs in [0, 1], sum to 1), the two are equal.
        assert np.all(w_anch[:, 1] >= w_full[:, 1] - 1e-12)
        assert np.all(w_anch[:, 1] <= 1.0 + 1e-12)


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


def test_poisson_anchor_labels_true_centrality():
    # The new (correct) labeling: a 5-10% cent bin at anchor 80 is the
    # 5-10% percentile range of the FULL cross section, NOT the
    # 5-10% percentile of the sim's coverage.  Verify by direct CDF
    # inversion on the full axis.  The anchored row, renormalized across
    # the *covered* bins, reproduces the renormalized full-anchor row
    # restricted to the same bins.
    rng = np.random.RandomState(42)
    Q0 = rng.gamma(50.0, 5.0, size=2000)
    centbins = [(0, 5), (5, 10), (10, 20), (70, 80), (80, 100)]
    w = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=80.0)
    # each row sums to 1 (or 0 if no covered mass)
    nz = w.sum(axis=1) > 1e-12
    assert np.allclose(w[nz].sum(axis=1), 1.0, atol=1e-9)
    # 80-100% bin is beyond coverage -> 0
    assert w[:, 4].sum() < 1e-9
    # The anchored weights for the COVERED bins, restricted to the
    # covered region and renormalized, equal the full-anchor weights
    # restricted to the same region and renormalized.  This is the true-
    # centrality identity: the cent-bin labels do NOT shift with anchor.
    w_full = poisson_cdf_cent_weights(Q0, centbins, efficiency=1.0, anchor_cent=100.0)
    w_full_cov = w_full[:, :4]  # restrict to covered bins
    fs = w_full_cov.sum(axis=1, keepdims=True)
    fs = np.maximum(fs, 1e-30)
    w_full_cov_norm = np.where(fs > 1e-15, w_full_cov / fs, 0.0)
    assert np.allclose(w[:, :4], w_full_cov_norm, rtol=1e-9, atol=1e-12)


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
        # every event in exactly one bin (simple) / rows sum to 1 (poisson)
        if method == 'simple':
            assert (w_desc.sum(axis=1) == 1).all()
        else:
            nz = w_desc.sum(axis=1) > 1e-12
            assert np.allclose(w_desc[nz].sum(axis=1), 1.0)


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

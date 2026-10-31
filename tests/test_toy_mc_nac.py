"""Toy-MC validation harness for event-plane correlation NAC(n,m) (subevent).

Verifies the direct subevent NAC estimator against the per-event truth for
unit-weighted events with disjoint subevents.  Two layers of test:

  1. **Synthetic Q-vectors** (exact, machine-precision): construct
     Q_n^A = M_A · v_n_true · e^{i Psi_n^A} and Q_m^B = M_B · v_m_true
     · e^{i Psi_m^B} so the per-event |Q_n^A Q_m^{B*}|/M_A M_B reproduces
     the true event-plane correlation exactly.  NAC(qvec) must match the
     truth bit-exact.

  2. **No-flow sanity** (particle level): generate events with v_n = v_m
     = 0 and disjoint eta subevents; the estimator must return NAC ~ 0.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
from analysis.observables_2d import _core_nac_mn


def _nac_truth(v_n_A, v_m_B, psi_n_A, psi_m_B, w=None):
    """Per-event truth NAC(n,m) = <Re(v_n^A v_m^B e^{i(n Psi_n - m Psi_m)})>."""
    # per-event: Re(v_n * v_m * e^{i(n*psi_n - m*psi_m)})
    per_event = v_n_A * v_m_B * np.cos(
        n_test * psi_n_A - m_test * psi_m_B)  # placeholder; set via closure
    raise RuntimeError("use _nac_truth_nm instead")


def _nac_truth_nm(v_n_A, v_m_B, psi_n_A, psi_m_B, n, m, w=None):
    per = v_n_A * v_m_B * np.cos(n * psi_n_A - m * psi_m_B)
    if w is None:
        return per.mean()
    return (w * per).sum() / w.sum()


def test_synthetic_exact_n_equals_m():
    """NAC(n,n) with disjoint subevents and RANDOM independent event-planes.

    Construct Q_n^A = M_A v_n e^{i n Psi_n^A}, Q_n^B = M_B v_n e^{i n Psi_n^B}
    with independent Psi_n per subevent.  Per-event ratio is
    v_n^2 cos(n(Psi_A - Psi_B)), which has mean 0 and std v_n^2/sqrt(2N).
    For v_n=0.05, N=2000: std ~ 5.6e-5.  Assert |NAC| < 5e-4 (loose enough
    to be robust to RNG, tight enough to catch real bugs).
    """
    rng = np.random.RandomState(10)
    n_events = 2000
    M_A = rng.normal(150.0, 20.0, n_events).clip(20.0)
    M_B = rng.normal(150.0, 20.0, n_events).clip(20.0)
    n, m = 2, 2  # NAC(2,2) = v_2{2}ab
    v_n = np.full(n_events, 0.05)
    v_m = np.full(n_events, 0.05)
    psi_A = rng.uniform(0, 2 * np.pi, n_events)
    psi_B = rng.uniform(0, 2 * np.pi, n_events)
    Qn_A = M_A * v_n * np.exp(1j * n * psi_A)
    Qm_B = M_B * v_m * np.exp(1j * m * psi_B)
    Q1_A = np.zeros((n_events, 4), dtype=complex)
    Q1_B = np.zeros((n_events, 4), dtype=complex)
    Q1_A[:, 0] = M_A
    Q1_A[:, n] = Qn_A
    Q1_B[:, 0] = M_B
    Q1_B[:, m] = Qm_B

    nac = _core_nac_mn(Q1_A, Q1_B, None, n, m)
    assert abs(nac) < 5e-4, f"random-phase NAC should be ~0, got {nac}"


def test_synthetic_exact_correlated_phase():
    """NAC(n,m) with PERFECTLY correlated event-plane angles (Psi_n = Psi_m).

    Set Q_n^A phase = n*psi, Q_m^B phase = n*psi (same effective event-plane
    angle psi in both subevents).  Then n*Psi_n - m*Psi_m = n*psi - n*psi = 0,
    so cos(0) = 1 and NAC(n,m) = <v_n v_m> = v_n v_m for constant v.
    """
    rng = np.random.RandomState(11)
    n_events = 2000
    M_A = rng.normal(150.0, 20.0, n_events).clip(20.0)
    M_B = rng.normal(150.0, 20.0, n_events).clip(20.0)
    n, m = 3, 2  # NAC(3,2) with 2-subevent
    v_n, v_m = 0.04, 0.05
    psi = rng.uniform(0, 2 * np.pi, n_events)
    # Q_n phase = n*psi (event-plane angle = psi); Q_m phase = n*psi too
    # (event-plane angle = n*psi/m = 3/2*psi -- NOT the same as psi, but
    # n*Psi_n - m*Psi_m = 0).
    Qn_A = M_A * v_n * np.exp(1j * n * psi)
    Qm_B = M_B * v_m * np.exp(1j * n * psi)
    Q1_A = np.zeros((n_events, 5), dtype=complex)
    Q1_B = np.zeros((n_events, 5), dtype=complex)
    Q1_A[:, 0] = M_A
    Q1_A[:, n] = Qn_A
    Q1_B[:, 0] = M_B
    Q1_B[:, m] = Qm_B

    nac = _core_nac_mn(Q1_A, Q1_B, None, n, m)
    # expected: v_n * v_m * <cos(n*psi - n*psi)> = v_n * v_m
    assert abs(nac - v_n * v_m) < 1e-12, f"NAC should be {v_n*v_m}, got {nac}"


def test_synthetic_exact_ebe_fluctuation():
    """Per-event ebE v_n, v_m with correlated event-plane angles.

    The estimator uses the ratio-of-means convention (matching the existing
    vn2_gap core for consistency with the production pipeline):
        NAC = <Re(Q_n^A conj(Q_m^B))> / <M_A M_B>
    With shared effective event-plane angle, the per-event numerator is
    M_A M_B v_n v_m for n=m (cos(0)=1), so the ratio of means equals
    the multiplicity-weighted mean of v_n v_m, which differs from the
    unweighted <v_n v_m> by a multiplicity-weighting factor.

    For n != m with shared angle, the per-event cos is cos((n-m)*psi)
    which has mean 0, so NAC ~ 0.
    """
    rng = np.random.RandomState(12)
    n_events = 3000
    M_A = rng.normal(150.0, 20.0, n_events).clip(20.0)
    M_B = rng.normal(150.0, 20.0, n_events).clip(20.0)
    base, sig = 0.05, 0.01
    v_n = np.clip(rng.normal(base, sig, n_events), 0.001, 0.2)
    v_m = np.clip(rng.normal(base, sig, n_events), 0.001, 0.2)
    psi = rng.uniform(0, 2 * np.pi, n_events)

    # Case n != m: NAC ~ 0
    n, m = 3, 2
    Q1_A = np.zeros((n_events, 5), dtype=complex)
    Q1_B = np.zeros((n_events, 5), dtype=complex)
    Q1_A[:, 0] = M_A; Q1_A[:, n] = M_A * v_n * np.exp(1j * n * psi)
    Q1_B[:, 0] = M_B; Q1_B[:, m] = M_B * v_m * np.exp(1j * m * psi)
    nac = _core_nac_mn(Q1_A, Q1_B, None, n, m)
    assert abs(nac) < 0.01, f"n!=m shared-angle NAC should be ~0, got {nac}"

    # Case n = m: per-event numerator is M_A M_B v_n v_m (cos(0)=1).
    # Ratio of means = <M_A M_B v_n v_m> / <M_A M_B>.
    n = m = 2
    Q1_A = np.zeros((n_events, 4), dtype=complex)
    Q1_B = np.zeros((n_events, 4), dtype=complex)
    Q1_A[:, 0] = M_A; Q1_A[:, n] = M_A * v_n * np.exp(1j * n * psi)
    Q1_B[:, 0] = M_B; Q1_B[:, n] = M_B * v_m * np.exp(1j * m * psi)
    nac = _core_nac_mn(Q1_A, Q1_B, None, n, m)
    expected = (M_A * M_B * v_n * v_m).mean() / (M_A * M_B).mean()
    assert abs(nac - expected) < 1e-12, \
        f"n=m NAC should be {expected}, got {nac}"


def test_synthetic_event_weights():
    """Per-event centrality weight should be a weighted mean."""
    rng = np.random.RandomState(13)
    n_events = 2000
    M_A = rng.normal(150.0, 20.0, n_events).clip(20.0)
    M_B = rng.normal(150.0, 20.0, n_events).clip(20.0)
    n, m = 2, 2
    v_n, v_m = 0.05, 0.05
    psi = rng.uniform(0, 2 * np.pi, n_events)
    Q1_A = np.zeros((n_events, 4), dtype=complex)
    Q1_B = np.zeros((n_events, 4), dtype=complex)
    Q1_A[:, 0] = M_A; Q1_A[:, n] = M_A * v_n * np.exp(1j * psi)
    Q1_B[:, 0] = M_B; Q1_B[:, m] = M_B * v_m * np.exp(1j * psi)

    w = rng.uniform(0.5, 1.5, n_events)
    nac_w = _core_nac_mn(Q1_A, Q1_B, w, n, m)
    nac_uw = _core_nac_mn(Q1_A, Q1_B, None, n, m)
    # both should be ~v_n^2 = 0.0025 (with shared phase, n=m)
    assert abs(nac_w - v_n ** 2) < 1e-12
    assert abs(nac_uw - v_n ** 2) < 1e-12


def test_synthetic_zero_multiplicity_guard():
    """Events with M_A = 0 or M_B = 0 should be skipped, not crash."""
    rng = np.random.RandomState(14)
    n_events = 100
    M_A = np.full(n_events, 150.0)
    M_B = np.full(n_events, 150.0)
    M_A[0:10] = 0.0  # first 10 events have no particles in A
    n, m = 2, 3
    v_n, v_m = 0.05, 0.04
    psi = rng.uniform(0, 2 * np.pi, n_events)
    Q1_A = np.zeros((n_events, 5), dtype=complex)
    Q1_B = np.zeros((n_events, 5), dtype=complex)
    Q1_A[:, 0] = M_A; Q1_A[:, n] = M_A * v_n * np.exp(1j * psi)
    Q1_B[:, 0] = M_B; Q1_B[:, m] = M_B * v_m * np.exp(1j * psi)
    # Should not raise; result should be finite.
    nac = _core_nac_mn(Q1_A, Q1_B, None, n, m)
    assert np.isfinite(nac)


def test_no_flow_particle_level():
    """Particle-level no-flow: NAC must be ~0 with disjoint eta subevents."""
    from analysis.toy_mc import (ToyGenConfig, generate_events,
                                  make_qvector_data)
    from analysis.observables_2d import _compute_weighted_qvectors
    from analysis.pipeline_dispatch import _make_jacobian_list
    cfg = ToyGenConfig(n_events=2000, mult_mean=300, mult_sigma=20,
                        rap_max=1.0, seed=17, flow={})
    events, truth = generate_events(cfg)
    qv = make_qvector_data(events, truth, n_pt=20, n_harm=5)
    # Two disjoint eta windows within the production pT grid.
    jac_A = _make_jacobian_list(qv.pt_centers, qv.masses, -0.8, -0.4)
    jac_B = _make_jacobian_list(qv.pt_centers, qv.masses,  0.4,  0.8)
    Q1_A = _compute_weighted_qvectors(qv, jac_A, 1)
    Q1_B = _compute_weighted_qvectors(qv, jac_B, 1)
    w = np.ones(qv.n_events)
    nac_23 = _core_nac_mn(Q1_A, Q1_B, w, 2, 3)
    nac_24 = _core_nac_mn(Q1_A, Q1_B, w, 2, 4)
    # both must be tiny (no flow, disjoint windows -> no shared particles)
    assert abs(nac_23) < 1e-3, f"no-flow NAC(2,3) too large: {nac_23}"
    assert abs(nac_24) < 1e-3, f"no-flow NAC(2,4) too large: {nac_24}"


if __name__ == '__main__':
    tests = [fn for fn in sorted(dir()) if fn.startswith('test_')]
    n_pass = n_fail = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"  PASS  {name}\n")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}\n")
            n_fail += 1
    print(f"{n_pass}/{n_pass + n_fail} passed")
    sys.exit(1 if n_fail else 0)

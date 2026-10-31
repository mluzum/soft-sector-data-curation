"""Toy-MC validation harness for symmetric cumulant SC(n,m).

Verifies the direct-product SC estimator against the per-event truth for
unit-weighted events.  Two layers of test:

  1. **Synthetic Q-vectors** (exact, machine-precision): construct
     Q_n = M · v_n_true · e^{i Ψ_n} so that the per-event |Q_n|²/M²
     reproduces v_n_true² exactly.  SC(qvec) must equal SC(true) bit-exact.

  2. **No-flow sanity** (particle level): generate events with v_n = v_m = 0;
     the estimator must return SC ≈ 0 within statistical noise.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
from analysis.observables_2d import _core_sc_mn


def _sc_truth(v_n_sq, v_m_sq, w=None):
    """Per-event truth SC(n,m) = <v_n² v_m²> − <v_n²><v_m²>."""
    if w is None:
        return (v_n_sq * v_m_sq).mean() - v_n_sq.mean() * v_m_sq.mean()
    wsum = w.sum()
    a = (w * v_n_sq * v_m_sq).sum() / wsum
    b = (w * v_n_sq).sum() / wsum
    c = (w * v_m_sq).sum() / wsum
    return a - b * c


def test_synthetic_exact_fixed_v():
    """Fixed v_n, v_m per event -- SC must match truth bit-exact."""
    rng = np.random.RandomState(0)
    n_events = 2000
    M = rng.normal(300.0, 30.0, n_events).clip(50.0)
    v_n = np.full(n_events, 0.05)
    v_m = np.full(n_events, 0.04)
    psi_n = rng.uniform(0, 2 * np.pi, n_events)
    psi_m = rng.uniform(0, 2 * np.pi, n_events)
    Qn = M * v_n * np.exp(1j * psi_n)
    Qm = M * v_m * np.exp(1j * psi_m)
    Q1 = np.zeros((n_events, 4), dtype=complex)
    Q1[:, 0] = M
    Q1[:, 2] = Qn  # harmonic n=2
    Q1[:, 3] = Qm  # harmonic m=3

    sc_qvec = _core_sc_mn(Q1, event_weights=None, n=2, m=3)
    sc_true = _sc_truth(v_n ** 2, v_m ** 2)
    # fixed v: <v²>² = v⁴ and <v²v²> = v⁴ so SC = 0 exactly
    assert abs(sc_qvec) < 1e-12, f"fixed-v SC must be 0, got {sc_qvec}"
    assert abs(sc_true) < 1e-12, f"truth SC must be 0, got {sc_true}"


def test_synthetic_exact_ebe_fluctuation():
    """Per-event ebE v_n, v_m -- SC matches truth bit-exact via Q-vectors."""
    rng = np.random.RandomState(1)
    n_events = 3000
    M = rng.normal(300.0, 30.0, n_events).clip(50.0)
    # ebE fluctuating v_2, v_3 with positive correlation
    base_n, sig_n = 0.05, 0.01
    base_m, sig_m = 0.04, 0.008
    rho = 0.5  # correlation between v_n and v_m event-by-event
    cov = np.array([[sig_n ** 2, rho * sig_n * sig_m],
                    [rho * sig_n * sig_m, sig_m ** 2]])
    v_n, v_m = rng.multivariate_normal([base_n, base_m], cov, n_events).T
    v_n = v_n.clip(0.001, 0.2)
    v_m = v_m.clip(0.001, 0.2)
    psi_n = rng.uniform(0, 2 * np.pi, n_events)
    psi_m = rng.uniform(0, 2 * np.pi, n_events)
    # Construct Q-vectors that yield |Q_n|²/M² = v_n² EXACTLY:
    Qn = M * v_n * np.exp(1j * psi_n)
    Qm = M * v_m * np.exp(1j * psi_m)
    Q1 = np.zeros((n_events, 4), dtype=complex)
    Q1[:, 0] = M
    Q1[:, 2] = Qn
    Q1[:, 3] = Qm

    sc_qvec = _core_sc_mn(Q1, event_weights=None, n=2, m=3)
    sc_true = _sc_truth(v_n ** 2, v_m ** 2)
    # The synthetic Q-vectors are constructed so the per-event estimator
    # |Q_n|²/M² == v_n² exactly; the SC(qvec) is therefore bit-exact equal
    # to SC(true).
    assert abs(sc_qvec - sc_true) < 1e-12, \
        f"SC(qvec)={sc_qvec} != SC(true)={sc_true} (delta={sc_qvec - sc_true})"
    # sanity: with rho>0 and the per-event v magnitudes, SC should be > 0
    assert sc_true > 0


def test_synthetic_exact_anticorrelated():
    """Anti-correlated v_n, v_m (rho < 0) -- SC must be negative."""
    rng = np.random.RandomState(2)
    n_events = 2000
    M = rng.normal(300.0, 30.0, n_events).clip(50.0)
    base_n, sig_n = 0.05, 0.01
    base_m, sig_m = 0.04, 0.008
    rho = -0.6
    cov = np.array([[sig_n ** 2, rho * sig_n * sig_m],
                    [rho * sig_n * sig_m, sig_m ** 2]])
    v_n, v_m = rng.multivariate_normal([base_n, base_m], cov, n_events).T
    v_n = v_n.clip(0.001, 0.2)
    v_m = v_m.clip(0.001, 0.2)
    psi_n = rng.uniform(0, 2 * np.pi, n_events)
    psi_m = rng.uniform(0, 2 * np.pi, n_events)
    Qn = M * v_n * np.exp(1j * psi_n)
    Qm = M * v_m * np.exp(1j * psi_m)
    Q1 = np.zeros((n_events, 4), dtype=complex)
    Q1[:, 0] = M
    Q1[:, 2] = Qn
    Q1[:, 3] = Qm

    sc_qvec = _core_sc_mn(Q1, event_weights=None, n=2, m=3)
    sc_true = _sc_truth(v_n ** 2, v_m ** 2)
    assert abs(sc_qvec - sc_true) < 1e-12
    assert sc_true < 0


def test_synthetic_with_event_weights():
    """Per-event centrality weight should reduce to a weighted mean."""
    rng = np.random.RandomState(3)
    n_events = 2000
    M = rng.normal(300.0, 30.0, n_events).clip(50.0)
    v_n, v_m = 0.05, 0.04
    psi_n = rng.uniform(0, 2 * np.pi, n_events)
    psi_m = rng.uniform(0, 2 * np.pi, n_events)
    Qn = M * v_n * np.exp(1j * psi_n)
    Qm = M * v_m * np.exp(1j * psi_m)
    Q1 = np.zeros((n_events, 4), dtype=complex)
    Q1[:, 0] = M
    Q1[:, 2] = Qn
    Q1[:, 3] = Qm

    # fixed v => SC = 0 regardless of weights
    w = rng.uniform(0.5, 1.5, n_events)
    sc_w = _core_sc_mn(Q1, event_weights=w, n=2, m=3)
    assert abs(sc_w) < 1e-12, f"weighted fixed-v SC must be 0, got {sc_w}"


def test_synthetic_harmonic_3_4():
    """Same machinery works for any (n,m); spot-check (3,4)."""
    rng = np.random.RandomState(4)
    n_events = 1500
    M = rng.normal(300.0, 30.0, n_events).clip(50.0)
    base_n, sig_n = 0.03, 0.008
    base_m, sig_m = 0.02, 0.005
    rho = 0.3
    cov = np.array([[sig_n ** 2, rho * sig_n * sig_m],
                    [rho * sig_n * sig_m, sig_m ** 2]])
    v_n, v_m = rng.multivariate_normal([base_n, base_m], cov, n_events).T
    v_n = v_n.clip(0.001, 0.2)
    v_m = v_m.clip(0.001, 0.2)
    Q1 = np.zeros((n_events, 5), dtype=complex)
    Q1[:, 0] = M
    Q1[:, 3] = M * v_n * np.exp(1j * rng.uniform(0, 2 * np.pi, n_events))
    Q1[:, 4] = M * v_m * np.exp(1j * rng.uniform(0, 2 * np.pi, n_events))

    sc_qvec = _core_sc_mn(Q1, event_weights=None, n=3, m=4)
    sc_true = _sc_truth(v_n ** 2, v_m ** 2)
    assert abs(sc_qvec - sc_true) < 1e-12
    assert sc_true > 0  # rho > 0


def test_no_flow_particle_level():
    """Particle-level no-flow: SC must be ~0 within statistical noise.

    Generates events with v_n = v_m = 0 (flat phi), fills Q-vectors from
    particles via the toy_mc converter, computes SC.  The direct-product
    estimator has O(1/M) self-corr noise even at zero flow; with M~300
    and N_events=2000 the residual should be well below 1e-3.
    """
    from analysis.toy_mc import (ToyGenConfig, generate_events,
                                  make_qvector_data)
    cfg = ToyGenConfig(n_events=2000, mult_mean=300, mult_sigma=20,
                        rap_max=1.0, seed=7, flow={})  # no flow
    events, truth = generate_events(cfg)
    qv = make_qvector_data(events, truth, n_pt=20, n_harm=5)
    Q1 = qv.Qn_sum if hasattr(qv, 'Qn_sum') else None
    # qv.Qn has shape (n_events, n_species, n_harm, n_pT); we need a single
    # (n_events, n_harm) Q-vector.  Use the Qn sum helper or the Q-vectors
    # the toy_mc test_toy_mc_flow already validated.
    from analysis.pipeline_dispatch import PTCUTS
    # weighted sum over species and pT using Jacobian
    from analysis.observables_2d import _compute_weighted_qvectors
    # build a flat Jacobian = 1 (unit weights) over the production pT grid
    pt_edges = qv.pt_bin_edges
    n_pt = len(pt_edges) - 1
    jac = np.ones((qv.n_species, n_pt))
    # event weights = 1
    w = np.ones(qv.n_events)
    Q1 = _compute_weighted_qvectors(qv, jac, 1)  # (n_events, n_harm) complex
    sc_23 = _core_sc_mn(Q1, w, 2, 3)
    sc_24 = _core_sc_mn(Q1, w, 2, 4)
    # both must be tiny
    assert abs(sc_23) < 1e-3, f"no-flow SC(2,3) too large: {sc_23}"
    assert abs(sc_24) < 1e-3, f"no-flow SC(2,4) too large: {sc_24}"


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

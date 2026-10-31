"""Unit tests for observable computation functions with synthetic data.

Generates events with a known flow signal, builds Q-vectors, and verifies
the computed v_n{2} matches the input within statistical uncertainty.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.observables import vn2, vn2_gap
from scripts.compute_flow_vn import (
    compute_vn2_harmonics,
    compute_vn2_gap_harmonics,
)


# ── Synthetic event generators ──────────────────────────────────────────────

def generate_flow_events(n_events, n_particles, v_true, n_harm=6, seed=42):
    """Generate synthetic Q-vectors with known flow.

    Each event has n_particles with azimuthal angles distributed as:
      dN/dφ ∝ 1 + 2·v₂·cos(2(φ − ψ_EP))

    Returns
    -------
    Qn : (n_events, n_harm) complex — Q-vectors (index 0=Q₀, 1=Q₁, 2=Q₂, …)
    Q0 : (n_events,) float — multiplicities
    dQ0 : (n_events,) float — double-weighted self-correlation
    """
    rng = np.random.RandomState(seed)
    psi_EP = rng.uniform(0, np.pi, n_events)

    Qn = np.zeros((n_events, n_harm), dtype=np.complex128)
    Q0 = np.zeros(n_events)
    dQ0 = np.zeros(n_events)

    for i in range(n_events):
        # Rejection sample φ from flow-modulated distribution.
        # Over-sample by 3× to avoid fallback (which would dilute flow).
        n_trials = int(np.ceil(n_particles * 3))
        dphi = rng.uniform(0, 2 * np.pi, n_trials)
        w = 1 + 2 * v_true * np.cos(2 * dphi)
        accept = rng.uniform(0, 1 + 2 * v_true, n_trials) < w
        kept = dphi[accept][:n_particles]
        if len(kept) < n_particles:
            kept = np.concatenate([
                kept,
                rng.uniform(0, 2 * np.pi, n_particles - len(kept)),
            ])
        phi = (psi_EP[i] + kept[:n_particles]) % (2 * np.pi)

        M = len(phi)
        Q0[i] = M
        Qn[i, 0] = M
        for h in range(1, n_harm):
            Qn[i, h] = np.sum(np.exp(1j * h * phi))
        dQ0[i] = M  # w_i = 1 for all → Σ w² = M

    return Qn, Q0, dQ0


def generate_subevent_shared(n_events, n_particles, v_true, decorrelation=1.0,
                              n_harm=6, seed=42):
    """Generate two-subevent Q-vectors from the SAME particles (2D-like).

    In 2D boost-invariant sims, each particle contributes to BOTH subevents
    A and B via the Jacobian (same particle at different η).  This generator
    creates one set of particles per event and copies the Q-vectors to both
    subevents, simulating perfect overlap.

    The dQ_cross = sqrt(dQ0_A * dQ0_B) = M is then correct for the gap formula.

    decorrelation < 1.0: subevent B's event-plane is rotated by N(0, σ²)
    to simulate longitudinal decorrelation.
    """
    rng = np.random.RandomState(seed)
    psi_EP = rng.uniform(0, np.pi, n_events)

    if decorrelation >= 1.0:
        psi_B = psi_EP.copy()
    else:
        psi_B = psi_EP + rng.normal(0, np.sqrt(1 - decorrelation), n_events)

    Qn_A = np.zeros((n_events, n_harm), dtype=np.complex128)
    Qn_B = np.zeros((n_events, n_harm), dtype=np.complex128)
    Q0_A = np.zeros(n_events)
    Q0_B = np.zeros(n_events)

    for i in range(n_events):
        # Shared particles (same azimuthal modulation)
        # Over-sample by 3× to avoid fallback (which would dilute flow).
        n_trials = int(np.ceil(n_particles * 3))
        dphi = rng.uniform(0, 2 * np.pi, n_trials)
        w = 1 + 2 * v_true * np.cos(2 * dphi)
        accept = rng.uniform(0, 1 + 2 * v_true, n_trials) < w
        kept = dphi[accept][:n_particles]
        if len(kept) < n_particles:
            kept = np.concatenate([
                kept, rng.uniform(0, 2 * np.pi, n_particles - len(kept)),
            ])
        dphi_shared = kept[:n_particles]

        # Subevent A particles
        phi_A = (psi_EP[i] + dphi_shared) % (2 * np.pi)
        M_A = len(phi_A)
        Q0_A[i] = M_A
        Qn_A[i, 0] = M_A
        for h in range(1, n_harm):
            Qn_A[i, h] = np.sum(np.exp(1j * h * phi_A))

        # Subevent B: same particles, different event-plane angle
        phi_B = (psi_B[i] + dphi_shared) % (2 * np.pi)
        M_B = len(phi_B)
        Q0_B[i] = M_B
        Qn_B[i, 0] = M_B
        for h in range(1, n_harm):
            Qn_B[i, h] = np.sum(np.exp(1j * h * phi_B))

    dQ0_A = Q0_A.copy()  # w² = 1 for each → Σ w² = M
    dQ0_B = Q0_B.copy()

    return Qn_A, Q0_A, dQ0_A, Qn_B, Q0_B, dQ0_B


# ── v₂ index helpers ────────────────────────────────────────────────────────
# compute_vn2_harmonics returns array [v₁{2}, v₂{2}, v₃{2}, v₄{2}, v₅{2}]
# because it starts from h=1 (skips Q₀ at index 0).
V2_IDX = 1  # index of v₂ in the returned array
V_TRUE = 0.05


# ── Tests ───────────────────────────────────────────────────────────────────

def test_vn2_single_no_flow():
    """Zero flow → v₂ should be zero within noise."""
    n_events = 2000
    n_particles = 500
    Qn, Q0, dQ0 = generate_flow_events(n_events, n_particles, v_true=0.0, seed=1)
    weights = np.ones(n_events)
    vals = compute_vn2_harmonics(Qn, dQ0, weights)
    assert abs(vals[V2_IDX]) < 0.01, f"v₂ near 0 with no flow, got {vals[V2_IDX]:.6f}"


def test_vn2_single_known_v2():
    """v₂{2} should recover the input v₂ within statistical uncertainty."""
    v_true = V_TRUE
    n_events = 5000
    n_particles = 300
    Qn, Q0, dQ0 = generate_flow_events(n_events, n_particles, v_true=v_true, seed=2)
    vals = compute_vn2_harmonics(Qn, dQ0, np.ones(n_events))
    v2_rec = vals[V2_IDX]
    frac_err = abs(v2_rec - v_true) / v_true
    assert frac_err < 0.03, (
        f"v₂ recovery: true={v_true}, recovered={v2_rec:.6f}, error={frac_err:.2%}")


def test_vn2_single_larger_flow():
    """v₂{2}=0.1 should also be recoverable."""
    v_true = 0.1
    n_events = 2000
    n_particles = 200
    Qn, Q0, dQ0 = generate_flow_events(n_events, n_particles, v_true=v_true, seed=3)
    vals = compute_vn2_harmonics(Qn, dQ0, np.ones(n_events))
    v2_rec = vals[V2_IDX]
    frac_err = abs(v2_rec - v_true) / v_true
    assert frac_err < 0.05, (
        f"v₂={v_true}: recovered={v2_rec:.6f}, error={frac_err:.2%}")


def test_vn2_weighted():
    """v₂{2} with event weights should still recover the true value."""
    v_true = V_TRUE
    n_events = 5000
    n_particles = 300
    Qn, Q0, dQ0 = generate_flow_events(n_events, n_particles, v_true=v_true, seed=4)
    weights = Q0 / Q0.sum()
    vals = compute_vn2_harmonics(Qn, dQ0, weights)
    v2_rec = vals[V2_IDX]
    frac_err = abs(v2_rec - v_true) / v_true
    assert frac_err < 0.05, (
        f"Weighted v₂: true={v_true}, recovered={v2_rec:.6f}")


def test_vn2_single_with_net_charge():
    """Variable multiplicity should not bias v₂ recovery."""
    v_true = V_TRUE
    n_events = 3000
    rng = np.random.RandomState(7)
    psi_EP = rng.uniform(0, np.pi, n_events)
    n_particles = rng.randint(100, 500, n_events)

    Qn = np.zeros((n_events, 6), dtype=np.complex128)
    dQ0 = np.zeros(n_events)

    for i in range(n_events):
        M = n_particles[i]
        n_trials = int(np.ceil(M * 3))
        dphi = rng.uniform(0, 2 * np.pi, n_trials)
        w = 1 + 2 * v_true * np.cos(2 * dphi)
        accept = rng.uniform(0, 1 + 2 * v_true, n_trials) < w
        kept = dphi[accept][:M]
        if len(kept) < M:
            kept = np.concatenate([kept, rng.uniform(0, 2 * np.pi, M - len(kept))])
        phi = (psi_EP[i] + kept[:M]) % (2 * np.pi)

        Qn[i, 0] = M
        for h in range(1, 6):
            Qn[i, h] = np.sum(np.exp(1j * h * phi))
        dQ0[i] = M

    vals = compute_vn2_harmonics(Qn, dQ0, np.ones(n_events))
    v2_rec = vals[V2_IDX]
    frac_err = abs(v2_rec - v_true) / v_true
    assert frac_err < 0.05, (
        f"Variable mult: true={v_true}, recovered={v2_rec:.6f}")


def test_observables_no_nan():
    """All harmonics should return finite values."""
    n_events = 500
    n_particles = 100
    Qn, Q0, dQ0 = generate_flow_events(n_events, n_particles, v_true=0.02, seed=8)
    vals = compute_vn2_harmonics(Qn, dQ0, np.ones(n_events))
    assert np.all(np.isfinite(vals)), "All harmonics should be finite"


def test_dQ0_subtraction_effect():
    """Without dQ0, v₂ should be biased high at low multiplicity (zero flow)."""
    n_events = 2000
    n_particles = 20
    v_true = 0.0
    Qn, Q0, dQ0 = generate_flow_events(n_events, n_particles, v_true=v_true, seed=9)

    vals_correct = compute_vn2_harmonics(Qn, dQ0, np.ones(n_events))
    vals_biased = compute_vn2_harmonics(Qn, np.zeros(n_events), np.ones(n_events))

    v2_correct = abs(vals_correct[V2_IDX])
    v2_biased = abs(vals_biased[V2_IDX])
    assert v2_biased > v2_correct, (
        f"Without dQ0, v₂ should be HIGHER: correct={v2_correct:.4f}, "
        f"biased={v2_biased:.4f}")


# ── Gap (two-subevent) tests with SHARED particles ─────────────────────────

def test_vn2_gap_shared_no_decorrelation():
    """v₂{2}ab with decorrelation=1.0 and shared particles → v₂{2}ab = v₂{2}."""
    n_events = 3000
    n_particles = 200
    v_true = V_TRUE

    Qn_A, Q0_A, dQ0_A, Qn_B, Q0_B, dQ0_B = generate_subevent_shared(
        n_events, n_particles, v_true, decorrelation=1.0, seed=5)

    weights = np.ones(n_events)
    vals_single = compute_vn2_harmonics(Qn_A, dQ0_A, weights)
    # Shared particles with unit weights: δ_AB = Σ w_A·w_B = M per event.
    dQ_cross = Q0_A.copy()
    vals_gap = compute_vn2_gap_harmonics(Qn_A, Qn_B, dQ_cross, weights)

    v2_single = vals_single[V2_IDX]
    v2_gap = vals_gap[V2_IDX]

    diff = abs(v2_gap - v2_single)
    assert diff < 0.005, (
        f"v₂{{2}}ab ≈ v₂{{2}} with shared particles: "
        f"single={v2_single:.6f}, gap={v2_gap:.6f}, diff={diff:.6f}")


def test_vn2_gap_shared_decorrelation():
    """v₂{2}ab < v₂{2} with decorrelation < 1 (shared particles)."""
    n_events = 3000
    n_particles = 200
    v_true = V_TRUE

    Qn_A, Q0_A, dQ0_A, Qn_B, Q0_B, dQ0_B = generate_subevent_shared(
        n_events, n_particles, v_true, decorrelation=0.8, seed=6)

    weights = np.ones(n_events)
    vals_single = compute_vn2_harmonics(Qn_A, dQ0_A, weights)
    # Shared particles with unit weights: δ_AB = Σ w_A·w_B = M per event.
    dQ_cross = Q0_A.copy()
    vals_gap = compute_vn2_gap_harmonics(Qn_A, Qn_B, dQ_cross, weights)

    v2_single = vals_single[V2_IDX]
    v2_gap = vals_gap[V2_IDX]

    assert v2_gap < v2_single, (
        f"v₂{{2}}ab < v₂{{2}} with decorr=0.8: "
        f"single={v2_single:.6f}, gap={v2_gap:.6f}")


# ── Low-level wrapper tests ─────────────────────────────────────────────────

def test_vn2_wrapper():
    """analysis.observables.vn2 with known inputs."""
    n_events = 1000
    Qn = np.ones(n_events, dtype=complex)
    Q0 = np.ones(n_events) * 100
    dQ0 = np.ones(n_events) * 100
    result = vn2(Qn, Q0, dQ0)  # should be -0.1
    assert np.isclose(result, -0.1, atol=1e-10), f"Expected -0.1, got {result}"


def test_vn2_gap_wrapper():
    """analysis.observables.vn2_gap with known inputs (should be -0.1)."""
    n_events = 1000
    Qn_A = np.ones(n_events, dtype=complex)
    Qn_B = np.ones(n_events, dtype=complex)
    MA = np.ones(n_events) * 100
    MB = np.ones(n_events) * 100
    dQ_cross = np.ones(n_events) * 100
    result = vn2_gap(Qn_A, Qn_B, MA, MB, dQ_cross)
    assert np.isclose(result, -0.1, atol=1e-10), f"Expected -0.1, got {result}"


def test_vn2_gap_wrapper_positive():
    """vn2_gap with well-correlated Qn for a positive result."""
    n_events = 1000
    Qn_A = np.ones(n_events, dtype=complex) * 10
    Qn_B = np.ones(n_events, dtype=complex) * 10
    MA = np.ones(n_events) * 100
    MB = np.ones(n_events) * 100
    dQ_cross = np.ones(n_events) * 10
    result = vn2_gap(Qn_A, Qn_B, MA, MB, dQ_cross)
    assert result > 0, f"Expected positive v, got {result}"


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [fn for fn in dir() if fn.startswith('test_')]
    n_pass = 0
    n_fail = 0
    for name in sorted(tests):
        try:
            globals()[name]()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            n_fail += 1
    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    exit(1 if n_fail else 0)

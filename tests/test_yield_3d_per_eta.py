"""Toy MC test for prepare_yield_3d_per_eta and _wrap_yield_per_eta.

Generates events with a known per-eta shape and verifies that the
pipeline recovers dN/deta(eta_i) = N_in_bin_i / delta_eta per eta bin.
"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
from analysis.qvector import QGridData
from analysis.pipeline_dispatch import prepare_yield_3d_per_eta, _wrap_yield_per_eta


def make_synthetic_grid(eta_edges, counts, pT_edges=(0.1, 2.0)):
    """Build a QGridData. counts: shape (n_events, n_eta, n_pT) with only harmonic 0 set."""
    pT_edges_arr = np.array(pT_edges)
    n_events, n_eta, n_pT = counts.shape
    n_harm = 6
    Qn = np.zeros((n_events, n_eta, n_pT, n_harm), dtype=np.complex128)
    Qn[:, :, :, 0] = counts.astype(np.float64)
    grid = QGridData(Qn=Qn, eta_edges=eta_edges, pt_edges=pT_edges_arr,
                     cent_mult=np.ones(n_events),
                     nsamples=np.ones(n_events),
                     sim_type="3d")
    return grid


def test_uniform_distribution():
    """Flat in [0,5]: dN/deta in that range = 80/0.2 = 400."""
    np.random.seed(0)
    eta_edges = np.linspace(-5, 5, 51)  # 50 bins, 0.2 wide
    n_eta = len(eta_edges) - 1
    n_events = 300
    counts = np.zeros((n_events, n_eta, 1), dtype=np.float64)
    for e in range(n_events):
        h, _ = np.histogram(np.random.uniform(0, 5, 80), bins=eta_edges)
        counts[e, :, 0] = h
    grid = make_synthetic_grid(eta_edges, counts)
    delta_eta = 0.2
    n_bins_in_range = 25  # 50 bins over [0,5]
    N = 80
    expected = (N / n_bins_in_range) / delta_eta  # 80/25/0.2 = 16
    params = {
        'single_range': {'eta': (-5, 5), 'pT': (0.1, 2.0)},
        'differential_bins': list(zip(eta_edges[:-1], eta_edges[1:])),
        'particle_id': {'species': 'charged'},
    }
    data = prepare_yield_3d_per_eta(grid, params)
    out = _wrap_yield_per_eta(data, np.ones(n_events))
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    for i, ec in enumerate(eta_centers):
        if 0 <= ec <= 5:
            assert abs(out[i] - expected) / expected < 0.10, \
                f"bin {i} eta={ec:.2f}: got {out[i]:.2f}, expected {expected:.2f}"
        else:
            assert out[i] < 1.0, f"bin {i} eta={ec:.2f}: expected ~0, got {out[i]:.2f}"
    print(f"  test_uniform_distribution: PASS (mean in [0,5] = {out[(eta_centers >= 0) & (eta_centers <= 5)].mean():.2f}, expected {expected:.2f})")


def test_single_bin():
    """All particles in a single bin: that bin should peak, others zero."""
    eta_edges = np.linspace(-5, 5, 51)
    n_eta = 50
    n_events = 200
    counts = np.zeros((n_events, n_eta, 1), dtype=np.float64)
    # Place 50 particles directly in bin 25 (eta edges [0.0, 0.2])
    counts[:, 25, 0] = 50.0
    grid = make_synthetic_grid(eta_edges, counts)
    delta_eta = 0.2
    expected_peak = 50.0 / delta_eta  # 250
    params = {
        'single_range': {'eta': (-5, 5), 'pT': (0.1, 2.0)},
        'differential_bins': list(zip(eta_edges[:-1], eta_edges[1:])),
        'particle_id': {'species': 'charged'},
    }
    data = prepare_yield_3d_per_eta(grid, params)
    out = _wrap_yield_per_eta(data, np.ones(n_events))
    peak_idx = int(np.argmax(out))
    assert peak_idx == 25, f"expected peak at bin 25, got {peak_idx}"
    assert abs(out[peak_idx] - expected_peak) / expected_peak < 0.05, \
        f"peak {peak_idx} = {out[peak_idx]:.2f}, expected {expected_peak:.2f}"
    off_peak_max = max(out[i] for i in range(len(out)) if i != peak_idx)
    assert off_peak_max < 1.0, f"off-peak max {off_peak_max:.2f}"
    print(f"  test_single_bin: PASS (peak bin {peak_idx} = {out[peak_idx]:.2f}, expected {expected_peak:.2f}, off-peak<1)")


def test_no_particles():
    """Empty events: all zero."""
    eta_edges = np.linspace(-5, 5, 51)
    n_eta = 50
    counts = np.zeros((100, n_eta, 1), dtype=np.float64)
    grid = make_synthetic_grid(eta_edges, counts)
    params = {
        'single_range': {'eta': (-5, 5), 'pT': (0.1, 2.0)},
        'differential_bins': list(zip(eta_edges[:-1], eta_edges[1:])),
        'particle_id': {'species': 'charged'},
    }
    data = prepare_yield_3d_per_eta(grid, params)
    out = _wrap_yield_per_eta(data, np.ones(100))
    assert np.all(out == 0), f"expected all zero, got max {out.max()}"
    print(f"  test_no_particles: PASS (all zero)")


if __name__ == '__main__':
    print("=== test_yield_3d_per_eta ===")
    test_uniform_distribution()
    test_single_bin()
    test_no_particles()
    print("=== ALL PASSED ===")

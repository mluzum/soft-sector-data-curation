import numpy as np


def covariance_from_replicates(replicates):
    """The delete-one-block jackknife covariance, (B-1)/B sum_b (r_b - rbar)(r_b - rbar)^T,
    from (n_blocks, n_obs) replicates (run_pipeline jk_replicates): exactly
    the `covariance` run_pipeline writes with --covariance full (rank <= B-1)."""
    r = np.asarray(replicates, dtype=float)
    d = r - r.mean(axis=0)
    return (r.shape[0] - 1) / r.shape[0] * d.T @ d


def jackknife_blocks(n_events, n_blocks, event_ids=None):
    """Delete-one-block masks (False = the deleted block).

    Contiguous blocks in container order by default.  With ``event_ids``
    (one id per event), block b is the events whose id % n_blocks == b: in a
    campaign with common random numbers (event id k has the same seed at
    every design point) block b then holds the same seeds at every point,
    whatever events are missing or reordered, so replicates of different
    design points can be correlated directly.
    """
    if event_ids is not None:
        ids = np.asarray(event_ids).astype(np.int64)
        if ids.shape != (n_events,):
            raise ValueError(f'event_ids: {ids.shape} for {n_events} events')
        return [ids % n_blocks != b for b in range(n_blocks)]
    block_size = n_events // n_blocks
    blocks = []
    for i in range(n_blocks):
        lo = i * block_size
        hi = n_events if i == n_blocks - 1 else (i + 1) * block_size
        mask = np.ones(n_events, dtype=bool)
        mask[lo:hi] = False
        blocks.append(mask)
    return blocks


def jackknife_error(obs, n_blocks=None):
    obs = np.asarray(obs)
    n = obs.shape[0]
    if n_blocks is None:
        n_blocks = n
    blocks = jackknife_blocks(n, min(n_blocks, n))

    theta_hat = np.mean(obs, axis=0)
    theta_i = np.array([np.mean(obs[mask], axis=0) for mask in blocks])
    n_b = len(blocks)
    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - theta_hat)
    var = (n_b - 1) / n_b * np.sum((theta_i - np.mean(theta_i, axis=0)) ** 2, axis=0)
    return np.sqrt(var), bias


def jackknife_covariance(obs, n_blocks=None):
    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = obs[:, None]
    n, k = obs.shape
    if n_blocks is None:
        n_blocks = n
    n_b = min(n_blocks, n)
    blocks = jackknife_blocks(n, n_b)

    theta_hat = np.mean(obs, axis=0)
    theta_i = np.array([np.mean(obs[mask], axis=0) for mask in blocks])
    diff = theta_i - theta_hat
    cov = (n_b - 1) / n_b * np.dot(diff.T, diff)
    return cov


def jackknife_cross_covariance(obs_list):
    n = obs_list[0].shape[0]
    combined = np.column_stack(obs_list)
    return jackknife_covariance(combined, n_blocks=n)

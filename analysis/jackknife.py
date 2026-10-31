import numpy as np


def jackknife_blocks(n_events, n_blocks):
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

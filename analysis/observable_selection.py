"""Model-precision summary and observable selection for a campaign table.

An emulator (PCA plus Gaussian processes, and most others) needs every
observable at every design point, and an observable whose model statistical
noise dominates its spread across the design carries no information to
learn.  Selection is therefore per observable (a column of the runs x
observables table), over the whole design, never per point: dropping the
points where the model is sparse would leave a ragged training set and bias
it toward the regions of parameter space with many particles (central,
high-multiplicity design points).  The values themselves are never changed;
the per-point model error (jackknife) belongs in the emulator as
heteroscedastic noise and in the likelihood, next to the experimental error.

quality() summarizes each observable across the runs; select() applies the
criteria:
  undefined   undefined (NaN: no pairs, no particles) in more than
              max_undefined of the runs (default: any);
  no_data     no experimental value or error to compare with (require_data);
  noise       median model error / experimental error above max_error_ratio;
  rel_noise   median |model error / value| above max_rel_error.
Each rejected observable keeps the first criterion it fails.
"""
import numpy as np

CRITERIA = ('undefined', 'no_data', 'noise', 'rel_noise')


def quality(values, errors, exp_values, exp_errors):
    """Per-observable summary of a (n_runs, n_obs) table.

    Returns {'undefined_fraction', 'median_rel_error', 'median_error_ratio',
    'has_data'}, each (n_obs,)."""
    values = np.asarray(values, float)
    errors = np.asarray(errors, float)
    exp_values = np.asarray(exp_values, float)
    exp_errors = np.asarray(exp_errors, float)
    finite = np.isfinite(values) & np.isfinite(errors)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(finite, np.abs(errors / values), np.nan)
        ratio = np.where(finite, errors / exp_errors[None, :], np.nan)
    rel[~np.isfinite(rel)] = np.nan
    ratio[~np.isfinite(ratio)] = np.nan

    def median(a):
        out = np.full(a.shape[1], np.nan)
        ok = np.isfinite(a).any(axis=0)
        out[ok] = np.nanmedian(a[:, ok], axis=0)
        return out
    n_runs = max(values.shape[0], 1)
    return {'undefined_fraction': 1.0 - finite.sum(axis=0) / n_runs,
            'median_rel_error': median(rel),
            'median_error_ratio': median(ratio),
            'has_data': np.isfinite(exp_values) & np.isfinite(exp_errors) & (exp_errors > 0)}


def select(q, max_undefined=0.0, require_data=True, max_error_ratio=None, max_rel_error=None):
    """(selected (n_obs,) bool, rejected_by (n_obs,) str: '' or the first criterion failed)."""
    n = len(q['undefined_fraction'])
    reason = np.array([''] * n, dtype=object)

    def fail(mask, why):
        reason[(reason == '') & mask] = why
    fail(q['undefined_fraction'] > max_undefined, 'undefined')
    if require_data:
        fail(~q['has_data'], 'no_data')
    if max_error_ratio is not None:
        fail(~(q['median_error_ratio'] <= max_error_ratio), 'noise')
    if max_rel_error is not None:
        fail(~(q['median_rel_error'] <= max_rel_error), 'rel_noise')
    return reason == '', reason.astype(str)

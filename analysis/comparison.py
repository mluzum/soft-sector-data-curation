import numpy as np


def compute_chi2(sim_values, sim_errors, exp_values, exp_errors):
    diff = np.asarray(sim_values) - np.asarray(exp_values)
    err = np.sqrt(np.asarray(sim_errors) ** 2 + np.asarray(exp_errors) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2 = np.sum((diff / err) ** 2)
    ndof = len(diff)
    return chi2, ndof


def format_table(centralities, sim_values, sim_errors, exp_values, exp_errors,
                 label="Observable", ndigits=4):
    lines = []
    header = f"{'Centrality':>15}  {f'Sim {label}':>12}  {f'Exp {label}':>12}  {'Pull':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, cent in enumerate(centralities):
        sv = sim_values[i] if i < len(sim_values) else np.nan
        se = sim_errors[i] if i < len(sim_errors) else np.nan
        ev = exp_values[i] if i < len(exp_values) else np.nan
        ee = exp_errors[i] if i < len(exp_errors) else np.nan
        pull = (sv - ev) / np.sqrt(se ** 2 + ee ** 2) if (se > 0 and ee > 0) else np.nan
        lines.append(
            f"{cent:>15}  {sv:>12.{ndigits}f}  {ev:>12.{ndigits}f}  {pull:>+8.3f}"
        )
    chi2, ndof = compute_chi2(sim_values, sim_errors, exp_values, exp_errors)
    lines.append(f"\nχ²/ndof = {chi2:.2f} / {ndof}")
    return "\n".join(lines)


def format_table_pt(pt_bins, sim_values, sim_errors, exp_values, exp_errors,
                    label="v_n{2}", ndigits=4):
    lines = []
    header = f"{'pT':>10}  {f'Sim {label}':>12}  {f'Exp {label}':>12}  {'Pull':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i in range(len(pt_bins)):
        sv = sim_values[i] if i < len(sim_values) else np.nan
        se = sim_errors[i] if i < len(sim_errors) else np.nan
        ev = exp_values[i] if i < len(exp_values) else np.nan
        ee = exp_errors[i] if i < len(exp_errors) else np.nan
        pull = (sv - ev) / np.sqrt(se ** 2 + ee ** 2) if (se > 0 and ee > 0) else np.nan
        lines.append(
            f"{pt_bins[i]:>10.3f}  {sv:>12.{ndigits}f}  {ev:>12.{ndigits}f}  {pull:>+8.3f}"
        )
    chi2, ndof = compute_chi2(sim_values, sim_errors, exp_values, exp_errors)
    lines.append(f"\nχ²/ndof = {chi2:.2f} / {ndof}")
    return "\n".join(lines)


def make_comparison_dict(sim_values, sim_errors, exp_values, exp_errors,
                         centralities=None, pt_bins=None, eta_bins=None):
    return {
        "sim_values": np.asarray(sim_values).tolist(),
        "sim_errors": np.asarray(sim_errors).tolist(),
        "exp_values": np.asarray(exp_values).tolist(),
        "exp_errors": np.asarray(exp_errors).tolist(),
        "centralities": centralities,
        "pt_bins": pt_bins,
        "eta_bins": eta_bins,
        "chi2": compute_chi2(sim_values, sim_errors, exp_values, exp_errors),
    }

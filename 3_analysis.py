"""Reproduce the figures from 3_analysis_new.ipynb as a plain script.

Every plot is written to figures/3_analysis/ instead of being shown inline.

Adaptation vs. the notebook: the hybrid output files are now ordered greedily
(2_hybrid.py backward elimination), so the spatially-varying parameter SET at
each step differs (e.g. "km", "beta-kp-km", ...). The notebook sorted files by
string length as a proxy for the number of spatial parameters, which no longer
holds (e.g. "spatialkm" is shorter than "spatialnone"). Here we instead sort by
the actual number of spatial parameters parsed from the filename.
"""

import os
import pickle
from collections import Counter
from itertools import product

import matplotlib
matplotlib.use("Agg")  # no display; we only save figures
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from config import default_param_ranges

folder = "hybrid_outputs"  # 'hybrid_outputs_Tsense'
Tsense = ""                # 'Tsense_'
OUTDIR = "figures/3_analysis"

os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    """Full path for a figure saved under the 3_analysis output folder."""
    return os.path.join(OUTDIR, name)


def n_spatial_params(fname):
    """Number of spatially-varying parameters encoded in a hybrid output filename.

    Filenames end with '..._spatial<PARAMS>.pkl' where <PARAMS> is either 'none'
    (all parameters global) or a '-'-joined list of parameter names.
    """
    token = fname.rsplit("_spatial", 1)[-1]
    if token.endswith(".pkl"):
        token = token[:-len(".pkl")]
    if token == "none":
        return 0
    return len(token.split("-"))


def r2(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ---------------------------------------------------------------------------
# R2-vs-n_spatial line plots
# ---------------------------------------------------------------------------

def analyse_one(temp, fold, md, mt, sat, targets, n_spatial, train_val_test):
    files = [
        fname for fname in os.listdir(folder)
        if f"hybrid_{Tsense}temp{temp}_fold{fold}_md{md}_mt{mt}_sat{sat}_targets{targets}_" in fname
    ]
    # order by the number of spatially-varying parameters (0, 1, 2, ...)
    files = sorted(files, key=n_spatial_params)
    with open(f"{folder}/" + files[n_spatial], "rb") as f:
        results = pickle.load(f)
    test_R2s = {}
    for target in targets.split("-"):
        test_idx = results["split"] == train_val_test
        tar = results[f"target_{target}"][test_idx]
        pred = results[f"pred_{target}"][test_idx]
        r2_value = r2(tar.values, pred.values)
        test_R2s[target] = float(r2_value)
    return test_R2s


def get_all_n_sp(temp, fold, md, mt, sat, targets, train_val_test):
    all_scores = []
    for i in range(20):
        try:
            all_scores.append(analyse_one(temp, fold, md, mt, sat, targets, i, train_val_test))
        except Exception:
            pass
    return all_scores


def full_plot(temp, md, mt, sat, train_val_test, save_name):
    target_sets = ["SOC", "SOC-MIC", "SOC-MAOC", "SOC-MAOC-MIC"]  # 4 columns
    all_targets = ["SOC", "MAOC", "MIC"]  # 3 possible targets, drawn as lines
    target_colors = {"SOC": "brown", "MAOC": "yellow", "MIC": "orange"}
    fig, axs = plt.subplots(1, len(target_sets), figsize=(5 * len(target_sets), 4), squeeze=False)
    fig.suptitle(f"md: {md}, mt: {mt}, sat: {sat}", fontsize=16, y=1.05)
    for col_idx, targets in enumerate(target_sets):
        split_targets = targets.split("-")
        ax = axs[0, col_idx]
        for target in all_targets:
            if target in split_targets:
                all_scores = []
                for fold in range(10):
                    fold_results = get_all_n_sp(temp, fold, md, mt, sat, targets, train_val_test)
                    fold_target_r2s = [x[target] for x in fold_results if target in x]
                    all_scores.append(fold_target_r2s)
                for i, fold_scores in enumerate(all_scores):
                    ax.plot(
                        range(len(fold_scores)),
                        fold_scores,
                        color=target_colors[target],
                        alpha=0.5,
                        label=f"{target} fold {i+1}" if i == 0 else "",
                    )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("# spatially varying parameters")
        ax.set_ylabel("R2 Score")
        ax.set_title(f"Constraints {targets}")
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 8)
        handles, labels = ax.get_legend_handles_labels()
        new_handles_labels = {}
        for handle, label in zip(handles, labels):
            if label and label.split()[0] not in new_handles_labels:
                new_handles_labels[label.split()[0]] = handle
        ax.legend(new_handles_labels.values(), new_handles_labels.keys())
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out(save_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all3_constraints_grid(save_name):
    temp, train_val_test = "static", "test"
    constraints = "SOC-MAOC-MIC"
    all_targets = ["SOC", "MAOC", "MIC"]
    target_colors = {"SOC": "brown", "MAOC": "gold", "MIC": "orange"}

    mds = ["linear", "MM"]  # rows (RMM left out)
    combos = [
        ("linear", "no"), ("density_dependent", "no"),
        ("linear", "Langmuir"), ("density_dependent", "Langmuir"),
    ]  # columns (mt, sat)

    fig, axs = plt.subplots(len(mds), len(combos), figsize=(5 * len(combos), 4 * len(mds)), squeeze=False)
    for r, md in enumerate(mds):
        for c, (mt, sat) in enumerate(combos):
            ax = axs[r, c]
            for target in all_targets:
                all_scores = []
                for fold in range(10):
                    fold_results = get_all_n_sp(temp, fold, md, mt, sat, constraints, train_val_test)
                    all_scores.append([x[target] for x in fold_results if target in x])
                for i, fold_scores in enumerate(all_scores):
                    ax.plot(
                        range(len(fold_scores)), fold_scores,
                        color=target_colors[target], alpha=0.5,
                        label=target if i == 0 else "",
                    )
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("# spatially varying parameters")
            ax.set_ylabel("R2 Score")
            ax.set_title(f"md: {md}, mt: {mt}, sat: {sat}")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, 8)
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys())
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out(save_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_target_curves(save_name):
    temp, train_val_test = "static", "test"

    versions = [
        ("linear", "linear", "no"),
        ("linear", "linear", "Langmuir"),
        ("linear", "density_dependent", "no"),
        ("linear", "density_dependent", "Langmuir"),
        ("MM", "linear", "no"),
        ("MM", "linear", "Langmuir"),
        ("MM", "density_dependent", "no"),
        ("MM", "density_dependent", "Langmuir"),
        ("RMM", "linear", "no"),
        ("RMM", "linear", "Langmuir"),
        ("RMM", "density_dependent", "no"),
        ("RMM", "density_dependent", "Langmuir"),
    ]

    md_colors = {"linear": "crimson", "MM": "steelblue", "RMM": "seagreen"}
    mt_marker = {"linear": "o", "density_dependent": "^"}

    def mean_target_curve(md, mt, sat, constraints, targets, n_folds=10):
        all_scores = []
        for fold in range(n_folds):
            fold_results = get_all_n_sp(temp, fold, md, mt, sat, constraints, train_val_test)
            all_scores.append([
                np.mean([x[t] for t in targets])
                for x in fold_results
                if all(t in x for t in targets)
            ])
        max_len = max((len(s) for s in all_scores), default=0)
        if max_len == 0:
            return np.array([])
        arr = np.full((len(all_scores), max_len), np.nan)
        for i, s in enumerate(all_scores):
            arr[i, :len(s)] = s
        return np.nanmean(arr, axis=0)

    all_constraints_targets = [
        ("SOC", ["SOC"]),
        ("SOC-MIC", ["SOC", "MIC"]),
        ("SOC-MAOC", ["SOC", "MAOC"]),
        ("SOC-MAOC-MIC", ["SOC", "MAOC", "MIC"]),
    ]
    n_rows, n_cols = 2, 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10), squeeze=False)
    ylim = (0.75, 1)
    legend_handles_labels = None

    for idx, (constraints, targets) in enumerate(all_constraints_targets):
        row = idx // n_cols
        col = idx % n_cols
        ax = axs[row, col]
        target_label = " & ".join(targets)
        handles, labels = [], []
        for md, mt, sat in versions:
            curve = mean_target_curve(md, mt, sat, constraints, targets)
            if curve.size == 0:
                continue
            (l,) = ax.plot(
                range(len(curve)), curve,
                color=md_colors[md],
                linestyle="dotted" if sat == "Langmuir" else "-",
                marker=mt_marker[mt], markersize=4, alpha=0.9,
                label=f"{md}, {mt}, {sat}",
            )
            handles.append(l)
            labels.append(f"{md}, {mt}, {sat}")
        ax.set_xlabel("# spatially varying parameters")
        ax.set_ylabel(f"Mean {target_label} R²")
        ax.set_title(f"{constraints} constrained")
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 8)
        if idx == 0:
            legend_handles_labels = (handles, labels)

    plt.tight_layout(rect=[0, 0.14, 1, 1])

    if legend_handles_labels:
        handles_uniq, labels_uniq, seen = [], [], set()
        for h, l in zip(*legend_handles_labels):
            if l not in seen:
                handles_uniq.append(h)
                labels_uniq.append(l)
                seen.add(l)
        fig.legend(
            handles_uniq, labels_uniq,
            loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, fontsize=9, frameon=True,
        )

    fig.savefig(out(save_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Equifinality / parameter-relationship scatter plots
# ---------------------------------------------------------------------------

with open("1_preprocessed.pkl", "rb") as f:
    data = pickle.load(f)


def get_data(var):
    return data[var]


def get_output(var, md, mt, sat, constraints, output_type):
    all_files = os.listdir(folder)
    version_files = [
        fname for fname in all_files
        if f"_md{md}_mt{mt}_sat{sat}_targets{constraints}_spatial" in fname
    ]
    output_array = None
    indices = None
    for fold in range(10):
        fold_files = [fname for fname in version_files if f"fold{fold}" in fname]
        # order by number of spatial parameters (none -> 0 first, then 1, 2, ...)
        fold_files.sort(key=n_spatial_params)
        for file_idx, file in enumerate(fold_files):
            with open(os.path.join(folder, file), "rb") as f:
                output_data = pickle.load(f)
            indices = output_data.index
            if output_array is None:
                output_array = np.full((len(fold_files), 10, len(output_data)), np.nan)
            if output_type == "param":
                output_array[file_idx, fold, :] = output_data[f"param_{var}"]
            elif output_type == "target":
                output_array[file_idx, fold, :] = output_data[f"target_{var}"]
            elif output_type == "prediction":
                output_array[file_idx, fold, :] = output_data[f"pred_{var}"]
    return indices, output_array


def plot_all_nsp(x, y, x_name, y_name, path, parity=False, n_sp_to_plot=None):
    n_sp_total = y.shape[0]
    if n_sp_to_plot is None or n_sp_to_plot > n_sp_total:
        sp_indices = range(n_sp_total)
    else:
        sp_indices = range(n_sp_total - n_sp_to_plot, n_sp_total)
    fig, axes = plt.subplots(1, len(sp_indices), figsize=(len(sp_indices) * 5, 5))
    if len(sp_indices) == 1:
        axes = [axes]
    for i, nsp in enumerate(sp_indices):
        for fold in range(10):
            axes[i].scatter(
                x[nsp, fold, :] if x.ndim > 1 else x,
                y[nsp, fold, :] if y.ndim > 1 else y,
                alpha=0.1, label=f"Fold {fold}", c=f"C{fold}",
            )
        axes[i].set_title(f"n_sp {nsp}")
        axes[i].set_xlabel(x_name)
        axes[i].set_ylabel(y_name)
        if x_name in default_param_ranges.keys():
            axes[i].set_xlim([default_param_ranges[x_name]["min"], default_param_ranges[x_name]["max"]])
        if y_name in default_param_ranges.keys():
            axes[i].set_ylim([default_param_ranges[y_name]["min"], default_param_ranges[y_name]["max"]])
        if parity:
            all_min = min(np.nanmin(x if x.ndim > 1 else x), np.nanmin(y if y.ndim > 1 else y))
            all_max = max(np.nanmax(x if x.ndim > 1 else x), np.nanmax(y if y.ndim > 1 else y))
            axes[i].plot([all_min, all_max], [all_min, all_max], "k--", lw=2, alpha=0.5)
            axes[i].set_xlim([all_min, all_max])
            axes[i].set_ylim([all_min, all_max])
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_full_analysis(md, mt, sat, targets):
    prefix = f"full_analysis_{md}_{mt}_{sat}_{targets}"
    valid_indices, CUE = get_output("CUE", md, mt, sat, targets, "param")
    valid_indices, Qmax = get_output("Cg0m", md, mt, sat, targets, "param")
    valid_indices, beta = get_output("beta", md, mt, sat, targets, "param")
    valid_indices, tmb = get_output("tmb", md, mt, sat, targets, "param")
    valid_indices, kb = get_output("kb", md, mt, sat, targets, "param")
    valid_indices, Vmax_m = get_output("Vmax_m", md, mt, sat, targets, "param")
    valid_indices, Vmax_p = get_output("Vmax_p", md, mt, sat, targets, "param")
    valid_indices, Km_m = get_output("Km_m", md, mt, sat, targets, "param")
    valid_indices, Km_p = get_output("Km_p", md, mt, sat, targets, "param")

    C = np.exp(get_data("OC_avg_09_15_18"))
    N = np.exp(get_data("N_avg_09_15_18"))
    pH = get_data("pH_H2O_avg_09_15_18")
    CN = (C / N).reset_index(drop=True)
    CN = CN.iloc[valid_indices]
    C = C.iloc[valid_indices]
    pH = pH.iloc[valid_indices]

    Clay = get_data("Clay")
    Silt = get_data("Silt")
    ClaySilt = (Clay + Silt).reset_index(drop=True)
    ClaySilt = ClaySilt.iloc[valid_indices]

    plot_all_nsp(CUE, beta, "CUE", "beta", out(f"{prefix}_CUE_vs_beta.png"))
    plot_all_nsp(CUE, kb, "CUE", "kb", out(f"{prefix}_CUE_vs_kb.png"))
    plot_all_nsp(CUE, tmb, "CUE", "kb", out(f"{prefix}_CUE_vs_tmb.png"))
    plot_all_nsp(Vmax_m, Vmax_p, "Vmax_m", "Vmax_p", out(f"{prefix}_Vmax_m_vs_Vmax_p.png"))
    plot_all_nsp(Km_m, Km_p, "Km_m", "Km_p", out(f"{prefix}_Km_m_vs_Km_p.png"))

    plot_all_nsp(CN, CUE, "CN", "CUE", out(f"{prefix}_CN_vs_CUE.png"))
    plot_all_nsp(C, CUE, "C", "CUE", out(f"{prefix}_C_vs_CUE.png"))
    plot_all_nsp(pH, CUE, "pH", "CUE", out(f"{prefix}_pH_vs_CUE.png"))
    plot_all_nsp(ClaySilt, Qmax, "ClaySilt", "Qmax", out(f"{prefix}_ClaySilt_vs_Qmax.png"))

    if "SOC" in targets:
        _, SOC_tar = get_output("SOC", md, mt, sat, targets, "target")
        _, SOC_pred = get_output("SOC", md, mt, sat, targets, "prediction")
        plot_all_nsp(SOC_tar, SOC_pred, "SOC target", "SOC prediction", out(f"{prefix}_parity_SOC.png"), parity=True)
    if "MIC" in targets:
        _, MIC_tar = get_output("MIC", md, mt, sat, targets, "target")
        _, MIC_pred = get_output("MIC", md, mt, sat, targets, "prediction")
        plot_all_nsp(MIC_tar, MIC_pred, "MIC target", "MIC prediction", out(f"{prefix}_parity_MIC.png"), parity=True)
    if "MAOC" in targets:
        _, MAOC_tar = get_output("MAOC", md, mt, sat, targets, "target")
        _, MAOC_pred = get_output("MAOC", md, mt, sat, targets, "prediction")
        plot_all_nsp(MAOC_tar, MAOC_pred, "MAOC target", "MAOC prediction", out(f"{prefix}_parity_MAOC.png"), parity=True)


def plot_last_nsp_comparison(configs, path, targets_to_plot=("SOC", "MIC", "MAOC")):
    """Parity (target vs prediction) plots for ONLY the last n_sp of each config.

    Rows = configs, columns = targets. configs: list of (md, mt, sat, targets).
    """
    n_rows, n_cols = len(configs), len(targets_to_plot)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)

    xy_dict = {var: {"x": [], "y": []} for var in targets_to_plot}
    for row, (md, mt, sat, targets) in enumerate(configs):
        for col, var in enumerate(targets_to_plot):
            _, tar = get_output(var, md, mt, sat, targets, "target")
            _, pred = get_output(var, md, mt, sat, targets, "prediction")
            x = tar[-1]   # last n_sp -> (10 folds, n_samples)
            y = pred[-1]
            xy_dict[var]["x"].append(x)
            xy_dict[var]["y"].append(y)

    ax_limits = {}
    for var in targets_to_plot:
        if xy_dict[var]["x"]:
            all_x = np.concatenate([xx.flatten() for xx in xy_dict[var]["x"]])
            all_y = np.concatenate([yy.flatten() for yy in xy_dict[var]["y"]])
            all_vals = np.concatenate([all_x, all_y])
            ax_limits[var] = (np.nanmin(all_vals), np.nanmax(all_vals))
        else:
            ax_limits[var] = (0, 1)

    for row, (md, mt, sat, targets) in enumerate(configs):
        for col, var in enumerate(targets_to_plot):
            ax = axes[row][col]
            if var not in targets:
                ax.set_visible(False)
                continue
            _, tar = get_output(var, md, mt, sat, targets, "target")
            _, pred = get_output(var, md, mt, sat, targets, "prediction")
            x = tar[-1]
            y = pred[-1]
            for fold in range(10):
                ax.scatter(x[fold, :], y[fold, :], alpha=0.1, c=f"C{fold}", label=f"Fold {fold}", s=2)
            col_min, col_max = ax_limits[var]
            ax.plot([col_min, col_max], [col_min, col_max], "k--", lw=2, alpha=0.5)
            ax.set_xlim([col_min, col_max])
            ax.set_ylim([col_min, col_max])
            ax.set_xlabel(f"{var} target")
            ax.set_ylabel(f"{var} prediction")

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_Vmax_m_Vmax_p_combined(configs, path, x_name="Vmax_m", y_name="Vmax_p", parity=False, n_sp_to_plot=None):
    """Same as calling plot_all_nsp once per config, but all in ONE figure (one row each)."""
    xs, ys, row_labels = [], [], []
    for md, mt, sat, targets in configs:
        _, x = get_output(x_name, md, mt, sat, targets, "param")
        _, y = get_output(y_name, md, mt, sat, targets, "param")
        xs.append(x)
        ys.append(y)
        row_labels.append(f"{md}/{mt}/{sat}/{targets}")

    def sp_idx(y):
        n = y.shape[0]
        if n_sp_to_plot is None or n_sp_to_plot > n:
            return list(range(n))
        return list(range(n - n_sp_to_plot, n))

    n_rows = len(configs)
    n_cols = max(len(sp_idx(y)) for y in ys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)

    for r, (x, y) in enumerate(zip(xs, ys)):
        sp_indices = sp_idx(y)
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(sp_indices):
                ax.set_visible(False)
                continue
            nsp = sp_indices[c]
            for fold in range(10):
                ax.scatter(x[nsp, fold, :], y[nsp, fold, :], alpha=0.1, label=f"Fold {fold}", c=f"C{fold}")
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            if x_name in default_param_ranges:
                ax.set_xlim([default_param_ranges[x_name]["min"], default_param_ranges[x_name]["max"]])
            if y_name in default_param_ranges:
                ax.set_ylim([default_param_ranges[y_name]["min"], default_param_ranges[y_name]["max"]])
            if parity:
                all_min = min(np.nanmin(x), np.nanmin(y))
                all_max = max(np.nanmax(x), np.nanmax(y))
                ax.plot([all_min, all_max], [all_min, all_max], "k--", lw=2, alpha=0.5)
                ax.set_xlim([all_min, all_max])
                ax.set_ylim([all_min, all_max])

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _kde_contour_99(ax, x, y, color, grid_range, n_grid=120, max_pts=4000):
    """Draw a single 2D-KDE contour line enclosing ~99% of the (x, y) points."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 20 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    pts = np.vstack([x, y])
    if x.size > max_pts:  # subsample for speed
        sel = np.random.default_rng(0).choice(x.size, max_pts, replace=False)
        pts = pts[:, sel]
    try:
        kde = gaussian_kde(pts)
    except np.linalg.LinAlgError:
        return None
    dens_at_pts = kde(pts)
    level = np.percentile(dens_at_pts, 1)  # 99% of points lie above this density
    (xlo, xhi), (ylo, yhi) = grid_range
    xg = np.linspace(xlo, xhi, n_grid)
    yg = np.linspace(ylo, yhi, n_grid)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    return ax.contour(XX, YY, ZZ, levels=[level], colors=[color], linewidths=1.5, alpha=0.9)


def plot_param_relations_grid(save_name):
    """One figure, 2 rows x 3 columns (per-fold lines, no scatter, no histogram).

    Row 1:   Vmax_m vs Vmax_p -- one 2D-KDE contour per fold enclosing 99% of points.
    Row 2:   CUE distribution per fold -- KDE curve over the central 99% of values.
    Columns: MM/dd/no with 3 spatial params (SOC-MAOC-MIC), 7 spatial params
             (SOC-MAOC-MIC), and 3 spatial params (SOC only).
    """
    md, mt, sat = "MM", "density_dependent", "no"
    # (targets, n_spatial_params, column title)
    columns = [
        ("SOC-MAOC-MIC", 3, "MM/dd/no, SOC-MAOC-MIC\n3 spatial params"),
        ("SOC-MAOC-MIC", 7, "MM/dd/no, SOC-MAOC-MIC\n7 spatial params"),
        ("SOC", 3, "MM/dd/no, SOC\n3 spatial params"),
    ]

    fig, axes = plt.subplots(2, len(columns),
                             figsize=(len(columns) * 4, 8), squeeze=False)
    legend_handles = {}
    for c, (targets, nsp, title) in enumerate(columns):
        # --- row 1: Vmax_m vs Vmax_p as per-fold 99% KDE contours ---
        ax = axes[0][c]
        _, xvm = get_output("Vmax_m", md, mt, sat, targets, "param")
        _, xvp = get_output("Vmax_p", md, mt, sat, targets, "param")
        if xvm is not None and xvp is not None and nsp < xvm.shape[0] and nsp < xvp.shape[0]:
            xr = (default_param_ranges["Vmax_m"]["min"], default_param_ranges["Vmax_m"]["max"])
            yr = (default_param_ranges["Vmax_p"]["min"], default_param_ranges["Vmax_p"]["max"])
            for fold in range(10):
                _kde_contour_99(ax, xvm[nsp, fold, :], xvp[nsp, fold, :],
                                f"C{fold}", (xr, yr))
                if fold not in legend_handles:
                    legend_handles[fold] = plt.Line2D([], [], color=f"C{fold}", label=f"Fold {fold}")
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            ax.set_xlabel("Vmax_m")
            ax.set_ylabel("Vmax_p")
        else:
            ax.set_visible(False)
        ax.set_title(title)

        # --- row 2: CUE distribution as per-fold KDE curves (central 99%) ---
        ax = axes[1][c]
        _, cue = get_output("CUE", md, mt, sat, targets, "param")
        if cue is None or nsp >= cue.shape[0]:
            ax.set_visible(False)
            continue
        arr = cue[nsp]  # (10 folds, n_samples)
        flat = arr[np.isfinite(arr)].ravel()
        if flat.size == 0:
            ax.set_visible(False)
            continue
        cue_lo = default_param_ranges["CUE"]["min"]
        cue_hi = default_param_ranges["CUE"]["max"]
        lo, hi = np.percentile(flat, [0.5, 99.5])  # fit KDE on central 99% of values
        xs = np.linspace(cue_lo, cue_hi, 300)
        for fold in range(10):
            v = arr[fold]
            v = v[np.isfinite(v)]
            vc = v[(v >= lo) & (v <= hi)]
            if vc.size > 5 and np.ptp(vc) > 0:
                ax.plot(xs, gaussian_kde(vc)(xs), color=f"C{fold}", lw=1.5, alpha=0.9)
        ax.set_xlim(cue_lo, cue_hi)
        ax.set_xlabel("CUE")
        ax.set_ylabel("density")

    if legend_handles:
        ordered = [legend_handles[k] for k in sorted(legend_handles)]
        fig.legend(ordered, [h.get_label() for h in ordered], loc="lower center",
                   ncol=10, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out(save_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_param_covariate_grid(save_name):
    """One figure, 3 rows x 5 cols: parameter vs covariate scatter per fold.

    Version: MM/dd/Langmuir, SOC-MAOC-MIC constrained, 3 spatially-varying params.
    Rows: CUE, Vmax_m, Vmax_p.
    Columns: CN, pH_H2O, MAT, WAI2, SOC.
    Covariate on the x-axis (data range), parameter on the y-axis (limits from config).
    Points of all folds scattered in different colors.
    """
    md, mt, sat, targets, nsp = "MM", "density_dependent", "no", "SOC-MAOC-MIC", 3

    # parameter arrays (n_files, 10 folds, n_samples), aligned to valid_indices
    valid_indices, params = None, {}
    for pname in ("CUE", "Vmax_m", "Vmax_p"):
        idx, arr = get_output(pname, md, mt, sat, targets, "param")
        valid_indices = idx
        params[pname] = arr
    if any(a is None or nsp >= a.shape[0] for a in params.values()):
        print(f"plot_param_covariate_grid: no data for n_sp={nsp}")
        return

    def cov(series):
        return series.reset_index(drop=True).iloc[valid_indices].to_numpy(dtype=float)

    C = np.exp(get_data("OC_avg_09_15_18"))
    N = np.exp(get_data("N_avg_09_15_18"))
    covariates = {
        "CN": ("CN", cov(C / N)),
        "pH_H2O": ("pH_H2O", cov(get_data("pH_H2O_avg_09_15_18"))),
        "MAT": ("MAT [C]", cov(get_data("era5_land_t2m_avg_09_15_18")) - 273.15),
        "WAI2": ("WAI2", cov(get_data("WAI2_avg_09_15_18"))),
        "SOC": ("SOC", cov(C)),
    }

    # each row is a parameter, each column a covariate
    cov_keys = ["CN", "pH_H2O", "MAT", "WAI2", "SOC"]
    panels = [[(pname, ck) for ck in cov_keys]
              for pname in ("CUE", "Vmax_m", "Vmax_p")]

    fig, axes = plt.subplots(len(panels), len(panels[0]),
                             figsize=(len(panels[0]) * 4, len(panels) * 4), squeeze=False)
    for r, row in enumerate(panels):
        for c, (pname, cov_key) in enumerate(row):
            ax = axes[r][c]
            cname, cval = covariates[cov_key]
            parr = params[pname]
            prange = (default_param_ranges[pname]["min"], default_param_ranges[pname]["max"])
            finite = cval[np.isfinite(cval)]
            xr = tuple(np.percentile(finite, [0.5, 99.5])) if finite.size else (0.0, 1.0)
            for fold in range(10):
                ax.scatter(cval, parr[nsp, fold, :], alpha=0.1, c=f"C{fold}", s=6)
            ax.set_xlim(*xr)
            ax.set_ylim(*prange)
            ax.set_xlabel(cname)
            ax.set_ylabel(pname)

    plt.tight_layout()
    fig.savefig(out(save_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_panels(panels, nrows, ncols, path, parity=False):
    """panels: list of (x, y, x_name, y_name), each drawn for the LAST n_sp only."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (x, y, x_name, y_name) in zip(axes_flat, panels):
        nsp = y.shape[0] - 1  # last n_sp
        for fold in range(10):
            xf = x[nsp, fold, :] if getattr(x, "ndim", 1) > 1 else x
            yf = y[nsp, fold, :] if getattr(y, "ndim", 1) > 1 else y
            ax.scatter(xf, yf, alpha=0.1, label=f"Fold {fold}", c=f"C{fold}")
        ax.set_title(f"{x_name} vs {y_name} (n_sp {nsp})")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        if x_name in default_param_ranges:
            ax.set_xlim([default_param_ranges[x_name]["min"], default_param_ranges[x_name]["max"]])
        if y_name in default_param_ranges:
            ax.set_ylim([default_param_ranges[y_name]["min"], default_param_ranges[y_name]["max"]])
        if parity:
            all_min = min(np.nanmin(x), np.nanmin(y))
            all_max = max(np.nanmax(x), np.nanmax(y))
            ax.plot([all_min, all_max], [all_min, all_max], "k--", lw=2, alpha=0.5)
            ax.set_xlim([all_min, all_max])
            ax.set_ylim([all_min, all_max])

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_CN_ClaySilt_panels(save_name):
    md, mt, sat, targets = "MM", "density_dependent", "Langmuir", "SOC-MAOC-MIC"

    valid_indices, CUE = get_output("CUE", md, mt, sat, targets, "param")
    valid_indices, Qmax = get_output("Cg0m", md, mt, sat, targets, "param")
    valid_indices, Vmax_m = get_output("Vmax_m", md, mt, sat, targets, "param")
    valid_indices, Vmax_p = get_output("Vmax_p", md, mt, sat, targets, "param")
    valid_indices, Km_m = get_output("Km_m", md, mt, sat, targets, "param")
    valid_indices, Km_p = get_output("Km_p", md, mt, sat, targets, "param")

    C = np.exp(get_data("OC_avg_09_15_18"))
    N = np.exp(get_data("N_avg_09_15_18"))
    CN = (C / N).reset_index(drop=True)
    CN = CN.iloc[valid_indices]

    Clay = get_data("Clay")
    Silt = get_data("Silt")
    ClaySilt = (Clay + Silt).reset_index(drop=True)
    ClaySilt = ClaySilt.iloc[valid_indices]

    plot_panels([
        (Vmax_m, Vmax_p, "Vmax_m", "Vmax_p"),
        (Km_m, Km_p, "Km_m", "Km_p"),
        (CN, CUE, "CN", "CUE"),
        (ClaySilt, Qmax, "ClaySilt", "Qmax"),
    ], nrows=2, ncols=2, path=out(save_name))


# ---------------------------------------------------------------------------
# LaTeX table: greedily selected spatial parameters + test accuracy (color coded)
# ---------------------------------------------------------------------------

# md / mt / sat abbreviations (same scheme as 1_preprocess.ipynb)
_MD_S = {"linear": "lin", "MM": "MM", "RMM": "RMM"}
_MT_S = {"linear": "lin", "density_dependent": "dd"}
_SAT_S = {"no": "no", "Langmuir": "Lang"}
_TABLE_CMAP = matplotlib.colormaps["RdYlGn"]  # low R2 -> red, high R2 -> green


def selection_sequence(temp, fold, md, mt, sat, targets, train_val_test):
    """For one fold, return the greedy selection sequence ordered by ascending n_spatial.

    Returns a list of (spatial_set, mean_test_R2) with one entry per saved step, where
    the first entry (n_spatial=0) is the all-global baseline. As n_spatial grows, each
    step adds one spatially-varying parameter (the parameter "selected" at that step).
    """
    files = [
        fname for fname in os.listdir(folder)
        if f"hybrid_{Tsense}temp{temp}_fold{fold}_md{md}_mt{mt}_sat{sat}_targets{targets}_" in fname
    ]
    files = sorted(files, key=n_spatial_params)
    tgt_list = targets.split("-")
    seq = []
    for fname in files:
        token = fname.rsplit("_spatial", 1)[-1]
        if token.endswith(".pkl"):
            token = token[:-len(".pkl")]
        spatial = set() if token == "none" else set(token.split("-"))
        with open(os.path.join(folder, fname), "rb") as f:
            results = pickle.load(f)
        test_idx = results["split"] == train_val_test
        r2s = [
            r2(results[f"target_{t}"][test_idx].values, results[f"pred_{t}"][test_idx].values)
            for t in tgt_list
        ]
        seq.append((spatial, float(np.mean(r2s))))
    return seq


def _table_cell(text, r2_value, vmin, vmax):
    """Colored LaTeX cell: parameter name only, background shaded by the test R2."""
    if r2_value is None or np.isnan(r2_value):
        return "\\cellcolor[HTML]{000000}{\\color{white}--}"
    sc = min(max((r2_value - vmin) / (vmax - vmin), 0.0), 1.0) if vmax > vmin else 1.0
    r, g, b, _ = _TABLE_CMAP(sc)
    hexc = f"{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
    tcol = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"
    return f"\\cellcolor[HTML]{{{hexc}}}{{\\color{{{tcol}}}{text}}}"


def _esc(s):
    return str(s).replace("_", r"\_")


def make_selection_table(targets, temp="static", train_val_test="test", r2_range=(0.75, 1.0)):
    """Build a LaTeX table of greedily selected spatial params + test accuracy.

    Rows are model versions (md/mt/sat), columns are the parameters in the order they
    were greedily made spatial (position 1 = first selected). Each cell shows the
    selected parameter's name and the mean test R2 at that step, shaded by the R2.
    The leading 'all-global' column reports the baseline (0 spatial parameters).

    The selection order genuinely differs across folds, so each column reports the
    majority-vote parameter and the mean R2 across folds at that selection step;
    shading is clipped to r2_range so that catastrophic baselines don't wash out the
    color scale (R2 below the range -> red, above -> green).
    """
    versions = list(product(["linear", "MM", "RMM"],
                            ["linear", "density_dependent"],
                            ["no", "Langmuir"]))

    rows = []          # (label, baseline_r2, [(param_name, mean_r2), ...])
    max_cols = 0
    for md, mt, sat in versions:
        # gather per-fold sequences
        per_fold = []
        for fold in range(10):
            seq = selection_sequence(temp, fold, md, mt, sat, targets, train_val_test)
            if seq:
                per_fold.append(seq)
        label = f"{_MD_S[md]}/{_MT_S[mt]}/{_SAT_S[sat]}"
        if not per_fold:
            rows.append((label, None, []))
            continue

        # baseline (n_spatial = 0) mean R2 across folds
        baseline = np.nanmean([s[0][1] for s in per_fold])

        # per selection step: collect the newly added param + R2 across folds
        max_steps = max(len(s) for s in per_fold) - 1
        cols = []
        for pos in range(1, max_steps + 1):
            names, r2s = [], []
            for s in per_fold:
                if pos < len(s):
                    added = s[pos][0] - s[pos - 1][0]
                    names.append("+".join(sorted(added)) if added else "?")
                    r2s.append(s[pos][1])
            if not r2s:
                continue
            consensus = Counter(names).most_common(1)[0][0]
            mean_r2 = float(np.nanmean(r2s))
            cols.append((consensus, mean_r2))
        rows.append((label, float(baseline), cols))
        max_cols = max(max_cols, len(cols))

    vmin, vmax = r2_range

    if targets == "SOC-MAOC-MIC":
        caption = (
            "Actually spatially varying parameters according to $n_{sp}$ per version "
            "where SOC, MAOC and MIC were constrained. The color coding implies the "
            "representativeness ($\\overline{R^2}$), where the Red-Yellow-Green color bar "
            "was clipped to 0.75-1.0. Since hybrid versions were trained per fold, the "
            "original 10 selections were condensed using the respectively most found "
            "parameter at a given position, which explains finding some parameters twice "
            "in a row. Same tables, but not with all targets constrained are in supplement."
        )
    else:
        caption = (
            "Equivalent table to Tab. \\ref{tab:sel_SOC-MAOC-MIC}, "
            f"but with only {targets} constrained."
        )

    col_fmt = "l|" + "c" * max_cols
    header = ("Version & "
              + " & ".join(f"$n_{{sp}} = {i}$" for i in range(1, max_cols + 1)) + " \\\\")
    lines = [
        "\\begin{table}[ht]", "\\centering",
        "\\captionsetup{width=0.75\\textwidth}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:sel_{targets}}}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.2}",
        f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule", header, "\\midrule",
    ]
    for label, baseline, cols in rows:
        cells = [_esc(label)]
        for pos in range(max_cols):
            if pos < len(cols):
                name, val = cols[pos]
                cells.append(_table_cell(_esc(name), val, vmin, vmax))
            else:
                cells.append("")
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def print_selection_tables():
    header = ("% requires \\usepackage[table]{xcolor} (\\cellcolor), "
              "\\usepackage{amsmath} and \\usepackage{caption} (\\captionsetup)\n")
    for targets in ["SOC", "SOC-MIC", "SOC-MAOC", "SOC-MAOC-MIC"]:
        table = make_selection_table(targets)
        path = out(f"selection_table_{targets}.tex")
        with open(path, "w") as f:
            f.write(header + table + "\n")
        print(f"Saved {path}")


def main():
    # # --- R2 vs n_spatial line plots per model version (In[3]-In[7]) ---
    # full_plot("static", "linear", "linear", "no", "test", "full_plot_static_linear_linear_no.png")

    # full_plot("static", "MM", "linear", "no", "test", "full_plot_static_MM_linear_no.png")
    # full_plot("static", "RMM", "linear", "no", "test", "full_plot_static_RMM_linear_no.png")

    # full_plot("static", "linear", "density_dependent", "no", "test", "full_plot_static_linear_density_dependent_no.png")
    # full_plot("static", "MM", "density_dependent", "no", "test", "full_plot_static_MM_density_dependent_no.png")
    # full_plot("static", "RMM", "density_dependent", "no", "test", "full_plot_static_RMM_density_dependent_no.png")

    # full_plot("static", "linear", "linear", "Langmuir", "test", "full_plot_static_linear_linear_Langmuir.png")
    # full_plot("static", "MM", "linear", "Langmuir", "test", "full_plot_static_MM_linear_Langmuir.png")
    # full_plot("static", "RMM", "linear", "Langmuir", "test", "full_plot_static_RMM_linear_Langmuir.png")

    # full_plot("static", "linear", "density_dependent", "Langmuir", "test", "full_plot_static_linear_density_dependent_Langmuir.png")
    # full_plot("static", "MM", "density_dependent", "Langmuir", "test", "full_plot_static_MM_density_dependent_Langmuir.png")
    # full_plot("static", "RMM", "density_dependent", "Langmuir", "test", "full_plot_static_RMM_density_dependent_Langmuir.png")

    # --- LaTeX tables: greedily selected spatial params + test accuracy (color coded) ---
    print_selection_tables()

    # --- combined param-vs-param scatter grid (2 rows x 3 columns) ---
    plot_param_relations_grid("param_relations_grid.png")

    # --- parameter vs covariate grid (2 rows x 4 columns) ---
    plot_param_covariate_grid("param_covariate_grid.png")

    # # --- combined grids (In[8], In[9]) ---
    # plot_all3_constraints_grid("grid_SOC-MAOC-MIC_static.png")
    plot_mean_target_curves("mean_target_curves.png")

    # # --- equifinality / parameter-relationship scatter plots (In[10]-In[15]) ---
    # plot_full_analysis("MM", "density_dependent", "no", "SOC-MAOC-MIC")
    # plot_full_analysis("RMM", "density_dependent", "no", "SOC-MAOC-MIC")
    # plot_full_analysis("MM", "density_dependent", "Langmuir", "SOC-MAOC-MIC")
    # plot_full_analysis("MM", "density_dependent", "Langmuir", "SOC")
    # plot_full_analysis("MM", "linear", "Langmuir", "SOC-MAOC-MIC")
    # plot_full_analysis("RMM", "density_dependent", "Langmuir", "SOC-MAOC-MIC")

    # # --- parity comparison of last n_sp across configs (In[20]) ---
    # plot_last_nsp_comparison([
    #     ("linear", "linear", "no", "SOC-MAOC-MIC"),
    #     ("MM", "linear", "Langmuir", "SOC-MAOC-MIC"),
    #     ("MM", "density_dependent", "no", "SOC"),
    #     ("MM", "density_dependent", "no", "SOC-MAOC-MIC"),
    # ], path=out("comparison_last_nsp.png"))

    # # --- Vmax_m vs Vmax_p combined (In[17]) ---
    # plot_Vmax_m_Vmax_p_combined([
    #     ("MM", "density_dependent", "Langmuir", "SOC-MAOC-MIC"),
    #     ("MM", "density_dependent", "Langmuir", "SOC"),
    # ], path=out("Vmax_m_Vmax_p_combined.png"), n_sp_to_plot=2)

    # # --- CN / ClaySilt panels (In[18]) ---
    # plot_CN_ClaySilt_panels("panels_MM_density_dependent_Langmuir.png")

    print(f"All figures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()

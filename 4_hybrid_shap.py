"""SHAP analysis of the hybrid (NN + mechanistic) SOC model.

This script trains the hybrid model exactly as in `2_hybrid.py` (so it can
re-evaluate the network on perturbed inputs, which the saved `.pkl` outputs do
not allow) for the two mechanistic combinations:

    * MM-density_dependent-no   (md=MM,  mt=density_dependent, sat=no)
    * RMM-density_dependent-no  (md=RMM, mt=density_dependent, sat=no)

and produces SHAP beeswarm, effect (dependence) and interaction plots for three
relationships of the hybrid model:

    1. targets           ~ covariates        (full model: covariates -> NN -> mechanistic -> targets)
    2. latent parameters ~ covariates        (network only:           covariates -> NN -> params)
    3. targets           ~ latent parameters  (mechanistic only:       params -> targets)

"targets"           = SOC, MAOC, MIC (steady-state, static temperature).
"covariates"        = the NN input predictors (z-scored) plus the carbon-input
                      forcing I (=NPP), which directly sets the mechanistic input rate.
"latent parameters" = the spatially-varying parameters predicted by the NN
                      (plus I for the targets~latent-parameter view).

Because the model is a JAX function (not a tree), exact SHAP interaction values
are not available (those require `TreeExplainer`). We therefore use a
model-agnostic `shap.KernelExplainer` for the SHAP values, and approximate the
"interaction" plots with SHAP dependence plots (SHAP value of feature i vs the
value of feature i, coloured by feature j) — the same visual idea as the
tree-based interaction grid in `1_preprocess.ipynb`.

Nothing in the existing scripts is modified. `2_hybrid.py` is imported only to
reuse its `pools_to_loss_targets` / `TARGET_LABELS` (and its monkey-patch of
`utils.pools_to_loss_targets`) so that the training matches it exactly.

Run with the project env, e.g.:
    micromamba run -n hybrid-lucas python 4_hybrid_shap.py
"""

import os
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax as dfx
import shap

from models import craig_BA_adapt, analytical_steady_state
from config import default_param_ranges, TARGET_CONFIG
import utils
from utils import (
    vector_field,
    simulate_final_state,
    init_mlp,
    build_param_matrix,
    eval_loss,
    init_adam,
    eval_r2,
    train_step,
    mlp_forward,
    constrain_to_range,
)

# Import 2_hybrid (digit-leading module name) to reuse its pools_to_loss_targets
# + TARGET_LABELS and to apply the same monkey-patch on utils that it does.
_spec = importlib.util.spec_from_file_location("hybrid_mod", "2_hybrid.py")
hybrid_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hybrid_mod)
pools_to_loss_targets = hybrid_mod.pools_to_loss_targets
TARGET_LABELS = hybrid_mod.TARGET_LABELS

# Training hyper-parameters mirrored from 2_hybrid.py
DT0 = hybrid_mod.dt0
DEPTH = hybrid_mod.depth
WIDTH = hybrid_mod.width
BATCH_SIZE = hybrid_mod.batch_size

# Fixed experiment choices for this SHAP study
TEMP = "static"            # steady-state
TARGETS = "SOC,MAOC,MIC"   # all three targets
FOLD = "0"                 # validation fold used for explanations

# SHAP sampling sizes (kept moderate; the steady-state is closed-form & fast)
N_EXPLAIN = 150            # number of validation samples explained
N_BACKGROUND = 40          # background summary size (kmeans)
N_SAMPLES = 1000           # KernelExplainer coalition samples
N_INTERACTION_FEATS = 6    # top-k features shown in the interaction grids
RNG_SEED = 0

SHAP_ROOT = Path("figures/hybrid_shap")


# --------------------------------------------------------------------------- #
# Training (single spatial configuration: max spatial = all non-zero-sensitivity
# parameters are spatial, i.e. the i=0 case of the 2_hybrid global-iteration loop)
# --------------------------------------------------------------------------- #
def build_predictors(targets):
    """Replicate the predictor construction from 2_hybrid.py."""
    predictors = []
    for tar in targets.split(","):
        for pred in TARGET_CONFIG[tar]["selected_predictors"]:
            if any(year in pred for year in ["2009", "2015", "2018"]):
                if (pred.endswith("2009") or pred.endswith("2015") or pred.endswith("2018")) and pred != "Ox_Al_2018":
                    predictors.append(pred.replace("_2009", "_avg_09_15_18").replace("_2015", "_avg_09_15_18").replace("_2018", "_avg_09_15_18"))
                elif pred == "Ox_Al_2018":
                    predictors.append(pred)
                elif pred.startswith("lc1_2_"):
                    predictors.append(pred[-1] + "_avg_09_15_18")
                elif "-5_mean" in pred:
                    predictors.append(pred[:-12] + "_avg_09_15_18")
                else:
                    raise ValueError(f"Not yet know what to do with: {pred}")
            else:
                predictors.append(pred)
    return list(dict.fromkeys(predictors))


def train_hybrid(md, mt, sat):
    """Train one hybrid model and return everything needed for SHAP."""
    use_dynamic = TEMP == "dynamic"
    lr = 5e-4
    n_steps = 3000
    early_stop_patience = 500

    # --- global / spatial split (i=0: only zero-sensitivity params are global)
    sensitivities = pd.read_csv("figures/sensitivities.csv")
    model_sens = sensitivities[(sensitivities["md"] == md) & (sensitivities["mt"] == mt) & (sensitivities["sat"] == sat) & (sensitivities["temp"] == ("dynamic" if use_dynamic else "steady"))].iloc[0]
    param_sens = model_sens.drop(labels=["md", "mt", "sat", "temp", "y0_Cp", "y0_Cb", "y0_Cm"])
    global_names = [name for name, val in param_sens.items() if val == 0.0]

    param_names = list(default_param_ranges.keys())
    param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
    param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])
    global_mask = jnp.array([name in global_names for name in param_names])
    spatial_names = [name for name in param_names if name not in global_names and name != "I"]
    print(f"[{md}] global parameters: {global_names}")
    print(f"[{md}] spatial parameters: {spatial_names}")

    # --- mechanistic models
    batched_steady = jax.vmap(partial(analytical_steady_state, microbial_decomposition=md, microbial_turnover=mt, saturation=sat))
    t0, t1 = 0.0, 9.0
    solver = dfx.Euler()
    model_fn = partial(craig_BA_adapt, microbial_decomposition=md, microbial_turnover=mt, saturation=sat)
    term = dfx.ODETerm(partial(vector_field, model_fn))
    batched_sim = jax.vmap(lambda p, y0: simulate_final_state(p, y0, t0, t1, DT0, term, solver))

    # --- data preparation (mirrors 2_hybrid.py)
    df = pd.read_pickle("1_preprocessed.pkl")
    predictors = build_predictors(TARGETS)
    helper_df = df.copy()
    input_col = "input_avg_09_15_18"
    target_labels = TARGET_LABELS[TARGETS]
    target_columns = {"SOC": "SOC"}
    if use_dynamic:
        target_columns.update({"MIC": "MIC", "MAOC": "MAOC"})
    else:
        for label in ("MIC", "MAOC"):
            if label in target_labels:
                col = f"{label}i"
                if col not in helper_df.columns:
                    raise ValueError(f"Missing static target column {col}.")
                target_columns[label] = col
    target_source_cols = [target_columns[label] for label in target_labels]
    required_cols = ["SOC", "POC", "MIC", "MAOC"] + target_source_cols + predictors + [input_col, "era5_land_t2m_avg_09_15_18", "split"]
    helper_df = helper_df[list(dict.fromkeys(required_cols))]

    npp_mask = (helper_df[input_col].notna() & np.isfinite(helper_df[input_col]) & (helper_df[input_col] > 0)).to_numpy()
    original_idx = np.where(npp_mask)[0]
    helper_df = helper_df.loc[npp_mask].reset_index(drop=True)
    split_col = helper_df["split"].astype(str).to_numpy()

    target_values = np.column_stack([helper_df[target_columns[label]].to_numpy() for label in target_labels])
    label_mask = np.all(np.isfinite(target_values), axis=1)
    train_idx = np.where(label_mask & (split_col != "test") & (split_col != FOLD))[0]
    val_idx = np.where(label_mask & (split_col == FOLD))[0]
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("No train/validation rows with finite targets")
    helper_df = helper_df.fillna(helper_df.iloc[jax.device_get(train_idx)].median(numeric_only=True))
    targets = jnp.asarray(target_values)

    x_features = jnp.asarray(helper_df[predictors].to_numpy())
    npp_I_all = jnp.asarray(helper_df[input_col].to_numpy())
    x_train = x_features[train_idx]
    y_train = targets[train_idx]
    npp_I_train = npp_I_all[train_idx]
    x_val = x_features[val_idx]
    y_val = targets[val_idx]
    npp_I_val = npp_I_all[val_idx]

    x_mean = jnp.mean(x_train, axis=0)
    x_std = jnp.std(x_train, axis=0) + 1e-8
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_features = (x_features - x_mean) / x_std
    target_mean = jnp.mean(y_train, axis=0)
    target_std = jnp.std(y_train, axis=0) + 1e-8
    y0_true = jnp.asarray(helper_df[["POC", "MIC", "MAOC"]].to_numpy())
    y0_train, y0_val = y0_true[train_idx], y0_true[val_idx]

    # --- initialise & train
    net_params = init_mlp(jax.random.PRNGKey(0), [x_features.shape[1]] + [WIDTH] * DEPTH + [param_mins.size])
    global_raw = jnp.zeros((param_mins.size,))
    params = {"net": net_params, "global": global_raw}
    n_targ = int(targets.shape[1])
    target_mask = jnp.ones((n_targ,))
    loss_ema, ema_beta, weights = jnp.ones((n_targ,)) * target_mask, 0.9, jnp.ones((n_targ,)) * target_mask
    opt_state, best_params, best_test = init_adam(params), params, float("inf")
    best_step = 0

    common = dict(param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, targets_arg=TARGETS)
    for step in range(1, n_steps + 1):
        k = jax.random.PRNGKey(step)
        batch_idx = jax.random.choice(k, train_idx.size, shape=(min(BATCH_SIZE, train_idx.size),), replace=False)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]
        y0_batch = y0_train[batch_idx] if use_dynamic else jnp.zeros((batch_idx.size, 3))
        npp_I_batch = npp_I_train[batch_idx]
        warmup_scale = jnp.minimum(1.0, step / 200.0)
        lr_t = lr * warmup_scale * 0.5 * (1.0 + jnp.cos(jnp.pi * step / n_steps))
        params, opt_state, loss, per_component = train_step(params, opt_state, x_batch, npp_I_batch, y0_batch, y_batch, lr_t, step, weights, target_mean=target_mean, target_std=target_std, **common)
        loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * per_component
        weights = (1.0 / (loss_ema + 1e-8)) * target_mask
        if step % 50 == 0:
            val_loss = eval_loss(params, x_val, npp_I_val, y0_val, y_val, weights, target_mean=target_mean, target_std=target_std, **common)[0]
            if val_loss < best_test:
                best_test = float(val_loss)
                best_params = params
                best_step = step
            if step - best_step >= early_stop_patience:
                print(f"[{md}] early stopping at step {step}")
                break

    val_r2 = jax.device_get(eval_r2(best_params, x_val, npp_I_val, y0_val, y_val, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, targets_arg=TARGETS))
    print(f"[{md}] validation R2: " + " ".join(f"{lb} {val_r2[i]:.3f}" for i, lb in enumerate(target_labels)))

    # predicted parameter matrix on the validation samples (for targets~params view)
    p_val = build_param_matrix(best_params["net"], best_params["global"], x_val, npp_I_val, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask)

    return dict(
        md=md, mt=mt, sat=sat,
        net_params=best_params["net"],
        global_raw=best_params["global"],
        global_mask=global_mask,
        param_mins=param_mins,
        param_maxs=param_maxs,
        param_names=param_names,
        spatial_names=spatial_names,
        predictors=predictors,
        target_labels=target_labels,
        batched_steady=batched_steady,
        use_dynamic=use_dynamic,
        x_val=np.asarray(jax.device_get(x_val)),
        npp_I_val=np.asarray(jax.device_get(npp_I_val)),
        p_val=np.asarray(jax.device_get(p_val)),
        val_r2=val_r2,
    )


# --------------------------------------------------------------------------- #
# Model functions exposed to SHAP (numpy in -> numpy out)
# --------------------------------------------------------------------------- #
def make_model_functions(bundle):
    net_params = bundle["net_params"]
    global_raw = bundle["global_raw"]
    global_mask = bundle["global_mask"]
    param_mins = bundle["param_mins"]
    param_maxs = bundle["param_maxs"]
    param_names = bundle["param_names"]
    spatial_names = bundle["spatial_names"]
    batched_steady = bundle["batched_steady"]
    use_dynamic = bundle["use_dynamic"]

    spatial_idx = jnp.array([param_names.index(n) for n in spatial_names])
    global_params_const = constrain_to_range(global_raw, param_mins, param_maxs)
    # template parameter vector (global values; col 0 = I is overwritten per sample)
    template = jnp.where(global_mask, global_params_const, 0.0)
    # varying-parameter columns for the targets~latent-params view: I + spatial params
    var_idx = jnp.array([0] + [param_names.index(n) for n in spatial_names])

    def _params_from_covariates(x_pred_norm, npp):
        raw_local = jax.vmap(lambda x: mlp_forward(net_params, x))(x_pred_norm)
        local = constrain_to_range(raw_local, param_mins, param_maxs)
        glob = constrain_to_range(global_raw, param_mins, param_maxs)
        full = jnp.where(global_mask, glob, local)
        full = full.at[:, 0].set(npp)
        return full

    def _targets_from_params(full):
        if use_dynamic:
            raise NotImplementedError("This SHAP script targets the static/steady model.")
        pred_compare = batched_steady(full)
        derived = pools_to_loss_targets(pred_compare, jnp.zeros_like(pred_compare), False, TARGETS)
        return derived

    @jax.jit
    def _f_targets_from_cov(x_cov):
        x_pred_norm = x_cov[:, :-1]
        npp = x_cov[:, -1]
        full = _params_from_covariates(x_pred_norm, npp)
        return _targets_from_params(full)

    @jax.jit
    def _f_spatialparams_from_cov(x_pred_norm):
        raw_local = jax.vmap(lambda x: mlp_forward(net_params, x))(x_pred_norm)
        local = constrain_to_range(raw_local, param_mins, param_maxs)
        return local[:, spatial_idx]

    @jax.jit
    def _f_targets_from_latent(x_var):
        full = jnp.broadcast_to(template, (x_var.shape[0], template.shape[0]))
        full = full.at[:, var_idx].set(x_var)
        return _targets_from_params(full)

    def f_targets_from_cov(x):
        return np.asarray(jax.device_get(_f_targets_from_cov(jnp.asarray(x, dtype=jnp.float32))))

    def f_spatialparams_from_cov(x):
        return np.asarray(jax.device_get(_f_spatialparams_from_cov(jnp.asarray(x, dtype=jnp.float32))))

    def f_targets_from_latent(x):
        return np.asarray(jax.device_get(_f_targets_from_latent(jnp.asarray(x, dtype=jnp.float32))))

    return dict(
        f_targets_from_cov=f_targets_from_cov,
        f_spatialparams_from_cov=f_spatialparams_from_cov,
        f_targets_from_latent=f_targets_from_latent,
        var_param_names=["I"] + spatial_names,
        var_idx=np.asarray(jax.device_get(var_idx)),
    )


# --------------------------------------------------------------------------- #
# SHAP value helper + plotting
# --------------------------------------------------------------------------- #
def _per_output_list(shap_values, n_outputs):
    """Normalise KernelExplainer output to a list of (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_outputs)
        return [arr[:, :, k] for k in range(arr.shape[-1])]
    return [arr]


def run_shap(f, X, feature_names, output_names, view_name, out_dir):
    """Run KernelExplainer and produce beeswarm / effect / interaction plots."""
    rng = np.random.default_rng(RNG_SEED)
    X = np.asarray(X, dtype=np.float64)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    n = X.shape[0]
    expl_idx = rng.choice(n, size=min(N_EXPLAIN, n), replace=False)
    X_expl = X[expl_idx]
    background = shap.kmeans(X, min(N_BACKGROUND, n))

    print(f"    [{view_name}] explaining {X_expl.shape[0]} samples, {X.shape[1]} features, {len(output_names)} outputs ...")
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_expl, nsamples=N_SAMPLES, silent=True)
    sv_list = _per_output_list(shap_values, len(output_names))

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- beeswarm (one subplot per output) ----
    n_out = len(output_names)
    fig_bee, axes_bee = plt.subplots(1, n_out, figsize=(7 * n_out, 6), squeeze=False)
    for k, out_name in enumerate(output_names):
        plt.sca(axes_bee[0, k])
        shap.summary_plot(sv_list[k], X_expl, feature_names=feature_names, max_display=len(feature_names), show=False, plot_size=None)
        axes_bee[0, k].set_title(f"{out_name}")
    fig_bee.suptitle(f"SHAP beeswarm — {view_name}", fontsize=15)
    fig_bee.tight_layout()
    fig_bee.savefig(out_dir / f"{view_name}_beeswarm.png", dpi=140, bbox_inches="tight")
    plt.close(fig_bee)

    # ---- effect (dependence) + interaction plots per output ----
    for k, out_name in enumerate(output_names):
        sv = sv_list[k]
        nf = len(feature_names)
        order = np.argsort(-np.mean(np.abs(sv), axis=0))  # most important first

        # effect: SHAP value vs feature value, for every feature
        ncol = min(5, nf)
        nrow = int(np.ceil(nf / ncol))
        fig_e, axes_e = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
        for ax in axes_e.flatten():
            ax.axis("off")
        for plot_i, feat_i in enumerate(order):
            ax = axes_e[plot_i // ncol, plot_i % ncol]
            ax.axis("on")
            ax.scatter(X_expl[:, feat_i], sv[:, feat_i], s=12, alpha=0.6, c="#1f77b4")
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_xlabel(feature_names[feat_i])
            ax.set_ylabel(f"SHAP ({out_name})")
            ax.set_title(feature_names[feat_i], fontsize=9)
        fig_e.suptitle(f"SHAP effect — {view_name} : {out_name}", fontsize=14)
        fig_e.tight_layout()
        fig_e.savefig(out_dir / f"{view_name}_effect_{out_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig_e)

        # interactions: dependence of feature i coloured by feature j (top-k features)
        top = order[: min(N_INTERACTION_FEATS, nf)]
        m = len(top)
        fig_i, axes_i = plt.subplots(m, m, figsize=(3.4 * m, 3.0 * m), squeeze=False)
        for a, fi in enumerate(top):
            for b, fj in enumerate(top):
                ax = axes_i[a, b]
                if a == b:
                    ax.axis("off")
                    ax.text(0.5, 0.5, feature_names[fi], ha="center", va="center", fontsize=12, fontweight="bold")
                    continue
                sc = ax.scatter(X_expl[:, fi], sv[:, fi], c=X_expl[:, fj], cmap="viridis", s=12, alpha=0.7)
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                ax.set_xlabel(feature_names[fi], fontsize=8)
                ax.set_ylabel(f"SHAP({feature_names[fi]})", fontsize=8)
                cbar = fig_i.colorbar(sc, ax=ax)
                cbar.set_label(feature_names[fj], fontsize=8)
        fig_i.suptitle(f"SHAP interactions — {view_name} : {out_name}\n(SHAP of row-feature coloured by column-feature)", fontsize=13)
        fig_i.tight_layout()
        fig_i.savefig(out_dir / f"{view_name}_interactions_{out_name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig_i)

    # quick numeric summary for plausibility checking
    print(f"    [{view_name}] mean|SHAP| per output:")
    for k, out_name in enumerate(output_names):
        mean_abs = np.mean(np.abs(sv_list[k]), axis=0)
        ranked = sorted(zip(feature_names, mean_abs), key=lambda t: -t[1])[:5]
        print(f"        {out_name}: " + ", ".join(f"{nm}={v:.3g}" for nm, v in ranked))


# --------------------------------------------------------------------------- #
def analyse_combo(md, mt, sat):
    combo = f"{md}_{mt}_{sat}"
    print(f"\n=== Hybrid SHAP for {combo} ===")
    bundle = train_hybrid(md, mt, sat)
    fns = make_model_functions(bundle)
    out_dir = SHAP_ROOT / combo

    # View 1: targets ~ covariates (predictors + carbon input I)
    X_cov = np.column_stack([bundle["x_val"], bundle["npp_I_val"]])
    cov_names = list(bundle["predictors"]) + ["I_input(NPP)"]
    run_shap(fns["f_targets_from_cov"], X_cov, cov_names, bundle["target_labels"], "targets_vs_covariates", out_dir)

    # View 2: latent parameters ~ covariates (spatially-varying NN params)
    run_shap(fns["f_spatialparams_from_cov"], bundle["x_val"], list(bundle["predictors"]), bundle["spatial_names"], "latentparams_vs_covariates", out_dir)

    # View 3: targets ~ latent parameters (mechanistic model only)
    X_var = bundle["p_val"][:, fns["var_idx"]]
    run_shap(fns["f_targets_from_latent"], X_var, fns["var_param_names"], bundle["target_labels"], "targets_vs_latentparams", out_dir)

    print(f"=== finished {combo}; figures in {out_dir} ===")


def main():
    SHAP_ROOT.mkdir(parents=True, exist_ok=True)
    for md in ("MM", "RMM"):
        analyse_combo(md, "density_dependent", "no")


if __name__ == "__main__":
    main()

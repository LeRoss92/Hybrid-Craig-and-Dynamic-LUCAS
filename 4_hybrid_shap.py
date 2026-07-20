"""SHAP effect grid of the hybrid (NN + mechanistic) SOC model.

This script trains the hybrid model exactly as in `2_hybrid.py` (so it can
re-evaluate the network on perturbed inputs, which the saved `.pkl` outputs do
not allow) for the single Craig version MM / density_dependent / no
(saturation) with all three targets (SOC, MAOC, MIC) applied, and produces a
*single* parent figure: SHAP dependence ("effect") plots of the NN-predicted
latent parameters against the covariates.

Layout (styled after `1_preprocess.ipynb`):

    * rows    = latent parameters  Vmax_m, Vmax_p, CUE
    * columns = the NN input predictors (covariates)
    * each cell scatters the covariate value (original space) against the SHAP
      value of that covariate for the row's latent parameter.

Because the model is a JAX function (not a tree), exact SHAP values are obtained
with a model-agnostic `shap.KernelExplainer`.

Nothing in the existing scripts is modified. `2_hybrid.py` is imported only to
reuse its `TARGET_LABELS` / hyper-parameters (and its monkey-patch of
`utils.pools_to_loss_targets`) so that the training matches it exactly.

Run with the project env, e.g.:
    micromamba run -n hybrid-lucas python 4_hybrid_shap.py
"""

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
from config import default_param_ranges
import utils
from utils import (
    build_hybrid_predictors,
    vector_field,
    simulate_final_state,
    init_mlp,
    eval_loss,
    init_adam,
    eval_r2,
    train_step,
    mlp_forward,
    constrain_to_range,
    _label_to_col,
)

# Import 2_hybrid (digit-leading module name) to reuse its TARGET_LABELS +
# hyper-parameters and to apply the same monkey-patch on utils that it does.
_spec = importlib.util.spec_from_file_location("hybrid_mod", "2_hybrid.py")
hybrid_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hybrid_mod)
TARGET_LABELS = hybrid_mod.TARGET_LABELS

# Training hyper-parameters mirrored from 2_hybrid.py
DT0 = hybrid_mod.dt0
DEPTH = hybrid_mod.depth
WIDTH = hybrid_mod.width
BATCH_SIZE = hybrid_mod.batch_size

# Fixed experiment choices for this SHAP study
TEMP = "static"            # steady-state
TARGETS = "SOC,MAOC,MIC"   # all three targets
FOLDS = list(range(10))    # validation folds explained (one colour each)
VERSION = ("MM", "density_dependent", "no")   # md, mt, sat

# Latent parameters shown as the rows of the parent figure
LATENT_ROWS = ["Vmax_m", "Vmax_p", "CUE"]

# SHAP sampling sizes (kept moderate; the steady-state is closed-form & fast)
N_EXPLAIN = 150            # number of validation samples explained
N_BACKGROUND = 40          # background summary size (kmeans)
N_SAMPLES = 1000           # KernelExplainer coalition samples
RNG_SEED = 0

OUT_DIR = Path("figures/4_hybrid_shap")


# --------------------------------------------------------------------------- #
# Training (single spatial configuration: max spatial = all non-zero-sensitivity
# parameters are spatial, i.e. the i=0 case of the 2_hybrid global-iteration loop)
# --------------------------------------------------------------------------- #
def train_hybrid(md, mt, sat, fold):
    """Train one hybrid model on `fold` and return everything needed for SHAP."""
    use_dynamic = TEMP == "dynamic"
    lr = 5e-4
    n_steps = 3000
    early_stop_patience = 500

    # --- global / spatial split: only the three plotted latent parameters
    #     (LATENT_ROWS) vary spatially; every other parameter is global.
    param_names = list(default_param_ranges.keys())
    param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
    param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])
    spatial_names = list(LATENT_ROWS)
    global_names = [name for name in param_names if name not in spatial_names and name != "I"]
    global_mask = jnp.array([name in global_names for name in param_names])
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
    predictors = build_hybrid_predictors(TARGETS)
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
    helper_df = helper_df.loc[npp_mask].reset_index(drop=True)
    split_col = helper_df["split"].astype(str).to_numpy()

    target_values = np.column_stack([helper_df[target_columns[label]].to_numpy() for label in target_labels])
    label_mask = np.all(np.isfinite(target_values), axis=1)
    fold_str = str(fold)
    train_idx = np.where(label_mask & (split_col != "test") & (split_col != fold_str))[0]
    val_idx = np.where(label_mask & (split_col == fold_str))[0]
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
    print(f"[{md} fold {fold}] validation R2: " + " ".join(f"{lb} {val_r2[i]:.3f}" for i, lb in enumerate(target_labels)))

    return dict(
        net_params=best_params["net"],
        param_mins=param_mins,
        param_maxs=param_maxs,
        param_names=param_names,
        predictors=predictors,
        x_val=np.asarray(jax.device_get(x_val)),
        x_mean=np.asarray(jax.device_get(x_mean)),
        x_std=np.asarray(jax.device_get(x_std)),
    )


# --------------------------------------------------------------------------- #
# SHAP model function: covariates (z-scored predictors) -> latent parameters
# --------------------------------------------------------------------------- #
def make_latent_fn(bundle, row_names):
    net_params = bundle["net_params"]
    param_mins = bundle["param_mins"]
    param_maxs = bundle["param_maxs"]
    param_names = bundle["param_names"]
    row_idx = jnp.array([param_names.index(n) for n in row_names])

    @jax.jit
    def _f(x_pred_norm):
        raw_local = jax.vmap(lambda x: mlp_forward(net_params, x))(x_pred_norm)
        local = constrain_to_range(raw_local, param_mins, param_maxs)
        return local[:, row_idx]

    def f(x):
        return np.asarray(jax.device_get(_f(jnp.asarray(x, dtype=jnp.float32))))

    return f


def _per_output_list(shap_values, n_outputs):
    """Normalise KernelExplainer output to a list of (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_outputs)
        return [arr[:, :, k] for k in range(arr.shape[-1])]
    return [arr]


def compute_shap_values(f, X, n_outputs):
    """Run KernelExplainer; return explained samples + per-output SHAP values."""
    rng = np.random.default_rng(RNG_SEED)
    X = np.asarray(X, dtype=np.float64)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    n = X.shape[0]
    expl_idx = rng.choice(n, size=min(N_EXPLAIN, n), replace=False)
    X_expl = X[expl_idx]
    background = shap.kmeans(X, min(N_BACKGROUND, n))
    print(f"    explaining {X_expl.shape[0]} samples, {X.shape[1]} features, {n_outputs} outputs ...")
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_expl, nsamples=N_SAMPLES, silent=True)
    return X_expl, _per_output_list(shap_values, n_outputs)


# --------------------------------------------------------------------------- #
def main():
    md, mt, sat = VERSION
    combo = f"{md}_{mt}_{sat}"
    print(f"\n=== Hybrid SHAP effect grid for {combo} ({len(FOLDS)} folds) ===")

    # short axis labels, styled after 1_preprocess.ipynb
    col_to_label = {v: k for k, v in _label_to_col().items()}
    cmap = plt.get_cmap("tab10")

    predictors = None
    fig = axes = None
    nrows = len(LATENT_ROWS)

    for fold in FOLDS:
        bundle = train_hybrid(md, mt, sat, fold)
        if predictors is None:
            predictors = bundle["predictors"]
            ncols = len(predictors)
            fig, axes = plt.subplots(nrows, ncols, figsize=(1.2 * ncols, 3.0 * nrows), squeeze=False, sharey="row")

        f = make_latent_fn(bundle, LATENT_ROWS)
        X_expl, sv_list = compute_shap_values(f, bundle["x_val"], len(LATENT_ROWS))
        # back to original (un-scaled) covariate space for the x-axis
        X_expl_orig = X_expl * bundle["x_std"] + bundle["x_mean"]
        color = cmap(fold % 10)
        for r in range(nrows):
            sv = sv_list[r]
            for i in range(len(predictors)):
                axes[r, i].scatter(X_expl_orig[:, i], sv[:, i], s=10, alpha=0.6, color=color, edgecolor="none")

    for r, row_name in enumerate(LATENT_ROWS):
        for i in range(len(predictors)):
            ax = axes[r, i]
            ax.set_xlabel(col_to_label.get(predictors[i], predictors[i]), fontsize=7)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
            ax.grid(True, ls="--", alpha=0.4)
            if i == 0:
                ax.set_ylabel(f"{row_name}\nSHAP value", fontsize=9)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"latentparams_vs_covariates_{combo}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"=== finished {combo}; figure saved to {out_path} ===")


if __name__ == "__main__":
    main()

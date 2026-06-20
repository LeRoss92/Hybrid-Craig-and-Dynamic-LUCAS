"""Dynamic parameter learning for MM / RMM – density_dependent – no saturation.

This is a self-contained rewrite that mirrors the *working* dynamic learning of
``2_hybrid.py`` (same Euler simulator, same utils helpers, same I=NPP forcing)
and the SHAP style of ``4_hybrid_shap.py``.

Idea
----
The static experiments (``2_hybrid.py`` with ``--temp static``) already fit every
mechanistic parameter spatially to reproduce the steady-state SOC / MAOC / MIC.
Here we *freeze* all of those per-site static parameters and progressively
re-learn spatial parameters in order of dynamic sensitivity (most sensitive
first, then cumulatively adding the next), mirroring the spatial/global sweep
in ``2_hybrid.py``.

Initial conditions  (from the end of 1_preprocess.ipynb)
    SOC_minus_4.5y is split into the three pools using the predicted indices
    MICi / MAOCi:
        Cb0 = MICi               * SOC_minus_4.5y      (microbial biomass)
        Cm0 = MAOCi              * SOC_minus_4.5y      (mineral-associated)
        Cp0 = (1-MICi-MAOCi)     * SOC_minus_4.5y      (particulate)

Target
    Δ SOC = SOC_plus_4.5y − SOC_minus_4.5y     (change over the 9-year window)

Fixed parameters
    Median across the 10 folds of the most-spatial static run
    ``...targetsSOC-MAOC-MIC_spatialCUE-beta-tmb-Vmax_p-Vmax_m-Km_p-Km_m-kb.pkl``
    (I is always overwritten by the carbon input NPP, exactly as in 2_hybrid.py).

Predictors
    All selected_predictors for dSOC from selected_predictors.json — the same
    covariates used to build pred_dSOC_median_XGB-n and hence SOC_plus/minus_4.5y.

Parameter sets
    Non-zero dynamic sensitivities (excluding I), sorted by |sensitivity|
    descending.  We re-learn them cumulatively: the most sensitive parameter
    alone, then + the second, then + the third, … up to all spatial parameters.

Outputs
-------
figures/dynamic_hybrid/{md}_{mt}_{sat}/
    r2_per_parameter.png                       bar chart of val R²(ΔSOC) for each
                                               cumulative sensitivity step
    {label}_learned_{param}_beeswarm.png       SHAP: learned param ~ covariates
    {label}_learned_{param}_effect.png
    {label}_learned_{param}_interactions.png
    {label}_dSOC_beeswarm.png                  SHAP: ΔSOC ~ covariates
    {label}_dSOC_effect.png
    {label}_dSOC_interactions.png
    (label = parameter set, e.g. "CUE" or "CUE+Vmax_m")
hybrid_outputs_dynamic/
    dynamic_{md}_{mt}_{sat}_{label}.pkl        per-site validation predictions

Run with:
    micromamba run -n hybrid-lucas python 5_dynamic_hybrid.py
"""

import os
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shap

import jax
import jax.numpy as jnp

from models import craig_BA_adapt
from config import default_param_ranges
from utils import (
    get_selected_predictors,
    init_mlp,
    mlp_forward,
    init_adam,
    clip_by_global_norm,
)

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────
MODELS = [
    ("MM",  "density_dependent", "no"),
    ("RMM", "density_dependent", "no"),
]

# Most-spatial static run whose per-site parameters we freeze.
STATIC_SPATIAL_SUFFIX = "CUE-beta-tmb-Vmax_p-Vmax_m-Km_p-Km_m-kb"
N_STATIC_FOLDS = 10

FOLD_VAL = "0"            # validation fold (matches 2_hybrid / 4_hybrid_shap)
T0, T1, DT0 = 0.0, 9.0, 0.05   # Euler step (matches the dynamic sensitivity run)

# Small NN that maps covariates -> the single re-learned parameter
DEPTH = 3
WIDTH = 48
LR = 5e-4
N_STEPS = 2000
EARLY_STOP_PATIENCE = 400
BATCH_SIZE = 512
LOG_EVERY = 200           # progress print cadence within a parameter's training

# SHAP sampling sizes (kept moderate; ΔSOC view runs the ODE per evaluation)
N_EXPLAIN = 120
N_BG = 40
N_SHAP = 400
RNG_SEED = 0

PARAM_NAMES = list(default_param_ranges.keys())
PARAM_MINS = np.array([default_param_ranges[n]["min"] for n in PARAM_NAMES], dtype=np.float32)
PARAM_MAXS = np.array([default_param_ranges[n]["max"] for n in PARAM_NAMES], dtype=np.float32)
PARAM_COLS = [f"param_{n}" for n in PARAM_NAMES]

OUT_FIG = Path("figures/dynamic_hybrid")
OUT_DATA = Path("hybrid_outputs_dynamic")
SKIP_SHAP = os.environ.get("HYBRID_SKIP_SHAP", "0").lower() in ("1", "true", "yes")


# ───────────────────────────────────────────────────────────────────────────
# Predictors: every dSOC selected predictor (same set as the XGB dSOC target).
# ───────────────────────────────────────────────────────────────────────────
def _build_dynamic_predictors() -> list[str]:
    return list(dict.fromkeys(get_selected_predictors("dSOC")))


DYNAMIC_PREDICTORS = _build_dynamic_predictors()

# Re-learned parameters move around the per-site static fit, not from mid-range.
RESIDUAL_SPAN_FRAC = 0.5


# ───────────────────────────────────────────────────────────────────────────
# Static parameters: median over the 10 folds of the most-spatial static run
# ───────────────────────────────────────────────────────────────────────────
def load_static_params(md: str, mt: str, sat: str) -> pd.DataFrame:
    per_fold = []
    for fold in range(N_STATIC_FOLDS):
        fname = (
            f"hybrid_outputs/hybrid_tempstatic_fold{fold}_md{md}_mt{mt}_sat{sat}"
            f"_targetsSOC-MAOC-MIC_spatial{STATIC_SPATIAL_SUFFIX}.pkl"
        )
        per_fold.append(pd.read_pickle(fname)[PARAM_COLS])

    common_idx = per_fold[0].index
    for df_f in per_fold[1:]:
        common_idx = common_idx.intersection(df_f.index)

    arrays = np.stack([df_f.loc[common_idx].to_numpy() for df_f in per_fold], axis=0)
    median_arr = np.median(arrays, axis=0)
    return pd.DataFrame(median_arr, index=common_idx, columns=PARAM_COLS)


# ───────────────────────────────────────────────────────────────────────────
# Data preparation
# ───────────────────────────────────────────────────────────────────────────
def prepare_data(md: str, mt: str, sat: str) -> dict:
    df = pd.read_pickle("1_preprocessed.pkl")
    static_df = load_static_params(md, mt, sat)
    df = df.join(static_df, how="left")

    input_col = "input_avg_09_15_18"

    has_npp = df[input_col].notna() & np.isfinite(df[input_col]) & (df[input_col] > 0)
    has_ic = (df["SOC_minus_4.5y"].notna() & np.isfinite(df["SOC_minus_4.5y"])
              & (df["SOC_minus_4.5y"] > 0))
    has_tgt = (df["SOC_plus_4.5y"].notna() & np.isfinite(df["SOC_plus_4.5y"])
               & (df["SOC_plus_4.5y"] > 0))
    has_idx = (df["MICi"].notna() & np.isfinite(df["MICi"])
               & df["MAOCi"].notna() & np.isfinite(df["MAOCi"]))
    has_stat = df[PARAM_COLS].notna().all(axis=1)

    # Predictors are NOT required to be non-NaN here — any missing values are
    # imputed below with the training-fold median, exactly as in 2_hybrid.py.
    # has_obs (OC_linreg_slope) is NOT required for training — targets come from
    # predicted SOC_plus/minus_4.5y.  We keep the column only for reporting r2_slope.
    mask = has_npp & has_ic & has_tgt & has_idx & has_stat
    original_idx = np.where(mask)[0]
    ds = df.loc[mask].reset_index(drop=True)

    split_col = ds["split"].astype(str).to_numpy()

    # Initial conditions: split SOC_minus_4.5y by the predicted indices.
    maoci = ds["MAOCi"].to_numpy(dtype=np.float32)
    mici = ds["MICi"].to_numpy(dtype=np.float32)
    soc0 = ds["SOC_minus_4.5y"].to_numpy(dtype=np.float32)

    # Clip so the two indices sum to at most 0.98 (Cp0 stays positive).
    total = maoci + mici
    scale = np.where(total > 0.98, 0.98 / (total + 1e-10), 1.0)
    maoci = maoci * scale
    mici = mici * scale

    Cp0 = np.clip((1.0 - maoci - mici) * soc0, 1e-4, None)
    Cb0 = np.clip(mici * soc0, 1e-4, None)
    Cm0 = np.clip(maoci * soc0, 1e-4, None)
    y0 = np.column_stack([Cp0, Cb0, Cm0]).astype(np.float32)

    soc1 = ds["SOC_plus_4.5y"].to_numpy(dtype=np.float32)
    delta_target = (soc1 - soc0).astype(np.float32)
    obs_slope = ds["OC_linreg_slope"].to_numpy(dtype=np.float32)

    # Impute missing predictor values with the global median (a per-split
    # re-imputation is then done inside train_combo using the training fold only).
    x_raw = ds[DYNAMIC_PREDICTORS].copy()
    x_raw = x_raw.fillna(x_raw.median(numeric_only=True))
    x_dyn = x_raw.to_numpy(dtype=np.float32)

    npp_I = ds[input_col].to_numpy(dtype=np.float32)
    static_params = ds[PARAM_COLS].to_numpy(dtype=np.float32)

    return dict(
        split_col=split_col,
        original_idx=original_idx,
        y0=y0,
        delta_target=delta_target,
        obs_slope=obs_slope,
        x_dyn=x_dyn,
        npp_I=npp_I,
        static_params=static_params,
    )


# ───────────────────────────────────────────────────────────────────────────
# Fast fixed-step Euler simulator (lax.scan).
#
# This reproduces diffrax's `Euler` solver with the same dt0 used in 2_hybrid.py,
# but reverse-mode AD through a `lax.scan` stores the (short) trajectory directly
# instead of diffrax's checkpoint-recompute adjoint, which makes the gradient
# evaluation that dominates training markedly cheaper.
# ───────────────────────────────────────────────────────────────────────────
def make_batched_sim(md: str, mt: str, sat: str):
    model_fn = partial(craig_BA_adapt, microbial_decomposition=md,
                       microbial_turnover=mt, saturation=sat)
    n_steps = int(round((T1 - T0) / DT0))

    def sim_one(p, y0):
        def body(y, _):
            return y + DT0 * model_fn(0.0, y, p), None
        yT, _ = jax.lax.scan(body, y0, None, length=n_steps)
        return yT

    return jax.vmap(sim_one)


# ───────────────────────────────────────────────────────────────────────────
# Training for one re-learned parameter
# ───────────────────────────────────────────────────────────────────────────
def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def learned_params(raw, static_b, param_idxs, p_mins, p_maxs):
    """Perturb frozen static parameters, starting at the static fit (raw=0)."""
    p_base = static_b[:, param_idxs]
    span = p_maxs - p_mins
    p_dyn = p_base + span * RESIDUAL_SPAN_FRAC * jnp.tanh(raw)
    return jnp.clip(p_dyn, p_mins, p_maxs)


def make_train_fns(batched_sim, param_idxs, p_mins, p_maxs):
    """Build a single fused jitted train-step plus eval/predict closures.

    ``param_idxs`` / ``p_mins`` / ``p_maxs`` are arrays (length k = number of
    parameters re-learned together).  The NN outputs k residual adjustments
    around the per-site static parameters (zero at initialisation).

    Mirrors the structure of ``utils.train_step`` in 2_hybrid.py: the loss, its
    gradient, gradient clipping and the Adam update are all compiled into one
    XLA program so there is no per-step Python dispatch overhead.
    """
    param_idxs = jnp.asarray(param_idxs)
    p_mins = jnp.asarray(p_mins)
    p_maxs = jnp.asarray(p_maxs)

    def predict_delta(net_p, x_b, static_b, npp_b, y0_b):
        raw = jax.vmap(lambda xi: mlp_forward(net_p, xi))(x_b)   # (n, k)
        p_dyn = learned_params(raw, static_b, param_idxs, p_mins, p_maxs)
        # Freeze static params, overwrite the learned columns and I (=NPP).
        p_mat = static_b.at[:, param_idxs].set(p_dyn).at[:, 0].set(npp_b)
        pf = batched_sim(p_mat, y0_b)
        return jnp.sum(pf - y0_b, axis=-1)        # predicted Δ SOC

    def loss_fn(net_p, x_b, static_b, npp_b, y0_b, delta_b, tgt_mean, tgt_std):
        pred = predict_delta(net_p, x_b, static_b, npp_b, y0_b)
        pn = (pred - tgt_mean) / (tgt_std + 1e-8)
        tn = (delta_b - tgt_mean) / (tgt_std + 1e-8)
        diff = pn - tn
        dh = 0.5
        huber = jnp.where(jnp.abs(diff) <= dh,
                          0.5 * diff ** 2,
                          dh * (jnp.abs(diff) - 0.5 * dh))
        return jnp.mean(huber)

    vg = jax.value_and_grad(loss_fn)

    @jax.jit
    def step_fn(net_p, opt_state, step, lr_t,
                x_b, static_b, npp_b, y0_b, delta_b, tgt_mean, tgt_std):
        loss, grads = vg(net_p, x_b, static_b, npp_b, y0_b, delta_b, tgt_mean, tgt_std)
        grads = clip_by_global_norm(grads, 1.0)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m, v = opt_state
        m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1 - b2) * g ** 2, v, grads)
        mh = jax.tree_util.tree_map(lambda mi: mi / (1 - b1 ** step), m)
        vh = jax.tree_util.tree_map(lambda vi: vi / (1 - b2 ** step), v)
        net_p = jax.tree_util.tree_map(
            lambda p, mi, vi: p - lr_t * mi / (jnp.sqrt(vi) + eps), net_p, mh, vh)
        return net_p, (m, v), loss

    return step_fn, jax.jit(loss_fn), jax.jit(predict_delta)


def train_combo(data: dict, param_names: list[str], batched_sim) -> dict | None:
    label = "+".join(param_names)
    pidxs = [PARAM_NAMES.index(n) for n in param_names]
    p_mins = np.array([PARAM_MINS[i] for i in pidxs], dtype=np.float32)
    p_maxs = np.array([PARAM_MAXS[i] for i in pidxs], dtype=np.float32)
    k = len(param_names)

    split = data["split_col"]
    train_idx = np.where((split != "test") & (split != FOLD_VAL))[0]
    val_idx = np.where(split == FOLD_VAL)[0]
    if train_idx.size < 50 or val_idx.size < 10:
        print(f"    [{label}] skip – too few rows")
        return None

    # Re-impute any remaining NaNs using training-fold median, then normalise.
    x_raw = data["x_dyn"].copy()
    train_medians = np.nanmedian(x_raw[train_idx], axis=0)
    nan_mask = ~np.isfinite(x_raw)
    x_raw[nan_mask] = np.take(train_medians, np.where(nan_mask)[1])
    x_all = jnp.asarray(x_raw)
    x_mean = jnp.mean(x_all[train_idx], axis=0)
    x_std = jnp.std(x_all[train_idx], axis=0) + 1e-8
    x_all = (x_all - x_mean) / x_std

    delta_all = jnp.asarray(data["delta_target"])
    tgt_mean = jnp.mean(delta_all[train_idx])
    tgt_std = jnp.std(delta_all[train_idx]) + 1e-8
    y0_all = jnp.asarray(data["y0"])
    npp_all = jnp.asarray(data["npp_I"])
    static_all = jnp.asarray(data["static_params"])

    n_feats = x_all.shape[1]
    net_p = init_mlp(jax.random.PRNGKey(0), [n_feats] + [WIDTH] * DEPTH + [k])
    opt_state = init_adam(net_p)
    best_p, best_loss, best_step = net_p, float("inf"), 0

    step_fn, eval_fn, predict_delta = make_train_fns(batched_sim, pidxs, p_mins, p_maxs)
    train_idx_j = jnp.asarray(train_idx)

    t0 = time.perf_counter()
    for step in range(1, N_STEPS + 1):
        key = jax.random.PRNGKey(step)
        bsel = jax.random.choice(
            key, train_idx_j.size,
            shape=(min(BATCH_SIZE, train_idx_j.size),), replace=False)
        gi = train_idx_j[bsel]
        wup = jnp.minimum(1.0, step / 200.0)
        lr_t = LR * wup * 0.5 * (1.0 + jnp.cos(jnp.pi * step / N_STEPS))
        step_j = jnp.asarray(step, dtype=jnp.float32)

        net_p, opt_state, loss = step_fn(
            net_p, opt_state, step_j, lr_t,
            x_all[gi], static_all[gi], npp_all[gi],
            y0_all[gi], delta_all[gi], tgt_mean, tgt_std,
        )

        if step % 50 == 0:
            vl = float(eval_fn(
                net_p, x_all[val_idx], static_all[val_idx], npp_all[val_idx],
                y0_all[val_idx], delta_all[val_idx], tgt_mean, tgt_std,
            ))
            if vl < best_loss:
                best_loss, best_p, best_step = vl, net_p, step
            elif step - best_step >= EARLY_STOP_PATIENCE:
                print(f"    [{label}] early stop at step {step}", flush=True)
                break
        if step % LOG_EVERY == 0:
            print(f"    [{label}] step {step}/{N_STEPS}  "
                  f"train_loss={float(loss):.4f}  best_val={best_loss:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # Validation R² of ΔSOC, plus R² against the observed OC slope (per year).
    pred_val = np.asarray(jax.device_get(predict_delta(
        best_p, x_all[val_idx], static_all[val_idx],
        npp_all[val_idx], y0_all[val_idx])))
    targ_val = np.asarray(jax.device_get(delta_all[val_idx]))
    r2_delta = r2_score(pred_val, targ_val)

    # Approximate log-OC slope (pred_dSOC units) from predicted ΔSOC and SOC at t0.
    soc0_val = np.asarray(jax.device_get(jnp.sum(y0_all[val_idx], axis=-1)))
    pred_log_slope = np.log1p(pred_val / np.clip(soc0_val, 1e-4, None)) / (T1 - T0)
    obs_slope_val = data["obs_slope"][val_idx]
    # r2_slope is secondary and only computed where the observed slope exists.
    obs_mask = np.isfinite(obs_slope_val) & np.isfinite(pred_log_slope)
    r2_slope = r2_score(obs_slope_val[obs_mask], pred_log_slope[obs_mask]) if obs_mask.sum() > 10 else float("nan")

    print(f"    [{label}] val R²(ΔSOC)={r2_delta:+.3f}  "
          f"R²(slope)={r2_slope:+.3f}  time={time.perf_counter() - t0:.0f}s")

    return dict(
        label=label,
        param_names=list(param_names),
        param_idxs=pidxs,
        p_mins=p_mins,
        p_maxs=p_maxs,
        n_learned=k,
        r2_delta=r2_delta,
        r2_slope=r2_slope,
        best_params=best_p,
        x_mean=np.asarray(jax.device_get(x_mean)),
        x_std=np.asarray(jax.device_get(x_std)),
        val_idx=val_idx,
        train_idx=train_idx,
        pred_delta_val=pred_val,
        tgt_delta_val=targ_val,
        pred_slope_val=pred_log_slope,
        obs_slope_val=obs_slope_val,
    )


# ───────────────────────────────────────────────────────────────────────────
# R² bar chart
# ───────────────────────────────────────────────────────────────────────────
def static_baseline_r2(data: dict, batched_sim, val_idx: np.ndarray) -> float:
    """R²(ΔSOC) with every parameter frozen at the median static fit."""
    static = jnp.asarray(data["static_params"])
    static = static.at[:, 0].set(jnp.asarray(data["npp_I"]))
    y0 = jnp.asarray(data["y0"])
    pred = np.asarray(jax.device_get(batched_sim(static, y0)))
    pred_delta = np.sum(pred - np.asarray(y0), axis=1)
    targ = data["delta_target"][val_idx]
    return r2_score(pred_delta[val_idx], targ)


def plot_r2_bars(results: list, out_dir: Path, combo: str, *, static_r2: float) -> None:
    # Results are in cumulative sensitivity order (1 param → all).
    if results:
        results[-1]["is_ceiling"] = True

    names = [r["label"] for r in results]
    r2_delta = [r["r2_delta"] for r in results]
    is_ceiling = [r.get("is_ceiling", False) for r in results]
    is_multi = [r["n_learned"] > 1 and not r.get("is_ceiling", False) for r in results]

    colors = []
    for v, multi, ceiling in zip(r2_delta, is_multi, is_ceiling):
        if ceiling:
            colors.append("#2E7D32")
        elif multi:
            colors.append("#1565C0" if v >= 0 else "#B71C1C")
        else:
            colors.append("#64B5F6" if v >= 0 else "#EF9A9A")

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.45), 5.5))
    bars = ax.bar(range(len(names)), r2_delta, color=colors,
                  edgecolor="white", linewidth=0.6)
    for bar, multi, ceiling in zip(bars, is_multi, is_ceiling):
        if ceiling:
            bar.set_hatch("xx")
            bar.set_edgecolor("#1B5E20")
        elif multi:
            bar.set_hatch("//")
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.axhline(static_r2, color="#616161", lw=1.0, ls=":",
               label=f"static only ({static_r2:+.2f})")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Validation R² of the SOC change (Δ SOC over 9 yr)")
    ax.set_xlabel("Cumulative re-learned parameters (most sensitive first; others frozen)")
    ax.set_title(
        f"Dynamic parameter learning v2 — R² of change per cumulative parameter step\n{combo}"
    )
    lo = max(min(r2_delta) - 0.05, -1.2)
    hi = max(max(r2_delta) + 0.08, 0.15)
    ax.set_ylim(lo, hi)
    for i, (v, ceiling) in enumerate(zip(r2_delta, is_ceiling)):
        if v >= lo + 0.02 or ceiling:
            ax.text(i, min(v + 0.02, hi - 0.02), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)
    legend_handles = [
        Patch(facecolor="#2E7D32", edgecolor="#1B5E20", hatch="xx",
              label="all spatial params (final step)"),
        Patch(facecolor="#64B5F6", edgecolor="white", label="1st step (most sensitive)"),
        Patch(facecolor="#1565C0", edgecolor="white", hatch="//",
              label="cumulative intermediate steps"),
        plt.Line2D([0], [0], color="#616161", ls=":", label=f"static only ({static_r2:+.2f})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    fig.tight_layout()
    path = out_dir / "r2_per_parameter.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → R² bar chart: {path}")


# ───────────────────────────────────────────────────────────────────────────
# SHAP (beeswarm + effect), mirroring the style of 4_hybrid_shap.py
# ───────────────────────────────────────────────────────────────────────────
def _beeswarm(sv, X, feat_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, max(3, 1.2 + 0.7 * len(feat_names))))
    plt.sca(ax)
    shap.summary_plot(sv, X, feature_names=feat_names,
                      max_display=len(feat_names), show=False, plot_size=None)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _effect(sv, X, feat_names, out_label, title, save_path):
    nf = len(feat_names)
    order = np.argsort(-np.mean(np.abs(sv), axis=0))
    ncol = min(nf, 3)
    nrow = int(np.ceil(nf / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for pi, fi in enumerate(order):
        ax = axes[pi // ncol, pi % ncol]
        ax.axis("on")
        ax.scatter(X[:, fi], sv[:, fi], s=14, alpha=0.6, c="#1f77b4")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(feat_names[fi], fontsize=9)
        ax.set_ylabel(f"SHAP ({out_label})", fontsize=9)
        ax.set_title(feat_names[fi], fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _interactions(sv, X, feat_names, title, save_path):
    """Approximate SHAP interaction grid (style of 4_hybrid_shap.py).

    Off-diagonal cell (i, j): SHAP value of feature i vs feature i, coloured by
    feature j — reveals how feature j modulates feature i's effect.
    """
    nf = len(feat_names)
    order = np.argsort(-np.mean(np.abs(sv), axis=0))
    top = order[:nf]
    m = len(top)
    fig, axes = plt.subplots(m, m, figsize=(3.4 * m, 3.0 * m), squeeze=False)
    for a, fi in enumerate(top):
        for b, fj in enumerate(top):
            ax = axes[a, b]
            if a == b:
                ax.axis("off")
                ax.text(0.5, 0.5, feat_names[fi], ha="center", va="center",
                        fontsize=12, fontweight="bold")
                continue
            sc = ax.scatter(X[:, fi], sv[:, fi], c=X[:, fj], cmap="viridis",
                            s=12, alpha=0.7)
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_xlabel(feat_names[fi], fontsize=8)
            ax.set_ylabel(f"SHAP({feat_names[fi]})", fontsize=8)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(feat_names[fj], fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _as_output_list(sv, n_out):
    """Normalise a KernelExplainer return into a list of (n_samples, n_feat)."""
    if isinstance(sv, list):
        return sv
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return [sv[:, :, k] for k in range(sv.shape[-1])]
    return [sv]


def run_shap(result: dict, data: dict, batched_sim, out_dir: Path) -> None:
    label = result["label"]
    pnames = result["param_names"]
    pidxs = jnp.asarray(result["param_idxs"])
    p_mins = jnp.asarray(result["p_mins"])
    p_maxs = jnp.asarray(result["p_maxs"])
    net_p = result["best_params"]
    x_mean = result["x_mean"]
    x_std = result["x_std"]
    val_idx = result["val_idx"]

    x_all_n = ((data["x_dyn"] - x_mean) / (x_std + 1e-8)).astype(np.float64)
    x_val_n = x_all_n[val_idx]
    n = x_val_n.shape[0]

    rng = np.random.default_rng(RNG_SEED)
    expl_idx = rng.choice(n, size=min(N_EXPLAIN, n), replace=False)
    X_expl = x_val_n[expl_idx]
    bg = shap.kmeans(x_val_n, min(N_BG, n))

    # Median validation-site auxiliary inputs for KernelExplainer batches.
    static_med = jnp.asarray(np.median(data["static_params"][val_idx], axis=0), dtype=jnp.float32)
    npp_med = jnp.asarray(np.median(data["npp_I"][val_idx]), dtype=jnp.float32)
    y0_med = jnp.asarray(np.median(data["y0"][val_idx], axis=0), dtype=jnp.float32)
    n_param = static_med.shape[0]

    # ── View 1: learned parameter(s) ~ dynamic covariates (network only) ──────
    @jax.jit
    def _net_params(x_batch):
        xb = jnp.asarray(x_batch, dtype=jnp.float32)
        raw = jax.vmap(lambda xi: mlp_forward(net_p, xi))(xb)           # (n, k)
        static_b = jnp.broadcast_to(static_med, (xb.shape[0], static_med.shape[0]))
        return learned_params(raw, static_b, pidxs, p_mins, p_maxs)

    def f_params(x):
        return np.asarray(jax.device_get(_net_params(x)))

    sv1_list = _as_output_list(
        shap.KernelExplainer(f_params, bg).shap_values(X_expl, nsamples=N_SHAP, silent=True),
        len(pnames))
    for j, pn in enumerate(pnames):
        sv = sv1_list[j]
        _beeswarm(sv, X_expl, DYNAMIC_PREDICTORS,
                  f"Learned {pn} ~ dynamic covariates  [{label}]",
                  out_dir / f"{label}_learned_{pn}_beeswarm.png")
        _effect(sv, X_expl, DYNAMIC_PREDICTORS, pn,
                f"Learned {pn} ~ dynamic covariates  [{label}]",
                out_dir / f"{label}_learned_{pn}_effect.png")
        _interactions(sv, X_expl, DYNAMIC_PREDICTORS,
                      f"Learned {pn} ~ dynamic covariates (interactions)  [{label}]",
                      out_dir / f"{label}_learned_{pn}_interactions.png")

    # ── View 2: Δ SOC ~ dynamic covariates (full dynamic model via learned params),
    # evaluated at a representative (median) validation site so the auxiliary
    # inputs (frozen static params, NPP, initial pools) are well defined for any
    # batch size the KernelExplainer feeds in. ─────────────────────────────────
    @jax.jit
    def _dsoc(x_batch):
        xb = jnp.asarray(x_batch, dtype=jnp.float32)
        nb = xb.shape[0]
        raw = jax.vmap(lambda xi: mlp_forward(net_p, xi))(xb)        # (nb, k)
        static_b = jnp.broadcast_to(static_med, (nb, n_param))
        p_dyn = learned_params(raw, static_b, pidxs, p_mins, p_maxs)
        p_mat = (static_b
                 .at[:, pidxs].set(p_dyn).at[:, 0].set(npp_med))
        y0b = jnp.broadcast_to(y0_med, (nb, 3))
        pf = batched_sim(p_mat, y0b)
        return jnp.sum(pf - y0b, axis=-1, keepdims=True)

    def f_dsoc(x):
        return np.asarray(jax.device_get(_dsoc(x)))

    sv2 = _as_output_list(
        shap.KernelExplainer(f_dsoc, bg).shap_values(X_expl, nsamples=N_SHAP, silent=True),
        1)[0]
    _beeswarm(sv2, X_expl, DYNAMIC_PREDICTORS,
              f"Δ SOC ~ dynamic covariates (via learned {label})",
              out_dir / f"{label}_dSOC_beeswarm.png")
    _effect(sv2, X_expl, DYNAMIC_PREDICTORS, "ΔSOC",
            f"Δ SOC ~ dynamic covariates (via learned {label})",
            out_dir / f"{label}_dSOC_effect.png")
    _interactions(sv2, X_expl, DYNAMIC_PREDICTORS,
                  f"Δ SOC ~ dynamic covariates (interactions, via learned {label})",
                  out_dir / f"{label}_dSOC_interactions.png")

    print(f"    [{label}] SHAP saved to {out_dir}")


# ───────────────────────────────────────────────────────────────────────────
# Per-combo analysis
# ───────────────────────────────────────────────────────────────────────────
def _clear_label_plots(out_dir: Path, label: str) -> None:
    """Remove stale SHAP / output PNGs for one parameter set before re-running it."""
    for path in out_dir.glob(f"{label}_*.png"):
        path.unlink()


def analyse_combo(md: str, mt: str, sat: str) -> None:
    combo = f"{md}_{mt}_{sat}"
    out_dir = OUT_FIG / combo
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"Dynamic hybrid  —  {combo}")
    print(f"Dynamic predictors: {DYNAMIC_PREDICTORS}")
    print(f"{'=' * 64}")

    data = prepare_data(md, mt, sat)
    split_col = data["split_col"]
    n_train = int(np.sum((split_col != "test") & (split_col != FOLD_VAL)))
    n_val = int(np.sum(split_col == FOLD_VAL))
    n_test = int(np.sum(split_col == "test"))
    print(f"  Rows: {len(split_col)}  (train={n_train}, val={n_val}, test={n_test})")
    print(f"  Δ SOC  mean={np.mean(data['delta_target']):.3f}  "
          f"std={np.std(data['delta_target']):.3f}  "
          f"[{np.min(data['delta_target']):.3f}, {np.max(data['delta_target']):.3f}]")

    # Build the forward (Euler) simulator for this variant.
    batched_sim = make_batched_sim(md, mt, sat)

    # Parameters to re-learn: non-zero *dynamic* sensitivity, excluding I.
    sens = pd.read_csv("figures/sensitivities.csv")
    row = sens[(sens["md"] == md) & (sens["mt"] == mt) &
               (sens["sat"] == sat) & (sens["temp"] == "dynamic")].iloc[0]
    param_sens = row.drop(labels=["md", "mt", "sat", "temp", "y0_Cp", "y0_Cb", "y0_Cm"])
    test_params = [n for n in PARAM_NAMES
                   if n != "I" and float(param_sens.get(n, 0.0)) != 0.0]
    sorted_params = sorted(test_params,
                           key=lambda name: abs(float(param_sens[name])),
                           reverse=True)
    param_sets = [sorted_params[:k] for k in range(1, len(sorted_params) + 1)]
    print(f"  Dynamic sensitivities (desc): "
          f"{', '.join(f'{n}={float(param_sens[n]):.3g}' for n in sorted_params)}")
    print(f"  Cumulative steps: {len(param_sets)} "
          f"(+{sorted_params[0]} → … → all)")

    val_idx = np.where(data["split_col"] == FOLD_VAL)[0]
    static_r2 = static_baseline_r2(data, batched_sim, val_idx)
    print(f"  Static-only baseline val R²(ΔSOC) = {static_r2:+.3f}")

    results = []
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    for step, pset in enumerate(param_sets, start=1):
        label = "+".join(pset)
        print(f"\n  ── Step {step}/{len(param_sets)}: re-learning {label} spatially ──")
        _clear_label_plots(out_dir, label)
        res = train_combo(data, pset, batched_sim)
        if res is None:
            continue
        res["step"] = step
        results.append(res)

        val_idx = res["val_idx"]
        pd.DataFrame({
            "original_idx": data["original_idx"][val_idx],
            "pred_delta_soc": res["pred_delta_val"],
            "tgt_delta_soc": res["tgt_delta_val"],
            "pred_slope": res["pred_slope_val"],
            "obs_slope": res["obs_slope_val"],
            "param_label": label,
            "n_learned": res["n_learned"],
            "r2_delta": res["r2_delta"],
            "r2_slope": res["r2_slope"],
        }).to_pickle(OUT_DATA / f"dynamic_{combo}_{label}.pkl")

    if not results:
        print(f"  No successful results for {combo}!")
        return

    plot_r2_bars(results, out_dir, combo, static_r2=static_r2)

    print(f"\n  Summary for {combo} (cumulative sensitivity order):")
    print(f"  {'Step':>4}  {'Parameter set':<30}  {'R²(ΔSOC)':>10}")
    print(f"  {'-' * 48}")
    for res in results:
        note = " (final)" if res["step"] == len(param_sets) else ""
        print(f"  {res['step']:>4}  {res['label']:<30}  {res['r2_delta']:>+10.4f}{note}")
    print(f"  {'':>4}  {'static only':<30}  {static_r2:>+10.4f}  (baseline)")

    print(f"\n  Running SHAP analysis (beeswarm + effect + interactions) ...")
    if SKIP_SHAP:
        print("  (skipped — set HYBRID_SKIP_SHAP=1 to disable)")
    shap_targets = results[:]
    for res in shap_targets:
        if SKIP_SHAP:
            break
        print(f"  ── SHAP for {res['label']} ──", flush=True)
        try:
            run_shap(res, data, batched_sim, out_dir)
        except Exception as exc:  # noqa: BLE001 - keep going on a single failure
            print(f"    SHAP failed for {res['label']}: {exc}")

    print(f"\n  Finished {combo}  →  {out_dir}")


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    print("Dynamic hybrid parameter learning")
    print(f"Predictors (dSOC selected_predictors from selected_predictors.json):")
    print(f"  {DYNAMIC_PREDICTORS}")

    for md, mt, sat in MODELS:
        analyse_combo(md, mt, sat)

    print("\nDone.")


if __name__ == "__main__":
    main()

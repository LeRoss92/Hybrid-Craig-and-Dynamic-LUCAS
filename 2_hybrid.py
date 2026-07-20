import argparse
import time
import os
import jax
import jax.numpy as jnp
import diffrax as dfx
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models import craig_BA_adapt, analytical_steady_state
from config import default_param_ranges, TARGET_CONFIG
import utils
from utils import vector_field, simulate_final_state, init_mlp, build_param_matrix, eval_loss, init_adam, eval_r2, train_step, build_hybrid_predictors


def pools_to_loss_targets(y_cmp, y0, use_dynamic, targets_arg):
    """Map model pools [Cp, Cb, Cm] to SOC plus requested subfraction targets."""
    y_fin = y_cmp + y0 if use_dynamic else y_cmp
    soc_sum = jnp.sum(y_fin, axis=-1, keepdims=True)
    if not use_dynamic:
        soc_safe = soc_sum + 1e-12
        mic_target = y_fin[:, 1:2] / soc_safe
        maoc_target = y_fin[:, 2:3] / soc_safe
    else:
        mic_target = y_fin[:, 1:2]
        maoc_target = y_fin[:, 2:3]
    if targets_arg == "SOC":
        return soc_sum
    if targets_arg == "SOC,MIC":
        return jnp.concatenate([soc_sum, mic_target], axis=-1)
    if targets_arg == "SOC,MAOC":
        return jnp.concatenate([soc_sum, maoc_target], axis=-1)
    if targets_arg == "SOC,MAOC,MIC":
        return jnp.concatenate([soc_sum, maoc_target, mic_target], axis=-1)
    raise KeyError(targets_arg)


utils.pools_to_loss_targets = pools_to_loss_targets

dt0 = 0.025  # years
depth = 7
width = 128
batch_size = 1024

TARGET_LABELS = {
    "SOC": ["SOC"],
    "SOC,MIC": ["SOC", "MIC"],
    "SOC,MAOC": ["SOC", "MAOC"],
    "SOC,MAOC,MIC": ["SOC", "MAOC", "MIC"],
}

MIN_STATIC_TARGET_ROWS = 1000

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--temp"); parser.add_argument("--fold"); parser.add_argument("--md"); parser.add_argument("--mt"); parser.add_argument("--sat"); parser.add_argument("--targets"); args = parser.parse_args()
    start_time = time.perf_counter()
    use_dynamic = args.temp == "dynamic"

    if use_dynamic:
        # dynamic
        lr = 5e-4
        n_steps = 1000 # epochs
        early_stop_patience = 100 # epochs
    else:
        # static
        lr = 5e-4
        n_steps = 3000 # epochs
        early_stop_patience = 500 # epochs

    sensitivities = pd.read_csv("figures/sensitivities.csv") # load importances

    desired_targets = TARGET_LABELS[args.targets]  # e.g. ["SOC"], ["SOC", "MIC"], ["SOC", "MAOC", "MIC"]
    model_sens = sensitivities[
        (sensitivities["md"] == args.md)
        & (sensitivities["mt"] == args.mt)
        & (sensitivities["sat"] == args.sat)
        & (sensitivities["target"].isin(desired_targets))
    ]  # one row per requested target for this version
    if model_sens.empty:
        raise ValueError(f"No sensitivity rows for md={args.md} mt={args.mt} sat={args.sat} targets={desired_targets}")
    meta_cols = ["md", "mt", "sat", "target"]
    param_cols = [c for c in sensitivities.columns if c not in meta_cols]
    param_sens = model_sens[param_cols].abs().max(axis=0)  # combined score across targets

    param_names = list(default_param_ranges.keys())
    param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
    param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])

    # sensitivities.csv is now used ONLY to decide which params are sensitive. Non-sensitive params
    # (zero sensitivity across all requested targets) are always global. I is the forcing (never spatial).
    always_global = [n for n, v in param_sens.items() if v == 0.0 and n != "I" and n in param_names]
    # sensitive params are the candidates greedily promoted to global one at a time
    sensitive_params = [n for n, v in param_sens.items() if v != 0.0 and n != "I" and n in param_names]
    print('always-global (not sensitive):', always_global)
    print('sensitive (candidates for promotion):', sensitive_params)

    def build_masks(global_names):
        cleaned = [name.strip() for name in global_names if name.strip()]
        unknown = sorted(set(cleaned) - set(param_names))
        if unknown:
            raise ValueError(f"Unknown global params: {', '.join(unknown)}")
        mask = jnp.array([name in cleaned for name in param_names])
        spatial = [name for name in param_names if name not in cleaned and name != "I"]
        return mask, spatial



    # build mechanistic models (independent of which params are global)
    batched_steady = jax.vmap(partial(analytical_steady_state, microbial_decomposition=args.md, microbial_turnover=args.mt, saturation=args.sat)) # vmap analytical solution
    t0, t1 = 0.0, 9.0
    solver = dfx.Euler()
    model_fn = partial(craig_BA_adapt, microbial_decomposition=args.md, microbial_turnover=args.mt, saturation=args.sat)
    term = dfx.ODETerm(partial(vector_field, model_fn))
    batched_sim = jax.vmap(lambda p, y0: simulate_final_state(p, y0, t0, t1, dt0, term, solver)) # vmap solver

    # preprocess: get data, log some features & calculate stocks, get split indices, impute, create targets, normalize
    df = pd.read_pickle("1_preprocessed.pkl") # get data
    targets = args.targets.split(',') # target(s)
    # Average versions of the selected index predictors (same normalize_hybrid_predictor
    # mapping used in 1_preprocess to build the MICi/MAOCi targets) so inputs match the targets.
    predictors = build_hybrid_predictors(targets)
    helper_df = df.copy()
    input_col = "input_avg_09_15_18"
    ta = args.targets
    target_labels = TARGET_LABELS[ta]
    target_columns = {"SOC": "SOC"}
    if use_dynamic:
        target_columns.update({"MIC": "MIC", "MAOC": "MAOC"})
    else:
        for label in ("MIC", "MAOC"):
            if label in target_labels:
                col = f"{label}i"
                if col not in helper_df.columns:
                    raise ValueError(
                        f"Missing static target column {col}. Rerun preprocessing so {col} is "
                        f"filled from the median predicted {label} index."
                    )
                target_columns[label] = col
    target_source_cols = [target_columns[label] for label in target_labels]
    # Full set of outputs to save (targets + predictions), even when a label is unconstrained.
    output_labels = ["SOC", "MAOC", "MIC"]
    output_columns = {}
    for label in output_labels:
        if label == "SOC":
            output_columns[label] = "SOC"
        elif use_dynamic:
            output_columns[label] = label
        else:
            col = f"{label}i"
            output_columns[label] = col if col in helper_df.columns else None
    output_source_cols = [c for c in output_columns.values() if c is not None]
    required_cols = ["SOC", "POC", "MIC", "MAOC"] + target_source_cols + output_source_cols + predictors + [input_col, 'era5_land_t2m_avg_09_15_18', 'split']
    helper_df = helper_df[list(dict.fromkeys(required_cols))]

    npp_mask = (helper_df[input_col].notna() & np.isfinite(helper_df[input_col]) & (helper_df[input_col] > 0)).to_numpy()
    original_idx = np.where(npp_mask)[0]
    helper_df = helper_df.loc[npp_mask].reset_index(drop=True)
    split_col = helper_df["split"].astype(str).to_numpy() # use same splits as in prediction
    if not use_dynamic:
        min_rows = max(MIN_STATIC_TARGET_ROWS, int(0.1 * len(helper_df)))
        for label in ("MIC", "MAOC"):
            if label in target_labels:
                col = target_columns[label]
                n_finite = int(np.isfinite(helper_df[col].to_numpy()).sum())
                if n_finite < min_rows:
                    raise ValueError(
                        f"Static target column {col} is not dense enough ({n_finite}/{len(helper_df)} finite). "
                        f"Rerun preprocessing so {col} is filled from the median predicted {label} index."
                    )

    target_values = np.column_stack([helper_df[target_columns[label]].to_numpy() for label in target_labels])
    # Observed values for all output labels, captured before fillna so unobserved rows stay NaN.
    all_target_values = np.column_stack([
        helper_df[output_columns[label]].to_numpy() if output_columns[label] is not None
        else np.full(len(helper_df), np.nan)
        for label in output_labels
    ])
    label_mask = np.all(np.isfinite(target_values), axis=1)
    train_idx = np.where(label_mask & (split_col != "test") & (split_col != str(args.fold)))[0] # train on all folds except validation fold (and also not test)
    val_idx = np.where(label_mask & (split_col == str(args.fold)))[0] # validation indices
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError(f"No train/validation rows with finite targets for {args.targets}")
    helper_df = helper_df.fillna(helper_df.iloc[jax.device_get(train_idx)].median(numeric_only=True)) # impute predictors and initial-state helpers only
    targets = jnp.asarray(target_values)

    # split
    x_features = jnp.asarray(helper_df[predictors].to_numpy()) # df to np
    npp_I_all = jnp.asarray(helper_df[input_col].to_numpy())
    x_train = x_features[train_idx]
    y_train = targets[train_idx]
    npp_I_train = npp_I_all[train_idx]
    x_val = x_features[val_idx]
    y_val = targets[val_idx]
    npp_I_val = npp_I_all[val_idx]
    # normalize features
    x_mean = jnp.mean(x_train, axis=0)
    x_std = jnp.std(x_train, axis=0) + 1e-8
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_features = (x_features - x_mean) / x_std
    target_mean = jnp.mean(y_train, axis=0)
    target_std = jnp.std(y_train, axis=0) + 1e-8
    # initial conditions (effectifly only used by dynamic)
    y0_true = jnp.asarray(helper_df[["POC", "MIC", "MAOC"]].to_numpy())
    y0_train, y0_val = y0_true[train_idx], y0_true[val_idx]

    def train_model(global_mask):
        """Train a hybrid model for the given global_mask; return (best_params, mean_val_r2)."""
        # initilize ML (fresh, deterministic per fold so trials are comparable)
        init_key = jax.random.fold_in(jax.random.PRNGKey(0), int(args.fold))
        net_params = init_mlp(init_key, [x_features.shape[1]] + [width] * depth + [param_mins.size]) # set up NN
        global_raw = jnp.zeros((param_mins.size,))
        params = {"net": net_params, "global": global_raw}
        n_targ = int(targets.shape[1])
        target_mask = jnp.ones((n_targ,))
        loss_ema, ema_beta, weights = jnp.ones((n_targ,)) * target_mask, 0.9, jnp.ones((n_targ,)) * target_mask # set up loss
        opt_state, best_params, best_test = init_adam(params), params, float("inf") # init optimizer and identification of best epoch
        best_step = 0
        best_val_r2 = None

        # training
        init_y_loss = eval_loss(params, x_train, npp_I_train, y0_train, y_train, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std, targets_arg=args.targets)[0]
        print(f"init y_loss {init_y_loss:.6f}")
        for step in range(1, n_steps + 1):
            k = jax.random.PRNGKey(step)
            batch_idx = jax.random.choice(k, train_idx.size, shape=(min(batch_size, train_idx.size),), replace=False)
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            y0_batch = y0_train[batch_idx] if use_dynamic else jnp.zeros((batch_idx.size, 3))
            npp_I_batch = npp_I_train[batch_idx]
            warmup_scale = jnp.minimum(1.0, step / 200.0)
            lr_t = lr * warmup_scale * 0.5 * (1.0 + jnp.cos(jnp.pi * step / n_steps))
            params, opt_state, loss, per_component = train_step(params, opt_state, x_batch, npp_I_batch, y0_batch, y_batch, lr_t, step, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std, targets_arg=args.targets)
            loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * per_component
            weights = (1.0 / (loss_ema + 1e-8)) * target_mask
            if step % 50 == 0:
                val_r2 = eval_r2(params, x_val, npp_I_val, y0_val, y_val, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, targets_arg=args.targets)
                val_r2_np = jax.device_get(val_r2)
                val_loss = eval_loss(params, x_val, npp_I_val, y0_val, y_val, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std, targets_arg=args.targets)[0]
                if val_loss < best_test:
                    best_test = float(val_loss)
                    best_params = params
                    best_step = step
                    best_val_r2 = val_r2_np  # R2 at the selected (best val loss) checkpoint
                r2_str = " ".join(f"{lb} {val_r2_np[i]:.3f}" for i, lb in enumerate(TARGET_LABELS[args.targets]))
                print(f"step {step} loss {loss:.6f} | R2 {r2_str}")
                if step - best_step >= early_stop_patience:
                    print(f"early stopping at step {step} (no val loss improvement for {early_stop_patience} epochs)")
                    break
        if best_val_r2 is None:  # never hit an eval step; fall back to final params
            best_val_r2 = jax.device_get(eval_r2(best_params, x_val, npp_I_val, y0_val, y_val, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, targets_arg=args.targets))
        return best_params, float(np.mean(best_val_r2))

    def save_output(best_params, global_mask, spatial_names):
        """Predict for all samples with best_params and save the per-step output pickle."""
        # predict (train/eval/test) and save outputs in original scale
        p_pred = build_param_matrix(  # predict parameters for all samples
            best_params["net"],  # network weights
            best_params["global"],  # global raw params (if any)
            x_features,  # normalized features
            npp_I_all,  # NPP forcing
            param_mins=param_mins,  # parameter lower bounds
            param_maxs=param_maxs,  # parameter upper bounds
            global_mask=global_mask)  # which params are global
        if use_dynamic:  # if dynamic mode
            pred_final = batched_sim(p_pred, y0_true)  # simulate final state
            pred_compare = pred_final - y0_true  # convert to delta for targets
        else:  # if steady-state
            pred_final = batched_steady(p_pred)  # compute steady-state
            pred_compare = pred_final  # targets for steady-state
        params_all, pred_final_all, pred_compare_all = map(jax.device_get, (p_pred, pred_final, pred_compare))  # move to numpy
        y0_derived = y0_true if use_dynamic else jnp.zeros_like(y0_true)
        # Derive predictions for all output labels (SOC, MAOC, MIC), even when unconstrained.
        pred_derived = jax.device_get(pools_to_loss_targets(jnp.asarray(pred_compare_all), y0_derived, use_dynamic, "SOC,MAOC,MIC"))
        lbl = output_labels
        out_cols = [f"target_{x}" for x in lbl] + [f"pred_{x}" for x in lbl] + [f"pred_final_{p}" for p in ["Cp", "Cb", "Cm"]] + [f"param_{n}" for n in param_names]
        df_out = pd.DataFrame(np.c_[all_target_values, pred_derived, pred_final_all, params_all], index=original_idx, columns=out_cols)
        df_out["split"] = split_col

        # save results
        os.makedirs("hybrid_outputs", exist_ok=True)  # ensure folder exists
        file_name = f"hybrid_temp{args.temp}_fold{args.fold}_md{args.md}_mt{args.mt}_sat{args.sat}_targets{args.targets.replace(',', '-')}_spatial{'none' if not spatial_names else '-'.join(spatial_names)}.pkl"
        df_out.to_pickle(os.path.join("hybrid_outputs", file_name)) # pickle
        print(f"total_time_sec {time.perf_counter() - start_time:.0f}: {file_name}")

    # greedy backward elimination: start with all sensitive params spatially varying and, at each step,
    # promote to global (learned as a single value) the remaining sensitive param whose promotion causes
    # the smallest drop (or largest gain) in mean validation R2. Continue until all sensitive params are global.
    max_steps = None
    if os.environ.get("HYBRID_MAX_GLOBAL_ITERS"):
        max_steps = int(os.environ["HYBRID_MAX_GLOBAL_ITERS"])  # cap number of saved steps (for quick testing)

    current_global = list(always_global)  # non-sensitive params are always global
    remaining = list(sensitive_params)    # sensitive params still spatial, candidates for promotion
    steps_done = 0

    # step 0: baseline with all sensitive params spatial
    global_mask, spatial_names = build_masks(current_global)
    print(f"[step 0] all-spatial | global: {current_global}")
    print(f"[step 0] spatial: {spatial_names}")
    best_params, baseline_r2 = train_model(global_mask)
    print(f"[step 0] mean val R2 {baseline_r2:.4f}")
    save_output(best_params, global_mask, spatial_names)
    steps_done += 1

    while remaining and (max_steps is None or steps_done < max_steps):
        trials = []  # (mean_val_r2, candidate, best_params, global_mask, spatial_names)
        for cand in remaining:
            trial_global = current_global + [cand]
            trial_mask, trial_spatial = build_masks(trial_global)
            print(f"[step {steps_done}] trial: promote {cand} to global -> global {trial_global}")
            trial_params, trial_r2 = train_model(trial_mask)
            print(f"[step {steps_done}] trial {cand}: mean val R2 {trial_r2:.4f}")
            trials.append((trial_r2, cand, trial_params, trial_mask, trial_spatial))
        # pick the promotion with the highest resulting mean val R2 (smallest drop / largest gain)
        best_r2, best_cand, best_params, best_mask, best_spatial = max(trials, key=lambda t: t[0])
        current_global.append(best_cand)
        remaining.remove(best_cand)
        print(f"[step {steps_done}] PROMOTE {best_cand} to global | mean val R2 {best_r2:.4f} | global now {current_global}")
        save_output(best_params, best_mask, best_spatial)
        steps_done += 1

if __name__ == "__main__":
    main()
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
from config import default_param_ranges, default_Q10_ranges, TARGET_CONFIG
import utils
from utils import vector_field, simulate_final_state, init_mlp, build_param_matrix, eval_loss, init_adam, eval_r2, train_step, constrain_to_range, build_hybrid_predictors


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
depth = 5
width = 20
batch_size = 1024
Q10_NAMES = list(default_Q10_ranges.keys())

TARGET_LABELS = {
    "SOC": ["SOC"],
    "SOC,MIC": ["SOC", "MIC"],
    "SOC,MAOC": ["SOC", "MAOC"],
    "SOC,MAOC,MIC": ["SOC", "MAOC", "MIC"],
}

MIN_STATIC_TARGET_ROWS = 1000
FORCED_GLOBAL_TSENSE = set(Q10_NAMES)

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
    model_sens = sensitivities[(sensitivities["md"] == args.md) & (sensitivities["mt"] == args.mt) & (sensitivities["sat"] == args.sat) & (sensitivities["temp"] == ("dynamic" if use_dynamic else "steady"))].iloc[0] # pick for this combination
    param_sens = model_sens.drop(labels=["md", "mt", "sat", "temp", "y0_Cp", "y0_Cb", "y0_Cm"] + [n for n in Q10_NAMES if n in model_sens.index]) # pick only parameters
    nonzero_params = [name for name, val in param_sens.items() if name != "I" and val != 0.0] # create list of non 0.0 (excluding I)
    sorted_params = sorted(nonzero_params, key=lambda name: abs(param_sens[name])) # sort this list by abs()
    n_global_iters = len(sorted_params) + 1
    if os.environ.get("HYBRID_MAX_GLOBAL_ITERS"):
        n_global_iters = min(n_global_iters, int(os.environ["HYBRID_MAX_GLOBAL_ITERS"]))
    q10_mins = jnp.array([default_Q10_ranges[n]["min"] for n in Q10_NAMES])
    q10_maxs = jnp.array([default_Q10_ranges[n]["max"] for n in Q10_NAMES])
    q10_raw_init = jnp.zeros((len(Q10_NAMES),))
    for i in range(n_global_iters): # loop over list
        global_names = [n for n, v in param_sens.items() if v == 0.0] + sorted_params[:i] # create which are to use global (0.0 and 0,1,2,3... of the ones in the list)
        param_names = list(default_param_ranges.keys())
        param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
        param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])
        global_names = sorted(set(name.strip() for name in global_names if name.strip()) | FORCED_GLOBAL_TSENSE)
        unknown_globals = sorted(set(global_names) - set(param_names))
        if unknown_globals:
            raise ValueError(f"Unknown global params: {', '.join(unknown_globals)}")
        global_mask = jnp.array([name in global_names for name in param_names])
        spatial_names = [name for name in param_names if name not in global_names and name != "I"]
        print('global parameters:', global_names)
        print('spatial parameters:', spatial_names)

        # build mechanistic models
        batched_steady = jax.vmap(partial(analytical_steady_state, microbial_decomposition=args.md, microbial_turnover=args.mt, saturation=args.sat)) # vmap analytical solution
        t0, t1 = 0.0, 9.0
        solver = dfx.Euler()
        model_fn = partial(craig_BA_adapt, microbial_decomposition=args.md, microbial_turnover=args.mt, saturation=args.sat)
        term = dfx.ODETerm(partial(vector_field, model_fn))
        batched_sim = jax.vmap(lambda p, y0: simulate_final_state(p, y0, t0, t1, dt0, term, solver)) # vmap solver

        # preprocess: get data, log some features & calculate stocks, get split indices, impute, create targets, normalize
        df = pd.read_pickle("1_preprocessed.pkl") # get data
        targets = args.targets.split(',') # target(s)
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
        required_cols = ["SOC", "POC", "MIC", "MAOC"] + target_source_cols + predictors + [input_col, 'era5_land_t2m_avg_09_15_18', 'split']
        helper_df = helper_df[list(dict.fromkeys(required_cols))]

        npp_mask = (helper_df[input_col].notna() & np.isfinite(helper_df[input_col]) & (helper_df[input_col] > 0)).to_numpy()
        original_idx = np.where(npp_mask)[0]
        helper_df = helper_df.loc[npp_mask].reset_index(drop=True)
        split_col = helper_df["split"].astype(str).to_numpy() # use same splits as in prediction
        if not use_dynamic:
            min_rows = max(MIN_STATIC_TARGET_ROWS, int(0.5 * len(helper_df)))
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
        label_mask = np.all(np.isfinite(target_values), axis=1)
        train_idx = np.where(label_mask & (split_col != "test") & (split_col != str(args.fold)))[0] # train on all folds except validation fold (and also not test)
        val_idx = np.where(label_mask & (split_col == str(args.fold)))[0] # validation indices
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError(f"No train/validation rows with finite targets for {args.targets}")
        helper_df = helper_df.fillna(helper_df.iloc[jax.device_get(train_idx)].median(numeric_only=True)) # impute predictors and initial-state helpers only
        if ta == "SOC":
            targets = jnp.asarray(target_values)
        if ta == "SOC,MIC":
            targets = jnp.asarray(target_values)
        if ta == "SOC,MAOC":
            targets = jnp.asarray(target_values)
        if ta == "SOC,MAOC,MIC":
            targets = jnp.asarray(target_values)

        # split 
        x_features = jnp.asarray(helper_df[predictors].to_numpy()) # df to np
        npp_I_all = jnp.asarray(helper_df[input_col].to_numpy())
        temp_all = jnp.asarray(helper_df["era5_land_t2m_avg_09_15_18"].to_numpy())
        x_train = x_features[train_idx]
        y_train = targets[train_idx]
        npp_I_train = npp_I_all[train_idx]
        temp_train = temp_all[train_idx]
        x_val = x_features[val_idx]
        y_val = targets[val_idx]
        npp_I_val = npp_I_all[val_idx]
        temp_val = temp_all[val_idx]
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

        # initilize ML
        net_params = init_mlp(jax.random.PRNGKey(0), [x_features.shape[1]] + [width] * depth + [param_mins.size]) # set up NN
        global_raw = jnp.zeros((param_mins.size,))
        params = {"net": net_params, "global": global_raw, "q10": q10_raw_init}
        n_targ = int(targets.shape[1])
        target_mask = jnp.ones((n_targ,))
        loss_ema, ema_beta, weights = jnp.ones((n_targ,)) * target_mask, 0.9, jnp.ones((n_targ,)) * target_mask # set up loss
        opt_state, best_params, best_test = init_adam(params), params, float("inf") # init optimizer and identification of best epoch
        best_step = 0
        
        # training
        init_y_loss = eval_loss(params, x_train, npp_I_train, y0_train, y_train, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std, targets_arg=args.targets, temp_batch=temp_train)[0]
        print(f"init y_loss {init_y_loss:.6f}")
        for step in range(1, n_steps + 1):
            k = jax.random.PRNGKey(step)
            batch_idx = jax.random.choice(k, train_idx.size, shape=(min(batch_size, train_idx.size),), replace=False)
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            y0_batch = y0_train[batch_idx] if use_dynamic else jnp.zeros((batch_idx.size, 3))
            npp_I_batch = npp_I_train[batch_idx]
            temp_batch = temp_train[batch_idx]
            warmup_scale = jnp.minimum(1.0, step / 200.0)
            lr_t = lr * warmup_scale * 0.5 * (1.0 + jnp.cos(jnp.pi * step / n_steps))
            params, opt_state, loss, per_component = train_step(params, opt_state, x_batch, npp_I_batch, y0_batch, y_batch, lr_t, step, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std, targets_arg=args.targets, temp_batch=temp_batch)
            loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * per_component
            weights = (1.0 / (loss_ema + 1e-8)) * target_mask
            if step % 50 == 0:
                val_r2 = eval_r2(params, x_val, npp_I_val, y0_val, y_val, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, targets_arg=args.targets, temp_batch=temp_val)
                val_r2_np = jax.device_get(val_r2)
                val_loss = eval_loss(params, x_val, npp_I_val, y0_val, y_val, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std, targets_arg=args.targets, temp_batch=temp_val)[0]
                if val_loss < best_test:
                    best_test = float(val_loss)
                    best_params = params
                    best_step = step
                r2_str = " ".join(f"{lb} {val_r2_np[i]:.3f}" for i, lb in enumerate(TARGET_LABELS[args.targets]))
                print(f"step {step} loss {loss:.6f} | R2 {r2_str}")
                if step - best_step >= early_stop_patience:
                    print(f"early stopping at step {step} (no val loss improvement for {early_stop_patience} epochs)")
                    break

        # predict (train/eval/test) and save outputs in original scale  # section header
        p_pred = build_param_matrix(  # predict parameters for all samples
            best_params["net"],  # network weights
            best_params["global"],  # global raw params (if any)
            x_features,  # normalized features
            npp_I_all,  # NPP forcing
            param_mins=param_mins,  # parameter lower bounds
            param_maxs=param_maxs,  # parameter upper bounds
            global_mask=global_mask,  # which params are global
            q10_raw=best_params["q10"],
            temp_batch=temp_all,
        )
        if use_dynamic:  # if dynamic mode
            pred_final = batched_sim(p_pred, y0_true)  # simulate final state
            pred_compare = pred_final - y0_true  # convert to delta for targets
        else:  # if steady-state
            pred_final = batched_steady(p_pred)  # compute steady-state
            pred_compare = pred_final  # targets for steady-state
        params_all, pred_final_all, pred_compare_all = map(jax.device_get, (p_pred, pred_final, pred_compare))  # move to numpy
        y0_derived = y0_true if use_dynamic else jnp.zeros_like(y0_true)
        pred_derived = jax.device_get(pools_to_loss_targets(jnp.asarray(pred_compare_all), y0_derived, use_dynamic, args.targets))
        lbl = TARGET_LABELS[args.targets]
        q10_vals = jax.device_get(constrain_to_range(best_params["q10"], q10_mins, q10_maxs))
        out_cols = [f"target_{x}" for x in lbl] + [f"pred_{x}" for x in lbl] + [f"pred_final_{p}" for p in ["Cp", "Cb", "Cm"]] + [f"param_{n}" for n in param_names] + [f"Q10_{n}" for n in Q10_NAMES]
        df_out = pd.DataFrame(np.c_[jax.device_get(targets), pred_derived, pred_final_all, params_all, np.tile(q10_vals, (len(original_idx), 1))], index=original_idx, columns=out_cols)
        df_out["split"] = split_col

        # save results
        os.makedirs("hybrid_outputs", exist_ok=True)  # ensure folder exists
        file_name = f"hybrid_Tsense_temp{args.temp}_fold{args.fold}_md{args.md}_mt{args.mt}_sat{args.sat}_targets{args.targets.replace(',', '-')}_spatial{'none' if not spatial_names else '-'.join(spatial_names)}.pkl"
        df_out.to_pickle(os.path.join("hybrid_outputs", file_name)) # pickle
        print(f"total_time_sec {time.perf_counter() - start_time:.0f}: {file_name}")

if __name__ == "__main__":
    main()

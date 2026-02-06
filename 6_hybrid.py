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
from hybrid_models import craig_BA_adapt, analytical_steady_state
from hybrid_config import default_param_ranges, predictors_dynamic, predictors_2015, predictors_2018, log_cols
from hybrid_utils import vector_field, simulate_final_state, init_mlp, build_param_matrix, normalize_targets, eval_loss, init_adam, eval_r2, train_step

dt0 = 0.05 # years
depth = 5
width = 512
lr = 2e-4
batch_size = 1024
n_steps = 3000 # epochs

def load_sensitivities():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "sensitivities.csv")
    pkl_path = os.path.join(base_dir, "sensitivities.pkl")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    try:
        return pd.read_pickle(pkl_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load sensitivities. "
            "If this is a pandas/pickle version mismatch, rerun "
            "`sensitivity_analysis.py` to regenerate `sensitivities.csv`."
        ) from exc

def plot_parity(test_df, true_prefix, pred_prefix, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=False)
    for ax, label in zip(axes, ["Cp", "Cb", "Cm"]):
        true_vals = test_df[f"{true_prefix}{label}"].to_numpy()
        pred_vals = test_df[f"{pred_prefix}{label}"].to_numpy()
        ax.hexbin(true_vals, pred_vals, gridsize=45, mincnt=1, cmap="viridis")
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray")
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        ax.set_title(f"{label} {title} (test)\nR2={r2:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    plt.tight_layout()
    fig.savefig(filename)

def log_and_stocks(df):
    helper_df = df.copy()
    for col in log_cols:
        helper_df[col] = np.log1p(helper_df[col])
    helper_df["y2015_Cm"] = helper_df["BD 0-20_2015_pred_median"] * helper_df["OC_sc_g_kg_2015_pred_median"]
    helper_df["y2015_Cb"] = helper_df["BD 0-20_2015_pred_median"] * helper_df["Cmic_2015_pred_median"]
    helper_df["y2015_Cp"] = helper_df["BD 0-20_2015_pred_median"] * (helper_df["OC_2015"] - helper_df["OC_sc_g_kg_2015_pred_median"] - helper_df["Cmic_2015_pred_median"])
    helper_df["y2018_Cm"] = helper_df["BD 0-20_2018_pred_median"] * helper_df["OC_sc_g_kg_2018_pred_median"]
    helper_df["y2018_Cb"] = helper_df["BD 0-20_2018_pred_median"] * helper_df["Cmic_2018_pred_median"]
    helper_df["y2018_Cp"] = helper_df["BD 0-20_2018_pred_median"] * (helper_df["OC_2018"] - helper_df["OC_sc_g_kg_2018_pred_median"] - helper_df["Cmic_2018_pred_median"])
    return helper_df

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--temp"); parser.add_argument("--fold"); parser.add_argument("--md"); parser.add_argument("--mt"); parser.add_argument("--sat"); parser.add_argument("--targets"); args = parser.parse_args()
    start_time = time.perf_counter()
    use_dynamic = args.temp == "2015_2018"
    target_mask = {
        "Cp,Cb,Cm": jnp.array([1.0, 1.0, 1.0]),
        "Cp,Cb": jnp.array([1.0, 1.0, 0.0]),
        "Cp,Cm": jnp.array([1.0, 0.0, 1.0]),
        "Cp": jnp.array([1.0, 0.0, 0.0]),
    }[args.targets]

    sensitivities = load_sensitivities() # load importances
    model_sens = sensitivities[(sensitivities["md"] == args.md) & (sensitivities["mt"] == args.mt) & (sensitivities["sat"] == args.sat) & (sensitivities["temp"] == ("dynamic" if use_dynamic else "steady"))].iloc[0] # pick for this combination
    param_sens = model_sens.drop(labels=["md", "mt", "sat", "temp"]) # pick only parameters
    nonzero_params = [name for name, val in param_sens.items() if name != "I" and val != 0.0] # create list of non 0.0 (excluding I)
    sorted_params = sorted(nonzero_params, key=lambda name: abs(param_sens[name]), reverse=True) # sort this list by abs()
    for i, name in enumerate(sorted_params, start=0): # loop over list
        global_names = [n for n, v in param_sens.items() if n != "I" and v == 0.0] + sorted_params[:i] # create which are to use global (0.0 and 0,1,2,3... of the ones in the list)
        print(global_names)

        param_names = list(default_param_ranges.keys())
        param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
        param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])
        global_names = [name.strip() for name in global_names if name.strip()]
        unknown_globals = sorted(set(global_names) - set(param_names))
        if unknown_globals:
            raise ValueError(f"Unknown global params: {', '.join(unknown_globals)}")
        global_mask = jnp.array([name in global_names for name in param_names])

        # build mechanistic models
        batched_steady = jax.vmap(partial(analytical_steady_state, microbial_decomposition=args.md, microbial_turnover=args.mt, saturation=args.sat)) # vmap analytical solution
        t0, t1 = 0.0, 3.0
        solver = dfx.Euler()
        model_fn = partial(craig_BA_adapt, microbial_decomposition=args.md, microbial_turnover=args.mt, saturation=args.sat)
        term = dfx.ODETerm(partial(vector_field, model_fn))
        batched_sim = jax.vmap(lambda p, y0: simulate_final_state(p, y0, t0, t1, dt0, term, solver)) # vmap solver

        # preprocess: get data, log some features & calculate stocks, get split indices, impute, create targets, normalize
        df = pd.read_pickle("5_with_predictions.pkl") # get data
        helper_df = log_and_stocks(df) # log specified variables and calculate stocks
        predictors = {"2015_2018": predictors_dynamic, "2015": predictors_2015, "2018": predictors_2018}[args.temp] # use only respective predictors
        npp_year = "2015" if args.temp in {"2015", "2015_2018"} else "2018"
        npp_col = f"MODIS_NPP_{npp_year}gps_{npp_year}"
        npp_mask = (
            helper_df[npp_col].notna()
            & np.isfinite(helper_df[npp_col])
            & (helper_df[npp_col] > 0)
        ).to_numpy()
        original_idx = np.where(npp_mask)[0]
        helper_df = helper_df.loc[npp_mask].reset_index(drop=True)
        split_col = helper_df["split"].astype(str).to_numpy() # use same splits as in prediction
        train_idx = np.where((split_col != "test") & (split_col != str(args.fold)))[0] # train on all folds ecxept validation fold (and also not test)
        val_idx = np.where(split_col == str(args.fold))[0] # validation indices
        helper_df = helper_df.fillna(helper_df.iloc[jax.device_get(train_idx)].median(numeric_only=True)) # impute empty with median
        if use_dynamic: # create targets
            targets = jnp.asarray(helper_df[["y2018_Cp", "y2018_Cb", "y2018_Cm"]].to_numpy() - helper_df[["y2015_Cp", "y2015_Cb", "y2015_Cm"]].to_numpy())
        else:
            targets = jnp.asarray(helper_df[[f"y{args.temp}_Cp", f"y{args.temp}_Cb", f"y{args.temp}_Cm"]].to_numpy())
        # split 
        x_features = jnp.asarray(helper_df[predictors].to_numpy()) # df to np
        npp_I_all = jnp.asarray(helper_df[npp_col].to_numpy())
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
        y0_true = jnp.asarray(helper_df[["y2015_Cp", "y2015_Cb", "y2015_Cm"]].to_numpy()) 
        y0_train, y0_val = y0_true[train_idx], y0_true[val_idx]

        # initilize ML
        net_params = init_mlp(jax.random.PRNGKey(0), [x_features.shape[1]] + [width] * depth + [param_mins.size]) # set up NN
        global_raw = jnp.zeros((param_mins.size,))
        params = {"net": net_params, "global": global_raw}
        loss_ema, ema_beta, weights = jnp.ones((3,)) * target_mask, 0.9, jnp.ones((3,)) * target_mask # set up loss
        opt_state, best_params, best_test = init_adam(params), params, float("inf") # init optimizer and identification of best epoch
        best_step = 0
        early_stop_patience = 500
        
        # training
        init_y_loss = eval_loss(params, x_train, npp_I_train, y0_train, y_train, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std)[0]
        print(f"init y_loss {init_y_loss:.6f}")
        for step in range(1, n_steps + 1):
            k = jax.random.PRNGKey(step)
            batch_idx = jax.random.choice(k, train_idx.size, shape=(batch_size,), replace=False)
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            y0_batch = y0_train[batch_idx] if use_dynamic else jnp.zeros((batch_idx.size, 3))
            npp_I_batch = npp_I_train[batch_idx]
            warmup_scale = jnp.minimum(1.0, step / 200.0)
            lr_t = lr * warmup_scale * 0.5 * (1.0 + jnp.cos(jnp.pi * step / n_steps))
            params, opt_state, loss, per_component = train_step(params, opt_state, x_batch, npp_I_batch, y0_batch, y_batch, lr_t, step, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std)
            loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * per_component
            weights = (1.0 / (loss_ema + 1e-8)) * target_mask
            if step % 50 == 0:
                val_r2 = eval_r2(params, x_val, npp_I_val, y0_val, y_val, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady)
                val_r2_np = jax.device_get(val_r2)
                val_loss = eval_loss(params, x_val, npp_I_val, y0_val, y_val, weights, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask, use_dynamic=use_dynamic, batched_sim=batched_sim, batched_steady=batched_steady, target_mean=target_mean, target_std=target_std)[0]
                if val_loss < best_test:
                    best_test = float(val_loss)
                    best_params = params
                    best_step = step
                print(f"step {step} loss {loss:.6f} | R2 Cp {val_r2_np[0]:.3f} Cb {val_r2_np[1]:.3f} Cm {val_r2_np[2]:.3f}")
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
            global_mask=global_mask)  # which params are global
        if use_dynamic:  # if dynamic mode
            pred_final = batched_sim(p_pred, y0_true)  # simulate final state
            pred_compare = pred_final - y0_true  # convert to delta for targets
        else:  # if steady-state
            pred_final = batched_steady(p_pred)  # compute steady-state
            pred_compare = pred_final  # targets for steady-state
        params_all, pred_final_all, pred_compare_all = map(jax.device_get, (p_pred, pred_final, pred_compare))  # move to numpy
        out_cols = [f"target_{l}" for l in ["Cp", "Cb", "Cm"]] + [f"pred_{l}" for l in ["Cp", "Cb", "Cm"]] + [f"pred_final_{l}" for l in ["Cp", "Cb", "Cm"]] + [f"param_{n}" for n in param_names]  # output columns
        df_out = pd.DataFrame(np.c_[jax.device_get(targets), pred_compare_all, pred_final_all, params_all], index=original_idx, columns=out_cols)

        # save results
        os.makedirs("6_hybrid_outputs", exist_ok=True)  # ensure folder exists
        file_name = f"hybrid_temp{args.temp}_fold{args.fold}_md{args.md}_mt{args.mt}_sat{args.sat}_targets{args.targets.replace(',', '-')}_globals{'none' if not global_names else '-'.join(global_names)}.pkl"
        df_out.to_pickle(os.path.join("6_hybrid_outputs", file_name)) # pickle
        print(f"total_time_sec {time.perf_counter() - start_time:.0f}: {file_name}")

if __name__ == "__main__":
    main()
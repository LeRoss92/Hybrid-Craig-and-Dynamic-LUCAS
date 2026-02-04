import jax
import jax.numpy as jnp
import diffrax as dfx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def craig_BA_adapt(
    t,
    y,
    p,
    microbial_decomposition="linear",
    microbial_turnover="linear",
    carbon_use_efficiency="constant",
    saturation="no",
    ):
    Cp, Cb, Cm = y
    (
        I,
        CUE,
        beta,
        tmb,
        Cg0b,
        Cg0m,
        qx,
        Vmax_p,
        Vmax_m,
        Km_p,
        Km_m,
        kp,
        kb,
        km,
    ) = p

    if microbial_decomposition == "linear":
        def mic_dec(k_i, Vmax_i, Cb_i, Km_i, C_i):
            return k_i * C_i
    elif microbial_decomposition == "MM":
        def mic_dec(k_i, Vmax_i, Cb_i, Km_i, C_i):
            return C_i * Vmax_i * Cb_i / (Km_i + C_i)
    elif microbial_decomposition == "RMM":
        def mic_dec(k_i, Vmax_i, Cb_i, Km_i, C_i):
            return C_i * Vmax_i * Cb_i / (Km_i + Cb_i)

    if microbial_turnover == "linear":
        def mic_tur(k_b, C_b, beta_i):
            return k_b * C_b
    elif microbial_turnover == "density_dependent":
        def mic_tur(k_b, C_b, beta_i):
            return k_b * C_b ** beta_i

    if carbon_use_efficiency == "constant":
        def ca_us_ef(CUE_i, Cg0b_i, C_b):
            return CUE_i
    elif carbon_use_efficiency == "density_dependent":
        def ca_us_ef(CUE_i, Cg0b_i, C_b):
            return CUE_i * (1 - C_b / Cg0b_i)

    if saturation == "no":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            return tmb_i
    elif saturation == "Langmuir":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            return 1 - C_m / Cg0m_i
    elif saturation == "exponential":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            return tmb_i * jnp.exp(-C_m / Cg0m_i)
    elif saturation == "MM":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            return tmb_i * Cg0m_i / (Cg0m_i + C_m)
    elif saturation == "freundlich":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            n = 0.7 * qx_i
            return tmb_i * (1 - (C_m / Cg0m_i) ** n)
    elif saturation == "step":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            k = 50.0 * qx_i
            return tmb_i / (1 + jnp.exp(k * (C_m - Cg0m_i) / Cg0m_i))
    elif saturation == "logistic":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            k = 10.0 * qx_i
            return tmb_i / (1 + jnp.exp(k * (C_m - Cg0m_i) / Cg0m_i))
    elif saturation == "power":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            alpha = 0.5 * qx_i
            return tmb_i * (1 - (C_m / Cg0m_i) ** alpha)
    elif saturation == "linear_threshold":
        def sat(tmb_i, Cg0m_i, C_m, qx_i):
            threshold = 0.8 * qx_i * Cg0m_i
            k = 30.0 * qx_i
            weight_before = 1.0 / (1 + jnp.exp(k * (C_m - threshold) / Cg0m_i))
            weight_after = 1.0 - weight_before

            value_before = tmb_i
            denom = jnp.maximum(Cg0m_i - threshold, 1e-10)
            linear_factor = jnp.maximum(0, 1 - (C_m - threshold) / denom)
            value_after = tmb_i * linear_factor

            return value_before * weight_before + value_after * weight_after

    saturation_fraction = sat(tmb, Cg0m, Cm, qx)
    total_turnover = mic_tur(kb, Cb, beta)

    to_Cm = saturation_fraction * total_turnover
    to_Cp = total_turnover - to_Cm

    dCpdt = I - mic_dec(kp, Vmax_p, Cb, Km_p, Cp) + to_Cp
    dCbdt = (
        ca_us_ef(CUE, Cg0b, Cb) * mic_dec(kp, Vmax_p, Cb, Km_p, Cp)
        + ca_us_ef(CUE, Cg0b, Cb) * mic_dec(km, Vmax_m, Cb, Km_m, Cm)
        - total_turnover
    )
    dCmdt = to_Cm - mic_dec(km, Vmax_m, Cb, Km_m, Cm)

    return jnp.array([dCpdt, dCbdt, dCmdt])


def vector_field(t, y, args):
    return craig_BA_adapt(t, y, args)


def simulate_final_state(params, y0, t0, t1, dt0, term, solver):
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=params,
        saveat=dfx.SaveAt(t1=True),
    )
    return solution.ys[0]

def init_mlp(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes))
    last_idx = len(sizes) - 2
    for i, (k, (in_dim, out_dim)) in enumerate(zip(keys, zip(sizes[:-1], sizes[1:]))):
        if i == last_idx:
            w = jax.random.normal(k, (in_dim, out_dim)) * 0.01
        else:
            w = jax.random.normal(k, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
        b = jnp.zeros((out_dim,))
        params.append((w, b))
    return params

def mlp_forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return x @ w + b

def constrain_to_range(raw, mins, maxs):
    return mins + (maxs - mins) * jax.nn.sigmoid(raw)

def main():
    # Setup
    key = jax.random.PRNGKey(0)
    t0 = 0.0
    t1 = 3.0
    dt0 = 0.1
    term = dfx.ODETerm(vector_field)
    solver = dfx.Euler()

    default_param_ranges = {
        "I": {"min": 0.05, "max": 2.0},                             # "I": {"min": 0.0, "max": 0.5},
        "CUE": {"min": 0.1, "max": 0.8},                            # "CUE": {"min": 0.2, "max": 0.8},
        "beta": {"min": 0.5, "max": 2.5},                           # "beta": {"min": 0.8, "max": 2.0},
        "tmb": {"min": 0.1, "max": 1.0},                            # "tmb": {"min": 0.3, "max": 1.2},
        "Cg0b": {"min": 0.0005, "max": 10.0},                       # "Cg0b": {"min": 5.0, "max": 20.0},
        "Cg0m": {"min": 0.5 * 1.3, "max": 150 * 1.3},               # "Cg0m": {"min": 5.0, "max": 20.0},
        "qx": {"min": 0.1, "max": 10.0},                            # "qx": {"min": 0.5, "max": 2.0},
        "Vmax_p": {"min": 88 * 0.1, "max": 88 * 10.0},              # "Vmax_p": {"min": 0.1, "max": 1.0},
        "Vmax_m": {"min": 171 * 0.1, "max": 171 * 10.0},            # "Vmax_m": {"min": 0.1, "max": 1.0},
        "Km_p": {"min": 144 * 0.1 * 1.3, "max": 144 * 10.0 * 1.3},  # "Km_p": {"min": 0.5, "max": 2.0},
        "Km_m": {"min": 936 * 0.1 * 1.3, "max": 936 * 10.0 * 1.3},  # "Km_m": {"min": 0.5, "max": 2.0},
        "kp": {"min": 0.3 * 0.1, "max": 0.3 * 10.0},                # "kp": {"min": 0.01, "max": 0.2},
        "kb": {"min": 2 * 0.01, "max": 2 * 1.0},                    # "kb": {"min": 0.01, "max": 0.2},
        "km": {"min": 0.09 * 0.1, "max": 0.09 * 10.0},              # "km": {"min": 0.01, "max": 0.2},
    }
    default_state_bounds = {
        "Cp": {"min": 1.0, "max": 200.0},                           # "Cp": {"min": 0.1, "max": 2.0},
        "Cb": {"min": 0.00005, "max": 6.0},                         # "Cb": {"min": 0.1, "max": 2.0},
        "Cm": {"min": 0.05, "max": 80.0},                           # "Cm": {"min": 0.1, "max": 2.0},
    }
    param_names = list(default_param_ranges.keys())
    y0_names = list(default_state_bounds.keys())
    param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
    param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])
    y0_mins = jnp.array([default_state_bounds[name]["min"] for name in y0_names])
    y0_maxs = jnp.array([default_state_bounds[name]["max"] for name in y0_names])

    # Dataset generation from ground truth data
    df = pd.read_pickle("5_with_predictions.pkl")

    predictors = (
        ["Clay", "Silt", "Sand", "Coarse"]
        + ["Ox_Al_2018", "Ox_Fe_2018"]
        + [x + "_2015" for x in ["Calc", "pH_c", "pH"]]
        + [x + "_2015" for x in ["CaCO3_c"]]  # logged
        + [f"MODIS_NPP_2015gps_{y}" for y in [2014, 2014, 2015]]
        + [f"BIOCLIM_2015gps_{i}" for i in range(1, 20)]
        + [x + "_2015" for x in ["N", "P", "K", "CNr", "CPr", "CKr"]]  # logged
        + [f"AE{str(i).zfill(2)}_2018gps_2017" for i in range(64)]
        + [f"AE{str(i).zfill(2)}_2018gps_2018" for i in range(64)]
    )

    log_cols = [
        "CaCO3_c_2015",
        "N_2015",
        "P_2015",
        "K_2015",
        "CNr_2015",
        "CPr_2015",
        "CKr_2015",
    ]
    df = df.copy()
    for col in log_cols:
        df[col] = np.log1p(df[col])

    df["y0_Cm"] = df["BD 0-20_2015_pred_median"] * df["OC_sc_g_kg_2015_pred_median"]
    df["y0_Cb"] = df["BD 0-20_2015_pred_median"] * df["Cmic_2015_pred_median"]
    df["y0_Cp"] = df["BD 0-20_2015_pred_median"] * (
        df["OC_2015"] - df["OC_sc_g_kg_2015_pred_median"] - df["Cmic_2015_pred_median"]
    )

    df["final_Cm"] = df["BD 0-20_2018_pred_median"] * df["OC_sc_g_kg_2018_pred_median"]
    df["final_Cb"] = df["BD 0-20_2018_pred_median"] * df["Cmic_2018_pred_median"]
    df["final_Cp"] = df["BD 0-20_2018_pred_median"] * (
        df["OC_2018"] - df["OC_sc_g_kg_2018_pred_median"] - df["Cmic_2018_pred_median"]
    )

    predictors = predictors + ["y0_Cp", "y0_Cb", "y0_Cm"]

    required_cols = list(
        dict.fromkeys(
            predictors
            + [
                "y0_Cp",
                "y0_Cb",
                "y0_Cm",
                "final_Cp",
                "final_Cb",
                "final_Cm",
            ]
        )
    )
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    df = df[required_cols].copy()

    n_samples = df.shape[0]
    test_frac = 0.2
    key, split_key = jax.random.split(key)
    perm = jax.random.permutation(split_key, n_samples)
    n_test = max(1, int(n_samples * test_frac))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_df = df.iloc[jax.device_get(train_idx)]
    train_medians = train_df.median(numeric_only=True)
    df = df.fillna(train_medians)
    df = df.rename(
        columns={
            "final_Cp": "final_true_Cp",
            "final_Cb": "final_true_Cb",
            "final_Cm": "final_true_Cm",
        }
    )

    x_features = jnp.asarray(df[predictors].to_numpy())
    feature_dim = x_features.shape[1]
    y0_true = jnp.asarray(df[["y0_Cp", "y0_Cb", "y0_Cm"]].to_numpy())
    targets = jnp.asarray(
        df[["final_true_Cp", "final_true_Cb", "final_true_Cm"]].to_numpy()
        - df[["y0_Cp", "y0_Cb", "y0_Cm"]].to_numpy()
    )

    batched_sim = jax.vmap(
        lambda p, y0: simulate_final_state(p, y0, t0, t1, dt0, term, solver)
    )

    # Training
    key, init_key = jax.random.split(key)
    depth = 5
    width = 512
    net_params = init_mlp(
        init_key, [feature_dim] + [width] * depth + [param_mins.size]
    )

    lr = 2e-4
    batch_size = 1024
    n_steps = 3000
    key, idx_key = jax.random.split(key)

    feature_cols = predictors
    x_train = x_features[train_idx]
    y_train = targets[train_idx]
    x_test = x_features[test_idx]
    y_test = targets[test_idx]

    x_mean = jnp.mean(x_train, axis=0)
    x_std = jnp.std(x_train, axis=0) + 1e-8
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    x_features = (x_features - x_mean) / x_std

    target_mean = jnp.mean(y_train, axis=0)
    target_std = jnp.std(y_train, axis=0) + 1e-8

    def normalize_targets(y):
        return (y - target_mean) / target_std

    def eval_components(params, x_batch, y0_batch, y_target):
        raw = jax.vmap(lambda x: mlp_forward(params, x))(x_batch)
        p_pred = constrain_to_range(raw, param_mins, param_maxs)
        y_pred = batched_sim(p_pred, y0_batch)
        y_pred_delta = y_pred - y0_batch
        y_pred_norm = normalize_targets(y_pred_delta)
        y_target_norm = normalize_targets(y_target)
        diff = y_pred_norm - y_target_norm
        delta = 0.5
        abs_diff = jnp.abs(diff)
        huber = jnp.where(abs_diff <= delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
        return jnp.mean(huber, axis=0)

    def eval_loss(params, x_batch, y0_batch, y_target, weights):
        per_component = eval_components(params, x_batch, y0_batch, y_target)
        weights = weights / (jnp.sum(weights) + 1e-8)
        loss = jnp.sum(per_component * weights)
        return loss, per_component

    def init_adam(params):
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        return m, v

    def clip_by_global_norm(grads, max_norm):
        leaves = jax.tree_util.tree_leaves(grads)
        norm = jnp.sqrt(jnp.sum(jnp.array([jnp.sum(g ** 2) for g in leaves])))
        scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
        return jax.tree_util.tree_map(lambda g: g * scale, grads)

    def l2_norm(params):
        leaves = jax.tree_util.tree_leaves(params)
        return jnp.sum(jnp.array([jnp.sum(p ** 2) for p in leaves]))

    def eval_r2(params, x_batch, y0_batch, y_target):
        raw = jax.vmap(lambda x: mlp_forward(params, x))(x_batch)
        p_pred = constrain_to_range(raw, param_mins, param_maxs)
        y_pred = batched_sim(p_pred, y0_batch)
        y_pred_delta = y_pred - y0_batch
        ss_res = jnp.sum((y_target - y_pred_delta) ** 2, axis=0)
        ss_tot = jnp.sum((y_target - jnp.mean(y_target, axis=0)) ** 2, axis=0)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    @jax.jit
    def train_step(params, opt_state, x_batch, y0_batch, y_batch, lr_t, step, weights):
        (loss, per_component), grads = jax.value_and_grad(eval_loss, has_aux=True)(
            params, x_batch, y0_batch, y_batch, weights
        )
        weight_decay = 1e-4
        loss = loss + weight_decay * l2_norm(params)
        grads = clip_by_global_norm(grads, 1.0)
        m, v = opt_state
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i, m, grads)
        v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1 - beta2) * (g_i ** 2), v, grads)
        m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1 - beta1 ** step), m)
        v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1 - beta2 ** step), v)
        params = jax.tree_util.tree_map(
            lambda p, m_i, v_i: p - lr_t * m_i / (jnp.sqrt(v_i) + eps),
            params,
            m_hat,
            v_hat,
        )
        return params, (m, v), loss, per_component

    loss_ema = jnp.ones((3,))
    ema_beta = 0.9
    weights = jnp.ones((3,))
    init_y_loss = eval_loss(net_params, x_train, y0_true[train_idx], y_train, weights)[0]
    print(f"init y_loss {init_y_loss:.6f}")

    opt_state = init_adam(net_params)
    best_params = net_params
    best_test = float("inf")

    for step in range(1, n_steps + 1):
        idx_key, k = jax.random.split(idx_key)
        batch_idx = jax.random.choice(k, train_idx.size, shape=(batch_size,), replace=False)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]
        y0_batch = y0_true[train_idx][batch_idx]
        warmup = 200.0
        warmup_scale = jnp.minimum(1.0, step / warmup)
        lr_t = lr * warmup_scale * 0.5 * (1.0 + jnp.cos(jnp.pi * step / n_steps))
        net_params, opt_state, loss, per_component = train_step(
            net_params, opt_state, x_batch, y0_batch, y_batch, lr_t, step, weights
        )
        loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * per_component
        weights = 1.0 / (loss_ema + 1e-8)
        if step % 50 == 0:
            test_r2 = eval_r2(net_params, x_test, y0_true[test_idx], y_test)
            test_r2_np = jax.device_get(test_r2)
            print(
                f"step {step} loss {loss:.6f} | "
                f"R2 Cp {test_r2_np[0]:.3f} Cb {test_r2_np[1]:.3f} Cm {test_r2_np[2]:.3f}"
            )
        if step % 200 == 0:
            test_loss = eval_loss(net_params, x_test, y0_true[test_idx], y_test, weights)[0]
            if test_loss < best_test:
                best_test = float(test_loss)
                best_params = net_params

    final_train_loss = eval_loss(best_params, x_train, y0_true[train_idx], y_train, weights)[0]
    final_test_loss = eval_loss(best_params, x_test, y0_true[test_idx], y_test, weights)[0]
    print(f"Final train loss: {final_train_loss:.6f}")
    print(f"Final test loss: {final_test_loss:.6f}")

    raw = jax.vmap(lambda x: mlp_forward(best_params, x))(x_features)
    p_pred = constrain_to_range(raw, param_mins, param_maxs)
    pred_final = batched_sim(p_pred, y0_true)
    pred_delta = pred_final - y0_true

    abs_err = jnp.abs(pred_delta - targets)
    rmse = jnp.sqrt(jnp.mean((pred_delta - targets) ** 2, axis=0))
    mae = jnp.mean(abs_err, axis=0)
    mae_np = jax.device_get(mae)
    rmse_np = jax.device_get(rmse)
    print(
        "Delta MAE (Cp, Cb, Cm): "
        f"{mae_np[0]:.4f}, {mae_np[1]:.4f}, {mae_np[2]:.4f}"
    )
    print(
        "Delta RMSE (Cp, Cb, Cm): "
        f"{rmse_np[0]:.4f}, {rmse_np[1]:.4f}, {rmse_np[2]:.4f}"
    )
    pred_delta_norm = normalize_targets(pred_delta)
    targets_norm = normalize_targets(targets)
    test_idx_np = jax.device_get(test_idx)
    split = np.array(["train"] * n_samples, dtype=object)
    split[test_idx_np] = "test"
    new_cols = {
        "final_pred_Cp": jax.device_get(pred_final[:, 0]),
        "final_pred_Cb": jax.device_get(pred_final[:, 1]),
        "final_pred_Cm": jax.device_get(pred_final[:, 2]),
        "delta_true_Cp_norm": jax.device_get(targets_norm[:, 0]),
        "delta_true_Cb_norm": jax.device_get(targets_norm[:, 1]),
        "delta_true_Cm_norm": jax.device_get(targets_norm[:, 2]),
        "delta_pred_Cp_norm": jax.device_get(pred_delta_norm[:, 0]),
        "delta_pred_Cb_norm": jax.device_get(pred_delta_norm[:, 1]),
        "delta_pred_Cm_norm": jax.device_get(pred_delta_norm[:, 2]),
        "split": split,
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(df)

    test_df = df[df["split"] == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=False)
    pairs = [
        ("final_true_Cp", "final_pred_Cp", "Cp"),
        ("final_true_Cb", "final_pred_Cb", "Cb"),
        ("final_true_Cm", "final_pred_Cm", "Cm"),
    ]
    for ax, (true_col, pred_col, label) in zip(axes, pairs):
        true_vals = test_df[true_col].to_numpy()
        pred_vals = test_df[pred_col].to_numpy()
        ax.hexbin(true_vals, pred_vals, gridsize=45, mincnt=1, cmap="viridis")
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray")
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        ax.set_title(f"{label} parity (test)\nR2={r2:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    plt.tight_layout()
    fig.savefig("figures/parity.png")

    test_df = test_df.copy()
    for pool in ["Cp", "Cb", "Cm"]:
        test_df[f"delta_true_{pool}"] = test_df[f"final_true_{pool}"] - test_df[f"y0_{pool}"]
        test_df[f"delta_pred_{pool}"] = test_df[f"final_pred_{pool}"] - test_df[f"y0_{pool}"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=False)
    delta_pairs = [
        ("delta_true_Cp", "delta_pred_Cp", "Cp"),
        ("delta_true_Cb", "delta_pred_Cb", "Cb"),
        ("delta_true_Cm", "delta_pred_Cm", "Cm"),
    ]
    for ax, (true_col, pred_col, label) in zip(axes, delta_pairs):
        true_vals = test_df[true_col].to_numpy()
        pred_vals = test_df[pred_col].to_numpy()
        ax.hexbin(true_vals, pred_vals, gridsize=45, mincnt=1, cmap="viridis")
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray")
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        ax.set_title(f"{label} delta parity (test)\nR2={r2:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    plt.tight_layout()
    fig.savefig("figures/parity_delta.png")


if __name__ == "__main__":
    main()

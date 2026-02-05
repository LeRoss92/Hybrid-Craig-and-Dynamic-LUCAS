from functools import partial

import jax
import jax.numpy as jnp
import diffrax as dfx


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


def normalize_targets(y, target_mean, target_std):
    return (y - target_mean) / target_std


def build_param_matrix(
    net_params,
    global_raw,
    x_batch,
    npp_I_batch,
    *,
    param_mins,
    param_maxs,
    global_mask,
):
    raw_local = jax.vmap(lambda x: mlp_forward(net_params, x))(x_batch)
    local_params = constrain_to_range(raw_local, param_mins, param_maxs)
    if global_raw is not None:
        global_params = constrain_to_range(global_raw, param_mins, param_maxs)
        local_params = jnp.where(global_mask, global_params, local_params)
    local_params = local_params.at[:, 0].set(npp_I_batch)
    return local_params


def eval_components(
    params,
    x_batch,
    npp_I_batch,
    y0_batch,
    y_target,
    *,
    param_mins,
    param_maxs,
    global_mask,
    use_dynamic,
    batched_sim,
    batched_steady,
    target_mean,
    target_std,
    ):
    p_pred = build_param_matrix(
        params["net"],
        params["global"],
        x_batch,
        npp_I_batch,
        param_mins=param_mins,
        param_maxs=param_maxs,
        global_mask=global_mask,
    )
    if use_dynamic:
        y_pred = batched_sim(p_pred, y0_batch)
        y_pred_compare = y_pred - y0_batch
    else:
        y_pred_compare = batched_steady(p_pred)
    y_pred_norm = normalize_targets(y_pred_compare, target_mean, target_std)
    y_target_norm = normalize_targets(y_target, target_mean, target_std)
    diff = y_pred_norm - y_target_norm
    delta = 0.5
    abs_diff = jnp.abs(diff)
    huber = jnp.where(abs_diff <= delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    return jnp.mean(huber, axis=0)


def eval_loss(
    params,
    x_batch,
    npp_I_batch,
    y0_batch,
    y_target,
    weights,
    *,
    param_mins,
    param_maxs,
    global_mask,
    use_dynamic,
    batched_sim,
    batched_steady,
    target_mean,
    target_std,
    ):
    per_component = eval_components(
        params,
        x_batch,
        npp_I_batch,
        y0_batch,
        y_target,
        param_mins=param_mins,
        param_maxs=param_maxs,
        global_mask=global_mask,
        use_dynamic=use_dynamic,
        batched_sim=batched_sim,
        batched_steady=batched_steady,
        target_mean=target_mean,
        target_std=target_std,
    )
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


def eval_r2(
    params,
    x_batch,
    npp_I_batch,
    y0_batch,
    y_target,
    *,
    param_mins,
    param_maxs,
    global_mask,
    use_dynamic,
    batched_sim,
    batched_steady,
):
    p_pred = build_param_matrix(
        params["net"],
        params["global"],
        x_batch,
        npp_I_batch,
        param_mins=param_mins,
        param_maxs=param_maxs,
        global_mask=global_mask,
    )
    if use_dynamic:
        y_pred = batched_sim(p_pred, y0_batch)
        y_pred_compare = y_pred - y0_batch
    else:
        y_pred_compare = batched_steady(p_pred)
    ss_res = jnp.sum((y_target - y_pred_compare) ** 2, axis=0)
    ss_tot = jnp.sum((y_target - jnp.mean(y_target, axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@partial(jax.jit, static_argnames=("use_dynamic", "batched_sim", "batched_steady"))
def train_step(
    params,
    opt_state,
    x_batch,
    npp_I_batch,
    y0_batch,
    y_batch,
    lr_t,
    step,
    weights,
    *,
    param_mins,
    param_maxs,
    global_mask,
    use_dynamic,
    batched_sim,
    batched_steady,
    target_mean,
    target_std,
):
    (loss, per_component), grads = jax.value_and_grad(eval_loss, has_aux=True)(
        params,
        x_batch,
        npp_I_batch,
        y0_batch,
        y_batch,
        weights,
        param_mins=param_mins,
        param_maxs=param_maxs,
        global_mask=global_mask,
        use_dynamic=use_dynamic,
        batched_sim=batched_sim,
        batched_steady=batched_steady,
        target_mean=target_mean,
        target_std=target_std,
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


def vector_field(model_fn, t, y, args):
    return model_fn(t, y, args)


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

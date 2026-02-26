import jax
import jax.numpy as jnp
from itertools import product
import diffrax as dfx
import numpy as np
import pandas as pd

from config import default_param_ranges, default_state_bounds
from hybrid_models import craig_BA_adapt, analytical_steady_state

# get ranges
state_names = list(default_state_bounds.keys())
state_mins = jnp.array([default_state_bounds[name]["min"] for name in state_names])
state_maxs = jnp.array([default_state_bounds[name]["max"] for name in state_names])
state_diffs = state_maxs - state_mins  # difference between max and min for state variables
SOC_diff = jnp.sum(state_diffs)
param_names = list(default_param_ranges.keys())
param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])
param_diffs = param_maxs - param_mins  # difference between max and min for parameters
# create random initial conditions and params
key_params, key_states = jax.random.split(jax.random.PRNGKey(0), 2)
n_samples = 1000000
random_params = jax.random.uniform(key_params, shape=(n_samples, len(param_names))) # 0-1
random_states = jax.random.uniform(key_states, shape=(n_samples, len(state_names))) # 0-1
random_params = param_mins + (param_maxs - param_mins) * random_params # scaled right
random_states = state_mins + (state_maxs - state_mins) * random_states # scaled right
# loop over versions & over temps
results = []
for dec, tur, sat, mode in product(
        ["linear", "MM", "RMM"],
        ["linear", "density_dependent"],
        ["no", "Langmuir"],
        ["steady", "dynamic"]):
    version = dict(microbial_decomposition=dec,microbial_turnover=tur,saturation=sat,) # get model
    p_senss = np.full((n_samples, len(param_names)), jnp.nan)
    s_senss = np.full((n_samples, len(state_names)), jnp.nan)
    outputs = np.full((n_samples,), jnp.nan)
    if mode == "steady":
        def S_steady(param_set):
            return jnp.sum(analytical_steady_state(param_set, **version))
        outputs = jax.vmap(S_steady)(random_params)
        p_senss =  jax.vmap(jax.grad(S_steady))(random_params)
        s_senss = np.full((n_samples, len(state_names)), jnp.nan)  # no state sensitivity in steady
    elif mode == "dynamic":
        solver, saveat = dfx.Euler(), dfx.SaveAt(t0=False, t1=True)
        def rhs(t, y, args): 
            return craig_BA_adapt(t, y, args, **version)
        def delta_S_dynamic(p, y0):
            sol = dfx.diffeqsolve(dfx.ODETerm(rhs),solver,
                                  t0=0.0, t1=3.0, dt0=0.05,
                                  y0=y0, args=p, saveat=saveat)
            yT = sol.ys[-1]
            return jnp.sum(yT) - jnp.sum(y0)
        grad_delta_S_dynamic = jax.vmap(lambda p, y0: jax.grad(delta_S_dynamic, argnums=(0,1))(p, y0),in_axes=(0, 0))
        delta_S_dynamic_vmap = jax.vmap(delta_S_dynamic, in_axes=(0, 0))
        outputs = delta_S_dynamic_vmap(random_params, random_states)
        p_senss, s_senss = grad_delta_S_dynamic(random_params, random_states)
    print(f"[{mode}] md={dec} mt={tur} sat={sat} -- completed")
    p_senss = np.abs(p_senss) # so importance doesn't cancel out
    s_senss = np.abs(s_senss) # so importance doesn't cancel out
    p_senss = p_senss * param_diffs / SOC_diff # normalize
    s_senss = s_senss * state_diffs / SOC_diff # normalize
    p_senss_median = np.nanmedian(p_senss, axis=0) # get median
    s_senss_median = np.nanmedian(s_senss, axis=0) # get median
    results.append(dict(md=dec, mt=tur, sat=sat, temp=mode,
             **dict(zip(param_names, p_senss_median)),
             **{f"y0_{s}": v for s, v in zip(state_names, s_senss_median)}))
df_results = pd.DataFrame(results)
df_results = df_results.round(5)
df_results.to_csv("sensitivities.csv", index=False)
print(df_results)
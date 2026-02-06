import time
import os
import jax
import jax.numpy as jnp
import pandas as pd
from itertools import product


def sensitivity_order(
    model_fn,
    steady_state_fn,
    param_ranges,
    version_kwargs,
    mode="steady",
    years=3.0,
    y0=None,
    dt=0.05,
  ):
    param_names = list(param_ranges.keys())
    p0 = jnp.array(
        [(v["min"] + v["max"]) / 2 for v in param_ranges.values()]
    )

    def S_steady(p):
        return jnp.sum(steady_state_fn(p, **version_kwargs))

    def delta_S_dynamic(p):
        if y0 is None:
            raise ValueError("y0 must be provided for dynamic sensitivity")

        steps = int(years / dt)

        def step(y, _):
            y_next = y + dt * model_fn(0.0, y, p, **version_kwargs)
            return y_next, y_next

        yT, _ = jax.lax.scan(step, y0, None, length=steps)
        return jnp.sum(yT) - jnp.sum(y0)

    output_fn = S_steady if mode == "steady" else delta_S_dynamic

    def log_output(log_p):
        val = output_fn(jnp.exp(log_p))
        safe = jnp.sqrt(val * val + 1e-12)
        return jnp.log(safe)

    sens = jax.grad(log_output)(jnp.log(p0))

    ranking = sorted(
        zip(param_names, sens),
        key=lambda x: jnp.abs(x[1]),
        reverse=True,
    )

    return ranking


def sensitivity_values(
    model_fn,
    steady_state_fn,
    param_ranges,
    version_kwargs,
    mode="steady",
    years=3.0,
    y0=None,
    dt=0.05,
    p0=None,
    ):
    param_names = list(param_ranges.keys())
    if p0 is None:
        p0 = jnp.array(
            [(v["min"] + v["max"]) / 2 for v in param_ranges.values()]
        )
    else:
        p0 = jnp.array(p0)

    def S_steady(p):
        return jnp.sum(steady_state_fn(p, **version_kwargs))

    def delta_S_dynamic(p):
        if y0 is None:
            raise ValueError("y0 must be provided for dynamic sensitivity")

        steps = int(years / dt)

        def step(y, _):
            y_next = y + dt * model_fn(0.0, y, p, **version_kwargs)
            return y_next, y_next

        yT, _ = jax.lax.scan(step, y0, None, length=steps)
        return jnp.sum(yT) - jnp.sum(y0)

    output_fn = S_steady if mode == "steady" else delta_S_dynamic

    def log_output(log_p):
        val = output_fn(jnp.exp(log_p))
        safe = jnp.sqrt(val * val + 1e-12)
        return jnp.log(safe)

    sens = jax.grad(log_output)(jnp.log(p0))
    return param_names, sens


def run_all_sensitivities(
    model_fn,
    steady_state_fn,
    param_ranges,
    decomposition_options,
    turnover_options,
    saturation_options,
    p0s,
    y0s,
    years=3.0,
  ):
    param_names = list(param_ranges.keys())
    rows = []
    n_samples = len(p0s) if p0s is not None else 0

    for dec, tur, sat in product(
        decomposition_options,
        turnover_options,
        saturation_options,
    ):
        version = dict(
            microbial_decomposition=dec,
            microbial_turnover=tur,
            saturation=sat,
        )

        for mode in ("steady", "dynamic"):
            if p0s is None:
                raise ValueError("p0s must be provided for sensitivity analysis")
            if mode == "dynamic" and y0s is None:
                raise ValueError("y0s must be provided for dynamic sensitivity")
            if y0s is not None and len(y0s) != len(p0s):
                raise ValueError("p0s and y0s must have the same length")

            sens_samples = []
            for i, p0 in enumerate(p0s):
                y0 = None if mode == "steady" else y0s[i]
                print(
                    f"[{mode}] md={dec} mt={tur} sat={sat} "
                    f"sample {i + 1}/{n_samples}"
                )
                names, sens = sensitivity_values(
                    model_fn=model_fn,
                    steady_state_fn=steady_state_fn,
                    param_ranges=param_ranges,
                    version_kwargs=version,
                    mode=mode,
                    years=years,
                    y0=y0,
                    p0=p0,
                )
                sens_samples.append(sens)

            sens_median = jnp.nanmedian(
                jnp.stack(sens_samples, axis=0), axis=0
            )
            sens_by_param = {
                name: float(val) for name, val in zip(names, sens_median)
            }
            row = {
                "md": dec,
                "mt": tur,
                "sat": sat,
                "temp": mode,
            }
            row.update({name: sens_by_param.get(name, 0.0) for name in param_names})
            rows.append(row)

    return pd.DataFrame(rows, columns=["md", "mt", "sat", "temp", *param_names])


from hybrid_models import craig_BA_adapt, analytical_steady_state
from hybrid_config import default_param_ranges, default_state_bounds

decomposition_options = ["linear", "MM", "RMM"]
turnover_options = ["linear", "density_dependent"]
saturation_options = ["no", "Langmuir"]

state_names = list(default_state_bounds.keys())
state_mins = jnp.array([default_state_bounds[name]["min"] for name in state_names])
state_maxs = jnp.array([default_state_bounds[name]["max"] for name in state_names])
param_names = list(default_param_ranges.keys())
param_mins = jnp.array([default_param_ranges[name]["min"] for name in param_names])
param_maxs = jnp.array([default_param_ranges[name]["max"] for name in param_names])

key = jax.random.PRNGKey(0)
key_params, key_states = jax.random.split(key, 2)
n_samples = 100
random_params = jax.random.uniform(key_params, shape=(n_samples, len(param_names)))
random_states = jax.random.uniform(key_states, shape=(n_samples, len(state_names)))
p0s = param_mins + (param_maxs - param_mins) * random_params
y0s = state_mins + (state_maxs - state_mins) * random_states

start_time = time.time()

results = run_all_sensitivities(
    model_fn=craig_BA_adapt,
    steady_state_fn=analytical_steady_state,
    param_ranges=default_param_ranges,
    decomposition_options=decomposition_options,
    turnover_options=turnover_options,
    saturation_options=saturation_options,
    p0s=p0s,
    y0s=y0s,
)

results = results.sort_values("temp")
print(results)
base_dir = os.path.dirname(os.path.abspath(__file__))
results.to_csv(os.path.join(base_dir, "sensitivities.csv"), index=False)
print(f"time: {time.time() - start_time:.2f}s")
"""Verification harness for the analytical steady states in models.py.

The corrected closed-form steady states for the 12 craig_BA_adapt model versions
live in models.analytical_steady_state. This script samples parameters from the
config ranges, plugs each analytical steady state back into craig_BA_adapt and
checks that dCp/dt = dCb/dt = dCm/dt = 0 for every version.

The 12 versions combine
  * microbial decomposition (dec): "linear", "MM", "RMM"
  * microbial turnover      (tur): "linear", "density_dependent"
  * saturation              (sat): "no", "Langmuir"
and assume constant carbon-use efficiency (the case the analytics are derived for).
See the docstring of models.analytical_steady_state for the derivation.
"""

import jax
import jax.numpy as jnp

# Iterable of the 12 (dec, tur, sat) model versions.
MODEL_VERSIONS = [
    (dec, tur, sat)
    for dec in ("linear", "MM", "RMM")
    for tur in ("linear", "density_dependent")
    for sat in ("no", "Langmuir")
]


def _test(N=1000, seed=0, tol=1e-6):
    """Sample parameters from config ranges and verify dC/dt = 0 for all 12 versions."""
    import numpy as np
    jax.config.update("jax_enable_x64", True)  # float64 so residuals are meaningful

    from config import default_param_ranges
    from models import craig_BA_adapt, analytical_steady_state

    names = list(default_param_ranges.keys())
    lows = np.array([default_param_ranges[k]["min"] for k in names], dtype=float)
    highs = np.array([default_param_ranges[k]["max"] for k in names], dtype=float)
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.uniform(lows, highs, size=(N, len(names))))

    print(f"Verifying analytical steady states (models.py) over {N} samples\n")
    header = f"{'md':6s} | {'mt':17s} | {'sat':8s} | feasible | not-0 | worst |dC/dt|"
    print(header)
    print("-" * len(header))

    total_fail = 0
    for dec, tur, sat in MODEL_VERSIONS:
        ss = jax.jit(jax.vmap(lambda q: analytical_steady_state(q, dec, tur, sat)))
        ode = jax.jit(jax.vmap(lambda q, y: craig_BA_adapt(
            0.0, y, q, dec, tur, "constant", sat)))

        y_star = ss(samples)
        dydt = np.asarray(ode(samples, y_star))
        y_np = np.asarray(y_star)

        # The ODE clamps pools at 1e-12, so the residual is only meaningful when all
        # pools are strictly positive (a real, feasible equilibrium).
        feas = np.all(np.isfinite(y_np) & (y_np > 0), axis=1)
        res = np.max(np.abs(dydt), axis=1)
        n_feas = int(feas.sum())
        n_fail = int(((res > tol) | ~np.isfinite(res))[feas].sum()) if n_feas else 0
        worst = float(np.max(res[feas])) if n_feas else float("nan")
        total_fail += n_fail

        print(f"{dec:6s} | {tur:17s} | {sat:8s} | {n_feas:8d} | {n_fail:5d} | {worst:.2e}")

    print("-" * len(header))
    print(f"Total feasible samples violating dC/dt = 0 (tol={tol:g}): {total_fail}")


if __name__ == "__main__":
    _test()

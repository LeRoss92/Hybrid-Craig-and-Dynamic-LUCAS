# SfT - Plot distribution of *time until steady state* (FAST)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import diffrax as dfx
from functools import partial
from hybrid_models import craig_BA_adapt, analytical_steady_state
from hybrid_utils import vector_field

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(_SCRIPT_DIR, "6_hybrid_outputs")
folds = 5
# ~10% of rows for speed (deterministic stride so progress totals stay correct)
SAMPLE_FRAC = 1.0#0.1
# Integration: sparse saves (was steps=True @ dt=0.2 → ~1500 steps/save)
T_END = 2500.0
DT_INTEGRATOR = 5.0
_SAVE_TS = jnp.linspace(0.0, T_END, int(T_END / DT_INTEGRATOR) + 1)
BATCH_CHUNK = 1024


def make_analytical_batch(md, mt, sat):
    """Vectorized steady state for fixed (md, mt, sat) — avoids Python loop over rows."""

    @jax.jit
    def batch(params_batch):
        return jax.vmap(
            lambda p: analytical_steady_state(
                p,
                microbial_decomposition=md,
                microbial_turnover=mt,
                saturation=sat,
            )
        )(params_batch)

    return batch


def subsample_df(df):
    n = len(df)
    if n == 0:
        return df
    step = max(1, int(round(1.0 / SAMPLE_FRAC)))
    return df.iloc[::step].reset_index(drop=True)

# From 7_analysis.ipynb adaptability grid (Cp-Cb-Cm, 2015_2018); override via selected_versions.csv
OPTIMAL_N_SP_DEFAULT = {
    "mdlinear_mtdensity_dependent_satLangmuir": 5,
    "mdlinear_mtdensity_dependent_satno": 6,
    "mdMM_mtdensity_dependent_satLangmuir": 5,
    "mdMM_mtdensity_dependent_satno": 5,
    "mdRMM_mtdensity_dependent_satLangmuir": 5,
    "mdRMM_mtdensity_dependent_satno": 5,
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3*3, 3)) # 3 subplots (3 pools)
versions = {}
for md in ["linear", "MM", "RMM"]:
    for mt in ["density_dependent"]:
        for sat in ["no", "Langmuir"]:
            file_sub_str = f"md{md}_mt{mt}_sat{sat}"
            color_map = {
                "linear": "black",
                "MM": "violet",
                "RMM": "red",
            }
            linewidth = 1 if md == "linear" else 3
            linestyle = ":" if sat == "no" else "-"
            versions[file_sub_str] = {
                "color": color_map[md],
                "texture": linestyle,
                "linewidth": linewidth,
                "legend": f"{md}-{mt}-{sat}",
            }

all_files = os.listdir(results_path)

_csv = os.path.join(_SCRIPT_DIR, "selected_versions.csv")
if os.path.isfile(_csv):
    selected = pd.read_csv(_csv)
else:
    selected = pd.DataFrame(
        [{"version": v, "optimal n_sp": n} for v, n in OPTIMAL_N_SP_DEFAULT.items()]
    )

# ------------------------------------------------------------
# JAX‑accelerated batch integration
# ------------------------------------------------------------
def make_batch_integrator(md, mt, sat):
    """Returns a JIT‑compiled function that integrates a batch of simulations."""
    model_fn = partial(craig_BA_adapt,
                       microbial_decomposition=md,
                       microbial_turnover=mt,
                       saturation=sat)

    @jax.jit
    def solve_batch(y0_batch, params_batch, ss_batch):
        # solve_one will be vmapped over the batch
        def solve_one(y0, params, ss):
            term = dfx.ODETerm(partial(vector_field, model_fn))
            solver = dfx.Euler()
            saveat = dfx.SaveAt(ts=_SAVE_TS)
            sol = dfx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=T_END,
                dt0=DT_INTEGRATOR,
                y0=y0,
                args=params,
                saveat=saveat,
            )
            ys = sol.ys        # (n_steps, 3)
            ts = sol.ts        # (n_steps,)

            # For each pool, find first time it's within 1e-3 relative tolerance
            close_enough = jnp.abs(ys - ss) < 1e-3 * (jnp.abs(ss) + 1e-6)
            def first_true(col):
                idx = jnp.argmax(col)
                return jnp.where(col[idx], ts[idx], jnp.nan)
            return jax.vmap(first_true, in_axes=1)(close_enough)

        # Vectorize over the batch
        batched = jax.vmap(solve_one, in_axes=(0, 0, 0))
        return batched(y0_batch, params_batch, ss_batch)

    return solve_batch

# ------------------------------------------------------------
# Process each version (same as before, but inner loop is batched)
# ------------------------------------------------------------
for version in versions:
    files = [f for f in all_files
            if version in f
            and f"{'Cp-Cb-Cm'}_" in f
            and f"temp2015_2018_fold" in f
            ]
    files = sorted(files, key=len, reverse=True)
    opt = selected[selected["version"] == version]["optimal n_sp"].values[0]
    files = files[folds*opt:folds*opt+folds]

    times_steady_all = [[] for _ in range(3)] # Cp, Cb, Cm
    # version format: md{md}_mt{mt}_sat{sat} — mt may contain underscores (e.g. density_dependent)
    parts = version.split("_")
    md = parts[0][2:]
    sat = parts[-1][3:]  # satno / satLangmuir / …
    mt = "_".join(parts[1:-1])[2:]  # strip leading "mt" from mtdensity_dependent…

    batch_integrator = make_batch_integrator(md, mt, sat)
    analytical_batch = make_analytical_batch(md, mt, sat)

    batches = []
    for file in files:
        data = subsample_df(pd.read_pickle(os.path.join(results_path, file)))
        batches.append(data)
    total_to_run = sum(len(d) for d in batches)

    counter = 0
    print(f"\nProcessing version: {version} (~{100 * SAMPLE_FRAC:.0f}% of rows)")

    for data in batches:
        params_arr = data[['param_I','param_CUE','param_beta','param_tmb','param_Cg0b','param_Cg0m','param_qx','param_Vmax_p','param_Vmax_m','param_Km_p','param_Km_m','param_kp','param_kb','param_km']].values.astype(np.float32)
        y0_arr = data[[f'pred_final_{pool}' for pool in ["Cp", "Cb", "Cm"]]].values.astype(np.float32)

        params_jax = jnp.asarray(params_arr)
        ss_parts = []
        for s in range(0, params_jax.shape[0], BATCH_CHUNK):
            e = min(s + BATCH_CHUNK, params_jax.shape[0])
            ss_parts.append(analytical_batch(params_jax[s:e]))
        ss_batch = jnp.concatenate(ss_parts, axis=0)

        y0_batch = jnp.array(y0_arr)

        nloc = y0_arr.shape[0]
        time_chunks = []
        for s in range(0, nloc, BATCH_CHUNK):
            e = min(s + BATCH_CHUNK, nloc)
            time_chunks.append(
                batch_integrator(y0_batch[s:e], params_jax[s:e], ss_batch[s:e])
            )
        times_batch = jnp.concatenate(time_chunks, axis=0)

        times_batch_np = np.array(times_batch)
        for i in range(3):
            times_steady_all[i].extend(times_batch_np[:, i])

        counter += len(y0_arr)
        print(f"\r{version} {counter}/{total_to_run} ({100*counter/total_to_run:.1f}%)", end="", flush=True)

    print("") # finish line

    # Plot densities (exactly as before)
    for i, pool in enumerate(["Cp", "Cb", "Cm"]):
        ax = axes[i]
        arr = np.array(times_steady_all[i])
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        counts, bin_edges = np.histogram(arr, bins=25, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        style = versions[version]
        ax.plot(bin_centers, counts,
                label=version,
                color=style["color"], linestyle=style["texture"], linewidth=style["linewidth"])

# (Rest of plotting unchanged)
axes[0].set_ylabel('density')
for i, pool in enumerate(["POC", "MIC", "MAOC"]):
    ax = axes[i]
    ax.set_title(pool)
    ax.set_xlabel(f"{pool} time to steady state (years)")

legend_elements = []
for version, style in versions.items():
    legend_elements.append(
        Line2D(
            [0], [0],
            color=style["color"],
            linestyle=style["texture"],
            linewidth=style["linewidth"],
            label=style["legend"],
        )
    )

fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.,
    frameon=False)
plt.tight_layout()
_figdir = os.path.join(_SCRIPT_DIR, "figures")
os.makedirs(_figdir, exist_ok=True)
plt.savefig(os.path.join(_figdir, "time2steady.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {os.path.join(_figdir, 'time2steady.png')}")
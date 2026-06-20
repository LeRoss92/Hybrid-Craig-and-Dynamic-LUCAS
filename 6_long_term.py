"""Long-term steady-state analysis using parameters from 5_dynamic_hybrid.py.

For each model variant (MM / RMM, density_dependent, no saturation) we run the
same cumulative parameter steps as ``5_dynamic_hybrid.py``:

  - static_only          all mechanistic params frozen at the per-site static fit
  - CUE                  only the most sensitive param re-learned spatially
  - CUE+beta             top-2 re-learned; all others stay at static fit
  - …                    up to the full spatial set

Outputs
-------
figures/long_term/{md}_{mt}_{sat}/{label}/
    delta_to_steady.png, time_to_steady.png, delta_soc_vs_time.png
hybrid_outputs_long_term/
    long_term_{combo}_{label}.pkl
    weights_{combo}_{label}.pkl   (cached NN weights; not used for static_only)

Run with:
    micromamba run -n hybrid-lucas python 6_long_term.py
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from models import craig_BA_adapt, analytical_steady_state

# ── Re-use the working pipeline from 5_dynamic_hybrid.py ─────────────────────
_SPEC = importlib.util.spec_from_file_location(
    "dynamic_hybrid", Path(__file__).with_name("5_dynamic_hybrid.py")
)
_D5 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_D5)

MODELS = _D5.MODELS
PARAM_NAMES = _D5.PARAM_NAMES
OUT_DATA = Path("hybrid_outputs_long_term")
OUT_FIG = Path("figures/long_term")

# Long-term integration settings
DT_LONG = 1.0           # years per Euler step
MAX_YEARS = 15000.0    # integration horizon
RATE_TOL = 1e-2         # max |dy/dt| at quasi-steady state  (g C kg⁻¹ yr⁻¹)
POOL_TOL = 0.01         # max relative pool change per step
MIN_STEADY_YEARS = 10.0 # ignore convergence detections before this time

# Model state is mg C g⁻¹ soil; numerically identical to g C kg⁻¹
MG_G_TO_G_KG = 1.0
POOL_UNIT = "g C kg⁻¹"
HIST_BINS = 60
LIM_PAD = 0.05          # fractional padding on shared axis limits
LIM_PERCENTILES = (0.5, 99.5)  # robust shared range across steps


@dataclass(frozen=True)
class PlotLimits:
    dcp_xlim: tuple[float, float]
    dcb_xlim: tuple[float, float]
    dcm_xlim: tuple[float, float]
    dsoc_xlim: tuple[float, float]
    delta_ylim: tuple[float, float]
    time_xlim: tuple[float, float]
    time_ylim: tuple[float, float]
    scatter_xlim: tuple[float, float]
    scatter_ylim: tuple[float, float]


def _cumulative_param_sets(md: str, mt: str, sat: str) -> list[list[str]]:
    """Cumulative spatial steps from 5_dynamic_hybrid (most sensitive first)."""
    sens = pd.read_csv("figures/sensitivities.csv")
    row = sens[
        (sens["md"] == md)
        & (sens["mt"] == mt)
        & (sens["sat"] == sat)
        & (sens["temp"] == "dynamic")
    ].iloc[0]
    param_sens = row.drop(labels=["md", "mt", "sat", "temp", "y0_Cp", "y0_Cb", "y0_Cm"])
    test_params = [
        n for n in PARAM_NAMES if n != "I" and float(param_sens.get(n, 0.0)) != 0.0
    ]
    sorted_params = sorted(
        test_params, key=lambda name: abs(float(param_sens[name])), reverse=True
    )
    return [sorted_params[:k] for k in range(1, len(sorted_params) + 1)]


def _param_label(param_names: list[str] | None) -> str:
    if not param_names:
        return "static_only"
    return "+".join(param_names)


def _normalize_covariates(data: dict, train_idx: np.ndarray) -> jnp.ndarray:
    """Training-fold median imputation + z-score (matches train_combo)."""
    x_raw = data["x_dyn"].copy()
    train_medians = np.nanmedian(x_raw[train_idx], axis=0)
    nan_mask = ~np.isfinite(x_raw)
    x_raw[nan_mask] = np.take(train_medians, np.where(nan_mask)[1])
    x_all = jnp.asarray(x_raw)
    x_mean = jnp.mean(x_all[train_idx], axis=0)
    x_std = jnp.std(x_all[train_idx], axis=0) + 1e-8
    return (x_all - x_mean) / x_std, x_mean, x_std


def _jax_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _numpy_to_jax(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def build_static_param_matrix(data: dict) -> jnp.ndarray:
    """Per-site static fit with NPP as I — no dynamic re-learning."""
    static_all = jnp.asarray(data["static_params"])
    return static_all.at[:, 0].set(jnp.asarray(data["npp_I"]))


def load_or_train(
    data: dict, param_names: list[str], batched_sim, combo: str
) -> dict:
    """Train (or load cached) one cumulative parameter set from script 5."""
    label = _param_label(param_names)
    cache = _D5.OUT_DATA / f"weights_{combo}_{label}.pkl"
    if cache.exists():
        print(f"  Loading cached weights: {cache}")
        result = pd.read_pickle(cache)
        result["best_params"] = _numpy_to_jax(result["best_params"])
        result["x_mean"] = jnp.asarray(result["x_mean"])
        result["x_std"] = jnp.asarray(result["x_std"])
        return result

    print(f"  Training final parameter set ({label}) — no cache found …")
    result = _D5.train_combo(data, param_names, batched_sim)
    if result is None:
        raise RuntimeError(f"Training failed for {combo} / {label}")

    to_save = dict(result)
    to_save["best_params"] = _jax_to_numpy(result["best_params"])
    to_save["x_mean"] = np.asarray(jax.device_get(result["x_mean"]))
    to_save["x_std"] = np.asarray(jax.device_get(result["x_std"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(to_save, cache)
    print(f"  Cached weights → {cache}")
    return result


def build_learned_param_matrix(result: dict, data: dict) -> jnp.ndarray:
    """Per-site parameter matrix with learned spatial adjustments."""
    split = data["split_col"]
    train_idx = np.where((split != "test") & (split != _D5.FOLD_VAL))[0]
    x_all, _, _ = _normalize_covariates(data, train_idx)

    pidxs = jnp.asarray(result["param_idxs"])
    p_mins = jnp.asarray(result["p_mins"])
    p_maxs = jnp.asarray(result["p_maxs"])
    net_p = result["best_params"]
    static_all = jnp.asarray(data["static_params"])
    npp_all = jnp.asarray(data["npp_I"])

    raw = jax.vmap(lambda xi: _D5.mlp_forward(net_p, xi))(x_all)
    p_dyn = _D5.learned_params(raw, static_all, pidxs, p_mins, p_maxs)
    return static_all.at[:, pidxs].set(p_dyn).at[:, 0].set(npp_all)


def make_steady_state_simulator(md: str, mt: str, sat: str):
    """Batched Euler integrator: steady state = low |dy/dt| and stable pools."""
    model_fn = partial(
        craig_BA_adapt,
        microbial_decomposition=md,
        microbial_turnover=mt,
        saturation=sat,
    )
    n_steps = int(round(MAX_YEARS / DT_LONG))
    dt = jnp.float32(DT_LONG)
    min_t = jnp.float32(MIN_STEADY_YEARS)
    max_t = jnp.float32(MAX_YEARS)
    big = jnp.float32(1e9)

    @jax.jit
    def sim_batch(p_batch, y0_batch):
        def sim_one(p, y0):
            def scan_body(carry, _):
                y, t, conv_done, y_at_conv = carry
                dy = model_fn(t, y, p)
                y_next = jnp.maximum(y + dt * dy, 1e-12)
                rate = jnp.max(jnp.abs(dy))
                rel_chg = jnp.max(jnp.abs((y_next - y) / (jnp.abs(y) + 1e-8)))
                conv = (rate < RATE_TOL) & (rel_chg < POOL_TOL) & (t >= min_t)
                t_next = t + dt
                new_conv_done = conv_done | conv
                new_y_at_conv = jnp.where(conv & ~conv_done, y_next, y_at_conv)
                return (y_next, t_next, new_conv_done, new_y_at_conv), (conv, t_next)

            init = (y0, jnp.float32(0.0), False, y0)
            (y_final, _, conv_done, y_at_conv), (conv_flags, times) = jax.lax.scan(
                scan_body, init, None, length=n_steps
            )

            conv_time = jnp.min(jnp.where(conv_flags, times, big))
            never = ~conv_done
            conv_time = jnp.where(never, max_t, conv_time)
            y_ss = jnp.where(never, y_final, y_at_conv)
            return y_ss, conv_time, never

        return jax.vmap(sim_one)(p_batch, y0_batch)

    return sim_batch


def analytical_batch(md: str, mt: str, sat: str):
    return jax.vmap(
        partial(
            analytical_steady_state,
            microbial_decomposition=md,
            microbial_turnover=mt,
            saturation=sat,
        )
    )


def _to_gkg(y: np.ndarray) -> np.ndarray:
    return y * MG_G_TO_G_KG


def _padded_limits(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, LIM_PERCENTILES)
    lo, hi = float(lo), float(hi)
    if lo == hi:
        pad = max(abs(lo), 1.0) * 0.05
        return lo - pad, hi + pad
    span = hi - lo
    return lo - LIM_PAD * span, hi + LIM_PAD * span


def _hist_peak(vals: np.ndarray, xlim: tuple[float, float], bins: int = HIST_BINS) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    counts, _ = np.histogram(vals, bins=bins, range=xlim)
    return float(np.max(counts))


def compute_plot_limits(results: list[dict]) -> PlotLimits:
    """Shared axis limits across all parameter steps within one model combo."""
    dcp = np.concatenate([r["delta"][:, 0] for r in results])
    dcb = np.concatenate([r["delta"][:, 1] for r in results])
    dcm = np.concatenate([r["delta"][:, 2] for r in results])
    dsoc = np.concatenate([r["d_soc"] for r in results])
    conv = np.concatenate([r["conv_time"] for r in results])

    dcp_x = _padded_limits(dcp)
    dcb_x = _padded_limits(dcb)
    dcm_x = _padded_limits(dcm)
    dsoc_x = _padded_limits(dsoc)
    time_x = (0.0, MAX_YEARS)
    scatter_x = time_x
    scatter_y = dsoc_x

    delta_peaks = [
        _hist_peak(r["delta"][:, 0], dcp_x) for r in results
    ] + [
        _hist_peak(r["delta"][:, 1], dcb_x) for r in results
    ] + [
        _hist_peak(r["delta"][:, 2], dcm_x) for r in results
    ] + [
        _hist_peak(r["d_soc"], dsoc_x) for r in results
    ]
    time_peaks = [_hist_peak(r["conv_time"], time_x) for r in results]

    y_hi = max(delta_peaks + time_peaks + [1.0]) * (1.0 + LIM_PAD)
    delta_y = (0.0, y_hi)
    time_y = (0.0, max(_hist_peak(r["conv_time"], time_x) for r in results) * (1.0 + LIM_PAD)
              if results else 1.0)

    return PlotLimits(
        dcp_xlim=dcp_x,
        dcb_xlim=dcb_x,
        dcm_xlim=dcm_x,
        dsoc_xlim=dsoc_x,
        delta_ylim=delta_y,
        time_xlim=time_x,
        time_ylim=time_y,
        scatter_xlim=scatter_x,
        scatter_ylim=scatter_y,
    )


def plot_distributions(
    y0: np.ndarray,
    y_ss: np.ndarray,
    conv_time: np.ndarray,
    converged: np.ndarray,
    out_dir: Path,
    combo: str,
    param_label: str,
    limits: PlotLimits,
) -> None:
    pool_names = ["Cp", "Cb", "Cm"]
    pool_xlims = [limits.dcp_xlim, limits.dcb_xlim, limits.dcm_xlim]
    y0 = _to_gkg(y0)
    y_ss = _to_gkg(y_ss)
    delta = y_ss - y0
    delta_soc = delta.sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name, col, xlim in zip(axes.flat[:3], pool_names, range(3), pool_xlims):
        vals = delta[:, col]
        ax.hist(vals[np.isfinite(vals)], bins=HIST_BINS, range=xlim,
                color="#5C6BC0", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        finite = vals[np.isfinite(vals)]
        med = float(np.median(finite)) if finite.size else 0.0
        ax.axvline(med, color="#E53935", lw=1.2, ls="-",
                   label=f"median={med:.2f}")
        ax.set_xlim(xlim)
        ax.set_ylim(limits.delta_ylim)
        ax.set_xlabel(f"Δ{name}  (steady − initial)  [{POOL_UNIT}]")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(delta_soc[np.isfinite(delta_soc)], bins=HIST_BINS,
            range=limits.dsoc_xlim, color="#43A047", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    soc_finite = delta_soc[np.isfinite(delta_soc)]
    med_soc = float(np.median(soc_finite)) if soc_finite.size else 0.0
    ax.axvline(med_soc, color="#E53935", lw=1.2, ls="-",
               label=f"median={med_soc:.2f}")
    ax.set_xlim(limits.dsoc_xlim)
    ax.set_ylim(limits.delta_ylim)
    ax.set_xlabel(f"ΔSOC  (steady − initial)  [{POOL_UNIT}]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    n_conv = int(converged.sum())
    fig.suptitle(
        f"Changes from initial conditions to new steady state\n"
        f"{combo}  —  learned params: {param_label}\n"
        f"converged {n_conv}/{len(converged)} sites within {MAX_YEARS:.0f} yr",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "delta_to_steady.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    t_plot = conv_time[converged] if n_conv else conv_time
    ax.hist(t_plot[np.isfinite(t_plot)], bins=HIST_BINS, range=limits.time_xlim,
            color="#FF7043", edgecolor="white", alpha=0.85)
    if n_conv:
        ax.axvline(np.median(t_plot), color="#E53935", lw=1.2,
                   label=f"median={np.median(t_plot):.1f} yr")
    ax.set_xlim(limits.time_xlim)
    ax.set_ylim(limits.time_ylim)
    ax.set_xlabel("Time to reach steady state  [years]")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Convergence time  —  {combo}\n"
        f"(|dy/dt| < {RATE_TOL:g}, rel. pool change < {POOL_TOL:g}, dt={DT_LONG} yr)"
    )
    if n_conv:
        ax.legend()
    fig.tight_layout()
    path = out_dir / "time_to_steady.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ok = np.isfinite(delta_soc)
    sc = ax.scatter(conv_time[ok], delta_soc[ok], c=converged[ok].astype(float),
                    cmap="RdYlGn", s=8, alpha=0.5, vmin=0, vmax=1)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlim(limits.scatter_xlim)
    ax.set_ylim(limits.scatter_ylim)
    ax.set_xlabel("Time to steady state  [years]")
    ax.set_ylabel(f"ΔSOC  [{POOL_UNIT}]")
    ax.set_title(f"ΔSOC vs convergence time  —  {combo}\n{param_label}")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("converged within horizon")
    fig.tight_layout()
    path = out_dir / "delta_soc_vs_time.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def analyse_param_set(
    md: str,
    mt: str,
    sat: str,
    data: dict,
    param_names: list[str] | None,
    batched_sim,
    sim_batch,
    combo: str,
    step: int,
    n_steps: int,
) -> dict:
    label = _param_label(param_names)
    out_dir = OUT_FIG / combo / label
    out_dir.mkdir(parents=True, exist_ok=True)
    n_sites = len(data["y0"])

    note = " (final)" if step == n_steps else ""
    print(f"\n  ── Step {step}/{n_steps}: {label}{note} ──")

    if param_names is None:
        p_all = build_static_param_matrix(data)
    else:
        result = load_or_train(data, param_names, batched_sim, combo)
        p_all = build_learned_param_matrix(result, data)

    y0 = jnp.asarray(data["y0"])
    print(f"  Integrating up to {MAX_YEARS:.0f} yr  (dt={DT_LONG} yr) …")
    y_ss, conv_time, never = sim_batch(p_all, y0)

    y_ss_np = np.asarray(jax.device_get(y_ss))
    conv_time_np = np.asarray(jax.device_get(conv_time))
    never_np = np.asarray(jax.device_get(never))
    converged = ~never_np
    y0_np = np.asarray(data["y0"])

    y_ana = np.asarray(jax.device_get(analytical_batch(md, mt, sat)(p_all)))
    ana_finite = np.all(np.isfinite(y_ana), axis=1)
    if ana_finite.any():
        sim_delta = np.sum(y_ss_np - y0_np, axis=1)
        ana_delta = np.sum(y_ana - y0_np, axis=1)
        r2_ana = _D5.r2_score(sim_delta[ana_finite], ana_delta[ana_finite])
        print(f"  Simulated vs analytical ΔSOC  R² = {r2_ana:+.3f}")

    n_conv = int(converged.sum())
    print(f"  Converged: {n_conv}/{n_sites}  ({100 * n_conv / n_sites:.1f}%)")
    if n_conv:
        print(f"  Convergence time  median={np.median(conv_time_np[converged]):.1f} yr  "
              f"mean={np.mean(conv_time_np[converged]):.1f} yr  "
              f"[{np.min(conv_time_np[converged]):.1f}, "
              f"{np.max(conv_time_np[converged]):.1f}]")

    y0_gkg = _to_gkg(y0_np)
    y_ss_gkg = _to_gkg(y_ss_np)
    delta = y_ss_gkg - y0_gkg
    d_soc = delta.sum(axis=1)
    n_bad = int(np.sum(~np.isfinite(d_soc)))
    if n_bad:
        print(f"  Warning: {n_bad} sites with non-finite steady-state pools")
    print(f"  ΔSOC  median={np.nanmedian(d_soc):.3f}  "
          f"std={np.nanstd(d_soc):.3f}  [{POOL_UNIT}]")

    pd.DataFrame({
        "original_idx": data["original_idx"],
        "Cp0": y0_gkg[:, 0],
        "Cb0": y0_gkg[:, 1],
        "Cm0": y0_gkg[:, 2],
        "Cp_ss": y_ss_gkg[:, 0],
        "Cb_ss": y_ss_gkg[:, 1],
        "Cm_ss": y_ss_gkg[:, 2],
        "dCp": delta[:, 0],
        "dCb": delta[:, 1],
        "dCm": delta[:, 2],
        "dSOC": delta.sum(axis=1),
        "conv_time_yr": conv_time_np,
        "converged": converged,
        "param_label": label,
        "n_learned": 0 if param_names is None else len(param_names),
        "step": step,
    }).to_pickle(OUT_DATA / f"long_term_{combo}_{label}.pkl")

    return dict(
        label=label,
        step=step,
        y0_np=y0_np,
        y_ss_np=y_ss_np,
        conv_time=conv_time_np,
        converged=converged,
        delta=delta,
        d_soc=d_soc,
    )


def replot_combo(md: str, mt: str, sat: str) -> None:
    """Regenerate figures from saved pickles using shared axis limits."""
    combo = f"{md}_{mt}_{sat}"
    param_sets = _cumulative_param_sets(md, mt, sat)
    labels = ["static_only"] + ["+".join(ps) for ps in param_sets]

    results = []
    for label in labels:
        path = OUT_DATA / f"long_term_{combo}_{label}.pkl"
        if not path.exists():
            print(f"  Missing {path}, skipping replot for {combo}")
            return
        df = pd.read_pickle(path)
        delta = df[["dCp", "dCb", "dCm"]].to_numpy(dtype=np.float64)
        results.append(dict(
            label=label,
            step=int(df["step"].iloc[0]),
            y0_np=df[["Cp0", "Cb0", "Cm0"]].to_numpy(dtype=np.float64) / MG_G_TO_G_KG,
            y_ss_np=df[["Cp_ss", "Cb_ss", "Cm_ss"]].to_numpy(dtype=np.float64) / MG_G_TO_G_KG,
            conv_time=df["conv_time_yr"].to_numpy(dtype=np.float64),
            converged=df["converged"].to_numpy(dtype=bool),
            delta=delta,
            d_soc=df["dSOC"].to_numpy(dtype=np.float64),
        ))

    limits = compute_plot_limits(results)
    print(f"  Shared plot limits: ΔSOC [{limits.dsoc_xlim[0]:.1f}, {limits.dsoc_xlim[1]:.1f}], "
          f"time [0, {limits.time_xlim[1]:.0f}] yr")
    for res in results:
        out_dir = OUT_FIG / combo / res["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_distributions(
            res["y0_np"], res["y_ss_np"], res["conv_time"], res["converged"],
            out_dir, combo, res["label"], limits,
        )


def analyse_combo(md: str, mt: str, sat: str) -> None:
    combo = f"{md}_{mt}_{sat}"
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"Long-term steady state  —  {combo}")
    print(f"{'=' * 64}")

    t0 = time.perf_counter()
    data = _D5.prepare_data(md, mt, sat)
    n_sites = len(data["y0"])
    print(f"  Sites with valid IC + static params: {n_sites}")

    param_sets = _cumulative_param_sets(md, mt, sat)
    # static_only + cumulative steps (1 param, 2 params, …, all)
    steps: list[tuple[int, list[str] | None]] = [(0, None)]
    steps.extend((i, ps) for i, ps in enumerate(param_sets, start=1))
    n_steps = len(steps)
    print(f"  Parameter steps: static_only → {' → '.join('+'.join(ps) for ps in param_sets)}")

    batched_sim = _D5.make_batched_sim(md, mt, sat)
    sim_batch = make_steady_state_simulator(md, mt, sat)

    results = []
    for step, param_names in steps:
        results.append(analyse_param_set(
            md, mt, sat, data, param_names,
            batched_sim, sim_batch, combo, step, n_steps - 1,
        ))

    limits = compute_plot_limits(results)
    print(f"  Shared plot limits: ΔSOC [{limits.dsoc_xlim[0]:.1f}, {limits.dsoc_xlim[1]:.1f}], "
          f"time [0, {limits.time_xlim[1]:.0f}] yr")

    for res in results:
        out_dir = OUT_FIG / combo / res["label"]
        plot_distributions(
            res["y0_np"], res["y_ss_np"], res["conv_time"], res["converged"],
            out_dir, combo, res["label"], limits,
        )

    print(f"\n  Finished {combo} ({n_steps} steps) in "
          f"{time.perf_counter() - t0:.0f}s  →  {OUT_FIG / combo}")


def main() -> None:
    import sys

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    _D5.OUT_DATA.mkdir(parents=True, exist_ok=True)

    replot_only = "--replot" in sys.argv
    if replot_only:
        print("Replotting long-term figures from saved pickles")
        for md, mt, sat in MODELS:
            replot_combo(md, mt, sat)
        print("\nDone.")
        return

    print("Long-term steady-state analysis (parameters from 5_dynamic_hybrid)")
    for md, mt, sat in MODELS:
        analyse_combo(md, mt, sat)
    print("\nDone.")


if __name__ == "__main__":
    main()

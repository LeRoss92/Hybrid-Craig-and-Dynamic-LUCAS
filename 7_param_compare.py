"""Compare static (steady-state) hybrid parameters to dynamic hybrid parameters.

For each model variant and each cumulative dynamic-learning step from
``5_dynamic_hybrid.py``, scatter per-site static-fit parameters (x) against
the corresponding dynamic parameters (y) — one subplot per mechanistic
parameter.

Outputs
-------
figures/param_compare/{md}_{mt}_{sat}/{label}.png

Run with:
    micromamba run -n hybrid-lucas python 7_param_compare.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

# ── Re-use pipeline from 5_dynamic_hybrid.py ─────────────────────────────────
_D5_SPEC = importlib.util.spec_from_file_location(
    "dynamic_hybrid", Path(__file__).with_name("5_dynamic_hybrid.py")
)
_D5 = importlib.util.module_from_spec(_D5_SPEC)
sys.modules["dynamic_hybrid"] = _D5
assert _D5_SPEC.loader is not None
_D5_SPEC.loader.exec_module(_D5)

MODELS = _D5.MODELS
PARAM_NAMES = _D5.PARAM_NAMES
WEIGHTS_DIR = _D5.OUT_DATA
OUT_FIG = Path("figures/param_compare")


def _sorted_dynamic_params(md: str, mt: str, sat: str) -> list[str]:
    """Non-zero dynamic sensitivities (excluding I), most sensitive first."""
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
    return sorted(
        test_params, key=lambda name: abs(float(param_sens[name])), reverse=True
    )


def _cumulative_param_sets(md: str, mt: str, sat: str) -> list[list[str]]:
    sorted_params = _sorted_dynamic_params(md, mt, sat)
    return [sorted_params[:k] for k in range(1, len(sorted_params) + 1)]


def _param_label(param_names: list[str]) -> str:
    return "+".join(param_names)


def _normalize_covariates(data: dict, train_idx: np.ndarray):
    x_raw = data["x_dyn"].copy()
    train_medians = np.nanmedian(x_raw[train_idx], axis=0)
    nan_mask = ~np.isfinite(x_raw)
    x_raw[nan_mask] = np.take(train_medians, np.where(nan_mask)[1])
    x_all = jnp.asarray(x_raw)
    x_mean = jnp.mean(x_all[train_idx], axis=0)
    x_std = jnp.std(x_all[train_idx], axis=0) + 1e-8
    return (x_all - x_mean) / x_std


def build_static_param_matrix(data: dict) -> np.ndarray:
    static_all = jnp.asarray(data["static_params"])
    return np.asarray(static_all.at[:, 0].set(jnp.asarray(data["npp_I"])))


def build_learned_param_matrix(result: dict, data: dict) -> np.ndarray:
    split = data["split_col"]
    train_idx = np.where((split != "test") & (split != _D5.FOLD_VAL))[0]
    x_all = _normalize_covariates(data, train_idx)

    pidxs = jnp.asarray(result["param_idxs"])
    p_mins = jnp.asarray(result["p_mins"])
    p_maxs = jnp.asarray(result["p_maxs"])
    net_p = result["best_params"]
    static_all = jnp.asarray(data["static_params"])
    npp_all = jnp.asarray(data["npp_I"])

    raw = jax.vmap(lambda xi: _D5.mlp_forward(net_p, xi))(x_all)
    p_dyn = _D5.learned_params(raw, static_all, pidxs, p_mins, p_maxs)
    p_mat = static_all.at[:, pidxs].set(p_dyn).at[:, 0].set(npp_all)
    return np.asarray(p_mat)


def _load_weights(param_names: list[str], combo: str) -> dict:
    label = _param_label(param_names)
    cache = WEIGHTS_DIR / f"weights_{combo}_{label}.pkl"
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing cached weights for {combo} / {label}: {cache}\n"
            "Run 6_long_term.py first to train and cache dynamic weights."
        )
    print(f"  Loading {cache.name}")
    result = pd.read_pickle(cache)
    result["best_params"] = jax.tree_util.tree_map(jnp.asarray, result["best_params"])
    result["x_mean"] = jnp.asarray(result["x_mean"])
    result["x_std"] = jnp.asarray(result["x_std"])
    return result


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def plot_static_vs_dynamic(
    p_static: np.ndarray,
    p_dynamic: np.ndarray,
    param_names: list[str],
    combo: str,
    label: str,
    out_path: Path,
) -> None:
    """One row of subplots — only dynamically used params, in addition order."""
    n_params = len(param_names)
    fig, axes = plt.subplots(
        1, n_params, figsize=(3.2 * n_params, 3.6), squeeze=False
    )
    if n_params == 1:
        axes = np.array([axes])

    name_to_idx = {name: i for i, name in enumerate(PARAM_NAMES)}

    for ax, name in zip(axes.flat, param_names):
        i = name_to_idx[name]
        x = p_static[:, i]
        y = p_dynamic[:, i]
        ok = np.isfinite(x) & np.isfinite(y)

        ax.scatter(x[ok], y[ok], s=8, alpha=0.4, c="#E53935", edgecolors="none")

        if ok.any():
            lo = float(min(x[ok].min(), y[ok].min()))
            hi = float(max(x[ok].max(), y[ok].max()))
            if lo == hi:
                pad = max(abs(lo), 1e-6) * 0.05
                lo, hi = lo - pad, hi + pad
            else:
                pad = 0.05 * (hi - lo)
                lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

        r2 = _r2(x, y)
        ax.set_title(f"{name}\nR²={r2:+.3f}", fontsize=10)
        ax.set_xlabel("static", fontsize=9)
        if ax is axes.flat[0]:
            ax.set_ylabel("dynamic", fontsize=9)

    fig.suptitle(
        f"Static vs dynamic hybrid parameters  —  {combo}\n"
        f"cumulative step: {label}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def analyse_combo(md: str, mt: str, sat: str) -> None:
    combo = f"{md}_{mt}_{sat}"
    out_dir = OUT_FIG / combo

    print(f"\n{'=' * 64}")
    print(f"Static vs dynamic parameters  —  {combo}")
    print(f"{'=' * 64}")

    data = _D5.prepare_data(md, mt, sat)
    p_static = build_static_param_matrix(data)

    for param_names in _cumulative_param_sets(md, mt, sat):
        label = _param_label(param_names)
        print(f"\n  {label}")
        result = _load_weights(param_names, combo)
        p_dynamic = build_learned_param_matrix(result, data)
        plot_static_vs_dynamic(
            p_static,
            p_dynamic,
            param_names,
            combo=combo,
            label=label,
            out_path=out_dir / f"{label}.png",
        )


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    print("Static vs dynamic hybrid parameter comparison")
    for md, mt, sat in MODELS:
        analyse_combo(md, mt, sat)
    print("\nDone.")


if __name__ == "__main__":
    main()

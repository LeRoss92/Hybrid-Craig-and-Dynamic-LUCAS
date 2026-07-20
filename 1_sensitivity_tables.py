"""Standalone sensitivity analysis extracted from the end of 1_preprocess.ipynb.

Runs the reject-sampling parameter/state sensitivity computation for the 12 model
versions x {static, dynamic} and writes the 4 final sensitivity tables
(SOC, MAOC, MIC, dSOC) each as an individual LaTeX (.tex) file.

Output files (in figures/1_preprocess/):
    sensitivity_SOC.tex
    sensitivity_MAOC.tex
    sensitivity_MIC.tex
    sensitivity_dSOC.tex

Each .tex file requires, in the surrounding LaTeX document:
    \\usepackage[table]{xcolor}
    \\usepackage{graphicx}   % for \\rotatebox
    \\usepackage{booktabs}   % for \\toprule / \\midrule / \\bottomrule
"""

import os
from itertools import product

import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np
import pandas as pd
import matplotlib

from config import default_param_ranges
from models import craig_BA_adapt, analytical_steady_state

OUT_DIR = "figures/1_preprocess"

# ------------------------------------------------------------------ #
# 0. settings                                                        #
# ------------------------------------------------------------------ #
N_ACCEPT    = 1000       # accepted runs required per (version, mode)
BATCH       = 20000      # draws per rejection batch
MAX_BATCHES = 500        # safety cap -> avoids infinite while loop
T_YEARS     = 9.0        # dynamic integration horizon (matches old cell)
SEED        = 0

# ------------------------------------------------------------------ #
# 1. flat parameter ranges + empirical target ranges/distributions   #
# ------------------------------------------------------------------ #
df_t = pd.read_pickle("1_preprocessed.pkl")            # targets generated in the notebook

param_names = list(default_param_ranges.keys())
param_mins  = jnp.array([default_param_ranges[n]["min"] for n in param_names])
param_maxs  = jnp.array([default_param_ranges[n]["max"] for n in param_names])
param_diffs = param_maxs - param_mins
param_diffs_np = np.asarray(param_diffs)

state_names = ["Cp", "Cb", "Cm"]                       # model state order
pool_cols   = {"Cp": "POC", "Cb": "MIC", "Cm": "MAOC"} # state -> data column
i_Cb, i_Cm  = 1, 2

# empirical joint initial conditions (dynamic): whole rows, NaNs dropped jointly
y0_pool = jnp.asarray(
    df_t[[pool_cols[s] for s in state_names]].dropna().to_numpy(dtype=float))

# pool ranges (for normalising y0-sensitivities)
state_mins  = jnp.array([float(np.nanmin(df_t[pool_cols[s]])) for s in state_names])
state_maxs  = jnp.array([float(np.nanmax(df_t[pool_cols[s]])) for s in state_names])
state_diffs = np.asarray(state_maxs - state_mins)

# state-target min-max (acceptance + normalisation)
def _minmax(a): return float(np.nanmin(a)), float(np.nanmax(a))
SOC_lo,  SOC_hi  = _minmax(df_t["SOC"])
MIC_lo,  MIC_hi  = _minmax(df_t["MIC"])
MAOC_lo, MAOC_hi = _minmax(df_t["MAOC"])

# dSOC in %/y (log-SOC slope * 100); used for discard AND normalisation
dsoc_obs         = 100.0 * df_t["OC_linreg_slope"]
dSOC_lo, dSOC_hi = _minmax(dsoc_obs)

tgt_rng = {"SOC":  SOC_hi  - SOC_lo,
           "MIC":  MIC_hi  - MIC_lo,
           "MAOC": MAOC_hi - MAOC_lo,
           "dSOC": dSOC_hi - dSOC_lo}

def dsoc_pct(SOC_end, SOC0):        # model dSOC in %/y (mean per-year log change)
    return 100.0 * (jnp.log(SOC_end) - jnp.log(SOC0)) / T_YEARS

# ------------------------------------------------------------------ #
# 2. forward maps, target reducers and their gradients per version   #
# ------------------------------------------------------------------ #
def build_fns(version, mode):
    if mode == "static":
        def state_of(p):
            return analytical_steady_state(p, **version)
        fwd = jax.jit(jax.vmap(state_of))                       # (B,14)->(B,3)
        tfun = {"SOC":  lambda p: jnp.sum(state_of(p)),
                "MIC":  lambda p: state_of(p)[i_Cb],
                "MAOC": lambda p: state_of(p)[i_Cm]}
        grads = {t: jax.jit(jax.vmap(jax.grad(f))) for t, f in tfun.items()}
        return fwd, grads
    else:
        term = dfx.ODETerm(lambda t, y, a: craig_BA_adapt(t, y, a, **version))
        def state_of(p, y0):
            sol = dfx.diffeqsolve(term, dfx.Euler(), t0=0.0, t1=T_YEARS, dt0=0.05,
                                  y0=y0, args=p, saveat=dfx.SaveAt(t1=True))
            return sol.ys[-1]
        fwd = jax.jit(jax.vmap(state_of, in_axes=(0, 0)))       # (P,Y0)->(B,3)
        tfun = {"SOC":  lambda p, y0: jnp.sum(state_of(p, y0)),
                "MIC":  lambda p, y0: state_of(p, y0)[i_Cb],
                "MAOC": lambda p, y0: state_of(p, y0)[i_Cm],
                "dSOC": lambda p, y0: dsoc_pct(jnp.sum(state_of(p, y0)), jnp.sum(y0))}
        grads = {t: jax.jit(jax.vmap(jax.grad(f, argnums=(0, 1)), in_axes=(0, 0)))
                 for t, f in tfun.items()}
        return fwd, grads

def accept_static(SS):
    SOC, MIC, MAOC = SS.sum(1), SS[:, i_Cb], SS[:, i_Cm]
    return (np.isfinite(SOC) & np.isfinite(MIC) & np.isfinite(MAOC)
            & (SOC  >= SOC_lo)  & (SOC  <= SOC_hi)
            & (MIC  >= MIC_lo)  & (MIC  <= MIC_hi)
            & (MAOC >= MAOC_lo) & (MAOC <= MAOC_hi))

def accept_dynamic(SS, Y0):
    SOC, MIC, MAOC = SS.sum(1), SS[:, i_Cb], SS[:, i_Cm]
    S0 = Y0.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dS = 100.0 * (np.log(SOC) - np.log(S0)) / T_YEARS      # %/y, matches obs
    return (np.isfinite(SOC) & np.isfinite(MIC) & np.isfinite(MAOC) & np.isfinite(dS)
            & (SOC  >= SOC_lo)  & (SOC  <= SOC_hi)
            & (MIC  >= MIC_lo)  & (MIC  <= MIC_hi)
            & (MAOC >= MAOC_lo) & (MAOC <= MAOC_hi)
            & (dS   >= dSOC_lo) & (dS   <= dSOC_hi))            # <-- observed dSOC discard

# ------------------------------------------------------------------ #
# 3. loop over 12 versions x {static, dynamic}: reject-sample to 1000 #
# ------------------------------------------------------------------ #
def compute_sensitivities():
    results = []
    combo = 0
    for (dec, tur, sat), mode in product(
            product(["linear", "MM", "RMM"], ["linear", "density_dependent"], ["no", "Langmuir"]),
            ["static", "dynamic"]):
        version = dict(microbial_decomposition=dec, microbial_turnover=tur, saturation=sat)
        fwd, grads = build_fns(version, mode)
        key = jax.random.PRNGKey(SEED + combo); combo += 1

        P_acc  = np.empty((0, len(param_names)))
        Y0_acc = np.empty((0, len(state_names)))
        n_drawn = 0
        b = 0
        while P_acc.shape[0] < N_ACCEPT and b < MAX_BATCHES:        # <-- while until 1000 worked
            b += 1
            key, k1, k2 = jax.random.split(key, 3)
            P = param_mins + (param_maxs - param_mins) * jax.random.uniform(k1, (BATCH, len(param_names)))
            if mode == "dynamic":
                Y0 = y0_pool[jax.random.randint(k2, (BATCH,), 0, y0_pool.shape[0])]
                SS = np.asarray(fwd(P, Y0))
                ok = accept_dynamic(SS, np.asarray(Y0))
                Y0_acc = np.concatenate([Y0_acc, np.asarray(Y0)[ok]], axis=0)
            else:
                SS = np.asarray(fwd(P))
                ok = accept_static(SS)
            n_drawn += BATCH
            P_acc = np.concatenate([P_acc, np.asarray(P)[ok]], axis=0)

        n_ok = P_acc.shape[0]
        if n_ok < N_ACCEPT:
            print(f"[WARN] {mode} md={dec} mt={tur} sat={sat}: only {n_ok}/{N_ACCEPT} accepted "
                  f"after {MAX_BATCHES} batches -> using {n_ok}.")
        P_acc = jnp.asarray(P_acc[:N_ACCEPT])
        if mode == "dynamic":
            Y0_acc = jnp.asarray(Y0_acc[:N_ACCEPT])
        accept_rate = n_ok / max(n_drawn, 1)

        # --- gradients on accepted runs + normalisation ---
        for t in grads:
            if mode == "static":
                gp = np.abs(np.asarray(grads[t](P_acc)))
                ps = np.nanmedian(gp * param_diffs_np / tgt_rng[t], axis=0)
                ys = {f"y0_{s}": np.nan for s in state_names}
            else:
                gp, gy = grads[t](P_acc, Y0_acc)
                ps = np.nanmedian(np.abs(np.asarray(gp)) * param_diffs_np / tgt_rng[t], axis=0)
                ym = np.nanmedian(np.abs(np.asarray(gy)) * state_diffs   / tgt_rng[t], axis=0)
                ys = {f"y0_{s}": v for s, v in zip(state_names, ym)}
            results.append(dict(mode=mode, md=dec, mt=tur, sat=sat, target=t,
                                n_accept=n_ok, accept_rate=round(accept_rate, 5),
                                **dict(zip(param_names, ps)), **ys))
        print(f"[{mode}] md={dec} mt={tur} sat={sat} -- {n_ok} accepted (rate={accept_rate:.3g})")

    return pd.DataFrame(results).round(5)

# ------------------------------------------------------------------ #
# 4. build per-target LaTeX tables and write each to its own .tex     #
# ------------------------------------------------------------------ #
tables = [("SOC", "static"), ("MAOC", "static"), ("MIC", "static"), ("dSOC", "dynamic")]

md_s  = {"linear": "lin", "MM": "MM", "RMM": "RMM"}
mt_s  = {"linear": "lin", "density_dependent": "dd"}
sat_s = {"no": "no", "Langmuir": "Lang"}
versions = list(product(["linear", "MM", "RMM"], ["linear", "density_dependent"], ["no", "Langmuir"]))
vlabels  = [f"{md_s[d]}/{mt_s[t]}/{sat_s[s]}" for d, t, s in versions]

CMAP = matplotlib.colormaps["YlGnBu"]

def cell(v, vmin, vmax):
    if pd.isna(v) or float(v) == 0.0:                          # 0.0 / nan -> black
        return "\\cellcolor[HTML]{000000}{\\color{white}" + ("--" if pd.isna(v) else "0.000") + "}"
    sc = min(max((float(v) - vmin) / (vmax - vmin), 0.0), 1.0) if vmax > vmin else 1.0
    r, g, b, _ = CMAP(sc)
    hexc = f"{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
    tcol = "black" if (0.299*r + 0.587*g + 0.114*b) > 0.55 else "white"
    return f"\\cellcolor[HTML]{{{hexc}}}{{\\color{{{tcol}}}{float(v):.3f}}}"

def esc(s):
    return str(s).replace("_", r"\_")

def make_table(mats, tgt, mode):
    M = mats[(tgt, mode)]
    M = M[~((M.fillna(0.0) == 0.0).all(axis=1))]               # drop all-zero param rows

    # per-version (per-column) color scale, within this target's table
    colscale = {}
    for c in M.columns:
        nz = M[c].to_numpy(dtype=float)
        nz = nz[(~np.isnan(nz)) & (nz != 0.0)]
        if nz.size:
            lo, hi = float(nz.min()), float(nz.max())
            if lo == hi:
                lo = 0.0                                        # single value -> top shade
        else:
            lo, hi = 0.0, 1.0
        colscale[c] = (lo, hi)

    col_fmt = "l" + "c" * M.shape[1]
    head = "Param & " + " & ".join(f"\\rotatebox{{90}}{{{c}}}" for c in M.columns) + " \\\\"
    lines = [
        "\\begin{table}[ht]", "\\centering", "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.2}",
        f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule", head, "\\midrule",
    ]
    for p in M.index:
        lines.append(esc(p) + " & "
                     + " & ".join(cell(M.loc[p, c], *colscale[c]) for c in M.columns) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        f"\\caption{{Normalized parameter sensitivities for {tgt} "
        f"({'steady state' if mode == 'static' else 'dynamic'}); "
        f"YlGnBu shading per version, black = 0.}}",
        f"\\label{{tab:sens_{tgt}}}", "\\end{table}",
    ]
    return "\n".join(lines)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_all = compute_sensitivities()

    # ---- params x versions matrix per target ----
    mats = {}
    for tgt, mode in tables:
        cols = {}
        for (d, t, s), vl in zip(versions, vlabels):
            r = df_all[(df_all["mode"] == mode) & (df_all["md"] == d) & (df_all["mt"] == t)
                       & (df_all["sat"] == s) & (df_all["target"] == tgt)]
            cols[vl] = (r.iloc[0][param_names].astype(float).values if not r.empty
                        else np.full(len(param_names), np.nan))
        mats[(tgt, mode)] = pd.DataFrame(cols, index=param_names)   # params x versions

    header = ("% requires \\usepackage[table]{xcolor}, \\usepackage{graphicx} "
              "(for \\rotatebox) and \\usepackage{booktabs}\n")
    for tgt, mode in tables:
        out_path = os.path.join(OUT_DIR, f"sensitivity_{tgt}.tex")
        with open(out_path, "w") as f:
            f.write(header + "\n")
            f.write(make_table(mats, tgt, mode))
            f.write("\n")
        print(f"wrote {out_path}")

if __name__ == "__main__":
    main()

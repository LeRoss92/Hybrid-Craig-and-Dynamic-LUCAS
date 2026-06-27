import jax
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, QuantileRegressor, ElasticNet, BayesianRidge, Ridge, TweedieRegressor, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def craig_BA_adapt(t, y, p, microbial_decomposition="linear", microbial_turnover="linear", carbon_use_efficiency="constant", saturation="no"):
    Cp, Cb, Cm = y
    # Clamp pools for rate computations to avoid non-physical negative states
    # triggering unstable MM terms during integration.
    Cp_pos = jnp.maximum(Cp, 1e-12)
    Cb_pos = jnp.maximum(Cb, 1e-12)
    Cm_pos = jnp.maximum(Cm, 1e-12)
    I, CUE, beta, tmb, Cg0b, Cg0m, qx, Vmax_p, Vmax_m, Km_p, Km_m, kp, kb, km = p

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
            # Guard against non-physical negative biomass values causing NaNs for fractional beta.
            C_b_safe = jnp.maximum(C_b, 1e-12)
            return k_b * C_b_safe ** beta_i

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

    saturation_fraction = sat(tmb, Cg0m, Cm_pos, qx)
    total_turnover = mic_tur(kb, Cb_pos, beta)

    to_Cm = saturation_fraction * total_turnover
    to_Cp = total_turnover - to_Cm

    dCpdt = I - mic_dec(kp, Vmax_p, Cb_pos, Km_p, Cp_pos) + to_Cp
    dCbdt = (
        ca_us_ef(CUE, Cg0b, Cb_pos) * mic_dec(kp, Vmax_p, Cb_pos, Km_p, Cp_pos)
        + ca_us_ef(CUE, Cg0b, Cb_pos) * mic_dec(km, Vmax_m, Cb_pos, Km_m, Cm_pos)
        - total_turnover
    )
    dCmdt = to_Cm - mic_dec(km, Vmax_m, Cb_pos, Km_m, Cm_pos)

    return jnp.array([dCpdt, dCbdt, dCmdt])


def _positive_quadratic_root(a, b, c):
    """Unique positive root of a*x**2 + b*x + c = 0 for a > 0, c < 0."""
    disc = b * b - 4.0 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    return (-b + sqrt_disc) / (2.0 * a)


def analytical_steady_state(p, microbial_decomposition="linear", microbial_turnover="linear", saturation="no"):
    """Exact steady state [Cp*, Cb*, Cm*] of craig_BA_adapt (constant CUE).

    The three steady-state conditions dCp/dt = dCb/dt = dCm/dt = 0 are
        dec_p = I + (1 - S) * T      (particulate balance; S = saturation fraction)
        dec_m = S * T                (mineral balance)
        CUE * (dec_p + dec_m) = T    (biomass balance)
    where T is the total microbial turnover flux and dec_p, dec_m are the
    decomposition fluxes out of the particulate / mineral pools.

    Summing the three equations gives the mass-balance identity
        dec_p + dec_m = I + T   =>   CUE * (I + T) = T   =>   T = CUE * I / (1 - CUE),
    so T (and therefore the biomass Cb) is fixed by mass balance alone, independent
    of the decomposition and saturation mechanisms.

    Simplification: divide each decomposition balance by Cb to work with the
    *per-biomass* decomposition demand g = T / Cb (specific turnover; g = kb for
    linear turnover, g = kb * Cb**(beta-1) in general). With it,
        dm = dec_m / Cb = S * g                  (mineral demand per biomass)
        dp = dec_p / Cb = g * (1 - CUE * S) / CUE (particulate demand per biomass)
    using I / Cb = g * (1 - CUE) / CUE. Inverting the kinetics per unit biomass,
        linear : C = d * Cb / k
        MM     : C = d * Km / (Vmax - d)
        RMM    : C = d * (Km + Cb) / Vmax
    For MM with linear turnover g = kb is input-independent, so dm, dp and hence
    Cp*, Cm* are INDEPENDENT of I (the classic Michaelis-Menten substrate result);
    only the biomass Cb* ~ I carries the input. RMM keeps Cb inside (Km + Cb) and
    linear decomposition scales with I, so those pools do depend on I.

    Assumes constant carbon-use efficiency, matching craig_BA_adapt's default
    carbon_use_efficiency="constant". Parameter draws that admit no positive
    equilibrium yield non-physical (negative / NaN) values.
    """
    dec, tur, sat = microbial_decomposition, microbial_turnover, saturation
    I, CUE, beta, tmb, Cg0b, Cg0m, qx, Vmax_p, Vmax_m, Km_p, Km_m, kp, kb, km = p

    # Mass balance fixes total turnover T and biomass Cb (independent of sat/dec).
    T = CUE * I / (1.0 - CUE)

    if tur == "linear":
        Cb = T / kb
    elif tur == "density_dependent":
        Cb = (T / kb) ** (1.0 / beta)
    else:
        raise ValueError(f"unknown microbial_turnover: {tur}")

    g = T / Cb  # specific microbial turnover per unit biomass (= kb for linear turnover)

    # Mineral pool: per-biomass demand dm = S * g, with S = tmb (no) or 1 - Cm/Cg0m.
    if sat == "no":
        S = tmb
        dm = S * g
        if dec == "linear":
            Cm = S * T / km
        elif dec == "MM":
            Cm = dm * Km_m / (Vmax_m - dm)
        elif dec == "RMM":
            Cm = dm * (Km_m + Cb) / Vmax_m
        else:
            raise ValueError(f"unknown microbial_decomposition: {dec}")
    elif sat == "Langmuir":
        # S = 1 - Cm/Cg0m couples back into dm; solve for Cm per mechanism.
        if dec == "linear":
            Cm = T * Cg0m / (km * Cg0m + T)
        elif dec == "MM":
            a = g / Cg0m
            b = Vmax_m - g + g * Km_m / Cg0m
            c = -g * Km_m
            Cm = _positive_quadratic_root(a, b, c)
        elif dec == "RMM":
            Cm = g * (Km_m + Cb) * Cg0m / (Vmax_m * Cg0m + g * (Km_m + Cb))
        else:
            raise ValueError(f"unknown microbial_decomposition: {dec}")
        S = 1.0 - Cm / Cg0m
    else:
        raise ValueError(f"unknown saturation: {sat}")

    # Particulate pool: per-biomass demand dp = g * (1 - CUE * S) / CUE.
    dp = g * (1.0 - CUE * S) / CUE
    if dec == "linear":
        Cp = (I + (1.0 - S) * T) / kp
    elif dec == "MM":
        Cp = dp * Km_p / (Vmax_p - dp)
    elif dec == "RMM":
        Cp = dp * (Km_p + Cb) / Vmax_p

    return jnp.array([Cp, Cb, Cm])


def get_models(seed=4210):
    MODELS = {
        'LinReg': {
            'model': LinearRegression(),
            'params': {},
        },
        'Piecewise_Linear_Reg': {
            'model': Pipeline([
                ("spline", SplineTransformer(degree=1,n_knots=3,include_bias=False)),
                ("interactions", PolynomialFeatures(degree=2, interaction_only=True, # no squared terms (x1 with x1)
                    include_bias=False)), # already exists
                ("ridge", Ridge(alpha=1.0)) ]),
            'params': {
                'spline__n_knots': [3],    #, 5               # 2 = linear regression, 3=two pieces
                'spline__degree': [1],                       # linear vs polynomial splines
                'spline__include_bias': [False],                # True: x=y=0
                'interactions__degree': [2],    # 2            # 1: no interactions, 2: pairwise...
                'ridge__alpha': [0.01]},       #  0.001, 0.5          # 1e-6: minimal regularization for numerical stability
        },
        'XGB-n': {
            'model': XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                device="cpu",
                n_jobs=1,
                verbosity=0,
                random_state=seed),
            'params': {
                'n_estimators': [150],
                'max_depth': [5],
                'learning_rate': [0.05],
                'subsample': [0.8],
                'colsample_bytree': [0.8], # 0.5
                'min_child_weight': [5], # , 5
                'reg_alpha': [0.5],
                },
        },
        'XGB-1': {
            'model': XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                device="cpu",
                n_jobs=1,
                verbosity=0,
                random_state=seed),
            'params': {
                'n_estimators': [150],
                'max_depth': [1],
                'learning_rate': [0.05],
                'subsample': [0.8],
                'colsample_bytree': [0.8], # 0.5
                'min_child_weight': [5], # , 5
                # 'reg_lambda': [1.0, 5.0, 10.0],
                'reg_alpha': [0.5],
                },
        },
    }
    return MODELS

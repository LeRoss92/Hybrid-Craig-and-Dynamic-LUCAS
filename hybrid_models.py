import jax
import jax.numpy as jnp


def craig_BA_adapt(
    t,
    y,
    p,
    microbial_decomposition="linear",
    microbial_turnover="linear",
    carbon_use_efficiency="constant",
    saturation="no",):
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


def analytical_steady_state(
    p,
    microbial_decomposition="linear",
    microbial_turnover="linear",
    saturation="no",
    ):
    dec = microbial_decomposition
    tur = microbial_turnover
    sat = saturation

    # Unpack parameters
    I, CUE, beta, tmb, Cg0b, Cg0m, qx, Vmax_p, Vmax_m, Km_p, Km_m, kp, kb, km = p
    eps = 1e-12
    n_freundlich = 0.7 * qx

    def positive_quadratic_root(a, b, c):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        c = jnp.asarray(c)
        is_a_zero = jnp.abs(a) < eps
        root_lin = -c / b
        disc = b ** 2 - 4 * a * c
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0))
        r1 = (-b + sqrt_disc) / (2 * a + eps * is_a_zero)
        r2 = (-b - sqrt_disc) / (2 * a + eps * is_a_zero)
        nan = jnp.nan

        def choose_linear():
            return jnp.where(jnp.abs(b) < eps, nan, jnp.where(root_lin > 0, root_lin, nan))

        def choose_quadratic():
            valid_disc = disc >= 0
            candidates = jnp.stack(
                [
                    jnp.where((r1 > 0) & valid_disc, r1, nan),
                    jnp.where((r2 > 0) & valid_disc, r2, nan),
                ]
            )
            return jnp.nanmin(candidates)

        return jax.lax.cond(is_a_zero, choose_linear, choose_quadratic)

    def sat_value(Cm):
        if sat == "no":
            return tmb
        if sat == "Langmuir":
            return 1.0 - Cm / Cg0m
        if sat == "freundlich":
            return tmb * (1.0 - (Cm / Cg0m) ** n_freundlich)
        raise ValueError(f"Unsupported saturation option: {sat}")

    def bracketed_bisection(func, lower, upper, max_iter=200, tol=1e-12):
        xs = jnp.linspace(lower, upper, 200)
        fxs = jax.vmap(func)(xs)
        sign_change = jnp.diff(jnp.sign(fxs))
        has_crossing = jnp.any(sign_change != 0)
        idx_cross = jnp.argmax(sign_change != 0)

        def do_bisect(_):
            a = xs[idx_cross]
            b = xs[idx_cross + 1]
            fa = fxs[idx_cross]
            fb = fxs[idx_cross + 1]

            def body_fun(i, state):
                a_i, b_i, fa_i, fb_i, converged = state
                mid = 0.5 * (a_i + b_i)
                fm = func(mid)
                is_nan = jnp.isnan(fm)
                has_converged = jnp.abs(fm) < tol
                same_sign_as_a = jnp.sign(fm) == jnp.sign(fa_i)
                should_update = ~is_nan & ~converged
                new_a = jnp.where(should_update & same_sign_as_a, mid, a_i)
                new_fa = jnp.where(should_update & same_sign_as_a, fm, fa_i)
                new_b = jnp.where(should_update & ~same_sign_as_a, mid, b_i)
                new_fb = jnp.where(should_update & ~same_sign_as_a, fm, fb_i)
                new_converged = converged | has_converged
                return (new_a, new_b, new_fa, new_fb, new_converged)

            init_state = (a, b, fa, fb, False)
            final_a, final_b, _, _, _ = jax.lax.fori_loop(
                0, max_iter, body_fun, init_state
            )
            return 0.5 * (final_a + final_b)

        def fallback(_):
            return jnp.nan

        return jax.lax.cond(has_crossing, do_bisect, fallback, operand=None)

    Cp_star = jnp.nan
    Cb_star = jnp.nan
    Cm_star = jnp.nan

    if dec == "linear":
        Cp_star = jnp.where(kp > 0, I / kp, Cg0b)

        def solve_cm_linear():
            if sat == "no":
                denom = km * (1.0 - CUE * tmb)
                return jnp.where(denom <= eps, Cg0m, (tmb * CUE * I) / denom)
            if sat == "Langmuir":
                a = (CUE * km) / Cg0m
                b = (-CUE * km + km + (CUE * I) / Cg0m)
                c = -CUE * I
                return positive_quadratic_root(a, b, c)
            if sat == "freundlich":
                upper = jnp.maximum(Cg0m * (1.0 - 1e-9), 1.0)

                def func(cm):
                    cm_c = jnp.clip(cm, eps, Cg0m * (1.0 - 1e-9))
                    s_val = tmb * (1.0 - (cm_c / Cg0m) ** n_freundlich)
                    return s_val * CUE * (I + km * cm_c) - km * cm_c

                return bracketed_bisection(func, eps, upper)
            return jnp.nan

        Cm_star = solve_cm_linear()
        cm_valid = ~jnp.isnan(Cm_star)
        base = jnp.where(kb > 0, CUE * (I + km * Cm_star) / kb, Cg0b)
        base_valid = (base > 0) & cm_valid

        if tur == "linear":
            Cb_star = jnp.where(base_valid, base, Cg0b)
        elif tur == "density_dependent":
            Cb_star = jnp.where(base_valid, base ** (1.0 / beta), Cg0b)
        else:
            Cb_star = Cg0b

        Cb_star = jnp.where(cm_valid, Cb_star, Cg0b)
        Cm_star = jnp.where(cm_valid, Cm_star, Cg0m)

    elif dec == "MM":
        def solve_cm_mm_linear():
            if sat == "no":
                denom = Vmax_m - tmb * kb
                return jnp.where(denom <= eps, Cg0m, (tmb * kb * Km_m) / denom)
            if sat == "Langmuir":
                a = kb / Cg0m
                b = Vmax_m + (kb * Km_m) / Cg0m - kb
                c = -kb * Km_m
                return positive_quadratic_root(a, b, c)
            if sat == "freundlich":
                upper = jnp.maximum(Cg0m * (1.0 - 1e-9), 1.0)

                def func(cm):
                    cm_c = jnp.clip(cm, eps, Cg0m * (1.0 - 1e-9))
                    s_val = tmb * (1.0 - (cm_c / Cg0m) ** n_freundlich)
                    return s_val * kb - (Vmax_m * cm_c) / (Km_m + cm_c)

                return bracketed_bisection(func, eps, upper)
            return jnp.nan

        def solve_cm_mm_density():
            if sat == "no":
                S = tmb
                valid_S = (1.0 - CUE * S) > eps
                B = CUE * I / (kb * (1.0 - CUE * S) + eps * ~valid_S)
                valid_B = (B > 0) & valid_S
                exponent = (beta - 1.0) / beta
                E = B ** exponent
                denom = Vmax_m - E * S * kb
                valid_denom = (denom > eps) & valid_B
                result = (E * S * kb * Km_m) / (denom + eps * ~valid_denom)
                return jnp.where(valid_denom, result, Cg0m)

            def cm_equation(cm):
                cm_c = jnp.clip(cm, eps, Cg0m * (1.0 - 1e-9))
                S = sat_value(cm_c)
                valid_S = S > 0
                denom_B = kb * (1.0 - CUE * S)
                valid_denom_B = (denom_B > eps) & valid_S
                B = CUE * I / (denom_B + eps * ~valid_denom_B)
                valid_B = (B > 0) & valid_denom_B
                lhs = B ** ((beta - 1.0) / beta)
                rhs = (Vmax_m * cm_c) / (S * kb * (Km_m + cm_c))
                result = lhs - rhs
                return jnp.where(valid_B, result, jnp.nan)

            upper = (
                Cg0m * (1.0 - 1e-9)
                if sat in ("Langmuir", "freundlich")
                else jnp.maximum(10.0 * Km_m, 1.0)
            )
            return bracketed_bisection(cm_equation, eps, upper)

        if tur == "linear":
            Cm_star = solve_cm_mm_linear()
            cm_valid = ~jnp.isnan(Cm_star)
            denom_cb = kb - (CUE * Vmax_m * Cm_star) / (Km_m + Cm_star)
            cb_valid = (denom_cb > eps) & cm_valid
            Cb_star = jnp.where(cb_valid, (CUE * I) / denom_cb, Cg0b)
            Cm_star = jnp.where(cm_valid, Cm_star, Cg0m)
        elif tur == "density_dependent":
            Cm_star = solve_cm_mm_density()
            cm_valid = ~jnp.isnan(Cm_star)
            S = sat_value(Cm_star)
            s_valid = (S > 0) & cm_valid
            B = CUE * I / (kb * (1.0 - CUE * S))
            b_valid = (B > 0) & s_valid
            Cb_star = jnp.where(b_valid, B ** (1.0 / beta), Cg0b)
            Cm_star = jnp.where(cm_valid, Cm_star, Cg0m)
        else:
            Cb_star = Cg0b
            Cm_star = Cg0m

        denom_cp = Vmax_p * Cb_star - I
        Cp_star = jnp.where(denom_cp > eps, (I * Km_p) / denom_cp, Cg0b)

    elif dec == "RMM":
        def solve_cm_rmm_linear():
            if sat == "no":
                denom = kb * (1.0 - CUE * tmb)
                cb_valid = denom > eps
                Cb = jnp.where(cb_valid, (CUE * I) / denom, Cg0b)
                Cm = jnp.where(cb_valid, tmb * kb * (Km_m + Cb) / Vmax_m, Cg0m)
                return Cb, Cm

            if sat == "Langmuir":
                def equation(Cm):
                    Cm_c = jnp.clip(Cm, eps, Cg0m * (1.0 - 1e-9))
                    S = 1.0 - Cm_c / Cg0m
                    denom_cb = kb * (1.0 - CUE * S)
                    valid = denom_cb > eps
                    Cb = jnp.where(valid, (CUE * I) / denom_cb, jnp.nan)
                    Cm_calc = S * kb * (Km_m + Cb) / Vmax_m
                    return jnp.where(valid, Cm_c - Cm_calc, jnp.nan)

                upper = Cg0m * (1.0 - 1e-9)
                Cm = bracketed_bisection(equation, eps, upper)
                cm_valid = ~jnp.isnan(Cm)
                S = 1.0 - Cm / Cg0m
                Cb = jnp.where(
                    cm_valid, (CUE * I) / (kb * (1.0 - CUE * S)), Cg0b
                )
                return Cb, Cm

            if sat == "freundlich":
                def equation(Cm):
                    Cm_c = jnp.clip(Cm, eps, Cg0m * (1.0 - 1e-9))
                    S = tmb * (1.0 - (Cm_c / Cg0m) ** n_freundlich)
                    denom_cb = kb * (1.0 - CUE * S)
                    valid = denom_cb > eps
                    Cb = jnp.where(valid, (CUE * I) / denom_cb, jnp.nan)
                    Cm_calc = S * kb * (Km_m + Cb) / Vmax_m
                    return jnp.where(valid, Cm_c - Cm_calc, jnp.nan)

                upper = Cg0m * (1.0 - 1e-9)
                Cm = bracketed_bisection(equation, eps, upper)
                cm_valid = ~jnp.isnan(Cm)
                S = tmb * (1.0 - (Cm / Cg0m) ** n_freundlich)
                Cb = jnp.where(
                    cm_valid, (CUE * I) / (kb * (1.0 - CUE * S)), Cg0b
                )
                return Cb, Cm

            return Cg0b, Cg0m

        def solve_cm_rmm_density():
            if sat == "no":
                S = tmb
                denom = kb * (1.0 - CUE * S)
                valid_denom = denom > eps
                base = jnp.where(valid_denom, (CUE * I) / denom, Cg0b)
                valid_base = (base > 0) & valid_denom
                Cb = jnp.where(valid_base, base ** (1.0 / beta), Cg0b)
                Cm = jnp.where(
                    valid_base & (Cb > 0),
                    S * kb * base * (Km_m + Cb) / (Vmax_m * Cb),
                    Cg0m,
                )
                return Cb, Cm

            def equation(Cm):
                Cm_c = jnp.clip(Cm, eps, Cg0m * (1.0 - 1e-9))
                S = sat_value(Cm_c)
                valid_S = S > 0
                denom = kb * (1.0 - CUE * S)
                valid_denom = (denom > eps) & valid_S
                base = jnp.where(valid_denom, (CUE * I) / denom, jnp.nan)
                valid_base = (base > 0) & valid_denom
                Cb = jnp.where(valid_base, base ** (1.0 / beta), jnp.nan)
                Cm_calc = jnp.where(
                    valid_base & (Cb > 0),
                    S * kb * base * (Km_m + Cb) / (Vmax_m * Cb),
                    jnp.nan,
                )
                return jnp.where(valid_base, Cm_c - Cm_calc, jnp.nan)

            upper = (
                Cg0m * (1.0 - 1e-9)
                if sat in ("Langmuir", "freundlich")
                else jnp.maximum(10.0 * Km_m, 1.0)
            )
            Cm = bracketed_bisection(equation, eps, upper)
            cm_valid = ~jnp.isnan(Cm)
            S = sat_value(Cm)
            base = (CUE * I) / (kb * (1.0 - CUE * S))
            Cb = jnp.where(cm_valid & (base > 0), base ** (1.0 / beta), Cg0b)
            return Cb, Cm

        if tur == "linear":
            Cb_star, Cm_star = solve_cm_rmm_linear()
        elif tur == "density_dependent":
            Cb_star, Cm_star = solve_cm_rmm_density()
        else:
            Cb_star = Cg0b
            Cm_star = Cg0m

        valid_cb = (Cb_star > 0) & ~jnp.isnan(Cb_star)
        Cp_star = jnp.where(
            valid_cb, I * (Km_p + Cb_star) / (Vmax_p * Cb_star), Cg0b
        )

    return jnp.array([Cp_star, Cb_star, Cm_star])

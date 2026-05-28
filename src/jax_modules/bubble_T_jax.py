"""
Bubble-Point Temperature Solver in Pure JAX (Newton-Raphson).
=============================================================
Two-phase solver: Wilson Newton (robust init) → PR Newton (kij gradient).

Key design decisions:
- Phase A: Wilson K Newton (smooth, monotonic, always converges) → T_wilson
- Phase B: PR K Newton from T_wilson (kij gradient flows through φ_L, φ_V)
- lax.stop_gradient on T_wilson (Wilson K has no kij dependency)
- Vapor composition estimated with Wilson K (avoids K=1 collapse at high P)

Gradient path: kij → mixture_ab → φ_L, φ_V → K_PR → f(T) → T*
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
from jax import lax

from src.jax_modules.pr_eos_jax import (
    fugacity_coefficients as fugacity_coefficients_jax,
    wilson_K as wilson_K_jax,
)


def _wilson_bubble_obj(T_val, x, P, Tc, Pc, omega):
    """Wilson-only bubble-point objective: f(T) = Σ(x_i * K_wilson_i(T)) - 1.

    Smooth, monotonic, single root — Newton always converges.
    No kij dependency (Wilson K is a vapor pressure correlation).
    """
    K_w = wilson_K_jax(T_val, P, Tc, Pc, omega)
    return jnp.sum(x * K_w) - 1.0


def _pr_bubble_obj(T_val, x, P, Tc, Pc, omega, kij):
    """
    PR-EOS bubble-point objective: f(T) = Σ(x_i * K_PR_i(T)) - 1.

    K_PR from fugacity ratio, with Wilson K for vapor composition estimate.
    kij gradient flows through both φ_L(kij) and φ_V(kij) pathways.
    """
    # Liquid fugacity coefficients
    ln_phi_L = fugacity_coefficients_jax(T_val, P, x, Tc, Pc, omega, kij, 'liquid')

    # Vapor estimate from Wilson K (not PR K — avoids y→x collapse)
    K_w = wilson_K_jax(T_val, P, Tc, Pc, omega)
    y_est = x * K_w
    y_sum = jnp.sum(y_est)
    y_est = jnp.where(y_sum > 1e-30, y_est / y_sum, x)

    # Vapor fugacity at Wilson-estimated composition
    ln_phi_V = fugacity_coefficients_jax(T_val, P, y_est, Tc, Pc, omega, kij, 'vapor')

    # K from fugacity ratio
    K_pr = jnp.exp(ln_phi_L - ln_phi_V)
    K_pr = jnp.clip(K_pr, 1e-8, 1e8)

    return jnp.sum(x * K_pr) - 1.0


def solve_bubble_T_newton(x, P, T_guess, Tc, Pc, omega, kij,
                           max_iter_wilson=8, max_iter_pr=8):
    """
    Two-phase bubble-T solver: Wilson init → PR refinement.

    Phase A: Newton on Wilson K objective (robust, always converges)
    Phase B: Newton on PR K objective starting from T_wilson (kij gradient flows)

    Parameters
    ----------
    x : (nc,) liquid mole fractions
    P : scalar, pressure in bar
    T_guess : scalar, initial temperature guess in K
    Tc, Pc, omega : (nc,) pure component properties
    kij : (nc, nc) binary interaction parameters
    max_iter_wilson : int, Newton iterations for Wilson phase
    max_iter_pr : int, Newton iterations for PR refinement phase

    Returns
    -------
    T : scalar, bubble-point temperature in K
    """
    x = jnp.maximum(x, 1e-15)
    x = x / jnp.sum(x)

    # === Phase A: Wilson bubble-T (robust initialization) ===
    def wilson_obj(T_val):
        return _wilson_bubble_obj(T_val, x, P, Tc, Pc, omega)

    dwilson_dT = jax.grad(wilson_obj)

    T = T_guess
    for _ in range(max_iter_wilson):
        f_val = wilson_obj(T)
        df_val = dwilson_dT(T)
        dT = -f_val / jnp.where(jnp.abs(df_val) > 1e-30, df_val, 1e-30)
        dT = jnp.clip(dT, -30.0, 30.0)
        T = T + dT
        T = jnp.clip(T, 270.0, 530.0)

    # Stop gradient: Wilson K has no kij dependency, skip backward tracing
    T_wilson = lax.stop_gradient(T)

    # === Phase B: PR Newton refinement (kij gradient flows here) ===
    def pr_obj(T_val):
        return _pr_bubble_obj(T_val, x, P, Tc, Pc, omega, kij)

    dpr_dT = jax.grad(pr_obj)

    T = T_wilson
    for _ in range(max_iter_pr):
        f_val = pr_obj(T)
        df_val = dpr_dT(T)
        dT = -f_val / jnp.where(jnp.abs(df_val) > 1e-30, df_val, 1e-30)
        dT = jnp.clip(dT, -20.0, 20.0)  # Tighter clip for refinement
        T = T + dT
        T = jnp.clip(T, 270.0, 530.0)

    return T

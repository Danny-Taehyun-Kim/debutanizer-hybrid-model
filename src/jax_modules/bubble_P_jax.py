"""
Bubble-Point Pressure Solver in Pure JAX (Newton-Raphson).
==========================================================
Two-phase solver: Wilson Newton (robust init) → PR Newton (kij gradient).

Mirrors bubble_T_jax.py structure but solves for P at fixed T.

Key design decisions:
- Phase A: Wilson K Newton (smooth, monotonic, always converges) → P_wilson
- Phase B: PR K Newton from P_wilson (kij gradient flows through φ_L, φ_V)
- lax.stop_gradient on P_wilson (Wilson K has no kij dependency)
- Vapor composition estimated with Wilson K (avoids K=1 collapse)

Gradient path: kij → mixture_ab → φ_L, φ_V → K_PR → f(P) → P*

Usage:
    P_bub = solve_bubble_P_newton(x, T_fixed, P_guess, Tc, Pc, omega, kij)
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


def _wilson_bubble_P_obj(P_val, x, T, Tc, Pc, omega):
    """Wilson-only bubble-P objective: f(P) = Σ(x_i * K_wilson_i(P)) - 1.

    Smooth, monotonic in P, single root — Newton always converges.
    No kij dependency (Wilson K is a vapor pressure correlation).
    """
    K_w = wilson_K_jax(T, P_val, Tc, Pc, omega)
    return jnp.sum(x * K_w) - 1.0


def _pr_bubble_P_obj(P_val, x, T, Tc, Pc, omega, kij):
    """
    PR-EOS bubble-P objective: f(P) = Σ(x_i * K_PR_i(P)) - 1.

    K_PR from fugacity ratio, with Wilson K for vapor composition estimate.
    kij gradient flows through both φ_L(kij) and φ_V(kij) pathways.
    """
    # Liquid fugacity coefficients
    ln_phi_L = fugacity_coefficients_jax(T, P_val, x, Tc, Pc, omega, kij, 'liquid')

    # Vapor estimate from Wilson K (not PR K — avoids y→x collapse)
    K_w = wilson_K_jax(T, P_val, Tc, Pc, omega)
    y_est = x * K_w
    y_sum = jnp.sum(y_est)
    y_est = jnp.where(y_sum > 1e-30, y_est / y_sum, x)

    # Vapor fugacity at Wilson-estimated composition
    ln_phi_V = fugacity_coefficients_jax(T, P_val, y_est, Tc, Pc, omega, kij, 'vapor')

    # K from fugacity ratio
    K_pr = jnp.exp(ln_phi_L - ln_phi_V)
    K_pr = jnp.clip(K_pr, 1e-8, 1e8)

    return jnp.sum(x * K_pr) - 1.0


def solve_bubble_P_newton(x, T_fixed, P_guess, Tc, Pc, omega, kij,
                           max_iter_wilson=8, max_iter_pr=8):
    """
    Two-phase bubble-P solver: Wilson init → PR refinement.

    Phase A: Newton on Wilson K objective (robust, always converges)
    Phase B: Newton on PR K objective starting from P_wilson (kij gradient flows)

    Parameters
    ----------
    x : (nc,) liquid mole fractions
    T_fixed : scalar, fixed temperature in K
    P_guess : scalar, initial pressure guess in bar
    Tc, Pc, omega : (nc,) pure component properties
    kij : (nc, nc) binary interaction parameters
    max_iter_wilson : int, Newton iterations for Wilson phase
    max_iter_pr : int, Newton iterations for PR refinement phase

    Returns
    -------
    P : scalar, bubble-point pressure in bar
    """
    x = jnp.maximum(x, 1e-15)
    x = x / jnp.sum(x)

    # === Phase A: Wilson bubble-P (robust initialization) ===
    def wilson_obj(P_val):
        return _wilson_bubble_P_obj(P_val, x, T_fixed, Tc, Pc, omega)

    dwilson_dP = jax.grad(wilson_obj)

    P = P_guess
    for _ in range(max_iter_wilson):
        f_val = wilson_obj(P)
        df_val = dwilson_dP(P)
        dP = -f_val / jnp.where(jnp.abs(df_val) > 1e-30, df_val, 1e-30)
        dP = jnp.clip(dP, -3.0, 3.0)
        P = P + dP
        P = jnp.clip(P, 0.5, 30.0)

    # Stop gradient: Wilson K has no kij dependency, skip backward tracing
    P_wilson = lax.stop_gradient(P)

    # === Phase B: PR Newton refinement (kij gradient flows here) ===
    def pr_obj(P_val):
        return _pr_bubble_P_obj(P_val, x, T_fixed, Tc, Pc, omega, kij)

    dpr_dP = jax.grad(pr_obj)

    P = P_wilson
    for _ in range(max_iter_pr):
        f_val = pr_obj(P)
        df_val = dpr_dP(P)
        dP = -f_val / jnp.where(jnp.abs(df_val) > 1e-30, df_val, 1e-30)
        dP = jnp.clip(dP, -2.0, 2.0)  # Tighter clip for refinement
        P = P + dP
        P = jnp.clip(P, 0.5, 30.0)

    return P

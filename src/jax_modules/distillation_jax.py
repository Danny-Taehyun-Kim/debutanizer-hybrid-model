"""
JAX Differentiable Distillation Column Solver (2nd implementation)
==================================================================
Mirrors distillation.py (numpy) but uses JAX for automatic differentiation.

Key changes from 1st attempt:
- Bubble-T: pure JAX Newton (no brentq/custom_vjp) from bubble_T_jax.py
- Thomas: from thomas_jax.py
- No scipy dependency (except initialization)
- Gradient strategy handled separately (partial unrolling / implicit diff)
- vmap over stages for bubble-T and K-value updates (performance)
- JIT-compiled iteration function

Gradient path: kij -> mixture_ab -> fugacity -> K-values -> column solve -> xD, T
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
from scipy.optimize import brentq

from src.jax_modules.thomas_jax import thomas_solve as thomas_solve_jax
from src.jax_modules.bubble_T_jax import solve_bubble_T_newton
from src.jax_modules.pr_eos_jax import (
    fugacity_coefficients as fugacity_coefficients_jax,
    wilson_K as wilson_K_jax,
)
from src.pr_eos import wilson_K as wilson_K_np


# ============================================================
# Helper functions
# ============================================================

# Pre-computed damping schedule
_DAMP_SCHEDULE = jnp.array(
    [0.3] * 10 + [0.5] * 20 + [0.7] * 30 + [0.9] * 90
)
_N_ITER = 30


def _build_tridiagonal_coeffs(L, V, S, K_eff_i, f_stage, z_Fi, N):
    """Build tridiagonal coefficients for component i (vectorized over stages)."""
    j = jnp.arange(N)

    # Sub-diagonal: L[j-1] for j > 0, else 0
    a_coeff = jnp.where(j > 0, L[jnp.maximum(j - 1, 0)], 0.0)

    # Super-diagonal: V[j+1] * K_eff[j+1, i] for j < N-1, else 0
    j_plus = jnp.minimum(j + 1, N - 1)
    c_coeff = jnp.where(j < N - 1, V[j_plus] * K_eff_i[j_plus], 0.0)

    # Diagonal: -(L[j] + S[j] + V[j] * K_eff[j, i])
    b_coeff = -(L[j] + S[j] + V[j] * K_eff_i[j])

    # RHS: -f_stage[j] * z_F[i]
    d_coeff = -f_stage * z_Fi

    return a_coeff, b_coeff, c_coeff, d_coeff


def _normalize_rows(x):
    """Normalize each row of x to sum to 1, safely."""
    row_sums = jnp.sum(x, axis=1, keepdims=True)
    safe_sums = jnp.where(row_sums > 1e-30, row_sums, 1.0)
    return x / safe_sums


def _enforce_monotonicity(T):
    """Enforce T[j] >= T[j-1] + 0.1 (temperature increases top to bottom)."""
    def scan_fn(T_prev, T_j):
        T_corrected = jnp.maximum(T_j, T_prev + 0.1)
        return T_corrected, T_corrected

    _, T_mono = lax.scan(scan_fn, T[0], T[1:])
    return jnp.concatenate([T[:1], T_mono])


# ============================================================
# JIT-compiled iteration with vmap over stages
# ============================================================

def _one_damped_iteration(x, T, K, kij, P, L, V, S, f_stage, z_F,
                           Tc, Pc, omega, N, efficiency, damp):
    """One damped column iteration step (vmap over stages).

    1. Compute K_eff with Murphree efficiency
    2. Tridiagonal solve for each component (vmap over components)
    3. Bubble-T for each stage (vmap over stages)
    4. Update K-values (vmap over stages)

    Returns updated (x, T, K).
    """
    j = jnp.arange(N)

    # K_eff with Murphree efficiency (always apply - when eff=1.0, gives K)
    eff_mask = (j >= 1) & (j <= N - 2)
    K_eff = jnp.where(eff_mask[:, None], 1.0 + efficiency * (K - 1.0), K)

    # Step 1: Tridiagonal solve for each component (vmap over nc)
    def solve_one_component(K_eff_i, z_Fi):
        a_c, b_c, c_c, d_c = _build_tridiagonal_coeffs(
            L, V, S, K_eff_i, f_stage, z_Fi, N
        )
        x_i = thomas_solve_jax(a_c, b_c, c_c, d_c)
        return jnp.maximum(x_i, 1e-15)

    # K_eff: (N, nc) -> vmap over axis 1 gives nc calls with (N,) each
    # z_F: (nc,) -> vmap over axis 0 gives nc scalars
    x_new = jax.vmap(solve_one_component, in_axes=(1, 0))(K_eff, z_F)  # (nc, N)
    x = _normalize_rows(x_new.T)  # (N, nc)

    # Step 2: Bubble-T for all stages (vmap over stages)
    def bubble_T_one_stage(x_j, P_j, T_j):
        return solve_bubble_T_newton(x_j, P_j, T_j, Tc, Pc, omega, kij)

    T_bp = jax.vmap(bubble_T_one_stage)(x, P, T)  # (N,)
    # Reject bubble-T values that hit clip limit (solver diverged)
    T_bp_safe = jnp.where(T_bp >= 529.0, T, T_bp)
    T_new = T + damp * (T_bp_safe - T)
    T_new = jnp.clip(T_new, 270.0, 530.0)
    T_new = _enforce_monotonicity(T_new)

    # Step 3: K-value update (vmap over stages)
    def compute_K_one_stage(x_j, K_old_j, T_j, P_j):
        # Liquid fugacity
        ln_phi_L = fugacity_coefficients_jax(
            T_j, P_j, x_j, Tc, Pc, omega, kij, 'liquid')

        # Vapor estimate from current K
        y_est = K_old_j * x_j
        y_sum = jnp.sum(y_est)
        y_est = jnp.where(y_sum > 1e-30, y_est / y_sum, x_j)

        # Vapor fugacity
        ln_phi_V = fugacity_coefficients_jax(
            T_j, P_j, y_est, Tc, Pc, omega, kij, 'vapor')

        # K from fugacity ratio
        K_j_new = jnp.exp(ln_phi_L - ln_phi_V)
        K_j_new = jnp.clip(K_j_new, 1e-8, 1e8)

        # Damped K update: K = K_old * (K_new / K_old)^damp
        ratio = K_j_new / jnp.maximum(K_old_j, 1e-30)
        ratio = jnp.clip(ratio, 1e-4, 1e4)
        K_j_updated = K_old_j * ratio ** damp

        # Fallback: if NaN/Inf, use Wilson
        K_wilson = wilson_K_jax(T_j, P_j, Tc, Pc, omega)
        K_valid = jnp.all(jnp.isfinite(K_j_updated))
        K_j_updated = jnp.where(K_valid, K_j_updated, K_wilson)

        # K-value collapse guard: when PR gives K≈1 (single-root regime),
        # blend with Wilson K to maintain meaningful phase separation
        max_dev = jnp.max(jnp.abs(K_j_updated - 1.0))
        blend_alpha = jnp.where(max_dev < 0.05, 0.5, 0.0)
        K_j_updated = (1.0 - blend_alpha) * K_j_updated + blend_alpha * K_wilson

        return K_j_updated

    K_new = jax.vmap(compute_K_one_stage)(x, K, T_new, P)  # (N, nc)

    return x, T_new, K_new


# JIT-wrapped version for standalone use (outside outer JIT)
_one_damped_iteration_jit = partial(jax.jit, static_argnums=(13,))(_one_damped_iteration)


# ============================================================
# Column Solver
# ============================================================

def compute_T_init(N, z_F_np, P_top, P_bot, Tc_np, Pc_np, omega_np):
    """Compute initial temperature profile using Wilson bubble-T (numpy/scipy).

    This is NOT JAX-traceable. Must be called outside JIT.
    Returns numpy array of shape (N,).
    """
    try:
        def wilson_bp_obj(T_trial, P_trial, x_trial):
            K = wilson_K_np(T_trial, P_trial, Tc_np, Pc_np, omega_np)
            return float(np.sum(x_trial * K) - 1.0)

        T_top_est = brentq(lambda T: wilson_bp_obj(T, float(P_top), z_F_np),
                           280, 500, xtol=0.5)
        T_bot_est = brentq(lambda T: wilson_bp_obj(T, float(P_bot), z_F_np),
                           280, 500, xtol=0.5)
        return np.linspace(T_top_est, T_bot_est, N)
    except Exception:
        return np.linspace(340.0, 430.0, N)


def solve_column_jax(N, NF, F, z_F, D, R, P_top, P_bot,
                     Tc, Pc, omega, kij, q=1.0, efficiency=1.0,
                     n_iter=_N_ITER, n_iter_grad=None, T_init=None):
    """
    JAX differentiable distillation column solver.

    Parameters
    ----------
    N : int, total stages
    NF : int, feed stage (1-indexed)
    F, D, R : scalars, flow rates and reflux ratio
    z_F : array (nc,), feed composition
    P_top, P_bot : scalars, pressures (bar)
    Tc, Pc, omega : arrays (nc,)
    kij : array (nc, nc), binary interaction parameters
    q : scalar, feed quality
    efficiency : scalar, Murphree efficiency
    n_iter : int, total iteration count
    n_iter_grad : int or None, if set, only last n_iter_grad iterations
                  propagate gradient (partial unrolling strategy)
    T_init : array (N,), optional initial temperature profile.
             If None, uses Wilson bubble-T (requires scipy, not JIT-compatible).

    Returns
    -------
    dict with 'T', 'x', 'y', 'K', 'L', 'V', 'P', 'iterations'
    """
    nc = z_F.shape[0]
    z_F = jnp.maximum(z_F, 1e-15)
    z_F = z_F / jnp.sum(z_F)

    B_flow = F - D
    nf = NF - 1  # 0-indexed

    # Pressure profile
    P = jnp.linspace(P_top, P_bot, N)

    # CMO flow rates
    L_rect = R * D
    V_rect = (R + 1.0) * D
    L_strip = R * D + q * F
    V_strip = (R + 1.0) * D - (1.0 - q) * F

    j = jnp.arange(N)
    L = jnp.zeros(N)
    V = jnp.zeros(N)

    # Condenser
    L = L.at[0].set(R * D)
    V = V.at[0].set(0.0)
    # Rectifying
    rect_mask = (j >= 1) & (j < nf)
    L = jnp.where(rect_mask, L_rect, L)
    V = jnp.where(rect_mask, V_rect, V)
    # Stripping
    strip_mask = (j >= nf)
    L = jnp.where(strip_mask, L_strip, L)
    V = jnp.where(strip_mask, V_strip, V)

    # Side streams
    S = jnp.zeros(N)
    S = S.at[0].set(D)
    S = S.at[N - 1].set(B_flow)

    # Feed array
    f_stage = jnp.zeros(N)
    f_stage = f_stage.at[nf].set(F)

    # ---- Initialization (no gradient needed) ----
    if T_init is None:
        # Fallback: use scipy brentq (NOT JIT-compatible)
        Tc_np = np.array(Tc)
        Pc_np = np.array(Pc)
        omega_np = np.array(omega)
        z_F_np = np.array(lax.stop_gradient(z_F))
        T_init = compute_T_init(N, z_F_np, float(P_top), float(P_bot),
                                Tc_np, Pc_np, omega_np)

    T = jnp.array(T_init)

    # Initialize K from Wilson (vmap over stages)
    K = jax.vmap(lambda T_j, P_j: wilson_K_jax(T_j, P_j, Tc, Pc, omega))(T, P)

    # Initialize compositions
    x = jnp.broadcast_to(z_F, (N, nc)) + 0.0  # +0.0 to make a concrete copy

    # ---- Determine gradient split ----
    if n_iter_grad is None:
        n_iter_no_grad = 0
        n_iter_with_grad = n_iter
    else:
        n_iter_no_grad = max(0, n_iter - n_iter_grad)
        n_iter_with_grad = n_iter_grad

    # Convert efficiency to JAX scalar for tracing
    eff_val = jnp.float64(efficiency)

    # ---- lax.scan-based iteration (single loop body in XLA graph) ----
    # Closes over constants to avoid passing N as dynamic argument.
    # lax.scan creates ONE compiled loop body instead of 30 unrolled copies.

    def _scan_body(carry, damp_val):
        """One iteration as lax.scan body. N, kij, P, etc. captured via closure."""
        x_c, T_c, K_c = carry
        x_c, T_c, K_c = _one_damped_iteration(
            x_c, T_c, K_c, kij, P, L, V, S, f_stage, z_F,
            Tc, Pc, omega, N, eff_val, damp_val
        )
        return (x_c, T_c, K_c), None

    # Phase 1: Iterations WITHOUT gradient
    if n_iter_no_grad > 0:
        damps_no_grad = _DAMP_SCHEDULE[:n_iter_no_grad]
        (x, T, K), _ = lax.scan(_scan_body, (x, T, K), damps_no_grad)
        x = lax.stop_gradient(x)
        T = lax.stop_gradient(T)
        K = lax.stop_gradient(K)

    # Phase 2: Iterations WITH gradient (checkpoint for memory efficiency)
    damps_grad = _DAMP_SCHEDULE[n_iter_no_grad:n_iter_no_grad + n_iter_with_grad]
    _scan_body_ckpt = jax.checkpoint(_scan_body)
    (x, T, K), _ = lax.scan(_scan_body_ckpt, (x, T, K), damps_grad)

    # Output (always compute K_eff; when eff=1.0, result equals K)
    eff_mask = (jnp.arange(N) >= 1) & (jnp.arange(N) <= N - 2)
    K_eff = jnp.where(eff_mask[:, None], 1.0 + efficiency * (K - 1.0), K)
    y = K_eff * x
    y = _normalize_rows(y)

    return {
        'T': T, 'x': x, 'y': y, 'K': K,
        'L': L, 'V': V, 'P': P, 'S': S,
        'f_stage': f_stage, 'z_F': z_F,
        'iterations': n_iter,
    }

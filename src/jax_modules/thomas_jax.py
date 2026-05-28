"""
Thomas Algorithm (Tridiagonal Matrix Solver) in JAX.
=====================================================
Extracted from distillation_jax.py for modular Phase 2 implementation.
Uses jax.lax.scan for forward elimination and back substitution.
Verified against numpy thomas_solve: max error ~1e-16.
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax.numpy as jnp
from jax import lax


def thomas_solve(a, b, c, d):
    """
    Tridiagonal matrix solver (Thomas algorithm) in pure JAX.

    Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    with a[0] = 0, c[n-1] = 0.

    Uses lax.scan for forward elimination and back substitution.

    Parameters
    ----------
    a : (n,) sub-diagonal coefficients (a[0] unused)
    b : (n,) diagonal coefficients
    c : (n,) super-diagonal coefficients (c[n-1] unused)
    d : (n,) right-hand side

    Returns
    -------
    x : (n,) solution vector
    """
    n = d.shape[0]

    # Forward elimination
    def forward_step(carry, inputs):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, d_i = inputs

        m = b_i - a_i * cp_prev
        safe_m = jnp.where(jnp.abs(m) > 1e-30, m, 1e-30)
        cp_i = c_i / safe_m
        dp_i = (d_i - a_i * dp_prev) / safe_m

        return (cp_i, dp_i), (cp_i, dp_i)

    # Initial values
    safe_b0 = jnp.where(jnp.abs(b[0]) > 1e-30, b[0], 1e-30)
    cp0 = c[0] / safe_b0
    dp0 = d[0] / safe_b0

    # Scan over indices 1..n-1
    _, (cp_rest, dp_rest) = lax.scan(
        forward_step,
        (cp0, dp0),
        (a[1:], b[1:], c[1:], d[1:])
    )

    # Combine
    cp = jnp.concatenate([jnp.array([cp0]), cp_rest])
    dp = jnp.concatenate([jnp.array([dp0]), dp_rest])

    # Back substitution
    def backward_step(x_next, i):
        x_i = dp[i] - cp[i] * x_next
        return x_i, x_i

    x_last = dp[n - 1]
    indices = jnp.arange(n - 2, -1, -1)
    _, x_rest = lax.scan(backward_step, x_last, indices)

    # x_rest is in reverse order, flip it
    x = jnp.concatenate([x_rest[::-1], jnp.array([x_last])])

    return x

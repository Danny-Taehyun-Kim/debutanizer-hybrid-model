"""
Peng-Robinson Equation of State — JAX differentiable implementation.

Mirrors pr_eos.py (numpy) but uses JAX for automatic differentiation.
Key difference: solve_cubic_Z uses custom_vjp with Implicit Function Theorem
for differentiable root selection.

Units: R = 83.14472 cm3 bar/(mol K), P in bar, T in K, V in cm3/mol
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
from jax import custom_vjp

R_GAS = 83.14472  # cm3 bar / (mol K)


def compute_kappa(omega):
    """PR kappa parameter from acentric factor (vectorized)."""
    return 0.37464 + 1.54226 * omega - 0.26992 * omega**2


def compute_a_alpha(T, Tc, Pc, omega):
    """Pure component a*alpha(T) for PR EOS (vectorized over components)."""
    kappa = compute_kappa(omega)
    alpha = (1.0 + kappa * (1.0 - jnp.sqrt(T / Tc)))**2
    ac = 0.45724 * R_GAS**2 * Tc**2 / Pc
    return ac * alpha


def compute_b(Tc, Pc):
    """Pure component co-volume b (vectorized over components)."""
    return 0.07780 * R_GAS * Tc / Pc


def mixture_ab(T, z, Tc, Pc, omega, kij):
    """
    Compute mixture a_mix, b_mix and per-component arrays.

    All operations are vectorized (no Python loops).

    Parameters
    ----------
    T : scalar, temperature in K
    z : array (nc,), mole fractions
    Tc, Pc, omega : arrays (nc,), pure component properties
    kij : array (nc, nc), binary interaction parameters

    Returns
    -------
    a_mix : scalar
    b_mix : scalar
    ai : array (nc,), pure component a*alpha values
    bi : array (nc,), pure component b values
    aij : array (nc, nc), cross a_ij values
    """
    ai = compute_a_alpha(T, Tc, Pc, omega)
    bi = compute_b(Tc, Pc)

    # Cross terms: aij[i,j] = sqrt(ai[i]*ai[j]) * (1 - kij[i,j])
    sqrt_ai = jnp.sqrt(ai)
    aij = jnp.outer(sqrt_ai, sqrt_ai) * (1.0 - kij)

    # a_mix = sum_i sum_j z[i]*z[j]*aij[i,j]
    a_mix = jnp.dot(z, jnp.dot(aij, z))

    # b_mix = sum_i z[i]*bi[i]
    b_mix = jnp.dot(z, bi)

    return a_mix, b_mix, ai, bi, aij


# ============================================================
# Differentiable cubic solver via Implicit Function Theorem
# ============================================================

def _solve_cubic_cardano(A, B, is_vapor):
    """Solve PR cubic in pure JAX using Cardano/trigonometric method.

    Z^3 - (1-B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0

    Compatible with jit, vmap, and grad (used inside custom_vjp forward).
    """
    # Coefficients: Z^3 + c2*Z^2 + c1*Z + c0 = 0
    c2 = -(1.0 - B)
    c1 = A - 3.0 * B**2 - 2.0 * B
    c0 = -(A * B - B**2 - B**3)

    # Depressed cubic: t^3 + p*t + q = 0, where Z = t - c2/3
    p = c1 - c2**2 / 3.0
    q_dep = c0 - c1 * c2 / 3.0 + 2.0 * c2**3 / 27.0

    # Discriminant: D = (q/2)^2 + (p/3)^3
    D = (q_dep / 2.0)**2 + (p / 3.0)**3

    # --- Case 1: D >= 0 (one real root) - Cardano ---
    sqrt_D = jnp.sqrt(jnp.maximum(D, 0.0))
    u = jnp.cbrt(-q_dep / 2.0 + sqrt_D)
    v = jnp.cbrt(-q_dep / 2.0 - sqrt_D)
    t_cardano = u + v
    Z_single = t_cardano - c2 / 3.0

    # --- Case 2: D < 0 (three real roots) - Trigonometric ---
    safe_p = jnp.where(jnp.abs(p) > 1e-30, p, -1e-30)
    m = 2.0 * jnp.sqrt(jnp.maximum(-safe_p / 3.0, 0.0))
    # cos(theta) = 3q / (p * m), clamped to [-1, 1]
    safe_m = jnp.where(jnp.abs(m) > 1e-30, m, 1e-30)
    cos_arg = jnp.clip(3.0 * q_dep / (safe_p * safe_m), -1.0, 1.0)
    theta = jnp.arccos(cos_arg) / 3.0

    t0 = m * jnp.cos(theta)
    t1 = m * jnp.cos(theta - 2.0 * jnp.pi / 3.0)
    t2 = m * jnp.cos(theta - 4.0 * jnp.pi / 3.0)

    Z0 = t0 - c2 / 3.0
    Z1 = t1 - c2 / 3.0
    Z2 = t2 - c2 / 3.0

    # Sort: Z_max >= Z_mid >= Z_min
    Z_max = jnp.maximum(jnp.maximum(Z0, Z1), Z2)
    Z_min = jnp.minimum(jnp.minimum(Z0, Z1), Z2)

    # Select based on phase and discriminant
    # Three roots case (D < 0): vapor=max, liquid=min
    Z_three = jnp.where(is_vapor > 0.5, Z_max, Z_min)

    # Final selection based on discriminant
    Z = jnp.where(D >= 0, Z_single, Z_three)

    # Fallback for non-physical Z
    fallback = jnp.maximum(B + 0.01, 0.1)
    Z = jnp.where(Z > 0.0, Z, fallback)

    return Z


@custom_vjp
def solve_cubic_Z(A, B, is_vapor):
    """
    Solve PR cubic: Z^3 - (1-B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0

    Parameters
    ----------
    A, B : scalar (JAX), dimensionless PR parameters
    is_vapor : scalar, > 0.5 for vapor, <= 0.5 for liquid

    Returns
    -------
    Z : scalar (JAX), compressibility factor
    """
    return _solve_cubic_cardano(A, B, is_vapor)


def _solve_cubic_Z_fwd(A, B, is_vapor):
    Z = solve_cubic_Z(A, B, is_vapor)
    return Z, (Z, A, B)


def _solve_cubic_Z_bwd(res, g):
    """IFT-based backward pass.

    F(Z, A, B) = Z^3 - (1-B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0

    dZ/dA = -(dF/dA) / (dF/dZ)
    dZ/dB = -(dF/dB) / (dF/dZ)
    """
    Z, A, B = res

    # dF/dZ = 3Z^2 - 2(1-B)Z + (A - 3B^2 - 2B)
    dF_dZ = 3.0 * Z**2 - 2.0 * (1.0 - B) * Z + (A - 3.0 * B**2 - 2.0 * B)

    # dF/dA = Z - B
    dF_dA = Z - B

    # dF/dB = Z^2 + (-6B - 2)Z - (A - 2B - 3B^2)
    dF_dB = Z**2 + (-6.0 * B - 2.0) * Z - (A - 2.0 * B - 3.0 * B**2)

    # Avoid division by zero at double roots
    safe_dF_dZ = jnp.where(jnp.abs(dF_dZ) > 1e-30, dF_dZ, 1e-30)

    dZ_dA = -dF_dA / safe_dF_dZ
    dZ_dB = -dF_dB / safe_dF_dZ

    return (g * dZ_dA, g * dZ_dB, jnp.zeros_like(g))


solve_cubic_Z.defvjp(_solve_cubic_Z_fwd, _solve_cubic_Z_bwd)


# ============================================================
# Fugacity coefficients
# ============================================================

def fugacity_coefficients(T, P, z, Tc, Pc, omega, kij, phase='vapor'):
    """
    Compute ln(phi_i) for each component in the mixture.

    Fully vectorized (no Python loops over components).

    Parameters
    ----------
    T : scalar, temperature in K
    P : scalar, pressure in bar
    z : array (nc,), mole fractions
    Tc, Pc, omega : arrays (nc,), pure component properties
    kij : array (nc, nc), binary interaction parameters
    phase : str, 'vapor' or 'liquid'

    Returns
    -------
    ln_phi : array (nc,), ln(fugacity coefficient) for each component
    """
    a_mix, b_mix, ai, bi, aij = mixture_ab(T, z, Tc, Pc, omega, kij)

    A = a_mix * P / (R_GAS * T)**2
    B = b_mix * P / (R_GAS * T)

    is_vapor = jnp.array(1.0 if phase == 'vapor' else 0.0)
    Z = solve_cubic_Z(A, B, is_vapor)

    sqrt2 = jnp.sqrt(2.0)

    # sum_za[i] = sum_j z[j] * aij[i,j]  — vectorized
    sum_za = jnp.dot(aij, z)  # (nc,)

    # term1: (bi/b_mix) * (Z - 1)
    term1 = (bi / b_mix) * (Z - 1.0)

    # term2: -ln(Z - B)
    term2 = -jnp.log(jnp.maximum(Z - B, 1e-30))

    # term3: mixing rule contribution
    arg1 = Z + (1.0 + sqrt2) * B
    arg2 = Z + (1.0 - sqrt2) * B
    log_ratio = jnp.log(jnp.maximum(arg1, 1e-30) / jnp.maximum(arg2, 1e-30))
    term3 = -A / (2.0 * sqrt2 * jnp.maximum(B, 1e-30)) * \
            (2.0 * sum_za / jnp.maximum(a_mix, 1e-30) - bi / jnp.maximum(b_mix, 1e-30)) * log_ratio

    # When B is negligibly small, term3 should be zero
    term3 = jnp.where(B > 1e-15, term3, 0.0)

    ln_phi = term1 + term2 + term3

    return ln_phi


# ============================================================
# Wilson K-values
# ============================================================

def wilson_K(T, P, Tc, Pc, omega):
    """Wilson correlation K-values for initialization (vectorized)."""
    return (Pc / P) * jnp.exp(5.373 * (1.0 + omega) * (1.0 - Tc / T))

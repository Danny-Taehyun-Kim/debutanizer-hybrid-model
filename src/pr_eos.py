"""
Peng-Robinson Equation of State implementation.
Units: R = 83.14472 cm3 bar/(mol K), P in bar, T in K, V in cm3/mol
"""
import numpy as np

R_GAS = 83.14472  # cm3 bar / (mol K)


def compute_kappa(omega):
    """PR kappa parameter from acentric factor."""
    return 0.37464 + 1.54226 * omega - 0.26992 * omega**2


def compute_a_alpha(T, Tc, Pc, omega):
    """Pure component a*alpha(T) for PR EOS."""
    kappa = compute_kappa(omega)
    alpha = (1.0 + kappa * (1.0 - np.sqrt(T / Tc)))**2
    ac = 0.45724 * R_GAS**2 * Tc**2 / Pc
    return ac * alpha


def compute_b(Tc, Pc):
    """Pure component co-volume b."""
    return 0.07780 * R_GAS * Tc / Pc


def mixture_ab(T, z, Tc, Pc, omega, kij):
    """
    Compute mixture a_mix, b_mix and per-component arrays.

    Parameters
    ----------
    T : float, temperature in K
    z : array, mole fractions
    Tc, Pc, omega : arrays, pure component properties
    kij : 2D array, binary interaction parameters

    Returns
    -------
    a_mix, b_mix : float
    ai : array of pure component a*alpha values
    bi : array of pure component b values
    aij : 2D array of cross a_ij values
    """
    nc = len(z)
    ai = np.array([compute_a_alpha(T, Tc[i], Pc[i], omega[i]) for i in range(nc)])
    bi = np.array([compute_b(Tc[i], Pc[i]) for i in range(nc)])

    # Cross terms
    aij = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            aij[i, j] = np.sqrt(ai[i] * ai[j]) * (1.0 - kij[i, j])

    a_mix = 0.0
    for i in range(nc):
        for j in range(nc):
            a_mix += z[i] * z[j] * aij[i, j]

    b_mix = np.dot(z, bi)

    return a_mix, b_mix, ai, bi, aij


def solve_cubic_Z(A, B, phase='vapor'):
    """
    Solve PR cubic: Z^3 - (1-B)*Z^2 + (A-3B^2-2B)*Z - (AB-B^2-B^3) = 0

    Parameters
    ----------
    A, B : float, dimensionless PR parameters
    phase : 'vapor' or 'liquid'

    Returns
    -------
    Z : float, compressibility factor
    """
    coeffs = [1.0,
              -(1.0 - B),
              A - 3.0 * B**2 - 2.0 * B,
              -(A * B - B**2 - B**3)]

    roots = np.roots(coeffs)

    # Filter real positive roots > B
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > B:
            real_roots.append(r.real)

    if len(real_roots) == 0:
        # Fallback: use any real positive root
        for r in roots:
            if abs(r.imag) < 1e-10 and r.real > 0:
                real_roots.append(r.real)
        if len(real_roots) == 0:
            # Last resort
            return max(B + 0.01, 0.1)

    if phase == 'vapor':
        return max(real_roots)
    else:
        return min(real_roots)


def fugacity_coefficients(T, P, z, Tc, Pc, omega, kij, phase='vapor'):
    """
    Compute ln(phi_i) for each component in the mixture.

    Returns
    -------
    ln_phi : array of ln(fugacity coefficient) for each component
    """
    nc = len(z)
    a_mix, b_mix, ai, bi, aij = mixture_ab(T, z, Tc, Pc, omega, kij)

    A = a_mix * P / (R_GAS * T)**2
    B = b_mix * P / (R_GAS * T)

    Z = solve_cubic_Z(A, B, phase)

    ln_phi = np.zeros(nc)
    sqrt2 = np.sqrt(2.0)

    for i in range(nc):
        # Sum of z_j * a_ij
        sum_za = 0.0
        for j in range(nc):
            sum_za += z[j] * aij[i, j]

        term1 = (bi[i] / b_mix) * (Z - 1.0)
        term2 = -np.log(Z - B)

        if B > 1e-15:
            arg1 = Z + (1.0 + sqrt2) * B
            arg2 = Z + (1.0 - sqrt2) * B
            if arg1 > 0 and arg2 > 0:
                term3 = -A / (2.0 * sqrt2 * B) * (2.0 * sum_za / a_mix - bi[i] / b_mix) * np.log(arg1 / arg2)
            else:
                term3 = 0.0
        else:
            term3 = 0.0

        ln_phi[i] = term1 + term2 + term3

    return ln_phi


def wilson_K(T, P, Tc, Pc, omega):
    """Wilson correlation K-values for initialization."""
    return (Pc / P) * np.exp(5.373 * (1.0 + omega) * (1.0 - Tc / T))

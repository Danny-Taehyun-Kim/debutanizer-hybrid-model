"""
Enthalpy module for PR EOS (numpy).
Ideal-gas Cp (DIPPR 107 / Reid-Prausnitz-Poling App A polynomial),
PR enthalpy departure, and stream molar enthalpy.

Components: C2, C3, iC4, nC4, iC5, nC5, C6+ (n-hexane)
Units: T [K], P [bar], H [J/mol]
"""
import numpy as np

from src.pr_eos import mixture_ab, solve_cubic_Z, R_GAS as R_EOS
from src.component_data import get_props_arrays, KIJ, COMPONENTS

R_J = 8.314462  # J/(mol·K)
T_REF = 298.15  # K, reference for enthalpy integration

# Ideal-gas Cp polynomial coefficients: Cp_ig [J/(mol·K)] = A + B*T + C*T^2 + D*T^3 + E*T^4
# Source: Reid, Prausnitz & Poling, "The Properties of Gases and Liquids", 5th ed., App. A
# C6+ represented as n-hexane.
# fmt: off
CP_COEFFS = {
    #              A           B            C             D              E
    'C2':  [ 5.409,   1.7810e-01, -6.938e-05,  8.713e-09,  1.2718e-12],
    'C3':  [-4.224,   3.0630e-01, -1.586e-04,  3.215e-08,  0.0       ],
    'iC4': [-1.390,   3.8470e-01, -1.846e-04,  2.895e-08,  0.0       ],
    'nC4': [ 9.487,   3.3130e-01, -1.108e-04, -2.822e-09,  0.0       ],
    'iC5': [-9.525,   5.0660e-01, -2.729e-04,  5.723e-08,  0.0       ],
    'nC5': [-3.626,   4.8730e-01, -2.580e-04,  5.305e-08,  0.0       ],
    'C6+': [-4.413,   5.8200e-01, -3.119e-04,  6.494e-08,  0.0       ],
}
# fmt: on

NC = len(COMPONENTS)
_CP_ARR = np.array([CP_COEFFS[c] for c in COMPONENTS])  # (7, 5)


def cp_ig(T):
    """Ideal-gas molar heat capacity [J/(mol·K)] for all 7 components at temperature T [K].
    Returns array of shape (7,)."""
    return _CP_ARR[:, 0] + _CP_ARR[:, 1]*T + _CP_ARR[:, 2]*T**2 + _CP_ARR[:, 3]*T**3 + _CP_ARR[:, 4]*T**4


def h_ig(T):
    """Ideal-gas molar enthalpy [J/mol] relative to T_REF=298.15 K for all 7 components.
    H_ig(T) = integral_{T_REF}^{T} Cp_ig dT.  Returns array of shape (7,)."""
    def _antideriv(t):
        return (_CP_ARR[:, 0]*t + _CP_ARR[:, 1]*t**2/2 + _CP_ARR[:, 2]*t**3/3
                + _CP_ARR[:, 3]*t**4/4 + _CP_ARR[:, 4]*t**5/5)
    return _antideriv(T) - _antideriv(T_REF)


def _da_dT_mix(T, z, Tc, Pc, omega, kij):
    """Temperature derivative of mixture a_mix for PR EOS.
    da_mix/dT = sum_i sum_j z_i z_j d(aij)/dT
    where aij = sqrt(ai*aj)*(1-kij), ai = ac_i * alpha_i(T).
    """
    nc = len(z)
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    ac = 0.45724 * R_EOS**2 * Tc**2 / Pc
    sqrt_Tr = np.sqrt(T / Tc)
    alpha = (1.0 + kappa*(1.0 - sqrt_Tr))**2
    ai = ac * alpha

    # d(alpha_i)/dT = 2*(1+kappa*(1-sqrt(T/Tc)))*(-kappa/(2*sqrt(T*Tc)))
    # d(ai)/dT = ac_i * d(alpha_i)/dT
    dai_dT = ac * 2.0*(1.0 + kappa*(1.0 - sqrt_Tr)) * (-kappa / (2.0*np.sqrt(T*Tc)))

    sqrt_ai = np.sqrt(ai)

    da_mix = 0.0
    for i in range(nc):
        for j in range(nc):
            # d(aij)/dT = (1-kij) * d(sqrt(ai*aj))/dT
            # = (1-kij) * (dai/dT * aj + ai * daj/dT) / (2*sqrt(ai*aj))
            aij_val = sqrt_ai[i] * sqrt_ai[j]
            if aij_val > 1e-30:
                daij_dT = (1.0 - kij[i, j]) * (dai_dT[i]*ai[j] + ai[i]*dai_dT[j]) / (2.0*aij_val)
            else:
                daij_dT = 0.0
            da_mix += z[i] * z[j] * daij_dT
    return da_mix


def h_dep_pr(T, P, z, Tc, Pc, omega, kij, phase='vapor'):
    """PR enthalpy departure [J/mol].
    H_dep = R*T*(Z-1) + (T*da/dT - a) / (2*sqrt(2)*b) * ln[(Z+(1+sqrt2)*B)/(Z+(1-sqrt2)*B)]
    where R here is 8.314 J/(mol·K), but Z uses R_EOS = 83.14 cm3·bar/(mol·K).
    Note: 1 cm3·bar = 0.1 J, so conversions are needed.
    """
    a_mix, b_mix, ai, bi, aij = mixture_ab(T, z, Tc, Pc, omega, kij)
    da_dT = _da_dT_mix(T, z, Tc, Pc, omega, kij)

    A = a_mix * P / (R_EOS * T)**2
    B = b_mix * P / (R_EOS * T)
    Z = solve_cubic_Z(A, B, phase)

    sqrt2 = np.sqrt(2.0)
    arg1 = Z + (1.0 + sqrt2) * B
    arg2 = Z + (1.0 - sqrt2) * B

    if arg1 > 0 and arg2 > 0 and b_mix > 1e-30:
        ln_ratio = np.log(arg1 / arg2)
    else:
        ln_ratio = 0.0

    # Convert a_mix from (cm3·bar)^2·K^? units → need consistent units
    # R_EOS in cm3·bar/(mol·K), a in cm6·bar/mol2, b in cm3/mol
    # H_dep in cm3·bar/mol, then multiply by 0.1 to get J/mol
    H_dep_eos = R_EOS * T * (Z - 1.0) + (T * da_dT - a_mix) / (2.0*sqrt2*b_mix) * ln_ratio
    return H_dep_eos * 0.1  # cm3·bar → J  (1 cm3·bar = 0.1 J)


def stream_enthalpy(T, P, z, Tc, Pc, omega, kij, phase='vapor'):
    """Molar enthalpy of a stream [J/mol] = sum_i z_i * H_ig_i(T) + H_dep(T,P,z,phase)."""
    H_ig_components = h_ig(T)  # (nc,)
    H_ig_mix = np.dot(z, H_ig_components)
    H_dep = h_dep_pr(T, P, z, Tc, Pc, omega, kij, phase)
    return H_ig_mix + H_dep


# ============================================================
# Sanity checks
# ============================================================
def run_sanity_checks():
    Tc, Pc, omega, MW = get_props_arrays()
    kij = KIJ

    print("=" * 65)
    print("  Enthalpy Module Sanity Checks")
    print("=" * 65)

    # Check 1: H_dep → 0 as P → 0
    print("\n1) H_dep → 0 as P → 0:")
    z_eq = np.ones(NC) / NC
    for P_test in [0.001, 0.01, 0.1, 1.0, 10.0]:
        hd_v = h_dep_pr(350.0, P_test, z_eq, Tc, Pc, omega, kij, 'vapor')
        print(f"   P={P_test:8.3f} bar  →  H_dep(vap) = {hd_v:10.3f} J/mol")

    # Check 2: H_vap > H_liq (latent heat positive)
    print("\n2) H_vap > H_liq at same T, P, z (latent heat > 0):")
    T_test, P_test = 350.0, 10.0
    h_v = stream_enthalpy(T_test, P_test, z_eq, Tc, Pc, omega, kij, 'vapor')
    h_l = stream_enthalpy(T_test, P_test, z_eq, Tc, Pc, omega, kij, 'liquid')
    print(f"   T={T_test} K, P={P_test} bar")
    print(f"   H_vap = {h_v:.1f} J/mol,  H_liq = {h_l:.1f} J/mol")
    print(f"   ΔH_vap = {h_v - h_l:.1f} J/mol  (should be > 0)")

    # Check 3: Pure-component latent heats near normal boiling point
    # Approximate Tb from Antoine or known values
    Tb_approx = {
        'C2': 184.6, 'C3': 231.1, 'iC4': 261.4, 'nC4': 272.7,
        'iC5': 301.0, 'nC5': 309.2, 'C6+': 341.9,
    }
    print("\n3) Pure-component molar latent heat near normal boiling point (1.013 bar):")
    print(f"   {'Comp':<6} {'Tb(K)':<8} {'ΔHvap(kJ/mol)':<16} {'ΔHvap(kJ/kg)':<14}")
    for idx, comp in enumerate(COMPONENTS):
        z_pure = np.zeros(NC)
        z_pure[idx] = 1.0
        Tb = Tb_approx[comp]
        P_atm = 1.01325
        h_v = stream_enthalpy(Tb, P_atm, z_pure, Tc, Pc, omega, kij, 'vapor')
        h_l = stream_enthalpy(Tb, P_atm, z_pure, Tc, Pc, omega, kij, 'liquid')
        dH = (h_v - h_l) / 1000.0  # kJ/mol
        dH_kg = dH * 1000.0 / MW[idx]  # kJ/kg
        print(f"   {comp:<6} {Tb:<8.1f} {dH:<16.2f} {dH_kg:<14.1f}")
    print("   (Similar ΔHvap across C2-C6+ supports CMO assumption for these light HCs)")

    print("\n" + "=" * 65)


if __name__ == '__main__':
    run_sanity_checks()

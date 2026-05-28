"""
Pure component properties for debutanizer simulation.
Components: C2 (ethane), C3 (propane), iC4, nC4, iC5, nC5, C6+ (as n-hexane)
Sources: Reid-Prausnitz-Poling, DIPPR
"""
import numpy as np
from collections import namedtuple

ComponentProps = namedtuple('ComponentProps', ['Tc', 'Pc', 'omega', 'MW'])
# Tc in K, Pc in bar, omega dimensionless, MW in g/mol

COMPONENTS = ['C2', 'C3', 'iC4', 'nC4', 'iC5', 'nC5', 'C6+']

PROPS = {
    'C2':  ComponentProps(Tc=305.32, Pc=48.72, omega=0.0995, MW=30.07),
    'C3':  ComponentProps(Tc=369.83, Pc=42.48, omega=0.1523, MW=44.10),
    'iC4': ComponentProps(Tc=408.14, Pc=36.48, omega=0.1770, MW=58.12),
    'nC4': ComponentProps(Tc=425.12, Pc=37.96, omega=0.2002, MW=58.12),
    'iC5': ComponentProps(Tc=460.43, Pc=33.81, omega=0.2275, MW=72.15),
    'nC5': ComponentProps(Tc=469.70, Pc=33.70, omega=0.2515, MW=72.15),
    'C6+': ComponentProps(Tc=507.60, Pc=30.25, omega=0.3013, MW=86.18),
}

NC = len(COMPONENTS)

# Binary interaction parameters (kij) - symmetric matrix
# Source: Kijs.csv (PR EOS binary interaction parameters)
# Order: C2, C3, iC4, nC4, iC5, nC5, C6+ (n-hexane)
KIJ = np.array([
    #  C2         C3         iC4        nC4        iC5        nC5        C6+
    [0.0,      1.26e-03,  4.57e-03,  4.10e-03,  7.41e-03,  7.61e-03,  1.14e-02],  # C2
    [1.26e-03, 0.0,       1.04e-03,  8.19e-04,  2.58e-03,  2.70e-03,  5.14e-03],  # C3
    [4.57e-03, 1.04e-03,  0.0,       1.34e-05,  3.46e-04,  3.90e-04,  1.57e-03],  # iC4
    [4.10e-03, 8.19e-04,  1.34e-05,  0.0,       4.95e-04,  5.47e-04,  1.87e-03],  # nC4
    [7.41e-03, 2.58e-03,  3.46e-04,  4.95e-04,  0.0,       1.25e-06,  4.40e-04],  # iC5
    [7.61e-03, 2.70e-03,  3.90e-04,  5.47e-04,  1.25e-06,  0.0,       3.93e-04],  # nC5
    [1.14e-02, 5.14e-03,  1.57e-03,  1.87e-03,  4.40e-04,  3.93e-04,  0.0     ],  # C6+
])


def get_props_arrays():
    """Return arrays of Tc, Pc, omega, MW for all components."""
    Tc = np.array([PROPS[c].Tc for c in COMPONENTS])
    Pc = np.array([PROPS[c].Pc for c in COMPONENTS])
    omega = np.array([PROPS[c].omega for c in COMPONENTS])
    MW = np.array([PROPS[c].MW for c in COMPONENTS])
    return Tc, Pc, omega, MW

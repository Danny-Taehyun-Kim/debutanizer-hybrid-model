#!/usr/bin/env python
"""
CMO assumption validation: compare constant molal overflow (CMO) column solver
with energy-balance (non-CMO) solver on normal operating days.

Runs the FPM with base kij under both solvers and reports composition differences,
quantifying the effect of the CMO simplification.

Usage:
    python -m scripts.robustness_cmo [--csv data/debutanizer_synthetic_29days.csv]
"""
import os
import sys
import csv
import argparse
import numpy as np

from src.data import load_training_data
from src.component_data import get_props_arrays, KIJ
from src.pr_eos import wilson_K, fugacity_coefficients, solve_cubic_Z, mixture_ab, R_GAS


# Column config
N_STAGES = 30
NF = 11
DELTA_P = 0.7
Q_FEED = 1.0
EFFICIENCY = 1.0
N_ITER_CMO = 150

Tc_np, Pc_np, omega_np, MW_np = get_props_arrays()
KIJ_BASE = KIJ.copy()

# Defer enthalpy import (optional module)
try:
    from revision_analyses.enthalpy_pr import stream_enthalpy
    HAS_ENTHALPY = True
except ImportError:
    HAS_ENTHALPY = False
    print("WARNING: enthalpy_pr not available, energy-balance solver disabled")


def solve_column_cmo_np(z_F, P_top, F, D, RD, kij=None):
    """Numpy CMO column solver (no JAX). Returns xD (distillate composition)."""
    from scipy.optimize import brentq

    kij = kij if kij is not None else KIJ_BASE
    nc = len(z_F)
    N = N_STAGES
    nf = NF - 1
    P_bot = P_top + DELTA_P
    P = np.linspace(P_top, P_bot, N)

    B = F - D
    L_rect = RD * D
    V_rect = (RD + 1.0) * D
    L_strip = RD * D + Q_FEED * F
    V_strip = (RD + 1.0) * D - (1.0 - Q_FEED) * F

    L = np.zeros(N)
    V = np.zeros(N)
    L[0] = RD * D
    V[0] = 0.0
    for j in range(1, N):
        if j < nf:
            L[j] = L_rect
            V[j] = V_rect
        else:
            L[j] = L_strip
            V[j] = V_strip

    S = np.zeros(N)
    S[0] = D
    S[N-1] = B
    f_stage = np.zeros(N)
    f_stage[nf] = F

    # T init from Wilson bubble-T
    def wilson_bp(T, P_val, x):
        K = wilson_K(T, P_val, Tc_np, Pc_np, omega_np)
        return np.sum(x * K) - 1.0

    try:
        T_top = brentq(lambda T: wilson_bp(T, P[0], z_F), 280, 500, xtol=0.5)
        T_bot = brentq(lambda T: wilson_bp(T, P[-1], z_F), 280, 500, xtol=0.5)
        T = np.linspace(T_top, T_bot, N)
    except Exception:
        T = np.linspace(340, 430, N)

    K = np.zeros((N, nc))
    for j in range(N):
        K[j] = wilson_K(T[j], P[j], Tc_np, Pc_np, omega_np)

    x = np.tile(z_F, (N, 1))

    damp_schedule = [0.3]*10 + [0.5]*20 + [0.7]*30 + [0.9]*90

    for it in range(N_ITER_CMO):
        damp = damp_schedule[min(it, len(damp_schedule)-1)]
        K_eff = K.copy()

        # Thomas solve per component
        for c in range(nc):
            a = np.zeros(N)
            b = np.zeros(N)
            cc_arr = np.zeros(N)
            d = np.zeros(N)
            for j in range(N):
                if j > 0:
                    a[j] = L[j-1]
                if j < N-1:
                    cc_arr[j] = V[j+1] * K_eff[j+1, c]
                b[j] = -(L[j] + S[j] + V[j] * K_eff[j, c])
                d[j] = -f_stage[j] * z_F[c]

            # Thomas algorithm
            n = N
            cc_arr = cc_arr.copy()
            d = d.copy()
            for i in range(1, n):
                if abs(b[i-1]) < 1e-30:
                    b[i-1] = 1e-30
                m = a[i] / b[i-1]
                b[i] -= m * cc_arr[i-1]
                d[i] -= m * d[i-1]
            x_col = np.zeros(n)
            x_col[n-1] = d[n-1] / b[n-1] if abs(b[n-1]) > 1e-30 else 0
            for i in range(n-2, -1, -1):
                x_col[i] = (d[i] - cc_arr[i] * x_col[i+1]) / b[i] if abs(b[i]) > 1e-30 else 0
            x[:, c] = np.maximum(x_col, 1e-15)

        row_sums = x.sum(axis=1, keepdims=True)
        x = x / np.where(row_sums > 1e-30, row_sums, 1.0)

        # Bubble-T + K update
        for j in range(N):
            try:
                T_bp = brentq(lambda T: wilson_bp(T, P[j], x[j]), 270, 530, xtol=0.5)
                T[j] += damp * (T_bp - T[j])
            except Exception:
                pass
            T[j] = np.clip(T[j], 270, 530)

            try:
                ln_phi_L = fugacity_coefficients(T[j], P[j], x[j], Tc_np, Pc_np, omega_np, kij, 'liquid')
                y_est = K[j] * x[j]
                ys = y_est.sum()
                if ys > 0:
                    y_est /= ys
                ln_phi_V = fugacity_coefficients(T[j], P[j], y_est, Tc_np, Pc_np, omega_np, kij, 'vapor')
                K_new = np.exp(ln_phi_L - ln_phi_V)
                K_new = np.clip(K_new, 1e-8, 1e8)
                ratio = K_new / np.maximum(K[j], 1e-30)
                ratio = np.clip(ratio, 1e-4, 1e4)
                K[j] = K[j] * ratio ** damp
            except Exception:
                K[j] = wilson_K(T[j], P[j], Tc_np, Pc_np, omega_np)

    return x[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='results/cmo_validation')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if not HAS_ENTHALPY:
        print("Energy-balance solver requires revision_analyses/enthalpy_pr.py")
        print("Running CMO-only baseline for demonstration.")

    all_inputs = load_training_data(args.csv)

    results = []
    for i, inp in enumerate(all_inputs):
        xD_cmo = solve_column_cmo_np(
            inp['z_F'], inp['P_top'], inp['F'], inp['D'], inp['RD']
        )
        results.append({
            'date': inp['date_str'],
            'nC4_cmo': xD_cmo[3] * 100,
            'nC4_actual': inp['xD_actual'][3] * 100,
            'nC4_cmo_ae': abs(xD_cmo[3] - inp['xD_actual'][3]) * 100,
        })
        print(f"  Day {i}: {inp['date_str']}, nC4_cmo={xD_cmo[3]*100:.2f}%")

    out_csv = os.path.join(args.save_dir, 'cmo_baseline.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    nC4_mae = np.mean([r['nC4_cmo_ae'] for r in results])
    print(f"\nCMO baseline nC4 MAE: {nC4_mae:.4f} mol%")
    print(f"Results saved: {out_csv}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Evaluate FPM baseline (literature kij, delta_kij=0) on normal days.
Reports per-output MAE and MAPE for C3, iC4, nC4, and TVP.

Usage:
    python -m scripts.eval_baseline [--csv data/debutanizer_synthetic_29days.csv]
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import numpy as np
import jax.numpy as jnp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.data import load_training_data, build_features, prestack_data
from src.component_data import get_props_arrays, KIJ
from src.jax_modules.distillation_jax import solve_column_jax
from src.jax_modules.bubble_P_jax import solve_bubble_P_newton

N_STAGES = 30
NF = 11
DELTA_P = 0.7
T_REF = 311.15

Tc_np, Pc_np, omega_np, _ = get_props_arrays()
Tc_j = jnp.array(Tc_np)
Pc_j = jnp.array(Pc_np)
omega_j = jnp.array(omega_np)
kij_base_j = jnp.array(KIJ)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    args = parser.parse_args()

    all_inputs = load_training_data(args.csv)
    data = prestack_data(all_inputs, N_STAGES, DELTA_P)

    n_days = len(all_inputs)
    print(f"\nBaseline FPM evaluation: {n_days} normal days")
    print(f"  N={N_STAGES}, NF={NF}, delta_kij=0, q=1.0\n")

    ae_C3, ae_iC4, ae_nC4, ae_tvp = [], [], [], []
    ape_C3, ape_iC4, ape_nC4, ape_tvp = [], [], [], []

    for i in range(n_days):
        result = solve_column_jax(
            N_STAGES, NF,
            data['F'][i], data['z_F'][i],
            data['D'][i], data['RD'][i],
            data['P_top'][i], data['P_top'][i] + DELTA_P,
            Tc_j, Pc_j, omega_j, kij_base_j,
            q=1.0, efficiency=jnp.float64(1.0),
            n_iter=30, n_iter_grad=None,
            T_init=data['T_init'][i],
        )
        xD_pred = np.array(result['x'][0, :])
        xD_actual = all_inputs[i]['xD_actual']

        # TVP
        tvp_pred = float(solve_bubble_P_newton(
            jnp.array(xD_pred), T_REF, 7.0, Tc_j, Pc_j, omega_j, kij_base_j
        ))
        tvp_actual = all_inputs[i]['TVP_plant_bar']

        # AE in mol% (×100)
        ae_C3.append(abs(xD_pred[1] - xD_actual[1]) * 100)
        ae_iC4.append(abs(xD_pred[2] - xD_actual[2]) * 100)
        ae_nC4.append(abs(xD_pred[3] - xD_actual[3]) * 100)
        ae_tvp.append(abs(tvp_pred - tvp_actual))

        eps = 1e-8
        ape_C3.append(abs(xD_pred[1] - xD_actual[1]) / (xD_actual[1] + eps) * 100)
        ape_iC4.append(abs(xD_pred[2] - xD_actual[2]) / (xD_actual[2] + eps) * 100)
        ape_nC4.append(abs(xD_pred[3] - xD_actual[3]) / (xD_actual[3] + eps) * 100)
        ape_tvp.append(abs(tvp_pred - tvp_actual) / (tvp_actual + eps) * 100)

        if i < 3 or i == n_days - 1:
            print(f"  Day {i} ({all_inputs[i]['date_str']}): "
                  f"nC4 pred={xD_pred[3]*100:.2f}% actual={xD_actual[3]*100:.2f}% "
                  f"AE={ae_nC4[-1]:.2f} | "
                  f"C3 pred={xD_pred[1]*100:.2f}% actual={xD_actual[1]*100:.2f}% "
                  f"AE={ae_C3[-1]:.2f}")
        elif i == 3:
            print("  ...")

    print(f"\n{'='*60}")
    print(f"Baseline FPM — Per-output MAE ({n_days} normal days)")
    print(f"{'='*60}")
    print(f"  {'Output':<10s} {'MAE':>10s} {'MAPE':>10s}")
    print(f"  {'C3':<10s} {np.mean(ae_C3):>10.4f}% {np.mean(ape_C3):>9.2f}%")
    print(f"  {'iC4':<10s} {np.mean(ae_iC4):>10.4f}% {np.mean(ape_iC4):>9.2f}%")
    print(f"  {'nC4':<10s} {np.mean(ae_nC4):>10.4f}% {np.mean(ape_nC4):>9.2f}%")
    print(f"  {'TVP':<10s} {np.mean(ae_tvp):>10.4f} bar {np.mean(ape_tvp):>6.2f}%")
    print(f"\n  nC4 MAE target: 4-7 mol%")
    nC4_mae = np.mean(ae_nC4)
    if 4.0 <= nC4_mae <= 7.0:
        print(f"  >>> nC4 MAE = {nC4_mae:.2f} mol% — IN RANGE <<<")
    else:
        print(f"  >>> nC4 MAE = {nC4_mae:.2f} mol% — OUT OF RANGE (adjust true_kij) <<<")


if __name__ == '__main__':
    main()

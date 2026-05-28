#!/usr/bin/env python
"""
Robustness analysis: sensitivity to column configuration (N, NF, efficiency).

Retrains the hybrid model under alternative column configurations and reports
MAE for each, demonstrating robustness of the DDM correction mechanism.

Configurations tested:
  1. Baseline: N=30, NF=11, eff=1.0
  2. N=36, NF=13, eff=1.0  (more stages)
  3. N=30, NF=11, eff=0.8  (Murphree efficiency)

Usage:
    python -m scripts.robustness_sensitivity [--epochs 200]
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import csv
import numpy as np
import jax
import jax.numpy as jnp

from src.data import load_training_data, build_features, prestack_data
from src.train import train, DEFAULT_CONFIG
from src.hybrid import (
    ddm_forward, make_kij_matrix,
    IDX_C3, IDX_IC4, IDX_NC4,
    Tc_jax, Pc_jax, omega_jax,
)
from src.jax_modules.distillation_jax import solve_column_jax
from src.jax_modules.bubble_P_jax import solve_bubble_P_newton


CONFIGS = [
    {'label': 'N30_NF11_eff1.0', 'n_stages': 30, 'nf': 11, 'efficiency': 1.0},
    {'label': 'N36_NF13_eff1.0', 'n_stages': 36, 'nf': 13, 'efficiency': 1.0},
    {'label': 'N30_NF11_eff0.8', 'n_stages': 30, 'nf': 11, 'efficiency': 0.8},
]


def evaluate_model(ddm_params, all_inputs, feat_mean, feat_std, cfg):
    """Evaluate trained model on all days, return per-day MAE."""
    features_np = build_features(all_inputs)
    data = prestack_data(all_inputs, cfg['n_stages'], cfg['delta_p'])
    features_normed = (features_np - feat_mean) / feat_std
    data['features_normed'] = jnp.array(features_normed)

    results = []
    for i, inp in enumerate(all_inputs):
        feat_n = jnp.array(features_normed[i])
        delta_kij, q_val, delta_tvp = ddm_forward(ddm_params, feat_n)
        kij = make_kij_matrix(delta_kij)

        result = solve_column_jax(
            cfg['n_stages'], cfg['nf'],
            data['F'][i], data['z_F'][i],
            data['D'][i], data['RD'][i],
            data['P_top'][i], data['P_top'][i] + cfg['delta_p'],
            Tc_jax, Pc_jax, omega_jax, kij,
            q=q_val, efficiency=jnp.float64(cfg['efficiency']),
            n_iter=cfg['n_iter'], n_iter_grad=None,
            T_init=data['T_init'][i],
        )
        xD_pred = np.array(result['x'][0, :])
        xD_actual = inp['xD_actual']

        results.append({
            'date': inp['date_str'],
            'C3_ae': abs(xD_pred[IDX_C3] - xD_actual[IDX_C3]) * 100,
            'iC4_ae': abs(xD_pred[IDX_IC4] - xD_actual[IDX_IC4]) * 100,
            'nC4_ae': abs(xD_pred[IDX_NC4] - xD_actual[IDX_NC4]) * 100,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save-dir', type=str, default='results/sensitivity')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    all_inputs = load_training_data(args.csv)

    summary_rows = []

    for config in CONFIGS:
        label = config['label']
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        cfg = {**DEFAULT_CONFIG, **config, 'max_epochs': args.epochs}

        features_np = build_features(all_inputs)
        feat_mean = features_np.mean(axis=0)
        feat_std = features_np.std(axis=0) + 1e-8
        features_normed = (features_np - feat_mean) / feat_std

        data = prestack_data(all_inputs, cfg['n_stages'], cfg['delta_p'])
        data['features_normed'] = jnp.array(features_normed)
        data['feat_mean'] = feat_mean
        data['feat_std'] = feat_std

        best_params, _ = train(data, cfg=cfg, save_dir=os.path.join(args.save_dir, label))

        jax.clear_caches()

        day_results = evaluate_model(best_params, all_inputs, feat_mean, feat_std, cfg)

        nC4_mae = np.mean([r['nC4_ae'] for r in day_results])
        C3_mae = np.mean([r['C3_ae'] for r in day_results])
        iC4_mae = np.mean([r['iC4_ae'] for r in day_results])

        summary_rows.append({
            'config': label,
            'C3_MAE': f'{C3_mae:.4f}',
            'iC4_MAE': f'{iC4_mae:.4f}',
            'nC4_MAE': f'{nC4_mae:.4f}',
        })
        print(f"\n  {label}: C3={C3_mae:.4f}, iC4={iC4_mae:.4f}, nC4={nC4_mae:.4f} mol%")

        jax.clear_caches()

    out_csv = os.path.join(args.save_dir, 'sensitivity_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved: {out_csv}")


if __name__ == '__main__':
    main()

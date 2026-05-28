#!/usr/bin/env python
"""
Retrain hybrid model on all 29 days (normal + turndown).

Compares with the 24-day (normal-only) model to quantify how much
turndown-day accuracy improves when the model sees those regimes.

Usage:
    python -m scripts.retrain_29day [--epochs 500] [--eval-only]
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import csv
import numpy as np
import jax
import jax.numpy as jnp

from src.data import load_all_data, load_training_data, build_features, prestack_data
from src.train import train, DEFAULT_CONFIG
from src.hybrid import (
    ddm_forward, make_kij_matrix, load_trained_params,
    IDX_C3, IDX_IC4, IDX_NC4,
    Tc_jax, Pc_jax, omega_jax,
)
from src.jax_modules.distillation_jax import solve_column_jax
from src.jax_modules.bubble_P_jax import solve_bubble_P_newton


def evaluate_model(ddm_params, all_inputs, feat_mean, feat_std, cfg):
    """Evaluate model on all days, return per-day results."""
    features_np = build_features(all_inputs)
    features_normed = (features_np - feat_mean) / feat_std
    data = prestack_data(all_inputs, cfg['n_stages'], cfg['delta_p'])
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

        TVP_raw = float(solve_bubble_P_newton(
            jnp.array(xD_pred), cfg['t_ref'], 7.0, Tc_jax, Pc_jax, omega_jax, kij
        ))
        TVP_pred = TVP_raw + float(delta_tvp)

        results.append({
            'date': inp['date_str'],
            'regime': inp.get('regime', 'normal'),
            'C3_ae': abs(xD_pred[IDX_C3] - xD_actual[IDX_C3]) * 100,
            'iC4_ae': abs(xD_pred[IDX_IC4] - xD_actual[IDX_IC4]) * 100,
            'nC4_ae': abs(xD_pred[IDX_NC4] - xD_actual[IDX_NC4]) * 100,
            'nC4_pred': xD_pred[IDX_NC4] * 100,
            'nC4_actual': xD_actual[IDX_NC4] * 100,
            'tvp_pred': TVP_pred,
            'tvp_actual': inp['TVP_plant_bar'],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, load params from save-dir')
    parser.add_argument('--params-24day', type=str, default=None,
                        help='Path to 24-day trained params for comparison')
    parser.add_argument('--save-dir', type=str, default='results/retrain_29day')
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, 'max_epochs': args.epochs}
    os.makedirs(args.save_dir, exist_ok=True)

    # Load all 29 days
    all_inputs = load_all_data(args.csv)

    if args.eval_only:
        npz_path = os.path.join(args.save_dir, 'trained_params.npz')
        print(f"Loading 29-day params from {npz_path}")
        params_29, feat_mean, feat_std, _ = load_trained_params(npz_path)
    else:
        # Train on all 29 days
        features_np = build_features(all_inputs)
        feat_mean = features_np.mean(axis=0)
        feat_std = features_np.std(axis=0) + 1e-8
        features_normed = (features_np - feat_mean) / feat_std

        data = prestack_data(all_inputs, cfg['n_stages'], cfg['delta_p'])
        data['features_normed'] = jnp.array(features_normed)
        data['feat_mean'] = feat_mean
        data['feat_std'] = feat_std

        params_29, _ = train(data, cfg=cfg, save_dir=args.save_dir)

        jax.clear_caches()

    # Evaluate 29-day model
    print("\n\nEvaluating 29-day model on all days...")
    results_29 = evaluate_model(params_29, all_inputs, feat_mean, feat_std, cfg)

    # Report by group
    normal = [r for r in results_29 if r['regime'] == 'normal']
    turndown = [r for r in results_29 if r['regime'] == 'turndown']

    print(f"\n29-day model:")
    print(f"  Normal ({len(normal)} days)  nC4 MAE: "
          f"{np.mean([r['nC4_ae'] for r in normal]):.4f} mol%")
    if turndown:
        print(f"  Turndown ({len(turndown)} days) nC4 MAE: "
              f"{np.mean([r['nC4_ae'] for r in turndown]):.4f} mol%")

    # Compare with 24-day model if available
    if args.params_24day and os.path.exists(args.params_24day):
        print(f"\nComparing with 24-day model: {args.params_24day}")
        params_24, fm24, fs24, _ = load_trained_params(args.params_24day)
        results_24 = evaluate_model(params_24, all_inputs, fm24, fs24, cfg)

        normal_24 = [r for r in results_24 if r['regime'] == 'normal']
        turndown_24 = [r for r in results_24 if r['regime'] == 'turndown']

        print(f"  24-day model:")
        print(f"    Normal ({len(normal_24)} days)  nC4 MAE: "
              f"{np.mean([r['nC4_ae'] for r in normal_24]):.4f} mol%")
        if turndown_24:
            print(f"    Turndown ({len(turndown_24)} days) nC4 MAE: "
                  f"{np.mean([r['nC4_ae'] for r in turndown_24]):.4f} mol%")

    # Save per-day results
    out_csv = os.path.join(args.save_dir, 'retrain_29day_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_29[0].keys())
        writer.writeheader()
        writer.writerows(results_29)
    print(f"\nResults saved: {out_csv}")


if __name__ == '__main__':
    main()

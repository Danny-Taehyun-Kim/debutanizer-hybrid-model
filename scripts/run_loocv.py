#!/usr/bin/env python
"""
Leave-One-Out Cross-Validation (LOOCV) for the hybrid model.

For each of the N normal days, trains on N-1 days and evaluates on the held-out day.
Reports per-day and aggregate MAE/MAPE for C3, iC4, nC4, and TVP.

Usage:
    python -m scripts.run_loocv [--epochs 300] [--csv data/debutanizer_synthetic_29days.csv]
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import csv
import time
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


def evaluate_one_day(ddm_params, feat_normed, data, day_idx, cfg):
    """Run forward pass for a single day, return predictions."""
    delta_kij, q_val, delta_tvp = ddm_forward(ddm_params, feat_normed)
    kij = make_kij_matrix(delta_kij)

    result = solve_column_jax(
        cfg['n_stages'], cfg['nf'],
        data['F'][day_idx], data['z_F'][day_idx],
        data['D'][day_idx], data['RD'][day_idx],
        data['P_top'][day_idx], data['P_top'][day_idx] + cfg['delta_p'],
        Tc_jax, Pc_jax, omega_jax, kij,
        q=q_val, efficiency=jnp.float64(cfg['efficiency']),
        n_iter=cfg['n_iter'], n_iter_grad=None,
        T_init=data['T_init'][day_idx],
    )
    xD_pred = result['x'][0, :]

    TVP_raw = solve_bubble_P_newton(
        xD_pred, cfg['t_ref'], 7.0, Tc_jax, Pc_jax, omega_jax, kij
    )
    TVP_pred = float(TVP_raw) + float(delta_tvp)

    return np.array(xD_pred), TVP_pred, float(q_val), float(delta_tvp)


def main():
    parser = argparse.ArgumentParser(description='LOOCV for hybrid model')
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='results/loocv')
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, 'max_epochs': args.epochs}
    os.makedirs(args.save_dir, exist_ok=True)

    all_inputs = load_training_data(args.csv)
    n_days = len(all_inputs)

    print(f"\nLOOCV: {n_days} folds, {cfg['max_epochs']} epochs each\n")

    results = []
    t0_all = time.time()

    for fold in range(n_days):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_days}: held-out = {all_inputs[fold]['date_str']}")
        print(f"{'='*60}")

        # Split
        train_inputs = [inp for k, inp in enumerate(all_inputs) if k != fold]
        test_input = all_inputs[fold]

        # Build features on train set
        features_np = build_features(train_inputs)
        feat_mean = features_np.mean(axis=0)
        feat_std = features_np.std(axis=0) + 1e-8
        features_normed = (features_np - feat_mean) / feat_std

        data = prestack_data(train_inputs, cfg['n_stages'], cfg['delta_p'])
        data['features_normed'] = jnp.array(features_normed)
        data['feat_mean'] = feat_mean
        data['feat_std'] = feat_std

        # Train on N-1 days
        best_params, _ = train(data, cfg=cfg, seed=args.seed)

        # Clear JIT caches to avoid OOM across folds
        jax.clear_caches()

        # Evaluate on held-out day
        test_feat = np.array(list(test_input['z_F']) +
                             [test_input['P_top'], test_input['F'],
                              test_input['D'], test_input['RD']])
        test_feat_normed = jnp.array((test_feat - feat_mean) / feat_std)

        # Prestack just the test day
        test_data = prestack_data([test_input], cfg['n_stages'], cfg['delta_p'])
        test_data['features_normed'] = test_feat_normed[None, :]

        xD_pred, tvp_pred, q_val, dtvp_val = evaluate_one_day(
            best_params, test_feat_normed, test_data, 0, cfg
        )

        xD_actual = test_input['xD_actual']
        tvp_actual = test_input['TVP_plant_bar']

        results.append({
            'date': test_input['date_str'],
            'nC4_pred': xD_pred[IDX_NC4] * 100,
            'nC4_actual': xD_actual[IDX_NC4] * 100,
            'nC4_ae': abs(xD_pred[IDX_NC4] - xD_actual[IDX_NC4]) * 100,
            'C3_ae': abs(xD_pred[IDX_C3] - xD_actual[IDX_C3]) * 100,
            'iC4_ae': abs(xD_pred[IDX_IC4] - xD_actual[IDX_IC4]) * 100,
            'tvp_pred': tvp_pred,
            'tvp_actual': tvp_actual,
            'tvp_ae': abs(tvp_pred - tvp_actual),
            'q': q_val,
            'delta_tvp': dtvp_val,
        })

        print(f"  nC4: pred={xD_pred[IDX_NC4]*100:.2f}%, "
              f"actual={xD_actual[IDX_NC4]*100:.2f}%, "
              f"AE={abs(xD_pred[IDX_NC4]-xD_actual[IDX_NC4])*100:.2f}%")

        jax.clear_caches()

    # Save results
    out_csv = os.path.join(args.save_dir, 'loocv_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    nC4_mae = np.mean([r['nC4_ae'] for r in results])
    C3_mae = np.mean([r['C3_ae'] for r in results])
    iC4_mae = np.mean([r['iC4_ae'] for r in results])
    tvp_mae = np.mean([r['tvp_ae'] for r in results])

    print(f"\n{'='*60}")
    print(f"LOOCV Summary ({n_days} folds)")
    print(f"  nC4 MAE: {nC4_mae:.4f} mol%")
    print(f"  C3  MAE: {C3_mae:.4f} mol%")
    print(f"  iC4 MAE: {iC4_mae:.4f} mol%")
    print(f"  TVP MAE: {tvp_mae:.4f} bar")
    print(f"  Total time: {time.time()-t0_all:.0f}s")
    print(f"  Results: {out_csv}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Train the hybrid DDM-FPM model on normal-regime days.

Usage:
    python -m scripts.train_hybrid [--epochs 500] [--lr 0.001] [--seed 42]
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import numpy as np
import jax.numpy as jnp

from src.data import load_training_data, build_features, prestack_data
from src.train import train, DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Train hybrid DDM-FPM model')
    parser.add_argument('--csv', type=str, default=None, help='Path to data CSV')
    parser.add_argument('--epochs', type=int, default=500, help='Max training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='results/train',
                        help='Output directory')
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, 'max_epochs': args.epochs, 'lr': args.lr}

    print("=" * 60)
    print("Hybrid DDM-FPM Training")
    print(f"  N={cfg['n_stages']}, NF={cfg['nf']}, lr={cfg['lr']}, "
          f"max_epochs={cfg['max_epochs']}")
    print("=" * 60)

    # Load data
    normal_inputs = load_training_data(args.csv)
    features_np = build_features(normal_inputs)
    feat_mean = features_np.mean(axis=0)
    feat_std = features_np.std(axis=0) + 1e-8
    features_normed = (features_np - feat_mean) / feat_std

    data = prestack_data(normal_inputs, cfg['n_stages'], cfg['delta_p'])
    data['features_normed'] = jnp.array(features_normed)
    data['feat_mean'] = feat_mean
    data['feat_std'] = feat_std

    # Train
    best_params, loss_history = train(
        data, cfg=cfg, seed=args.seed, save_dir=args.save_dir
    )

    print(f"\nDone. Results in {args.save_dir}/")


if __name__ == '__main__':
    main()

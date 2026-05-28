"""
Data loading utilities for the debutanizer hybrid model.

Reads the plant CSV (or synthetic CSV), extracts simulation inputs,
and pre-stacks into JAX arrays for JIT-friendly training.

CSV expected columns:
  Date, Balance_Status, Regime,
  Feed_C2_mol%, Feed_C3_mol%, Feed_iC4_mol%, Feed_nC4_mol%,
  Feed_iC5_mol%, Feed_nC5_mol%, Feed_C6_mol%,
  LPG_C2_mol%, LPG_C3_mol%, LPG_iC4_mol%, LPG_nC4_mol%,
  LPG_iC5_mol%, LPG_nC5_mol%, LPG_C6+_mol%,
  PIC3000_barg, F_kmolh, D_kmolh, RD_ratio, LPG_TVP_kgcm2
"""
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp

from src.component_data import get_props_arrays


# Default data path (can be overridden)
_DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'debutanizer_synthetic_29days.csv'
)


def load_plant_data(csv_path=None):
    """Load plant CSV, filter to complete days."""
    csv_path = csv_path or _DEFAULT_CSV
    df = pd.read_csv(csv_path)
    df = df[df['Balance_Status'] == 'Complete'].reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def extract_inputs(row):
    """Extract simulation inputs from one DataFrame row."""
    z_F = np.array([
        row['Feed_C2_mol%'], row['Feed_C3_mol%'], row['Feed_iC4_mol%'],
        row['Feed_nC4_mol%'], row['Feed_iC5_mol%'], row['Feed_nC5_mol%'],
        row['Feed_C6_mol%'],
    ]) / 100.0

    xD_actual = np.array([
        row['LPG_C2_mol%'], row['LPG_C3_mol%'], row['LPG_iC4_mol%'],
        row['LPG_nC4_mol%'], row['LPG_iC5_mol%'], row['LPG_nC5_mol%'],
        row['LPG_C6+_mol%'],
    ]) / 100.0

    P_top = row['PIC3000_barg'] + 1.01325  # barg -> bar abs
    F = row['F_kmolh']
    D = row['D_kmolh']
    RD = row['RD_ratio']

    z_F = np.maximum(z_F, 1e-15)
    z_F /= z_F.sum()
    xD_actual = np.maximum(xD_actual, 1e-15)
    xD_actual /= xD_actual.sum()

    TVP_plant_bar = row['LPG_TVP_kgcm2'] * 0.980665  # kgf/cm2 -> bar

    return {
        'z_F': z_F, 'xD_actual': xD_actual,
        'P_top': P_top, 'F': F, 'D': D, 'RD': RD,
        'TVP_plant_bar': TVP_plant_bar,
        'Date': row['Date'],
    }


def load_all_data(csv_path=None):
    """Load all complete days with regime label.

    Returns list of dicts, each with 'is_turndown' bool based on Regime column.
    """
    df = load_plant_data(csv_path)
    all_inputs = []
    for _, row in df.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        inp = extract_inputs(row)
        inp['date_str'] = date_str
        inp['regime'] = row.get('Regime', 'normal')
        inp['is_turndown'] = (inp['regime'] == 'turndown')
        all_inputs.append(inp)
    n_td = sum(1 for inp in all_inputs if inp['is_turndown'])
    print(f"Loaded {len(all_inputs)} complete days "
          f"({len(all_inputs) - n_td} normal, {n_td} turndown)")
    return all_inputs


def load_training_data(csv_path=None):
    """Load only normal-regime days for training."""
    all_inputs = load_all_data(csv_path)
    normal = [inp for inp in all_inputs if not inp['is_turndown']]
    print(f"Training set: {len(normal)} normal days "
          f"(excluded {len(all_inputs) - len(normal)} turndown)")
    return normal


def build_features(all_inputs):
    """Build feature matrix from input dicts. Returns (n, 11) numpy array.

    Features: z_F(7) + P_top(1) + F(1) + D(1) + RD(1)
    """
    feats = []
    for inp in all_inputs:
        row = list(inp['z_F']) + [inp['P_top'], inp['F'], inp['D'], inp['RD']]
        feats.append(row)
    return np.array(feats)


def prestack_data(all_inputs, n_stages, delta_p):
    """Convert list of input dicts to stacked JAX arrays for JIT-friendly access.

    Also pre-computes T_init (Wilson bubble-T via scipy, not JIT-compatible).
    """
    from src.jax_modules.distillation_jax import compute_T_init

    Tc_np, Pc_np, omega_np, _ = get_props_arrays()

    n_days = len(all_inputs)
    z_F_all = jnp.stack([jnp.array(inp['z_F']) for inp in all_inputs])
    xD_actual_all = jnp.stack([jnp.array(inp['xD_actual']) for inp in all_inputs])
    F_all = jnp.array([inp['F'] for inp in all_inputs])
    D_all = jnp.array([inp['D'] for inp in all_inputs])
    RD_all = jnp.array([inp['RD'] for inp in all_inputs])
    P_top_all = jnp.array([inp['P_top'] for inp in all_inputs])

    print(f"  Pre-computing T_init for N={n_stages} stages...")
    T_init_list = []
    for inp in all_inputs:
        T_init = compute_T_init(
            n_stages, inp['z_F'], inp['P_top'], inp['P_top'] + delta_p,
            Tc_np, Pc_np, omega_np
        )
        T_init_list.append(T_init)
    T_init_all = jnp.stack([jnp.array(t) for t in T_init_list])

    TVP_plant_all = jnp.array([inp['TVP_plant_bar'] for inp in all_inputs])

    print(f"  Pre-stacked: {n_days} days, T_init shape={T_init_all.shape}")

    return {
        'z_F': z_F_all, 'xD_actual': xD_actual_all,
        'F': F_all, 'D': D_all, 'RD': RD_all,
        'P_top': P_top_all, 'T_init': T_init_all,
        'TVP_plant': TVP_plant_all,
    }

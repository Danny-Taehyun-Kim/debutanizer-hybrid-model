#!/usr/bin/env python
"""
Generate a synthetic 29-day debutanizer dataset for demonstration.

Produces data/debutanizer_synthetic_29days.csv with realistic operating ranges
(matching paper's Ghana Gas debutanizer) but NO real plant data.

Strategy: generate feed + operating conditions from specified distributions,
then run FPM with a "true_kij" (perturbed from literature) to produce
physically-consistent "plant" distillate compositions. The baseline FPM
(literature kij) will then naturally show ~4-7 mol% nC4 MAE.

24 normal days + 5 turndown days.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add repo root to path for src imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

np.random.seed(2025)

OUT_DIR = os.path.join(_REPO_ROOT, 'data')
os.makedirs(OUT_DIR, exist_ok=True)

N_NORMAL = 24
N_TURNDOWN = 5
N_TOTAL = N_NORMAL + N_TURNDOWN


def trunc_normal(mean, std, n, low=0.01):
    """Truncated normal: clamp below at `low`."""
    return np.maximum(np.random.normal(mean, std, n), low)


def normalize_composition(comp_dict, n):
    """Normalize component arrays so each sample sums to 100%."""
    keys = list(comp_dict.keys())
    total = sum(comp_dict[k] for k in keys)
    for k in keys:
        comp_dict[k] = comp_dict[k] / total * 100.0
    return comp_dict


def generate_plant_distillate(feed_rows, true_delta_kij):
    """Run FPM with true_kij to produce plant-like distillate + TVP.

    This makes the synthetic "plant measurements" physically consistent
    with the column model, and guarantees a known baseline MAE.
    """
    os.environ['JAX_ENABLE_X64'] = '1'
    import jax.numpy as jnp
    from src.jax_modules.distillation_jax import solve_column_jax, compute_T_init
    from src.jax_modules.bubble_P_jax import solve_bubble_P_newton
    from src.component_data import get_props_arrays, KIJ

    Tc_np, Pc_np, omega_np, _ = get_props_arrays()
    Tc_j = jnp.array(Tc_np)
    Pc_j = jnp.array(Pc_np)
    omega_j = jnp.array(omega_np)

    N_STAGES = 30
    NF = 11
    DELTA_P = 0.7
    T_REF = 311.15

    # Build true kij matrix
    PARAM_PAIRS = [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
    kij_true = KIJ.copy()
    for k, (i, j) in enumerate(PARAM_PAIRS):
        kij_true[i, j] += true_delta_kij[k]
        kij_true[j, i] = kij_true[i, j]
    kij_true_j = jnp.array(kij_true)

    xD_list = []
    tvp_list = []
    for idx, row in enumerate(feed_rows):
        z_F = row['z_F']
        P_top = row['P_top']
        F = row['F']
        D = row['D']
        RD = row['RD']

        T_init = compute_T_init(
            N_STAGES, z_F, P_top, P_top + DELTA_P, Tc_np, Pc_np, omega_np
        )

        result = solve_column_jax(
            N_STAGES, NF, F, jnp.array(z_F), D, RD,
            P_top, P_top + DELTA_P,
            Tc_j, Pc_j, omega_j, kij_true_j,
            q=1.0, efficiency=jnp.float64(1.0),
            n_iter=30, n_iter_grad=None,
            T_init=jnp.array(T_init),
        )
        xD = np.array(result['x'][0, :])

        # Add small measurement noise (±0.2 mol%, then renormalize)
        noise = np.random.normal(0, 0.002, 7)
        xD_noisy = np.maximum(xD + noise, 1e-6)
        xD_noisy /= xD_noisy.sum()

        # TVP = bubble-P of distillate at T_REF
        tvp = float(solve_bubble_P_newton(
            jnp.array(xD_noisy), T_REF, 7.0, Tc_j, Pc_j, omega_j, kij_true_j
        ))
        # Convert bar -> kgf/cm2
        tvp_kgcm2 = tvp / 0.980665

        xD_list.append(xD_noisy * 100.0)  # mol%
        tvp_list.append(tvp_kgcm2)

        if idx % 5 == 0:
            print(f"    Day {idx}: nC4={xD_noisy[3]*100:.1f}%, C3={xD_noisy[1]*100:.1f}%, "
                  f"TVP={tvp_kgcm2:.2f} kgf/cm2")

    return xD_list, tvp_list


def make_dataset(true_delta_kij=None):
    """Generate synthetic dataset.

    Parameters
    ----------
    true_delta_kij : array (6,) or None
        Perturbation to literature kij for generating "plant" distillate.
        If None, uses a default perturbation calibrated for ~5 mol% nC4 MAE.
    """
    if true_delta_kij is None:
        # Default: moderate perturbation to C6+ pairs
        # Calibrated so baseline FPM (lit kij) gives ~5 mol% nC4 MAE
        true_delta_kij = np.array([
            -0.02,   # C2-C6+
            +0.08,   # C3-C6+
            -0.02,   # iC4-C6+
            +0.12,   # nC4-C6+ (main driver of nC4 error)
            -0.10,   # iC5-C6+
            -0.09,   # nC5-C6+
        ])

    # ==================================================================
    # Normal days (24): Gaussian distributions
    # ==================================================================
    feed_n = normalize_composition({
        'C2':  trunc_normal(0.5, 0.2, N_NORMAL),
        'C3':  trunc_normal(8.8, 1.2, N_NORMAL),
        'iC4': trunc_normal(12.0, 1.5, N_NORMAL),
        'nC4': trunc_normal(28.0, 2.0, N_NORMAL),
        'iC5': trunc_normal(18.0, 2.0, N_NORMAL),
        'nC5': trunc_normal(13.0, 2.0, N_NORMAL),
        'C6':  trunc_normal(19.0, 3.0, N_NORMAL),
    }, N_NORMAL)

    P_top_n = trunc_normal(17.7, 0.2, N_NORMAL, low=16.0)    # barg
    F_n     = trunc_normal(305.0, 25.0, N_NORMAL, low=200.0)  # kmol/h
    RD_n    = trunc_normal(1.27, 0.10, N_NORMAL, low=0.5)

    # D/F ~ 0.16: at these conditions, D≈50 kmol/h gives LPG C3≈48%, nC4≈15%
    # (D/F=0.50 is incompatible with C3-rich LPG; mass balance forces nC4-rich distillate)
    D_n     = trunc_normal(50.0, 5.0, N_NORMAL, low=30.0)

    # Convert to absolute pressure for FPM
    P_top_abs_n = P_top_n + 1.01325

    # Build input dicts for FPM
    normal_rows = []
    for i in range(N_NORMAL):
        z_F = np.array([feed_n['C2'][i], feed_n['C3'][i], feed_n['iC4'][i],
                         feed_n['nC4'][i], feed_n['iC5'][i], feed_n['nC5'][i],
                         feed_n['C6'][i]]) / 100.0
        z_F = np.maximum(z_F, 1e-15)
        z_F /= z_F.sum()
        normal_rows.append({
            'z_F': z_F, 'P_top': P_top_abs_n[i],
            'F': F_n[i], 'D': D_n[i], 'RD': RD_n[i],
        })

    # ==================================================================
    # Turndown days (5): lower feed, higher reflux
    # ==================================================================
    feed_t = normalize_composition({
        'C2':  trunc_normal(0.3, 0.15, N_TURNDOWN),
        'C3':  trunc_normal(6.0, 1.5, N_TURNDOWN),
        'iC4': trunc_normal(9.0, 2.0, N_TURNDOWN),
        'nC4': trunc_normal(22.0, 3.0, N_TURNDOWN),
        'iC5': trunc_normal(22.0, 3.0, N_TURNDOWN),
        'nC5': trunc_normal(17.0, 3.0, N_TURNDOWN),
        'C6':  trunc_normal(24.0, 4.0, N_TURNDOWN),
    }, N_TURNDOWN)

    F_t  = np.random.uniform(60, 140, N_TURNDOWN)
    RD_t = np.random.uniform(2.0, 6.3, N_TURNDOWN)
    P_top_t = trunc_normal(17.7, 0.3, N_TURNDOWN, low=16.0)
    # Turndown D scales with F (same D/F ratio)
    D_t = trunc_normal(0.16, 0.02, N_TURNDOWN, low=0.08) * F_t

    P_top_abs_t = P_top_t + 1.01325

    turndown_rows = []
    for i in range(N_TURNDOWN):
        z_F = np.array([feed_t['C2'][i], feed_t['C3'][i], feed_t['iC4'][i],
                         feed_t['nC4'][i], feed_t['iC5'][i], feed_t['nC5'][i],
                         feed_t['C6'][i]]) / 100.0
        z_F = np.maximum(z_F, 1e-15)
        z_F /= z_F.sum()
        turndown_rows.append({
            'z_F': z_F, 'P_top': P_top_abs_t[i],
            'F': F_t[i], 'D': D_t[i], 'RD': RD_t[i],
        })

    # ==================================================================
    # Run FPM with true_kij to generate plant distillate
    # ==================================================================
    all_rows = normal_rows + turndown_rows
    print(f"  Running FPM with true_kij to generate plant distillate ({N_TOTAL} days)...")
    xD_all, tvp_all = generate_plant_distillate(all_rows, true_delta_kij)

    # ==================================================================
    # Assemble DataFrame
    # ==================================================================
    base_date = pd.Timestamp('2025-03-01')
    rows = []

    for i in range(N_NORMAL):
        rows.append({
            'Date': (base_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
            'Balance_Status': 'Complete',
            'Regime': 'normal',
            'Feed_C2_mol%':  feed_n['C2'][i],
            'Feed_C3_mol%':  feed_n['C3'][i],
            'Feed_iC4_mol%': feed_n['iC4'][i],
            'Feed_nC4_mol%': feed_n['nC4'][i],
            'Feed_iC5_mol%': feed_n['iC5'][i],
            'Feed_nC5_mol%': feed_n['nC5'][i],
            'Feed_C6_mol%':  feed_n['C6'][i],
            'LPG_C2_mol%':   xD_all[i][0],
            'LPG_C3_mol%':   xD_all[i][1],
            'LPG_iC4_mol%':  xD_all[i][2],
            'LPG_nC4_mol%':  xD_all[i][3],
            'LPG_iC5_mol%':  xD_all[i][4],
            'LPG_nC5_mol%':  xD_all[i][5],
            'LPG_C6+_mol%':  xD_all[i][6],
            'PIC3000_barg': P_top_n[i],
            'F_kmolh': F_n[i],
            'D_kmolh': D_n[i],
            'RD_ratio': RD_n[i],
            'LPG_TVP_kgcm2': tvp_all[i],
        })

    for i in range(N_TURNDOWN):
        day_offset = N_NORMAL + i
        rows.append({
            'Date': (base_date + pd.Timedelta(days=day_offset)).strftime('%Y-%m-%d'),
            'Balance_Status': 'Complete',
            'Regime': 'turndown',
            'Feed_C2_mol%':  feed_t['C2'][i],
            'Feed_C3_mol%':  feed_t['C3'][i],
            'Feed_iC4_mol%': feed_t['iC4'][i],
            'Feed_nC4_mol%': feed_t['nC4'][i],
            'Feed_iC5_mol%': feed_t['iC5'][i],
            'Feed_nC5_mol%': feed_t['nC5'][i],
            'Feed_C6_mol%':  feed_t['C6'][i],
            'LPG_C2_mol%':   xD_all[N_NORMAL + i][0],
            'LPG_C3_mol%':   xD_all[N_NORMAL + i][1],
            'LPG_iC4_mol%':  xD_all[N_NORMAL + i][2],
            'LPG_nC4_mol%':  xD_all[N_NORMAL + i][3],
            'LPG_iC5_mol%':  xD_all[N_NORMAL + i][4],
            'LPG_nC5_mol%':  xD_all[N_NORMAL + i][5],
            'LPG_C6+_mol%':  xD_all[N_NORMAL + i][6],
            'PIC3000_barg': P_top_t[i],
            'F_kmolh': F_t[i],
            'D_kmolh': D_t[i],
            'RD_ratio': RD_t[i],
            'LPG_TVP_kgcm2': tvp_all[N_NORMAL + i],
        })

    df = pd.DataFrame(rows)

    # Round for readability
    float_cols = [c for c in df.columns if c not in ('Date', 'Balance_Status', 'Regime')]
    for c in float_cols:
        df[c] = df[c].round(4)

    out_path = os.path.join(OUT_DIR, 'debutanizer_synthetic_29days.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSynthetic dataset written: {out_path}")
    print(f"  {N_NORMAL} normal + {N_TURNDOWN} turndown = {N_TOTAL} days")

    # Sanity: composition sums
    feed_cols = [c for c in df.columns if c.startswith('Feed_')]
    lpg_cols = [c for c in df.columns if c.startswith('LPG_') and c.endswith('mol%')]
    print(f"  Feed sum range: [{df[feed_cols].sum(axis=1).min():.1f}, "
          f"{df[feed_cols].sum(axis=1).max():.1f}]%")
    print(f"  LPG sum range:  [{df[lpg_cols].sum(axis=1).min():.1f}, "
          f"{df[lpg_cols].sum(axis=1).max():.1f}]%")

    # Normal-day statistics
    ndf = df[df['Regime'] == 'normal']
    print(f"\n  Normal-day means:")
    print(f"    Feed C3  = {ndf['Feed_C3_mol%'].mean():.1f} mol%")
    print(f"    Feed nC4 = {ndf['Feed_nC4_mol%'].mean():.1f} mol%")
    print(f"    F        = {ndf['F_kmolh'].mean():.0f} kmol/h")
    print(f"    D        = {ndf['D_kmolh'].mean():.0f} kmol/h (D/F={ndf['D_kmolh'].mean()/ndf['F_kmolh'].mean():.2f})")
    print(f"    P_top    = {ndf['PIC3000_barg'].mean():.1f} barg")
    print(f"    RD       = {ndf['RD_ratio'].mean():.2f}")
    print(f"    LPG C3   = {ndf['LPG_C3_mol%'].mean():.1f} mol%")
    print(f"    LPG iC4  = {ndf['LPG_iC4_mol%'].mean():.1f} mol%")
    print(f"    LPG nC4  = {ndf['LPG_nC4_mol%'].mean():.1f} mol%")
    print(f"    LPG iC5  = {ndf['LPG_iC5_mol%'].mean():.1f} mol%")
    print(f"    TVP      = {ndf['LPG_TVP_kgcm2'].mean():.2f} kgf/cm2")

    return df


if __name__ == '__main__':
    make_dataset()

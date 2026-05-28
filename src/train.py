"""
End-to-end training loop for the hybrid DDM-FPM model.

Split-JIT strategy: composition loss and TVP loss are compiled as separate
JIT functions to avoid LLVM OOM on the combined graph. Gradients are combined
via VJP chain rule through the DDM MLP.

Loss = MAPE(C3) + MAPE(iC4) + MAPE(nC4) + lambda_tvp * MAPE(TVP + delta_tvp)
       + reg_q * (q - 1)^2
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.hybrid import (
    make_kij_matrix, ddm_forward,
    IDX_C3, IDX_IC4, IDX_NC4,
    Tc_jax, Pc_jax, omega_jax,
)
from src.jax_modules.distillation_jax import solve_column_jax
from src.jax_modules.bubble_P_jax import solve_bubble_P_newton


# ============================================================
# Default hyperparameters
# ============================================================
DEFAULT_CONFIG = {
    'n_stages': 30,
    'nf': 11,
    'delta_p': 0.7,
    'efficiency': 1.0,
    'n_iter': 30,
    't_ref': 311.15,       # K (37.8 C, ASTM D1267)
    'lr': 0.001,
    'max_epochs': 500,
    'patience': 80,
    'reg_q': 0.001,
    'lambda_tvp': 0.5,
}


# ============================================================
# Loss functions (module-level for single JIT compilation)
# ============================================================

def _make_comp_loss_fn(cfg):
    """Create composition MAPE loss function closed over column config."""
    N = cfg['n_stages']
    NF = cfg['nf']
    DP = cfg['delta_p']
    EFF = jnp.float64(cfg['efficiency'])
    NI = cfg['n_iter']
    RQ = cfg['reg_q']

    def comp_loss_fn(delta_kij, q_val, day_idx, data):
        kij = make_kij_matrix(delta_kij)
        result = solve_column_jax(
            N, NF,
            data['F'][day_idx], data['z_F'][day_idx],
            data['D'][day_idx], data['RD'][day_idx],
            data['P_top'][day_idx], data['P_top'][day_idx] + DP,
            Tc_jax, Pc_jax, omega_jax, kij,
            q=q_val, efficiency=EFF,
            n_iter=NI, n_iter_grad=None,
            T_init=data['T_init'][day_idx],
        )
        xD_pred = result['x'][0, :]
        xD_actual = data['xD_actual'][day_idx]

        eps = 1e-8
        ape_C3 = jnp.abs(xD_pred[IDX_C3] - xD_actual[IDX_C3]) / (xD_actual[IDX_C3] + eps)
        ape_iC4 = jnp.abs(xD_pred[IDX_IC4] - xD_actual[IDX_IC4]) / (xD_actual[IDX_IC4] + eps)
        ape_nC4 = jnp.abs(xD_pred[IDX_NC4] - xD_actual[IDX_NC4]) / (xD_actual[IDX_NC4] + eps)

        return ape_C3 + ape_iC4 + ape_nC4 + RQ * (q_val - 1.0) ** 2

    return comp_loss_fn


def _make_tvp_loss_fn(cfg):
    """Create TVP MAPE loss function closed over column config."""
    N = cfg['n_stages']
    NF = cfg['nf']
    DP = cfg['delta_p']
    EFF = jnp.float64(cfg['efficiency'])
    NI = cfg['n_iter']
    TREF = cfg['t_ref']
    LTVP = cfg['lambda_tvp']

    def tvp_loss_fn(delta_kij, q_val, delta_tvp, day_idx, data):
        kij = make_kij_matrix(delta_kij)
        result = solve_column_jax(
            N, NF,
            data['F'][day_idx], data['z_F'][day_idx],
            data['D'][day_idx], data['RD'][day_idx],
            data['P_top'][day_idx], data['P_top'][day_idx] + DP,
            Tc_jax, Pc_jax, omega_jax, kij,
            q=q_val, efficiency=EFF,
            n_iter=NI, n_iter_grad=None,
            T_init=data['T_init'][day_idx],
        )
        xD_pred = result['x'][0, :]
        eps = 1e-8
        TVP_raw = solve_bubble_P_newton(xD_pred, TREF, 7.0, Tc_jax, Pc_jax, omega_jax, kij)
        TVP_pred = TVP_raw + delta_tvp
        TVP_plant = data['TVP_plant'][day_idx]
        return LTVP * jnp.abs(TVP_pred - TVP_plant) / (TVP_plant + eps)

    return tvp_loss_fn


def train(data, cfg=None, seed=42, save_dir=None):
    """Run end-to-end training.

    Parameters
    ----------
    data : dict from prestack_data, must also include 'features_normed'
    cfg : dict, hyperparameters (merged with DEFAULT_CONFIG)
    seed : int, random seed
    save_dir : str or None, directory to save params/loss history

    Returns
    -------
    best_params : dict, best DDM parameters (by total loss)
    feat_mean, feat_std : arrays used for feature normalization
    loss_history : list of (total, comp, tvp) tuples
    """
    from src.hybrid import init_ddm_params

    c = {**DEFAULT_CONFIG, **(cfg or {})}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    n_days = int(data['F'].shape[0])

    # Build JIT-compiled loss+grad functions
    comp_loss_fn = _make_comp_loss_fn(c)
    tvp_loss_fn = _make_tvp_loss_fn(c)
    comp_loss_and_grad = jax.jit(jax.value_and_grad(comp_loss_fn, argnums=(0, 1)))
    tvp_loss_and_grad = jax.jit(jax.value_and_grad(tvp_loss_fn, argnums=(0, 1, 2)))

    def e2e_loss_and_grad(ddm_params, day_idx):
        features = data['features_normed'][day_idx]
        (delta_kij, q_val, delta_tvp), vjp_fn = jax.vjp(
            lambda p: ddm_forward(p, features), ddm_params
        )
        cl, (gd_c, gq_c) = comp_loss_and_grad(delta_kij, q_val, day_idx, data)
        tl, (gd_t, gq_t, gdtvp_t) = tvp_loss_and_grad(
            delta_kij, q_val, delta_tvp, day_idx, data
        )
        (grad_ddm,) = vjp_fn((gd_c + gd_t, gq_c + gq_t, gdtvp_t))
        return float(cl) + float(tl), grad_ddm, float(cl), float(tl)

    # Initialize
    key = jax.random.PRNGKey(seed)
    ddm_params = init_ddm_params(key)

    optimizer = optax.adam(learning_rate=c['lr'])
    opt_state = optimizer.init(ddm_params)

    # JIT warm-up
    print("JIT warm-up (comp loss)...")
    t0 = time.time()
    _ = comp_loss_and_grad(jnp.zeros(6), jnp.array(1.0), 0, data)
    print(f"  Done ({time.time() - t0:.1f}s)")

    print("JIT warm-up (TVP loss)...")
    t0 = time.time()
    _ = tvp_loss_and_grad(jnp.zeros(6), jnp.array(1.0), jnp.array(0.0), 0, data)
    print(f"  Done ({time.time() - t0:.1f}s)\n")

    # Training loop
    loss_history = []
    best_total_loss = float('inf')
    best_params = ddm_params
    best_epoch = 0
    patience_counter = 0

    t0_train = time.time()
    for epoch in range(c['max_epochs']):
        t0_ep = time.time()
        total_loss = 0.0
        total_comp = 0.0
        total_tvp = 0.0
        total_grads = jax.tree.map(jnp.zeros_like, ddm_params)

        for i in range(n_days):
            loss_i, grad_i, comp_i, tvp_i = e2e_loss_and_grad(ddm_params, i)
            total_loss += loss_i
            total_comp += comp_i
            total_tvp += tvp_i
            total_grads = jax.tree.map(lambda a, b: a + b, total_grads, grad_i)

        avg_loss = total_loss / n_days
        avg_comp = total_comp / n_days
        avg_tvp = total_tvp / n_days
        avg_grads = jax.tree.map(lambda g: g / n_days, total_grads)

        updates, opt_state = optimizer.update(avg_grads, opt_state)
        ddm_params = optax.apply_updates(ddm_params, updates)

        loss_history.append((avg_loss, avg_comp, avg_tvp))
        dt = time.time() - t0_ep

        if avg_loss < best_total_loss:
            best_total_loss = avg_loss
            best_params = jax.tree.map(lambda x: x.copy(), ddm_params)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == c['max_epochs'] - 1:
            print(f"  Epoch {epoch:3d}: total={avg_loss:.6f}, comp={avg_comp:.6f}, "
                  f"tvp={avg_tvp:.6f}, time={dt:.1f}s")

        if patience_counter >= c['patience']:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {c['patience']} epochs)")
            break

    dt_train = time.time() - t0_train
    print(f"\nTraining done in {dt_train:.0f}s")
    print(f"  Best loss: {best_total_loss:.8f} at epoch {best_epoch}")

    # Save
    if save_dir:
        feat_mean = np.array(data.get('feat_mean', np.zeros(11)))
        feat_std = np.array(data.get('feat_std', np.ones(11)))
        params_np = jax.tree.map(np.array, best_params)
        npz_path = os.path.join(save_dir, 'trained_params.npz')
        np.savez(npz_path,
                 W1=params_np['W1'], b1=params_np['b1'],
                 W2=params_np['W2'], b2=params_np['b2'],
                 feat_mean=feat_mean, feat_std=feat_std,
                 best_epoch=best_epoch)
        print(f"  Params saved: {npz_path}")

        import csv
        loss_csv = os.path.join(save_dir, 'loss_history.csv')
        with open(loss_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_loss', 'comp_loss', 'tvp_loss'])
            for ep, (tl, cl, tv) in enumerate(loss_history):
                writer.writerow([ep, f'{tl:.10f}', f'{cl:.10f}', f'{tv:.10f}'])
        print(f"  Loss history saved: {loss_csv}")

    return best_params, loss_history

"""
Hybrid model components: DDM (data-driven model) + FPM (first-principles model).

DDM: MLP (11 -> 16 -> 8) predicting delta_kij(6) + q(1) + delta_tvp(1).
FPM: JAX differentiable distillation column + bubble-P solver.

Gradient path: ddm_params -> MLP -> (delta_kij, q, delta_tvp) -> column solver -> loss
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import jax.nn as jnn

from src.component_data import get_props_arrays, KIJ, COMPONENTS

# Component indices
IDX_C3 = 1
IDX_IC4 = 2
IDX_NC4 = 3
IDX_C6P = 6

# 6 tunable kij pairs: each lighter component paired with C6+ (pseudo-component)
PARAM_PAIRS = [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
PARAM_LABELS = [f'{COMPONENTS[i]}-{COMPONENTS[j]}' for i, j in PARAM_PAIRS]
N_PARAMS = len(PARAM_PAIRS)

# Component properties as JAX arrays (module-level for JIT)
Tc_np, Pc_np, omega_np, MW_np = get_props_arrays()
Tc_jax = jnp.array(Tc_np)
Pc_jax = jnp.array(Pc_np)
omega_jax = jnp.array(omega_np)
kij_base_jax = jnp.array(KIJ)

# DDM architecture
N_INPUT = 11    # z_F(7) + P_top + F + D + RD
N_HIDDEN = 16
N_OUTPUT = 8    # delta_kij(6) + q_raw(1) + delta_tvp_raw(1)


def make_kij_matrix(delta_params):
    """Apply 6 delta_kij values to C6+ pairs (symmetric)."""
    kij = kij_base_jax
    for k, (i, j) in enumerate(PARAM_PAIRS):
        kij = kij.at[i, j].set(kij_base_jax[i, j] + delta_params[k])
        kij = kij.at[j, i].set(kij_base_jax[i, j] + delta_params[k])
    return kij


def init_ddm_params(key):
    """Initialize DDM MLP parameters (11 -> 16 -> 8).

    Bias initialization:
      - delta_kij: 0 (no correction at init)
      - q_raw: -0.693 -> sigmoid(-0.693) ~ 0.333 -> q = 0.5 + 1.5*0.333 ~ 1.0
      - delta_tvp_raw: 0 -> sigmoid(0) = 0.5 -> delta_tvp = -1.25 + 2.5*0.5 = 0.0
    """
    k1, k2 = jax.random.split(key)
    params = {
        'W1': jax.random.normal(k1, (N_INPUT, N_HIDDEN)) * 0.1,
        'b1': jnp.zeros(N_HIDDEN),
        'W2': jax.random.normal(k2, (N_HIDDEN, N_OUTPUT)) * 0.01,
        'b2': jnp.concatenate([
            jnp.zeros(6),
            jnp.array([-0.693]),
            jnp.zeros(1),
        ]),
    }
    return params


def ddm_forward(ddm_params, features):
    """MLP forward pass: (11,) -> (delta_kij(6), q, delta_tvp).

    Transforms:
      q = 0.5 + 1.5 * sigmoid(q_raw)         -> [0.5, 2.0]
      delta_tvp = -1.25 + 2.5 * sigmoid(raw)  -> [-1.25, +1.25] bar
    """
    h = jnn.tanh(features @ ddm_params['W1'] + ddm_params['b1'])
    output = h @ ddm_params['W2'] + ddm_params['b2']

    delta_kij = output[:6]
    q = 0.5 + 1.5 * jnn.sigmoid(output[6])
    delta_tvp = -1.25 + 2.5 * jnn.sigmoid(output[7])

    return delta_kij, q, delta_tvp


def load_trained_params(npz_path):
    """Load trained DDM parameters from .npz file.

    Returns (ddm_params dict, feat_mean, feat_std, best_epoch).
    """
    d = dict(jnp.load(npz_path))
    ddm_params = {
        'W1': jnp.array(d['W1']),
        'b1': jnp.array(d['b1']),
        'W2': jnp.array(d['W2']),
        'b2': jnp.array(d['b2']),
    }
    feat_mean = d.get('feat_mean', None)
    feat_std = d.get('feat_std', None)
    best_epoch = int(d['best_epoch']) if 'best_epoch' in d else None
    return ddm_params, feat_mean, feat_std, best_epoch

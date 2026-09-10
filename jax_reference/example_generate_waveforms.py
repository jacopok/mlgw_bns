"""Example: generate gravitational-wave polarizations with mlgw_bns JAX.

This script shows how to use ``jax_import_n_predict.py`` to produce
frequency-domain waveforms (hp, hc) for binary neutron star systems.
It requires only JAX, NumPy, h5py — no mlgw_bns installed.

Run
---
    python example_generate_waveforms.py

Dependencies: jax, numpy, h5py, matplotlib (for the optional plot)
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_import_n_predict import load_predict

# ── 1. Load the model ──────────────────────────────────────────────────────────
#
# load_predict reads mlgw_bns_jax_model.h5 and returns a pure JAX function.
# The first call will JIT-compile the function; subsequent calls are fast.

predict = load_predict("mlgw_bns_jax_model.h5")

# ── 2. Define the frequency grid ───────────────────────────────────────────────
#
# Any 1-D array of frequencies in Hz between ~10 Hz and ~2048 Hz works.

f_min_hz = 20.0      # Hz  — lower frequency cutoff
f_max_hz = 2048.0    # Hz  — Nyquist frequency (srate = 4096 Hz)
n_freqs  = 4096

freqs = jnp.linspace(f_min_hz, f_max_hz, n_freqs)

# ── 3. Single waveform ────────────────────────────────────────────────────────
#
# params = [mass_ratio q, lambda_1, lambda_2, chi_1, chi_2]
#   q        : mass ratio m1/m2 >= 1  (dimensionless)
#   lambda_1 : tidal deformability of the heavier star  (dimensionless)
#   lambda_2 : tidal deformability of the lighter star  (dimensionless)
#   chi_1    : aligned spin of the heavier star  (dimensionless, |χ| ≤ 0.3)
#   chi_2    : aligned spin of the lighter star  (dimensionless, |χ| ≤ 0.3)

params = jnp.array([1.2,    # q
                    300.0,  # lambda_1
                    300.0,  # lambda_2
                    0.05,   # chi_1
                    0.05])  # chi_2

hp, hc = predict(
    params,
    freqs,
    total_mass   = jnp.array(2.8),    # total mass in solar masses
    distance_mpc = jnp.array(100.0),  # luminosity distance in Mpc
    inclination  = jnp.array(0.0),    # inclination angle in radians (0 = face-on)
)

print("Single waveform")
print(f"  hp shape : {hp.shape}  dtype: {hp.dtype}")
print(f"  |hp| max : {float(jnp.abs(hp).max()):.3e}")
print(f"  |hc| max : {float(jnp.abs(hc).max()):.3e}")

# ── 4. Batched waveforms with jax.vmap ────────────────────────────────────────
#
# vmap vectorises the function over a leading batch dimension — no Python loop.

batch_params = jnp.array([
    [1.0, 200.0, 200.0,  0.00,  0.00],   # equal mass, no spin, low tides
    [1.5, 500.0, 300.0,  0.10, -0.05],   # unequal mass, mild spin
    [2.0, 3000., 3000., -0.20,  0.20],   # high mass ratio, high tides
])

# vmap maps over axis 0 of params; freqs and scalars are broadcast.
batched_predict = jax.vmap(
    predict,
    in_axes=(0, None, None, None, None),   # batch only the params axis
)

hp_batch, hc_batch = batched_predict(
    batch_params,
    freqs,
    jnp.array(2.8),
    jnp.array(100.0),
    jnp.array(0.0),
)

print("\nBatched waveforms (vmap over 3 parameter sets)")
print(f"  hp_batch shape : {hp_batch.shape}")

# ── 5. JIT compilation ────────────────────────────────────────────────────────
#
# jit the batched function once and re-use for fast repeated evaluations.

batched_jit = jax.jit(batched_predict)

import time
_ = batched_jit(batch_params, freqs, jnp.array(2.8), jnp.array(100.0), jnp.array(0.0))  # warm-up

t0 = time.perf_counter()
for _i in range(50):
    hp_jit, _ = batched_jit(
        batch_params, freqs, jnp.array(2.8), jnp.array(100.0), jnp.array(0.0)
    )
hp_jit.block_until_ready()
dt_ms = (time.perf_counter() - t0) / 50 * 1e3
print(f"\nJIT-compiled batched call (3 waveforms, {n_freqs} freq bins): {dt_ms:.2f} ms/call")

# ── 6. Optional: plot amplitude and phase ────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    freqs_np = np.asarray(freqs)
    hp_np    = np.asarray(hp)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.loglog(freqs_np, np.abs(hp_np), color="#3a7abf")
    ax1.set_ylabel(r"$|\tilde{h}_+|$")
    ax1.set_title(f"mlgw_bns JAX — q={float(params[0]):.1f}, "
                  f"λ₁={float(params[1]):.0f}, λ₂={float(params[2]):.0f}")

    ax2.semilogx(freqs_np, np.unwrap(np.angle(hp_np)), color="#e07b39")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel(r"Phase $\Phi_+$ [rad]")

    fig.tight_layout()
    fig.savefig("example_waveform.png", dpi=150)
    plt.close(fig)
    print("\nWaveform plot saved to example_waveform.png")
except ImportError:
    pass

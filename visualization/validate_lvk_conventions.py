"""The LVK spin angles (mlgw_bns.spin_conversion) against IMRPhenomXPHM.

For a binary given by the LVK angles (theta_jn, phi_jl, tilt_i, phi_12, a_i,
phase at f_ref), this compares mlgw_bns' twisted co-precessing multipoles
(PrecessingModel.jax_predict_modes, c_+ and c_x times the co-precessing
mode) with IMRPhenomXPHM's, one co-precessing (l, m) at a time
(``mode_array``), for h_+ and h_x separately: the match maximised over time
only, and the phase of the overlap there. A convention error (the sign or
offset of phi_jl or phase, the line of sight) shows as an O(1) mismatch or as
a residual phase that depends on the scanned angle; the model difference is
a match of ~0.99 with a fixed phase.

What it shows: the phases do not depend on phase, phi_jl or theta_jn, and
those of h_+ and h_x differ by pi, which is Model.predict's known h_x sign
(CHANGELOG, Known issues); with h_x negated mlgw_bns and LALSuite agree.

Needs bilby and lalsuite. Run with:
python visualization/validate_lvk_conventions.py
"""
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from bilby.gw.source import _base_lal_cbc_fd_waveform

from mlgw_bns.batched_precession import batch_arguments
from mlgw_bns.model import Model
from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic

DF = 1 / 256
F_MIN, F_MAX, F_REF = 20.0, 1024.0, 20.0
FULL = np.arange(0, F_MAX + DF / 2, DF)
BAND = FULL >= F_MIN
FREQUENCIES = FULL[BAND]
MODES = [(2, 2), (2, 1), (3, 3), (4, 4)]

predict = jax.jit(PrecessingModel(Model.default_for_testing()).jax_predict_modes(modes=MODES))


def mlgw_modes(p):
    params = PrecessingParametersWithExtrinsic.from_lvk(
        **p, reference_frequency_hz=F_REF, lambda_1=5.0, lambda_2=5.0, distance_mpc=100.0)
    cop, c_plus, c_cross = (np.asarray(x)[0] for x in predict(*batch_arguments([params], FREQUENCIES)))
    return {key: (c_plus[j] * cop[j], c_cross[j] * cop[j]) for j, key in enumerate(MODES)}


def xphm_modes(p):
    out = {}
    for ell, emm in MODES:
        wf = _base_lal_cbc_fd_waveform(
            frequency_array=FULL, luminosity_distance=100.0, **p,
            waveform_approximant="IMRPhenomXPHM", reference_frequency=F_REF,
            minimum_frequency=F_MIN, maximum_frequency=F_MAX, mode_array=[[ell, emm]],
            catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
            pn_phase_order=-1, pn_amplitude_order=0)
        out[(ell, emm)] = (wf["plus"][BAND], wf["cross"][BAND])
    return out


def match(a, b):
    """|<a, b>| maximised over time, white noise, and the phase there."""
    n = 2 * 2 ** int(np.ceil(np.log2(len(FULL))))
    product = np.zeros(n, dtype=complex)
    product[np.flatnonzero(BAND)] = a * np.conj(b)
    overlap = np.fft.ifft(product) * n
    i = np.argmax(np.abs(overlap))
    norm = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
    return np.abs(overlap[i]) / norm, np.angle(overlap[i])


base = dict(mass_1=1.6, mass_2=1.3, a_1=0.3, tilt_1=1.2, a_2=0.2, tilt_2=2.0, phi_12=1.0,
            phi_jl=0.7, theta_jn=0.9, phase=0.4)
scans = [("base", {})] + [(f"{key}={value:.2f}", {key: value}) for key, values in (
    ("phase", [0.0, 1.5, 3.0, 4.5]), ("phi_jl", [0.0, 2.0, 4.0]), ("theta_jn", [0.3, 2.2]),
    ("tilt_1", [0.0])) for value in values]
print("match/phase of the overlap, per co-precessing mode and polarization")
for label, change in scans:
    p = dict(base, **change)
    start = time.time()
    ours, theirs = mlgw_modes(p), xphm_modes(p)
    line = [f"{label:14s}"]
    for key in MODES:
        for pol in (0, 1):
            m, phase = match(ours[key][pol], theirs[key][pol])
            line.append(f"{key}{'+x'[pol]} {m:.3f}/{phase:+.2f}")
    print("  ".join(line), f"({time.time() - start:.0f}s)", flush=True)

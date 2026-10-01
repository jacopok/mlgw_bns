r"""Anatomy of the precessing mismatch floor against TEOBResumS.

For each binary and line of sight, several sets of co-precessing multipoles
are twisted along the *same* Euler angles (ours) and projected the same
way, and each is compared with the :math:`h_+, h_\times` TEOBResumS returns
for that precessing binary (detector strain, maximised over time and a
global phase, as in :mod:`validate_precessing_against_teob`):

* ``surrogate`` --- the shipping pipeline, ``PrecessingModel.predict``;
* ``surrogate_interpolated_target`` --- the same, but against TEOBResumS'
  h+, hx interpolated onto the model grid, as
  :mod:`validate_precessing_against_teob` does (everything else here is
  compared on TEOBResumS' own frequency nodes);
* ``eob`` --- TEOBResumS' aligned-spin multipoles as the model is compared
  with them (``source="eob"``: its own call, referenced to the merger);
* ``teob_raw`` --- the multipoles TEOBResumS twisted in that very
  precessing call (its ``hflm``), in its own time and orbital-phase origin
  (the start of the integration, where the spins are imposed). Only our
  angles and our twist separate this from TEOBResumS' h+, hx;
* ``surrogate_rotated`` --- the surrogate multipoles moved to TEOBResumS'
  origin by one time shift and one orbital-phase rotation
  :math:`e^{i (2 \pi f t_0 + m \varphi_0)}`, both fitted on the
  *aligned-spin* multipoles alone (surrogate against ``teob_raw``), with
  no information from the precessing waveform;
* ``surrogate_best`` --- the surrogate with its orbital phase (the
  coalescence phase, a rotation :math:`e^{i m \varphi}` of the
  co-precessing multipoles) optimised directly against the precessing
  waveform: how close the model's family gets.

It also records, per binary, the phase offset of each multipole against
``teob_raw`` after the time shift alone, to check that it is
:math:`m \varphi_0` for every mode --- a rotation of the orbital phase ---
rather than an independent error per mode.

Run with: python visualization/precession_floor_anatomy.py [--n-binaries N]
    [--n-orientations K] [--processes P]
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import precession_error_vs_inplane_spin as scaling
import validate_precessing_against_teob as v
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import (
    PrecessingModel,
    PrecessingParametersWithExtrinsic,
    polarizations_from_inertial_modes,
    twist_modes_frequency_domain,
)

DATA_PATH = Path(__file__).with_name("precession_floor_anatomy.npz")
VARIANTS = ("surrogate", "surrogate_interpolated_target", "eob", "teob_raw",
            "surrogate_rotated", "surrogate_best")
MODE_KEYS = [(mode.l, mode.m) for mode in v.MODES]

_WORKER: dict = {}


def _setup():
    if not _WORKER:
        model = Model.default_for_testing(modes=v.MODES)
        v.SRATE_HZ = model.dataset.effective_srate_hz
        _WORKER["model"] = model
        _WORKER["precessing"] = PrecessingModel(model)
        _WORKER["validator"] = ValidateModel(model.mode_models[Mode(2, 2)])
    return _WORKER["model"], _WORKER["precessing"], _WORKER["validator"]


def rotated(modes: dict, frequencies: np.ndarray, time_shift: float, phase: float):
    """Multipoles shifted in time and rotated in orbital phase: ``e^{i(2 pi f t + m phi)}``."""
    return {
        (l, m): value * np.exp(2j * np.pi * frequencies * time_shift + 1j * m * phase)
        for (l, m), value in modes.items()
    }


def strain(modes, frequencies, angles, params, antenna):
    """Twist co-precessing multipoles along ``angles`` and project on a detector."""
    positive, negative = twist_modes_frequency_domain(
        modes, frequencies, angles, mass_sum_seconds=params.aligned().mass_sum_seconds
    )
    hp, hc = polarizations_from_inertial_modes(
        positive, negative, inclination=params.inclination, azimuth=params.azimuth
    )
    return antenna[0] * hp + antenna[1] * hc


def time_origin_offset(frequencies, teob_22, surrogate_22):
    """Seconds between TEOBResumS' (2,2) time origin and the surrogate's.

    The slope of their phase difference over a narrow window of TEOBResumS'
    own grid (``frequencies``), unwrapped sample to sample. The true offset
    (minus the time from TEOBResumS' start to the merger, ~ -1200 s) moves
    the phase by more than pi per sample, so this returns it modulo
    ``1 / df`` (2048 s): exact wherever it is applied, TEOBResumS' own
    nodes, all multiples of ``df``. The largest residual of the linear fit
    is returned too, as a check.
    """
    difference = np.unwrap(np.angle(teob_22 * np.conj(surrogate_22)))
    slope, intercept = np.polyfit(frequencies, difference, 1)
    residual = difference - (slope * frequencies + intercept)
    return slope / (2.0 * np.pi), float(np.max(np.abs(residual)))


def aligned_fit(model, validator, params, f_native, raw):
    """The aligned-spin multipoles of one binary, and how they relate.

    On the model's grid snapped to TEOBResumS' nodes ``f_native`` within the
    comparison band: the surrogate's and ``source="eob"`` multipoles, and
    TEOBResumS' own ``raw`` ones (as returned by ``teob_run``) moved by a
    pure time shift next to them. Then one time shift ``t0`` and one
    orbital-phase rotation ``phi0`` taking the surrogate's to TEOBResumS'
    (``e^{i (2 pi f t0 + m phi0)}``), fitted on these multipoles alone, and
    the residual phase of each multipole after the time shift alone.

    Returns
    -------
    nodes, fb, surrogate, eob, teob_raw, fit
    """
    df = f_native[1] - f_native[0]
    model_grid = validator.frequencies
    model_grid = model_grid[(model_grid >= max(v.BAND_LO, f_native[0]))
                            & (model_grid <= min(v.BAND_HI, f_native[-1]))]
    nodes = np.unique(np.rint((model_grid - f_native[0]) / df).astype(int))
    fb = f_native[nodes]
    surrogate = model.coprecessing_modes_dict(fb, params.aligned())
    eob = model.coprecessing_modes_dict(fb, params.aligned(), source="eob")

    window = (f_native >= 200.0) & (f_native <= 200.2)
    amplitude_22, phase_22 = raw[(2, 2)]
    offset_t, offset_residual = time_origin_offset(
        f_native[window],
        (amplitude_22 * np.exp(1j * phase_22))[window],
        model.coprecessing_modes_dict(f_native[window], params.aligned())[(2, 2)],
    )
    # TEOBResumS' own multipoles, moved by that time shift only: its
    # orbital-phase origin is untouched.
    teob_raw = rotated(
        {k: a[nodes] * np.exp(1j * ph[nodes]) for k, (a, ph) in raw.items()},
        fb, -offset_t, 0.0,
    )

    aligned_mismatch, t0, phi0 = validator.full_waveform_mismatch(
        teob_raw, surrogate, frequencies=fb, return_shifts=True)
    eob_aligned_mismatch, t_eob, phi_eob = validator.full_waveform_mismatch(
        teob_raw, eob, frequencies=fb, return_shifts=True)
    weights = np.gradient(fb) / validator.psd_at_frequencies(fb)
    shifted = rotated(surrogate, fb, t0, 0.0)
    offsets = [
        float(np.angle(np.sum(weights * teob_raw[k] * np.conj(shifted[k]))))
        for k in MODE_KEYS
    ]
    fit = dict(aligned_mismatch=aligned_mismatch, t0=t0, phi0=phi0,
               eob_aligned_mismatch=eob_aligned_mismatch, t_eob=t_eob,
               phi_eob=phi_eob, offsets=offsets, time_origin=offset_t,
               time_origin_residual=offset_residual)
    return nodes, fb, surrogate, eob, teob_raw, fit


def evaluate_binary(cases):
    """Every variant, for every orientation of one binary.

    Everything is evaluated on the model's grid snapped to TEOBResumS'
    own frequency nodes, so that TEOBResumS' output is used as it comes,
    never interpolated: its h+, hx are sums of multipoles which beat on
    ~1 / (t_44 - t_22) ~ 1e-3 Hz, a couple of its samples at 20 Hz.
    """
    model, precessing, validator = _setup()
    start = time.time()

    index, intrinsic, chi_1, chi_2, _ = cases[0]
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=chi_1, chi_2=chi_2,
        distance_mpc=v.DISTANCE_MPC, inclination=0.0, total_mass=v.TOTAL_MASS,
        reference_frequency_hz=v.SPIN_REFERENCE_FREQUENCY_HZ,
    )
    angles = precessing.euler_angles(params, float(validator.frequencies[0]))

    rows = []
    fit = None
    for _, _, _, _, orientation in cases:
        params.inclination = orientation["inclination"]
        params.azimuth = orientation["azimuth"]
        antenna = v.antenna_patterns(orientation["theta"], orientation["phi"],
                                     orientation["psi"])
        try:
            f_native, hp_native, hc_native, *raw = v.teob_run(
                intrinsic, chi_1, chi_2, params.inclination, params.azimuth,
                multipoles=fit is None,
            )
        except RuntimeError as error:  # TEOBResumS' root finder, occasionally
            print(f"binary {index}: TEOBResumS failed ({error})", flush=True)
            return rows
        target_native = antenna[0] * hp_native + antenna[1] * hc_native

        if fit is None:
            nodes, fb, surrogate, eob, teob_raw, fit = aligned_fit(
                model, validator, params, f_native, raw[0])
            model_grid = validator.frequencies[
                (validator.frequencies >= fb[0]) & (validator.frequencies <= fb[-1])]

        target = target_native[nodes]

        def mismatch(modes, reference=target, frequencies=fb):
            return validator.full_waveform_mismatch(
                {(2, 2): reference},
                {(2, 2): strain(modes, frequencies, angles, params, antenna)},
                frequencies=frequencies,
            )

        def with_orbital_phase(phase):
            return mismatch(rotated(surrogate, fb, 0.0, phase))

        grid = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        coarse = [with_orbital_phase(p) for p in grid]
        best = int(np.argmin(coarse))
        step = grid[1] - grid[0]
        refined = minimize_scalar(
            with_orbital_phase, bounds=(grid[best] - step, grid[best] + step),
            method="bounded", options={"xatol": 1e-4},
        )

        # What validate_precessing_against_teob measures: the same surrogate
        # waveform, against TEOBResumS' h+, hx interpolated (through
        # amplitude and unwrapped phase) onto the model's own grid.
        interpolated_target = v.interp_fd(model_grid, f_native, target_native)
        surrogate_on_model_grid = model.coprecessing_modes_dict(model_grid, params.aligned())

        mismatches = {
            "surrogate": mismatch(surrogate),
            "surrogate_interpolated_target": mismatch(
                surrogate_on_model_grid, interpolated_target, model_grid),
            "eob": mismatch(eob),
            "teob_raw": mismatch(teob_raw),
            "surrogate_rotated": mismatch(rotated(surrogate, fb, fit["t0"], fit["phi0"])),
            "surrogate_best": min(refined.fun, coarse[best]),
        }
        rows.append(dict(
            binary=index, inclination=params.inclination,
            opening_angle=float(angles.beta.max()),
            in_plane_spin=max(np.hypot(*chi_1[:2]), np.hypot(*chi_2[:2])),
            mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2, chi_1z=chi_1[2], chi_2z=chi_2[2],
            best_orbital_phase=float(refined.x),
            **{f"mismatch_{k}": m for k, m in mismatches.items()},
            **{k: val for k, val in fit.items() if k != "offsets"},
            **{f"offset_{l}{m}": o for (l, m), o in zip(MODE_KEYS, fit["offsets"])},
        ))
    print(
        f"binary {index}: beta {angles.beta.max():.3f}  phi0 {fit['phi0']:+.3f}  "
        f"aligned {fit['aligned_mismatch']:.1e}  "
        + "  ".join(f"{k} {np.median([r['mismatch_' + k] for r in rows]):.1e}"
                    for k in VARIANTS)
        + f"  ({time.time() - start:.0f}s)",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-binaries", type=int, default=32)
    parser.add_argument("--n-orientations", type=int, default=4)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()

    scaling.N_BINARIES = args.n_binaries
    scaling.N_ORIENTATIONS = args.n_orientations
    cases = scaling.draw_cases()
    per_binary = [
        [case for case in cases if case[0] == index] for index in range(args.n_binaries)
    ]
    with multiprocessing.Pool(args.processes) as pool:
        rows = [row for result in pool.imap_unordered(evaluate_binary, per_binary)
                for row in result]

    records = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(DATA_PATH, **records)
    print(f"\n{len(rows)} observations of {len(set(records['binary']))} binaries "
          f"written to {DATA_PATH}")
    for variant in VARIANTS:
        values = records[f"mismatch_{variant}"]
        print(f"{variant:>18}: median {np.nanmedian(values):.2e}, "
              f"90th pct {np.nanpercentile(values, 90):.2e}, worst {np.nanmax(values):.2e}")


if __name__ == "__main__":
    main()

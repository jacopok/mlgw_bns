"""Is the precessing-vs-TEOB floor (1.2e-4 since the 2026-11-17 switch of the
reference TEOBResumS, 7e-7 before) the model or the reference?

  run OUT.npz N MODEL    validate_model.precessing_mismatches on seeds 0..N-1
                         with the installed TEOBResumS, keeping its h+, hx and
                         the first sample of its dynamics
  compare A.npz B.npz MODEL   TEOB(A) vs TEOB(B) on the same binaries, and the
                         model against each

Run ``run`` once in each build's environment, then ``compare`` in either.
"""
import sys

import numpy as np

import validate_model as vm
from mlgw_bns.model import MODELS_AVAILABLE, Model
from mlgw_bns.higher_order_modes import Mode


def load(name):
    return Model.default_for_testing(name) if name in MODELS_AVAILABLE else vm.load_model(name)


def gamma0_from_torque():
    """Start gamma at alpha's initial value, the direction of the torque (the
    rhs of the spin equations), as TEOBResumS does since its 293ed39f,
    instead of the NLO formula (Eq. A5 of arXiv:2111.03675) it used before."""
    import mlgw_bns.batched_precession as bp

    original = bp._reference_state

    def reference_state(binary, chi_1, chi_2, reference_frequency_22):
        y0, alpha_reference = original(binary, chi_1, chi_2, reference_frequency_22)
        return y0.at[9].set(0.0 * alpha_reference), alpha_reference

    bp._reference_state = reference_state


def run(out, n, name):
    import os

    import mlgw_bns.dataset_generation as dg
    import EOBRun_module

    if os.environ.get("GAMMA0") == "torque":
        gamma0_from_torque()

    model = load(name)
    captured = []
    original = dg.eobrun

    def eobrun(params):
        result = original(params)
        dyn = result[-1]
        captured.append({k: float(np.asarray(dyn[k])[0]) for k in ("t", "MOmega", "phi") if k in dyn})
        return result

    dg.eobrun = eobrun
    hp_all, hc_all = [], []
    original_tpp = vm.teob_precessing_polarizations

    def tpp(*args):
        hp, hc = original_tpp(*args)
        hp_all.append(hp)
        hc_all.append(hc)
        return hp, hc

    vm.teob_precessing_polarizations = tpp
    res = vm.precessing_mismatches(model, n)
    np.savez(
        out, build=EOBRun_module.__file__, hp=np.array(hp_all), hc=np.array(hc_all),
        plus=res["plus"], cross=res["cross"], plus_no=res["plus_no_optimisation"],
        beta_max=res["beta_max"], start_momega=[c["MOmega"] for c in captured],
        start_phi=[c["phi"] for c in captured],
        reference_phase=[p.reference_phase for p in res["params"]],
    )
    print("start MOmega", np.round([c["MOmega"] for c in captured][:5], 8))


def compare(a, b, name):
    model = load(name)
    A, B = np.load(a), np.load(b)
    validator = vm.ValidateModel(model.mode_models[Mode(2, 2)])
    grid = validator.frequencies
    grid = grid[(grid >= vm.PRECESSING_MIN_FREQUENCY_HZ) & (grid < grid[-1])]
    for pol in ("hp", "hc"):
        mm = [validator.mismatch(x, y, frequencies=grid, max_delta_t=1e-3) for x, y in zip(A[pol], B[pol])]
        print(f"TEOB A vs TEOB B {pol}: median {np.median(mm):.2e}, max {np.max(mm):.2e}")
    for label, f in (("A", A), ("B", B)):
        print(f"model vs TEOB {label} ({f['build']}): h+ optimised median {np.median(f['plus']):.2e} "
              f"max {np.max(f['plus']):.2e}; not optimised median {np.median(f['plus_no']):.2e}")
    print("start MOmega A/B:", np.round(A["start_momega"][:5], 8), np.round(B["start_momega"][:5], 8))
    print("reference_phase A-B:", np.round((A["reference_phase"] - B["reference_phase"])[:8], 4))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4])

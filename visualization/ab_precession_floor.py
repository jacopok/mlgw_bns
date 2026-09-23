"""A/B of the two precession-floor fixes against TEOBResumS h+, hx.

Variants: the (2,2) phase `EulerAngles.reanchored` reads (the model's own,
or the old aliased np.unwrap(np.angle(h22)) on the model grid) x where the
spins are taken to be given (where TEOBResumS' FD path imposes them,
0.95 * initial_frequency, or the start of the PN integration, a few Hz). Reuses validate_precessing_against_teob.

Run with: python visualization/ab_precession_floor.py VARIANT N_BIN N_ORIENT
VARIANT in {aliased_start, aliased_teob, native_start, native_teob}.
"""
import sys
from pathlib import Path

import numpy as np

import validate_precessing_against_teob as v
from mlgw_bns.model import Model
from mlgw_bns.precessing_model import EulerAngles

variant = sys.argv[1]
phase_source, reference = variant.split("_")
if reference == "start":
    v.SPIN_REFERENCE_FREQUENCY_HZ = None
if phase_source == "aliased":
    native_reanchored = EulerAngles.reanchored

    def aliased_reanchored(self, frequencies_hz, reference_phase, *args, **kwargs):
        aliased = np.unwrap(np.angle(np.exp(1j * reference_phase)))
        return native_reanchored(self, frequencies_hz, aliased, *args, **kwargs)

    EulerAngles.reanchored = aliased_reanchored

model = Model.default_for_testing(modes=v.MODES)
v.SRATE_HZ = model.dataset.effective_srate_hz
records = v.validate(model, int(sys.argv[2]), int(sys.argv[3]))
np.savez(Path(__file__).with_name(f"ab_precession_floor_{variant}.npz"), **records)

beta = records["opening_angle"]
print(f"RESULT {variant}: median reanchored "
      f"{np.nanmedian(records['reanchored_mismatch']):.2e}, plain "
      f"{np.nanmedian(records['plain_mismatch']):.2e}")
for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .5)):
    s = (beta >= lo) & (beta < hi)
    if s.any():
        print(f"RESULT {variant}   beta {lo}-{hi}: n={s.sum()} reanchored "
              f"{np.nanmedian(records['reanchored_mismatch'][s]):.2e} plain "
              f"{np.nanmedian(records['plain_mismatch'][s]):.2e}")

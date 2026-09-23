"""Sign of h_x relative to h_+: Model.predict against TEOBResumS.

For an aligned-spin binary, negating h_x is the same as viewing it from
pi - iota, so no phase-maximised single-detector mismatch can see it.
This compares the phase-sensitive ratio h_x / h_+ directly (for the (2,2)
it is +-i 2 cos(iota) / (1 + cos^2 iota)), for Model.predict, for TEOBResumS
in the frequency domain (conjugated into the mlgw_bns convention) and for the
FFT of TEOBResumS' time-domain output (numpy's FFT is the mlgw_bns
convention; see compare_fd_twist_with_teob_formula.py).

Run with: python visualization/probe_hcross_sign.py
"""
import numpy as np
from EOBRun_module import EOBRunPy

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.mode_model import ParametersWithExtrinsic

model = Model.default_for_testing(modes=[Mode(2, 2)])
check = np.array([40.0, 100.0, 300.0])
for iota in (0.5, 1.0, 2.5):
    params = ParametersWithExtrinsic(1.3, 400.0, 600.0, 0.05, 0.0, 100.0, iota, 2.8)
    hp, hc = model.predict(check, params)
    base = dict(q=1.3, LambdaAl2=400.0, LambdaBl2=600.0, chi1=0.05, chi2=0.0, M=2.8,
                distance=100.0, inclination=iota, srate_interp=8192.0,
                use_geometric_units="no", interp_uniform_grid="yes", output_hpc="no",
                arg_out="no", use_mode_lm=[1], coalescence_angle=0.0)
    f, rp, ip, rc, ic = EOBRunPy(dict(base, domain=1, df=1 / 512, initial_frequency=15.0))
    ratio_fd = np.interp(check, f, np.conj((np.asarray(rc) + 1j * np.asarray(ic))
                                           / (np.asarray(rp) + 1j * np.asarray(ip))))
    t, hp_t, hc_t = EOBRunPy(dict(base, domain=0, initial_frequency=15.0))
    fr = np.fft.rfftfreq(len(t), t[1] - t[0])
    ratio_td = np.interp(check, fr, np.fft.rfft(hc_t) / np.fft.rfft(hp_t))
    expected = 2 * np.cos(iota) / (1 + np.cos(iota) ** 2)
    print(f"iota {iota}: |h_x/h_+| expected {expected:.3f}\n"
          f"  Model.predict   {np.round(hc / hp, 3)}\n"
          f"  TEOB FD (conj)  {np.round(ratio_fd, 3)}\n"
          f"  FFT of TEOB TD  {np.round(ratio_td, 3)}", flush=True)

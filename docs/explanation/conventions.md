(conventions)=
# Conventions: LALSuite and TEOBResumS

`mlgw_bns` is a surrogate of TEOBResumS, and is mostly used through
LALSuite-based inference codes. The two codes describe the same binary with
different conventions. Every predictor takes a `convention` argument,
`"lal"` (the default) or `"teob"`, which sets all of them at once
({mod}`mlgw_bns.conventions`).

## Quick reference

| | LALSuite (`"lal"`, default) | TEOBResumS (`"teob"`) |
|---|---|---|
| Fourier transform | $\tilde{h}(f) = \int h(t)\, e^{-2\pi i f t}\, \mathrm{d}t$ | $e^{+2\pi i f t}$ (its frequency-domain output): the complex conjugate |
| mode decomposition | $h_+ - i h_\times = \sum_{\ell m} h_{\ell m}\, {}_{-2}Y_{\ell m}(\iota, \Phi - \phi_{\rm ref})$ | its own, with `coalescence_angle` $c$ |
| sign of $h_\times$ relative to $h_+$ | that of the decomposition | the same |
| `azimuth` (precessing) | $\Phi$; $\pi/2$ for LALSuite's waveforms | `coalescence_angle` $c$ |
| `coalescence_phase` (aligned spins) | $\phi_{\rm ref}$, with $\Phi = \pi/2$ | `coalescence_angle` $c$ |
| same binary | $\Phi = \pi/2$, $\phi_{\rm ref} = \phi$ | $c = \pi$, `reference_phase` $= \phi$ |
| third Euler angle | $\gamma$: $D^\ell_{mm'} = e^{-im\alpha} d^\ell_{mm'}(\beta) e^{-im'\gamma}$, $\dot\gamma = -\dot\alpha\cos\beta$ | $-\gamma$ (IMRPhenomXPHM's $\epsilon$) |
| spins | LVK angles at $f_{\rm ref}$ ({func}`~mlgw_bns.spin_conversion.lvk_to_precessing`) | Cartesian, $\hat{z} = \hat{L}$, at $0.95\,$`initial_frequency` |
| where the orbital phase is set | $f_{\rm ref}$, each model in its own way | zero at the start of the integration ({meth}`~mlgw_bns.precessing_model.PrecessingModel.teob_reference_phase`) |
| masses | `mass_1`, `mass_2` | $q = m_1 / m_2 \geq 1$, body 1 the heavier |

The inputs that are the same in both conventions (masses, spins as vectors
in the frame of $\hat{L}$ at `reference_frequency_hz`, `reference_phase`,
`inclination`) are those of
{class}`~mlgw_bns.precessing_model.PrecessingParametersWithExtrinsic`.

## What the switch changes

**The Fourier convention.** LALSuite (and numpy's FFT) transforms with
$e^{-2\pi i f t}$, TEOBResumS' frequency-domain output with $e^{+2\pi i f t}$.
For real $h_+$, $h_\times$ the two transforms are complex conjugates, and with
`"teob"` the polarizations are conjugated. Nothing else in the polarizations
changes: in particular, the sign of $h_\times$ relative to $h_+$ is the same
in both codes, in the time domain and in the frequency domain.

**The azimuth of the line of sight.** For a given binary, with its spins in
the frame of $\hat{L}$, LALSuite's line of sight is at $\Phi = \pi/2$ (with
the orbital phase $\phi_{\rm ref}$ given separately), and TEOBResumS'
`coalescence_angle` for the same binary is $c = \pi$. In general $\Phi$ and
$c$ move the line of sight in opposite senses: $c = 3\pi/2 - \Phi$. With precessing spins the azimuth and the
orbital phase are separate inputs, `azimuth` and `reference_phase`, as in
{func}`~mlgw_bns.spin_conversion.lvk_to_precessing`; for aligned spins only
their combination matters, and `coalescence_phase` is LALSuite's
$\phi_{\rm ref}$ (at $\Phi = \pi/2$) or TEOBResumS' `coalescence_angle`.

**The third Euler angle.** The twist ({ref}`precession`) uses TEOBResumS'
$\gamma$, with the minimal-rotation condition $\dot\gamma = \dot\alpha
\cos\beta$. LALSuite's mode rotation (`XLALSimInspiralPrecessionRotateModes`,
the usual Wigner $D^\ell_{mm'}(\alpha, \beta, \gamma)$) has
$\dot\gamma = -\dot\alpha \cos\beta$: its $\gamma$ is minus TEOBResumS', with
the same $\alpha$ and $\beta$. IMRPhenomXPHM calls TEOBResumS' angle
$\epsilon$ (Pratten et al. 2021, appendix E), and its $\gamma = -\epsilon$ is
LALSuite's.
{meth}`EulerAngles.in_convention <mlgw_bns.precessing_model.EulerAngles.in_convention>`
returns $(\alpha, \beta, \gamma)$ in either convention. The angle tables
passed between the low-level functions (`angles=` arguments,
{func}`~mlgw_bns.batched_precession.precession_angles`) are always in
TEOBResumS' convention.

## What it does not change

**Where the orbital phase is set.** LALSuite approximants each define
$\phi_{\rm ref}$ at $f_{\rm ref}$ through their own phase and time
alignment: at the same $\phi_{\rm ref}$, IMRPhenomXHM, IMRPhenomXAS and
TaylorF2 differ by a constant phase that depends on the binary, and so do
IMRPhenomXPHM and IMRPhenomPv2. `mlgw_bns` defines it through the waveform
({ref}`precession-reference-phase`), and compared with any of them it agrees
up to such a constant. For precessing spins the orbital phase also orients
the separation relative to the spins, and the binary is the same in both
conventions for the same `reference_phase`; to reproduce a TEOBResumS run,
{meth}`~mlgw_bns.precessing_model.PrecessingModel.teob_reference_phase`
converts its phase origin.

**The spins.** Both conventions take the spins as vectors in the frame of
$\hat{L}$ at `reference_frequency_hz`.
{func}`~mlgw_bns.spin_conversion.lvk_to_precessing` converts the LVK angles
(`theta_jn`, `phi_jl`, `tilt_1`, `tilt_2`, `phi_12`, `a_1`, `a_2`, `phase`)
to them, as LALSuite's `SimInspiralTransformPrecessingNewInitialConditions`
does, and gives the `azimuth` in either convention. TEOBResumS imposes its
spins at $0.95$ times its `initial_frequency`.

**The low-level functions.** {func}`mlgw_bns.batched.mode_polarizations`
and the `azimuth` the low-level precession functions receive internally use
one fixed convention: the co-precessing multipole $m$ is rotated by
$e^{i m (a + \phi)}$, with $a$ = {func}`~mlgw_bns.conventions.internal_azimuth`
and $\phi$ the orbital phase, and the output is in LALSuite's Fourier
convention. {func}`~mlgw_bns.conventions.aligned_internal_phase` gives the
$a + \phi$ of an aligned-spin waveform.

## How this was checked

`tests/test_conventions.py` compares both polarizations at once,
maximising over a time shift only: a constant phase, a polarization angle or
a separate optimisation of each polarization absorbs the very signs at stake
(the sign of $h_\times$ went unnoticed until 2.0 for this reason).

- `"teob"` against TEOBResumS' frequency-domain precessing output, with
  its `coalescence_angle` as `azimuth`: overlap 0.9989; 0.14 with
  $-h_\times$.
- `"lal"` against IMRPhenomXPHM from the same LVK parameters: 0.996--0.998
  with a constant phase left free (see above), 0.05 with $-h_\times$.
- The two conventions give complex-conjugate polarizations for the same
  binary.

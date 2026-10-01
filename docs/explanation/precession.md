(precession)=
# Precession: the twist and its conventions

The surrogate of {class}`~mlgw_bns.model.Model` is aligned-spin. Precessing
waveforms are built from it, as TEOBResumS builds its own and as most
phenomenological models do, by "twisting":

1. take the aligned-spin multipoles as the multipoles in a *co-precessing*
   frame, one whose $z$ axis follows the orbital angular momentum
   $\hat{L}(t)$;
2. integrate the post-Newtonian (PN) spin-precession equations, which give
   $\hat{L}(t)$ and so the Euler angles $\alpha, \beta, \gamma$ of the
   co-precessing frame;
3. rotate the co-precessing multipoles into the inertial frame with those
   angles, and project them on the sky.

This page describes each step, the conventions they use, and how to
reproduce TEOBResumS' precessing waveforms with them. The code is
{mod}`mlgw_bns.precessing_model` (numpy, {class}`~mlgw_bns.precessing_model.PrecessingModel`)
and {mod}`mlgw_bns.batched_precession` (JAX, batched,
{meth}`PrecessingModel.jax_predict <mlgw_bns.precessing_model.PrecessingModel.jax_predict>`):
the two agree to mismatches of $10^{-11}$--$10^{-10}$, and the JAX one is
orders of magnitude faster.

## Parameters and frame

A precessing binary is described by
{class}`~mlgw_bns.precessing_model.PrecessingParametersWithExtrinsic`. Everything
is given at a reference frequency `reference_frequency_hz`, $f_{\rm ref}$, the
$(2, 2)$ gravitational-wave frequency (TEOBResumS' `initial_frequency`,
LALSuite's `f_ref`). At $f_{\rm ref}$:

- the $z$ axis is along the orbital angular momentum, $\hat{L}_0 = \hat{z}$;
- `chi_1`, `chi_2` are the two dimensionless spin vectors, in that frame;
- `reference_phase` is the orbital phase (see [below](precession-reference-phase));
- the line of sight has polar angle `inclination` $\iota$ and azimuth
  `azimuth` $\varphi$, also in that frame.

Precession rotates both the in-plane spins and $\hat{L}$ between any two
frequencies, so the same vectors given at two different reference frequencies
describe two different binaries. The co-precessing multipoles are those of
the aligned-spin binary with the $z$ components of the spins
({meth}`PrecessingParametersWithExtrinsic.aligned <mlgw_bns.precessing_model.PrecessingParametersWithExtrinsic.aligned>`).

## The Euler angles

The spins $\vec{S}_{A,B}$ and $\hat{L}$ are evolved with the PN
spin-precession equations TEOBResumS uses ({mod}`mlgw_bns.twist_waveform`
reproduces its `eob_spin_dyn_rhs_PN`), driven by a 3.5PN orbital-frequency
evolution $\dot\Omega$ with the leading tidal term. The angles are those of
$\hat{L}$,

$$
\hat{L} = (\sin\beta \cos\alpha, \sin\beta \sin\alpha, \cos\beta) \,,
$$

with the third one fixed by the minimal-rotation condition, which in
TEOBResumS' convention reads $\dot\gamma = \dot\alpha \cos\beta$. Initial
conditions, at $f_{\rm ref}$:

- $\beta = 0$, since $\hat{L}_0 = \hat{z}$;
- $\alpha$ is undefined there, and is given its limit from above (the
  direction in which $\hat{L}$ first moves off the $z$ axis);
- $\gamma$ is the next-to-leading-order estimate of $\alpha$ of
  [arXiv:2111.03675](https://arxiv.org/abs/2111.03675) (eq. A5), as in
  TEOBResumS.

The integration runs from $f_{\rm ref}$ upwards, to $1.1$ times the merger
orbital frequency of TEOBResumS' fits, and downwards, as far as the lowest
frequency the waveform is evaluated at needs: the $(\ell, m)$ multipole
reaches the frequency $f$ when the orbital frequency is $2 \pi f / m$. Below
$f_{\rm ref}$, as $\hat{L}$ passes through $\hat{z}$, $\alpha = {\rm
atan2}(L_y, L_x)$ jumps by $\pi$; $\gamma$ is continued with $\gamma + \pi$
there, which keeps the rotation continuous
({func}`~mlgw_bns.precessing_model.backward_gamma`).

The angles are tabulated against the orbital frequency $M\Omega$, which is
what the frequency-domain twist needs. The numpy path integrates in time with
an adaptive DOP853; the JAX one integrates in $\Omega$, with a fixed-step
DOP853 on a grid uniform in $\ln\Omega - \kappa\,\Omega_{\rm lo}/\Omega$
(fine at low frequencies, where $\hat{L}$ winds around $\vec{J}$ many times),
and integrates $\alpha - \gamma$ instead of $\gamma$, which is regular where
$\hat{L} = \hat{z}$.

## The twist in the frequency domain

In the time domain, with TEOBResumS' convention $h_{\ell m} = A_{\ell m}
e^{-i \phi_{\ell m}}$, the rotation is

$$
h^{\rm I}_{\ell m}(t) = e^{-i m \alpha} \sum_{n = -\ell}^{\ell}
d^{\ell}_{m n}(\beta)\, e^{i n \gamma}\, h^{\rm co}_{\ell n}(t) \,,
\qquad
h^{\rm co}_{\ell, -n} = (-1)^\ell \left(h^{\rm co}_{\ell n}\right)^* \,,
$$

with the angles at time $t$ ({func}`~mlgw_bns.twist_waveform.twist_modes`).

`mlgw_bns` works in the frequency domain, with $\tilde{h}(f) = \int h(t)
e^{2 \pi i f t} \mathrm{d}t$, under the stationary-phase approximation: the
co-precessing multipole $n$ at frequency $f$ comes from the time at which
its own frequency is $f$, which is when the orbital frequency is
$M\Omega = 2 \pi f M / n$. So
({func}`~mlgw_bns.precessing_model.twist_modes_frequency_domain`):

- each co-precessing multipole is twisted with the angles read at *its own*
  $M\Omega = 2\pi f M / n$;
- the multipoles with $n > 0$ have their support at $f > 0$ and those with
  $n < 0$ at $f < 0$, so the two halves of the sum are twisted separately,
  into $\tilde{h}^{\rm I}_{\ell m}(f)$ and $\tilde{h}^{\rm I}_{\ell m}(-f)$;
- the surrogate's multipoles are the complex conjugates of the transforms of
  the $h_{\ell m}$ above, so the rotation acting on them is the conjugate
  one: the same formula with $\alpha, \gamma \to -\alpha, -\gamma$.

Outside the range of the integration the angles are held at their end values.
The polarizations are then

$$
\tilde{h}_+ - i \tilde{h}_\times = \sum_{\ell m}
\tilde{h}^{\rm I}_{\ell m}(f)\, {}_{-2}Y_{\ell m}(\iota, \varphi) \,,
\qquad
\tilde{h}_+ + i \tilde{h}_\times = \sum_{\ell m}
\tilde{h}^{\rm I *}_{\ell m}(-f)\, {}_{-2}Y^*_{\ell m}(\iota, \varphi) \,,
$$

both needed, since precession breaks the equatorial symmetry that relates
them for aligned spins
({func}`~mlgw_bns.precessing_model.polarizations_from_inertial_modes`).
Without in-plane spins the whole construction is exactly
{meth}`Model.predict <mlgw_bns.model.Model.predict>`
({func}`~mlgw_bns.precessing_model.check_aligned_spin_limit`).

(precession-reference-phase)=
## Orbital phase at the reference frequency

The aligned-spin model puts the coalescence phase at the merger. Without
precession the orbital phase is a choice of observer: rotating the binary
about $\hat{z}$ is the same as moving the line of sight in azimuth. With
precession it is not, because it sets the angle between the orbital
separation and the in-plane spins. As in LALSuite (`phiRef` at `f_ref`),
NRSur7dq4, SEOBNR and TEOBResumS, it is therefore given at the reference
frequency, where the spins are: `reference_phase`, $\phi_0$.

`mlgw_bns` defines it through the waveform, like NRSur7dq4. The
stationary-phase transform of a frequency-domain multipole's phase
$\Psi_{\ell m}(f)$,

$$
X_{\ell m}(f) = \Psi_{\ell m}(f) - f\, \Psi_{\ell m}'(f) \,,
$$

does not change under a time shift. At $f = m F$, with $F$ the orbital
frequency, it equals the time-domain phase at the stationary time plus
$\pi / 4$:

$$
X_{\ell m}(m F) = m\, \phi_{\rm orb}(F) + \delta_{\ell m} + \frac{\pi}{4} \,,
$$

where $\delta_{\ell m}$ is the multipole's phase at zero orbital phase, taken
at its leading (Newtonian) order: $\delta_{22} = -\pi$, $\delta_{21} = \pi /
2$, $\delta_{33} = -\pi / 2$
({data}`~mlgw_bns.precessing_model.LEADING_ORDER_MODE_PHASES`).

{meth}`PrecessingModel.reference_orbital_phase <mlgw_bns.precessing_model.PrecessingModel.reference_orbital_phase>`
reads $\phi_{\rm orb}$ at $F = f_{\rm ref} / 2$ from the surrogate's own
$(2, 2)$. This gives it modulo $\pi$, and the branch comes from the $(2, 1)$,
or else the $(3, 3)$. Every co-precessing multipole is then rotated by
$e^{i m (\phi_0 - \phi_{\rm orb})}$, so that the orbital phase at $f_{\rm ref}$
is $\phi_0$. If `reference_frequency_hz` is `None`, the aligned-spin
convention is kept: `reference_phase` is the coalescence phase at the merger,
and the spins are given where the PN integration starts.

## Reproducing TEOBResumS

TEOBResumS' precessing waveforms are twisted in the same way, so the two
agree up to the accuracy of the aligned-spin surrogate once the conventions
are matched. For TEOBResumS called in the frequency domain (`domain=1`,
`use_spins=2`):

| | TEOBResumS | `mlgw_bns` |
|---|---|---|
| Fourier transform | $\tilde{h}$ | $\tilde{h}^*$ |
| cross polarization | $h_\times$ | $-h_\times$ |
| azimuth of the line of sight | `coalescence_angle` | `azimuth` $= $ `coalescence_angle` $- \pi / 2$ |
| where the spins are given | $0.95\,$`initial_frequency` | `reference_frequency_hz` |
| orbital phase | zero at the start of the integration | `reference_phase`, from {meth}`~mlgw_bns.precessing_model.PrecessingModel.teob_reference_phase` |
| multipoles | `use_mode_lm`, `use_mode_lm_inertial` | all those of the model, into every inertial one |

The orbital phase needs a conversion. TEOBResumS sets its orbital phase to
zero at the first sample of its integration. Its PN spin dynamics starts there
at $0.95\,$`initial_frequency`, while its EOB dynamics, which the waveform
follows, starts at a slightly different frequency (about $9.506$ Hz for
$9.5$). There, its $(2, 2)$ is not at its leading-order phase but a few
$10^{-3}$ rad off it, from its higher-order phase corrections.
{meth}`~mlgw_bns.precessing_model.PrecessingModel.teob_reference_phase`
accounts for both. It takes the EOB orbital frequency, the orbital phase and
the $(2, 2)$ phase at the first sample, as TEOBResumS returns them with
`arg_out="yes"`, and returns the `reference_phase` at the frequency where the
spins are given.

For TEOBResumS to have every multipole throughout the band, its
`initial_frequency` must be low enough: the $(\ell, m)$ multipole starts at
$(m / 2) \times 0.95\,$`initial_frequency`. With 10 Hz and the $(4, 4)$, the
comparison can start at 20 Hz.

```{literalinclude} ../examples/precessing_vs_teobresums.py
:language: python
```

This prints mismatches of $7.5 \times 10^{-8}$ for $h_+$ and
$7.9 \times 10^{-8}$ for $h_\times$, with the Einstein Telescope noise curve,
optimised over a time shift and a constant phase. Leaving `reference_phase`
at zero instead gives $6 \times 10^{-4}$.

Both optimisations are needed, and neither comes from precession: they are
the same with the in-plane spins set to zero, and with TEOBResumS' own
co-precessing multipoles in place of the surrogate's.

- The time shift is $\sim 70\,\mu$s. TEOBResumS' frequency-domain output puts
  its time origin there, relative to the merger that `mlgw_bns` references its
  waveforms to.
- The constant phase is a few $10^{-3}$ rad. It is the difference between the
  $(2, 2)$ phases at the reference frequency, at the very start of
  TEOBResumS' integration. Optimising only the time shift leaves $10^{-5}$.

(precession-regression)=
# Regressed precession angles and the two-spin nutation

The precessing waveforms of [the twist](precession.md) need the Euler angles of
the orbital plane, which {mod}`mlgw_bns.batched_precession` gets by
integrating the post-Newtonian (PN) spin-precession equations, two legs of
2048 fixed DOP853 steps per binary. That integration is ~95% of the cost of
a JAX precessing waveform. {mod}`mlgw_bns.precession_regression` replaces
most of it by a regressor; the hard part of doing so is the *nutation*, the
beat between the precession of the two spins. This page describes the
representation the regressor uses, why its first version failed near equal
masses, how the literature treats the same problem, and the closed form
for the nutation frequency that it now uses, derived for the precession
equations of {mod}`mlgw_bns.twist_waveform`.

Units: $G = c = M = 1$, with $M = M_A + M_B$ the total mass and
$M_A \geq M_B$; $q = M_A / M_B \geq 1$ (the opposite of the $q \leq 1$ of
most of the precession literature), $\nu = M_A M_B$, $\Omega$ the orbital
frequency and $v = \Omega^{1/3}$. The spins are $\vec{S}_X = \chi_X M_X^2
\hat{s}_X$, and $L$ is the magnitude of the orbital angular momentum.

## The representation

Regressing the Euler angles directly does not work. In the frame of the
reference frequency, where $\hat{L} = \hat{z}$, $\hat{L}$ comes back close to
$\hat{z}$ once per precession cycle and $\alpha$ swings by $\sim\pi$ each
time; in the frame of the total angular momentum $\vec{J}$ the angles are
smooth for one spin, but with two the winding number of $\alpha_J$ jumps
across parameter space, depending on which spin dominates the tilt of
$\hat{L}$ (see [Why the beat](precession-regression-beat)).

So in the frame of $\vec{J}$ at the reference frequency, the in-plane part
$\zeta = L_x + i L_y$ of $\hat{L}$ is written as a sum of two *carriers* with
slowly varying complex *envelopes*,

$$
\zeta(x) = c_0(x) + c_1(x)\, e^{i \Phi_1(x)} + c_2(x)\, e^{i \Phi_2(x)} \,,
\qquad
\Phi_k(x) = \int_{x_{\rm ref}}^{x} \lambda_k \,\mathrm{d}x' \,,
$$

where $x = \ln\Omega - \kappa\,\Omega_{\min}/\Omega$ is the integration
variable of {mod}`mlgw_bns.batched_precession`, and the third angle, through
$G = \alpha_J - \gamma_J$, is written similarly. The carrier rates
$\lambda_k$ are *computed*, not regressed, by
{func}`~mlgw_bns.precession_regression.carrier_rates`, and integrated by a
quadrature. Everything that winds is in the carriers. The regressor (PCA and
kernel ridge on the B-spline coefficients of the envelopes) only sees the
envelopes, which are smooth in $x$ and in the parameters, *provided the
carriers have the right phase*. A carrier phase error $\delta\Phi(x)$ ends
up in the envelopes as a factor $e^{-i\,\delta\Phi(x)}$, which winds if
$\delta\Phi$ grows to radians over the band, and the regression then fails.
Above $x_{\rm switch} = -5.5$ (~165 Hz for a binary neutron star) the
remaining few precession cycles are integrated instead.

(precession-regression-beat)=
## Why the beat

Each spin precesses mostly about $\hat{L}$, and $\hat{L}$ about $\vec{J}$.
With one spin this is a single rotation, *simple precession*
{cite:p}`apostolatosSpininducedOrbitalPrecession1994`. With two, the spins
precess at different rates, and the angle between their projections on the
orbital plane, and with it the magnitude of $\vec{S} = \vec{S}_A + \vec{S}_B$
and the tilt of $\hat{L}$ from $\vec{J}$, oscillates. This is the nutation, and
its frequency is the difference of the carrier rates,
$\lambda_1 - \lambda_2$: the *beat*.

### The linearized carriers

The first version of {func}`~mlgw_bns.precession_regression.carrier_rates`
took the carrier rates from the normal modes of the precession linearized in
the tilt. With $\dot{\vec{S}}_A = (w_A \hat{L} + h \vec{S}_B) \times
\vec{S}_A$ and the same for $B$ (the right-hand side below), the in-plane
components $s_{A, B}$ of the spins in the frame of $\vec{J}$ obey, to first
order in $S / L$, $\dot{s} = i \mathsf{M} s$ with

$$
\mathsf{M} = \begin{pmatrix}
    w_A (\cos\beta_J + S_A^J / L) + h S_B^J & S_A^J (w_A / L - h) \\
    S_B^J (w_B / L - h) & w_B (\cos\beta_J + S_B^J / L) + h S_A^J
\end{pmatrix} ,
$$

$S_X^J$ being the projections of the spins on $\vec{J}$ and $\beta_J$ the tilt of
$\hat{L}$ from it. The carrier rates are its eigenvalues, the beat their
difference

$$
\lambda_1 - \lambda_2 = \sqrt{(\mathsf{M}_{AA} - \mathsf{M}_{BB})^2 +
4 \mathsf{M}_{AB} \mathsf{M}_{BA}} \,.
$$

At leading PN order $w_X \propto (2 + 3 M_Y / (2 M_X))$, so the diagonal
splitting $\mathsf{M}_{AA} - \mathsf{M}_{BB}$ vanishes as $q \to 1$, and the
beat is then set entirely by the off-diagonal spin-spin couplings, which
the linearization handles only to first order: it averages the in-plane
part of $\vec{S}_A \cdot \vec{S}_B$ to zero, and holds the projections of the
spins on $\hat{L}$ at their reference values. Neither is right when the two
spins precess at nearly the same rate, because then their relative
orientation changes slowly and *is* the dynamics.

### What happens at equal masses

At $q = 1$, the 2PN orbit-averaged equations conserve $|\vec{S}|$
{cite:p}`gerosaEqualmassLimitPrecessing2017`: with equal spin-orbit
coefficients, $\dot{\vec{S}}_A + \dot{\vec{S}}_B = w \hat{L} \times \vec{S}$ is
perpendicular to $\vec{S}$. $\vec{S}$ and $\hat{L}$ then precess rigidly about
$\vec{J}$, while the two spins turn about $\vec{S}$ at

$$
\frac{\mathrm{d}\varphi'}{\mathrm{d}t} = -\frac{3 S}{r^3}
\left( 1 - \xi \sqrt{\frac{M}{r}} \right) ,
$$

a spin-spin rate $\propto r^{-3}$, slower than the $r^{-5/2}$ of the precession
about $\vec{J}$. This rate is the beat at $q = 1$, and it is proportional to
$S = |\vec{S}_A + \vec{S}_B|$, a nonlinear function of the in-plane spins
and of their relative azimuth. For $q$ slightly above one the beat goes over
continuously from this to the mass-asymmetry splitting.

### How wrong the linear beat is

`visualization/precession_beat_check.py` integrates binaries with
{func}`~mlgw_bns.batched_precession.integrate_angles`, measures the true beat
from the extrema of $\vec{S}_A \cdot \hat{L}$ (which oscillates at the beat for
every $q$, unlike $|\vec{S}|$), and integrates each model's beat rate between
successive extrema, which should give $\pi$. The largest relative error of
that phase, $|\delta\Phi| / \pi$, over bands of $\Omega$ up to the last one
before the switch to the integration ($M\Omega \simeq 3$–$5 \times
10^{-3}$), and in that last band, is
(`visualization/precession_beat_check.log`; *elliptic* is the closed form
derived below, from the reference values alone, *frozen* the nutation of the
full equations integrated at a fixed $\Omega$ at 6 frequencies, interpolated
between them):

| $q$ | in-plane $\chi$ | linear | elliptic | frozen | last band: linear | elliptic |
|-----|------|--------|----------|--------|--------|--------|
| 1.00 | 0.10 | —     | —     | —     | +168% | 0.0% |
| 1.00 | 0.25 | 34%   | 0.00% | 2.5%  | +34%  | 0.0% |
| 1.00 | 0.40 | 52%   | 0.00% | 1.0%  | +51%  | 0.0% |
| 1.02 | 0.10 | 16%   | 2.0%  | 2.1%  | +13%  | −4.2% |
| 1.02 | 0.25 | 25%   | 1.8%  | 1.9%  | +43%  | +3.3% |
| 1.02 | 0.40 | 26%   | 0.8%  | 1.1%  | +12%  | −0.9% |
| 1.10 | 0.10 | 2.7%  | 0.7%  | 1.1%  | +1.2% | −1.2% |
| 1.10 | 0.25 | 3.5%  | 0.04% | 0.4%  | +6.4% | +1.2% |
| 1.10 | 0.40 | 32%   | 1.7%  | 2.5%  | +45%  | +2.8% |
| 1.25 | 0.10–0.40 | 0.5–2.6% | ≤ 0.13% | ≤ 0.7% | +0.6 to +3.1% | ≤ 0.1% |
| 1.50 | 0.10–0.40 | ≤ 0.9% | ≤ 0.02% | ≤ 0.6% | ≤ 0.9% | ≤ 0.1% |
| 2.00 | 0.10–0.40 | ≤ 0.3% | ≤ 0.01% | ≤ 0.3% | ≤ 0.5% | 0.0% |

At $q = 1$ and in-plane spins of 0.1 the band holds a single half-cycle of
the beat. Near $q = 1.02$ the bands hold one half-cycle each, and the
measurement itself (the extrema of an anharmonic oscillation, $m \sim 0.9$)
scatters at the percent level, the same for the elliptic and the frozen
rates.

This is where the regressor failed: below $q \simeq 1.5$
its waveform mismatches stopped improving with more training data, while
above they kept falling as $N^{-0.6}$.

## How the literature treats it

**Integrating the dynamics.** NRSur7dq2 {cite:p}`blackmanNumericalRelativityWaveform2017`
and NRSur7dq4 {cite:p}`varmaSurrogateModelsPrecessing2019` fit the *time
derivatives* of the coprecessing-frame quaternion, of the orbital phase and of
the spins, and integrate them with a fourth-order Adams–Bashforth scheme from
the reference time: the beat comes out of the integration, as it does in
{mod}`mlgw_bns.batched_precession`. TEOBResumS
{cite:p}`akcayHybridPostNewtonianEffectiveonebody2021`, SEOBNRv5PHM and
IMRPhenomTPHM integrate the PN spin equations; so does
IMRPhenomXPHM-SpinTaylor {cite:p}`colleoniFastFrequencydomainGravitational2024`,
which offers integrated angles in place of the closed-form ones of
IMRPhenomXPHM (and finds them closer to numerical relativity), and on which
PhenomXPNR {cite:p}`hamiltonPhenomXPNRImprovedGravitational2025` builds: there,
the greatest improvement "is seen for systems at close to equal mass where
the two-spin oscillations are strongest".

**Regressing the frame directly.** The neural-network emulator of NRSur7dq4
{cite:p}`purrerFastAccurateDifferentiable2026` predicts the quaternion
time series directly from the parameters, with a 23-million-parameter
network for the quaternion alone. NRSur7dq4 covers the last ~4300 $M$,
about 20 orbits and a few precession cycles, so nothing winds; a binary
neutron star from 20 Hz goes through hundreds of precession cycles, and
this does not carry over.

**Multiple-scale analysis.** The closed-form angles come from the
separation between the precession timescale ($\propto v^{-5}$) and the
radiation-reaction one ($\propto v^{-8}$)
{cite:p}`chatziioannouGravitationalWaveformsPrecessing2013`.
{cite:t}`kesdenEffectivePotentialsMorphological2015` showed that on the
precession timescale, where $L$ is constant, the conserved quantities
$\vec{J}$, the spin magnitudes and the effective spin $\xi$ reduce the
problem to one variable, $S^2$, whose rate squared is a cubic,
$(\mathrm{d}S^2/\mathrm{d}t)^2 = -A^2 S^6 + B S^4 + C S^2 + D$.
{cite:t}`chatziioannouAnalyticGravitationalWaveforms2017` (and
{cite:t}`chatziioannouConstructingGravitationalWaves2017` in full) solved it
with Jacobi elliptic functions, $S^2 = S_+^2 + (S_-^2 - S_+^2)\,
\mathrm{sn}^2(\psi, m)$, added the radiation reaction by promoting the
constants to slowly varying functions, and built the frequency-domain
waveforms from them. These are the angles of IMRPhenomPv3
{cite:p}`khanPhenomenologicalModelGravitationalwave2019` and of the default
IMRPhenomXPHM {cite:p}`prattenComputationallyEfficientModels2021`, which falls
back to simpler (NNLO, single-spin) angles where the MSA fails to initialize.
Their nutation phase is $\psi \propto (M_A - M_B) / v^3$: in the $S^2$
variable the equal-mass limit is singular, because $S$ stops varying.

**Precession averaging.** {cite:t}`gerosaMultitimescaleAnalysisPhase2015`
evolve $J$ on the radiation-reaction timescale with the precession-averaged
$\mathrm{d}J/\mathrm{d}L$, which needs only the turning points of the
nutation; {cite:t}`schnittmanSpinorbitResonanceEvolution2004` found the
resonant families (the spins and $\hat{L}$ coplanar, $\Delta\Phi = 0, \pi$)
at which the nutation vanishes. {cite:t}`gerosaEfficientMultitimescaleDynamics2023`
reparametrize the precession by the *weighted spin difference* $\delta\chi$
introduced by {cite:t}`kleinEFPEEfficientFully2021`, which keeps varying at
$q = 1$: their cubic for $(\mathrm{d}\delta\chi/\mathrm{d}t)^2$ has no hidden
divergences, its spurious third root goes to infinity as $q \to 1$, and the
nutation period

$$
\tau = \frac{4 K(m)}{A \sqrt{\delta\chi_3 - (1 - q)\,\delta\chi_-}}
$$

(their $q \leq 1$) is regular everywhere. That is the formula this module
needs, but derived for the right-hand side it integrates.

## The nutation in closed form

### The precession at fixed orbital frequency

The right-hand side of {func}`~mlgw_bns.twist_waveform._pn_precession_derivatives`
(TEOBResumS' `SPIN_FLX_PN`, N4LO spin-orbit, from
{cite:t}`akcayHybridPostNewtonianEffectiveonebody2021`) is

$$
\dot{\vec{S}}_A = \left(a_A \hat{L} + h \vec{S}_B\right) \times \vec{S}_A \,,
\qquad
\dot{\vec{S}}_B = \left(a_B \hat{L} + h \vec{S}_A\right) \times \vec{S}_B \,,
$$

with $h = v^6 / 2$ and, writing $Y = (\vec{S}_A / q + \vec{S}_B) \cdot
\hat{L}$,

$$
a_A = w_A - 3 h Y \,, \qquad a_B = w_B - 3 h q Y \,,
\qquad
w_X = c^{(5)}_X v^5 + c^{(7)}_X v^7 + c^{(9)}_X v^9
$$

({func}`~mlgw_bns.twist_waveform._spin_orbit_coefficients`). For $\hat{L}$, to
leading order, $L \dot{\hat{L}} = -\dot{\vec{S}}_A - \dot{\vec{S}}_B$; the next
order multiplies each $\dot{\vec{S}}_X$ by $1 + \nu v^2 c_X$, with $c_X = -(3 +
1/M_X)/4$. For equal $c_X$ that is the leading order with $L$ replaced by
$L / (1 + \nu v^2 c)$, and so $L$ is taken to be that, at the mean of the
two $c_X$ (they are equal at $q = 1$, where this matters most). The rest of
$\dot{\hat{L}}$ is quadratic in the spins.

At fixed $\Omega$ (on the precession timescale) these equations conserve
$\vec{J} = L \hat{L} + \vec{S}_A + \vec{S}_B$ and the spin magnitudes. Write
$u_X = \vec{S}_X \cdot \hat{L}$ and $T = \hat{L} \cdot (\vec{S}_A \times
\vec{S}_B)$. Then

$$
\dot{u}_A = \dot{\vec{S}}_A \cdot \hat{L} + \vec{S}_A \cdot \dot{\hat{L}}
= h\, \hat{L} \cdot (\vec{S}_B \times \vec{S}_A)
- \frac{1}{L}\, \vec{S}_A \cdot \dot{\vec{S}}_B
= \left( \frac{a_B}{L} - h \right) T
$$

(only the $a_B \hat{L}$ part of $\dot{\vec{S}}_B$ survives the dot product
with $\vec{S}_A$), and likewise $\dot{u}_B = -(a_A / L - h)\, T$. With

$$
\alpha_X = \frac{w_X}{L} - h \,, \qquad k = \frac{3 h q}{2 L} \,,
$$

these read $\dot{u}_A = (\alpha_B - 2 k Y)\, T$ and $\dot{u}_B = -(\alpha_A -
2 k Y / q)\, T$, and

$$
F = \alpha_A u_A + \alpha_B u_B - k Y^2
$$

is conserved: $\partial F / \partial u_A = \alpha_A - 2 k Y / q$ and
$\partial F / \partial u_B = \alpha_B - 2 k Y$, so $\dot{F} = 0$ identically.
At leading PN order, $w_X / L = (2 + 3 M_Y / (2 M_X))\, v^6$ and $\alpha_A =
\alpha_B / q$, so $F$ is a function of $Y$ alone: $Y$ is then conserved, and
it is the effective spin $\xi$. In general $\delta\alpha = \alpha_A - \alpha_B
/ q$ is of relative order $v^2$, and $\dot{Y} = -\delta\alpha\, T$.

### One degree of freedom

$\vec{J}$, $|\vec{S}_A|$, $|\vec{S}_B|$ and $F$ leave one degree of freedom
(besides a rotation about $\vec{J}$, which does not enter the beat), and
$u_A$ describes it at every $q$. Given $u_A$:

- $Y$ follows from $F$. $F$ is quadratic in $Y$, but $Y$ varies over a
  nutation by $O(\delta\alpha)$ only, so it is linearized about its reference
  value $Y_\star$: $Y = y_0 + y_1 u_A$ with $\tilde\alpha = \alpha_B - 2 k
  Y_\star$,
  $y_0 = Y_\star - (\alpha_B Y_\star - k Y_\star^2 - F) / \tilde\alpha$ and
  $y_1 = -\delta\alpha / \tilde\alpha$;
- $u_B = Y - u_A / q$;
- $\vec{S}_A \cdot \vec{S}_B$ follows from $J^2 = L^2 + S_A^2 + S_B^2 +
  2 L (u_A + u_B) + 2\, \vec{S}_A \cdot \vec{S}_B$.

$T^2$ is the Gram determinant of $\hat{L}, \vec{S}_A, \vec{S}_B$, a function of
their dot products alone,

$$
T^2 = S_A^2 S_B^2 - (\vec{S}_A \cdot \vec{S}_B)^2 - u_A^2 S_B^2
+ 2 u_A u_B\, (\vec{S}_A \cdot \vec{S}_B) - u_B^2 S_A^2 \,,
$$

and with $u_B$ and $\vec{S}_A \cdot \vec{S}_B$ linear in $u_A$ it is a cubic,
$P(u_A)$. So

$$
\dot{u}_A^2 = \tilde\alpha^2\, P(u_A) \,,
$$

the analogue of the cubic of {cite:t}`kesdenEffectivePotentialsMorphological2015`.
Its cubic coefficient is $-2 L\, b (1 + b)$ with $b = y_1 - 1/q$, which
vanishes as $q \to 1$, where $b \to -1$: $P$ becomes a quadratic, and its
third root goes to infinity, as in {cite:t}`gerosaEfficientMultitimescaleDynamics2023`.
$P \geq 0$ only for configurations that exist, and at $u_A = \pm S_A$
($\vec{S}_A$ along $\pm\hat{L}$) $P = -(\vec{S}_A \cdot \vec{S}_B \mp S_A
u_B)^2 \leq 0$.

### The period

$u_A$ oscillates between the two roots $u_- < u_+$ of $P$ around its
maximum. Writing $P = (u - u_-)(u_+ - u)\, g(u)$, with $g$ linear and
positive on $[u_-, u_+]$ (its root is the spurious third one, beyond $u_+$),

$$
u_A(t) = u_- + (u_+ - u_-)\, \mathrm{sn}^2\!\left(\tfrac{1}{2}
|\tilde\alpha| \sqrt{g(u_-)}\, t,\ m\right) ,
\qquad
m = 1 - \frac{g(u_+)}{g(u_-)} \,,
$$

and the nutation period is

$$
\tau = 2 \int_{u_-}^{u_+} \frac{\mathrm{d}u}{|\tilde\alpha| \sqrt{P(u)}}
= \frac{4 K(m)}{|\tilde\alpha| \sqrt{g(u_-)}} \,.
$$

The beat is $2\pi / \tau$, in time; in $x$ it is multiplied by
$(\mathrm{d}\Omega / \mathrm{d}x) / \dot\Omega$, as the carrier rates are.
At $q = 1$, $g$ is constant, $m = 0$ and the oscillation is harmonic. Away
from it, $m$ measures the anharmonicity: up to $\sim 0.95$ at $q = 1.02$ and
$\sim 0.9$ at $q = 1.1$ for in-plane spins of 0.1–0.4, below 0.15 for
$q \geq 1.5$.

### Across the inspiral

At each frequency the formula needs $J^2$, $F$ and $Y_\star$, which
radiation reaction changes. From the reference values only:

- $|\vec{S}_A|$, $|\vec{S}_B|$ are exactly constant.
- $Y$: $\dot{Y} = -\delta\alpha\, T$, and $T$ averages to zero over a
  nutation, so $Y_\star = Y_{\rm ref}$.
- $F$ at each frequency is $\alpha_A u_{A, {\rm ref}} + \alpha_B u_{B,
  {\rm ref}} - k Y_{\rm ref}^2$, with that frequency's coefficients: exact
  where $\delta\alpha = 0$, off by $O(\delta\alpha)$ times the nutation of
  $u_A$ otherwise.
- $J$: radiation reaction changes $L$ along $\hat{L}$, so $\mathrm{d}(J^2) /
  \mathrm{d}L = 2\, \vec{J} \cdot \hat{L} = 2 L + 2 (u_A + u_B)$, and over the
  nutation

  $$
  \frac{\mathrm{d} J^2}{\mathrm{d} L} = 2 L + 2 \langle u_A + u_B \rangle ,
  \qquad
  \langle u_A + u_B \rangle = \langle Y \rangle
  + \left(1 - \frac{1}{q}\right) \langle u_A \rangle ,
  $$

  the precession-averaged evolution of
  {cite:t}`gerosaMultitimescaleAnalysisPhase2015`. The average of
  $\mathrm{sn}^2$ over its period is $(K - E) / (m K)$, so $\langle u_A \rangle =
  u_- + (u_+ - u_-)(K(m) - E(m)) / (m K(m))$. At $q = 1$ the integral is
  explicit, $J^2 = L^2 + 2 Y L + \text{const}$ (the $J(L)$ of
  {cite:t}`gerosaEqualmassLimitPrecessing2017`). In general it is
  integrated in $L$ from $J_{\rm ref}$ over the carrier quadrature nodes,
  by a few Picard iterations starting from $\langle u_A + u_B \rangle =
  u_{A, {\rm ref}} + u_{B, {\rm ref}}$.

Holding $u_A + u_B$ at its reference value instead, as the linear carriers
did, gets $J^2$ wrong by $2 (1 - 1/q)(u_{A, {\rm ref}} - \langle u_A
\rangle) (L - L_{\rm ref})$. By a rough estimate (in-plane spins of 0.4,
$u_A$ nutating by $\sim S_A$, $L - L_{\rm ref} \sim 1$) this can reach a
sizeable fraction of $S^2$ at $q \sim 1.1$, and $S^2$ is what sets the beat
there.

(precession-regression-carriers)=
### Into the carriers

Why the nutation frequency *is* the beat of the carriers: on the precession
timescale every quantity in the frame that turns with $\hat{L}$ about
$\vec{J}$ (the tilt $\beta_J$, the spins' projections) is a function of the one
degree of freedom $u_A$, and so periodic with period $\tau$; only the
azimuth $\phi_z$ of $\hat{L}$ about $\vec{J}$ also advances secularly, at its
average $\langle\Omega_z\rangle$. Then

$$
\zeta = \sin\beta_J\, e^{i\phi_z}
= e^{i \langle\Omega_z\rangle t} \times (\text{a function of period } \tau)
= \sum_n c_n\, e^{i (\langle\Omega_z\rangle + 2\pi n / \tau) t} ,
$$

so the frequencies in $\zeta$ are $\langle\Omega_z\rangle$ and its sidebands at
multiples of $2\pi/\tau$, exactly. The two carriers are two adjacent ones,
and the other harmonics, small unless $m$ is close to one, are left to the
envelopes (which reproduce the integrated rotation to $\sim 10^{-6}$ rad).
It also says what the *mean* of the carriers should be: one of them should
turn at $\langle\Omega_z\rangle$, the precession-averaged rate of
{cite:t}`chatziioannouAnalyticGravitationalWaveforms2017`.

{func}`~mlgw_bns.precession_regression.carrier_rates` keeps the mean of the
two carrier rates from the linearized modes (with their second-order
correction for the circling of $\hat{L}$), and replaces their difference by
the nutation frequency of
{func}`~mlgw_bns.precession_regression.nutation_frequency`. The mean is not
yet the precession-averaged $\langle\Omega_z\rangle$: if the regression near
$q = 1$ is still limited, that is the next thing to replace. Which of the two
is used is the `elliptic_beat` field of
{class}`~mlgw_bns.precession_regression.AngleGrid` (true by default; grids
saved before it was added load with the linear beat they were trained with).

### Numerics

Everything is vectorized and written for both numpy and JAX (no
data-dependent control flow). The nutation frequency is smooth in
$\Omega$, so it is computed at
{data}`~mlgw_bns.precession_regression.N_NUTATION_NODES` = 64 of the 1024
quadrature nodes and interpolated linearly in $\ln$ of both; this changes
the beat phase accumulated over the band by $\lesssim 3 \times 10^{-3}$ rad
(median $5 \times 10^{-4}$) for $q \leq 1.5$. The evolution of $J$ takes
{data}`~mlgw_bns.precession_regression.N_PICARD` = 6 Picard iterations:
each reduces the error by a factor of 5–10, and 6 leave
$\lesssim 2.4 \times 10^{-4}$ rad of beat phase (3 leave up to 0.06 rad).
At each node:

- the coefficients of $P$ come from its values at $u_A / S_A = -1, -1/3,
  1/3, 1$ (exact, since with $Y$ linearized $P$ is exactly cubic);
- its maximum is the stationary point with negative curvature,
  $z_{\max} = c_1 / (\sqrt{c_2^2 - 3 c_3 c_1} - c_2)$ for $P = \sum c_i z^i$,
  a form without cancellation as $c_3 \to 0$;
- $u_\pm$ are bisected (20 steps, then 3 Newton steps kept in the final
  bracket) in $[-S_A, u_{\max}]$ and $[u_{\max}, S_A]$, where $P$ changes
  sign; where $P(u_{\max}) \leq 0$ (a resonance) there is no nutation and
  $u_\pm = u_{\max}$;
- $g(u_\pm)$ follows from the coefficients and the two roots,
  $g(u_-) = -c_3 (2 z_- + z_+) - c_2$, without the third root;
- $K(m)$ and $(K - E)/(mK)$ come from 8 steps of the arithmetic–geometric
  mean, with $c_{n+1} = c_n^2 / (4 a_{n+1})$ so that nothing cancels as $m
  \to 0$;
- without in-plane spins (to $10^{-6}$ of $J$) there is no precession, the
  roots coincide on the boundary $u_A = S_A$ and the root finding is
  ill-conditioned (numpy and XLA differ at $10^{-6}$); there, and wherever the
  result is not a positive finite number (a spin of zero), the linear
  splitting is used instead. The carriers multiply envelopes of zero there,
  so the switch is harmless; for small but nonzero tilts the closed form is
  continuous down to zero tilt, and right (it matches the fixed-$\Omega$
  integration to $2 \times 10^{-4}$ with one spin aligned and the other tilted
  by $10^{-3}$, where at $q = 1.02$ the linear splitting is off by a factor of
  ~100).

The cost: with JAX on four CPU cores, in a batch of 128, the carrier table
goes from ~0.15 to ~1 ms a binary (a first version, at all 1024 nodes with 55
bisections, took 6 ms), against ~7.5 ms for a whole precessing waveform
with the regressed angles. Compilation goes from ~3 to ~15 s, the loops being
unrolled; `jax.lax.fori_loop` would bring that down if it matters.

## Validation

**At fixed $\Omega$**, with the conserved quantities taken from an
integrated state, the closed form agrees with the nutation of the full
right-hand side (integrated numerically with $\dot\Omega = 0$) to
$\lesssim 10^{-4}$ at $M\Omega \leq 10^{-3}$ and $\lesssim 9 \times 10^{-4}$
at $4 \times 10^{-3}$, for $q$ from 1 to 3 and in-plane spins of 0.1 and
0.4. Without the $1 + \nu v^2 c$ correction to $L$ the error is ~10 times
larger, and grows as $v^2$, as expected.

**Along the inspiral**, see the table [above](precession-regression-beat):
from the reference values alone, the closed form is as good as, and mostly
better than, the frozen-$\Omega$ nutation of the full equations sampled at 6
frequencies, and it removes the 10–50% errors of the linear beat near
equal masses.

### Effect on the regressor

Two regressors trained on the same 4096 binaries with $1 \leq q \leq 1.5$
(otherwise the default {class}`~mlgw_bns.precession_regression.TrainingRanges`,
no envelope refinement), one with each beat, compared on 256 others
(`visualization/precession_regression_study.py`, with `--mass-ratio 1 1.5`
and `--linear-beat`; total masses 2.4–3.2, random orientations, ET noise,
20–2048 Hz, nothing optimized):

|  | linear beat | elliptic beat |
|---|---|---|
| $\zeta$ error, $x < -10$: median / 90% | $8.3 \times 10^{-3}$ / $4.2 \times 10^{-2}$ | $4.4 \times 10^{-3}$ / $1.9 \times 10^{-2}$ |
| $\zeta$ error, $x > -10$: median / 90% | $1.2 \times 10^{-2}$ / $6.3 \times 10^{-2}$ | $7.4 \times 10^{-3}$ / $4.1 \times 10^{-2}$ |
| $G$ error, $x < -10$: median / 90% | $1.5 \times 10^{-2}$ / $7.4 \times 10^{-2}$ | $1.6 \times 10^{-2}$ / $7.0 \times 10^{-2}$ |
| $G$ error, $x > -10$: median / 90% | $1.7 \times 10^{-2}$ / $6.6 \times 10^{-2}$ | $1.8 \times 10^{-2}$ / $6.7 \times 10^{-2}$ |
| waveform mismatch: median / 90% / max | $5.8 \times 10^{-4}$ / $1.6 \times 10^{-2}$ / 0.38 | $4.9 \times 10^{-4}$ / $7.9 \times 10^{-3}$ / 0.31 |
| binaries with mismatch $> 0.1$ | 4 | 1 |

The errors of $\zeta$, the tilt of $\hat{L}$, which the carriers describe,
halve, and so does the upper tail of the mismatches. Those of $G = \alpha_J
- \gamma_J$ do not move: its envelopes, and its secular baseline
({func}`~mlgw_bns.precession_regression.g_baseline`, also built on the linear
modes), are now the limit near equal masses, together with the mean of the
carriers (see [Into the carriers](precession-regression-carriers)). The
improvement is real but not yet the order of magnitude that the beat alone
would suggest.

(precession-regression-error)=
## Where the error is, and what reduces it

Replacing, one at a time, each stage of the pipeline by the exact quantity
(`visualization/precession_regression_study.py validate --oracle`) shows that
the error is all in the regression: with the fitted envelopes themselves the
median mismatch is $5 \times 10^{-11}$, with their principal components (64
of each) $1.2 \times 10^{-7}$, against $4.5 \times 10^{-4}$ for kernel ridge
on 4096 binaries; 128 components change nothing. Kernel ridge's own
leave-one-out error says the same: at 4096 binaries it explains only about
65% of the variance of the $G$ envelopes.

The targets, not the regressor's capacity, are what is hard. With the 128
cells of the default {class}`~mlgw_bns.precession_regression.AngleGrid` the
envelopes resolve the slow beats of $q \simeq 1$ binaries, so that how the
nutation is split between $c_1$ and $c_2$ is set by a tiny smoothing penalty
and jumps across parameter space. Pinning it --- a stronger smoothing
(`AngleGrid.smoothing = 1e-2`), fewer cells, or the refinement of
{func}`~mlgw_bns.precession_regression.refine_envelopes`, which refits each
binary's envelopes with the prediction of a regressor trained on the others
as a prior --- is worth about four times the data. A perceptron
({class}`~mlgw_bns.jax_mlp.JaxMLP`, 4 hidden layers of 256, on the same
principal components) does no better than kernel ridge on the raw targets,
but on smoothed and refined ones it is the best regressor by far, and
improves with the data about as $N^{-0.9}$. Median, 90th percentile and
fraction above $10^{-2}$ of the waveform mismatch, $1 \leq q \leq 1.5$, 1024
held-out binaries (`visualization/precession_learning_curve.py`):

| targets | kernel ridge, 4096 | kernel ridge, 16384 | perceptron, 4096 | perceptron, 16384 |
|---|---|---|---|---|
| as fitted | $4.5 \times 10^{-4}$, $9.2 \times 10^{-3}$, 9.8% | $3.1 \times 10^{-4}$, $7.6 \times 10^{-3}$, 8.6% | $3.8 \times 10^{-4}$, $9.3 \times 10^{-3}$, 9.5% | $3.0 \times 10^{-4}$, $8.9 \times 10^{-3}$, 9.2% |
| smoothed | $3.0 \times 10^{-4}$, $7.3 \times 10^{-3}$, 8.0% | $1.8 \times 10^{-4}$, $4.8 \times 10^{-3}$, 6.9% | | |
| smoothed, refined twice | $2.4 \times 10^{-4}$, $7.1 \times 10^{-3}$, 7.8% | $1.5 \times 10^{-4}$, $4.5 \times 10^{-3}$, 7.0% | $1.5 \times 10^{-4}$, $5.7 \times 10^{-3}$, 7.0% | $5.7 \times 10^{-5}$, $2.9 \times 10^{-3}$, 4.9% |

(One seed each; the perceptron at 4096 in single precision, the others in
double, which made no difference beyond that seed's scatter.) The largest
mismatches, up to ~0.7, are of binaries with $q$ within a few thousandths of
one, where the two carriers straddle the true mean precession rate; neither
more data nor any of the following has moved them.

Three ideas from the literature did not help as they stand: more
*sidebands* (carriers at $\Phi_1 + k(\Phi_1 - \Phi_2)$, worse with each one
added), training sets *oversampled* at large in-plane spins (the errors there
are not for lack of data), and carriers at the *precession-averaged* rate
$\langle \Omega_z \rangle$ (in closed form,
{func}`~mlgw_bns.precession_regression.mean_precession_rate`), which fix
the windings of $q < 1.05$ binaries but are slightly off where the
linearized carriers were exact, and are worse overall.

So the way forward is the perceptron on well-posed targets, trained on as
many binaries as can be made: {mod}`mlgw_bns.precession_dataset` keeps
training sets of millions of binaries on disk, made, refined and trained on
in parallel batch jobs, about 25 ms of a core a binary; see
[](cluster-training).

## References

The papers this page uses, in the order of the argument:

- {cite:t}`apostolatosSpininducedOrbitalPrecession1994`: simple precession,
  the single-spin limit.
- {cite:t}`schnittmanSpinorbitResonanceEvolution2004`: spin-orbit resonances,
  where the nutation vanishes.
- {cite:t}`chatziioannouGravitationalWaveformsPrecessing2013`: multiple-scale
  analysis of the precession equations, small-spin expansion.
- {cite:t}`kesdenEffectivePotentialsMorphological2015`: the reduction of
  the conservative precession to one variable and a cubic; morphologies.
- {cite:t}`gerosaMultitimescaleAnalysisPhase2015`: precession-averaged
  evolution of $J$ on the radiation-reaction timescale.
- {cite:t}`chatziioannouAnalyticGravitationalWaveforms2017` and
  {cite:t}`chatziioannouConstructingGravitationalWaves2017`: the
  elliptic-function (MSA) solution, with radiation reaction, and the
  frequency-domain waveforms built from it.
- {cite:t}`gerosaEqualmassLimitPrecessing2017`: the equal-mass limit, where
  $S$ is conserved and the spins turn about $\vec{S}$.
- {cite:t}`kleinEFPEEfficientFully2021` and
  {cite:t}`gerosaEfficientMultitimescaleDynamics2023`: the weighted spin
  difference, and the nutation period regular at $q = 1$.
- {cite:t}`khanPhenomenologicalModelGravitationalwave2019` (IMRPhenomPv3) and
  {cite:t}`prattenComputationallyEfficientModels2021` (IMRPhenomXPHM): MSA
  angles in phenomenological models.
- {cite:t}`colleoniFastFrequencydomainGravitational2024` (IMRPhenomXPHM-SpinTaylor)
  and {cite:t}`hamiltonPhenomXPNRImprovedGravitational2025` (PhenomXPNR):
  back to integrating the PN equations, with the largest gains near equal
  masses.
- {cite:t}`blackmanNumericalRelativityWaveform2017` (NRSur7dq2) and
  {cite:t}`varmaSurrogateModelsPrecessing2019` (NRSur7dq4): surrogates that
  fit and integrate the time derivatives of the precession dynamics.
- {cite:t}`purrerFastAccurateDifferentiable2026`: a neural-network emulator
  of NRSur7dq4 that regresses the frame directly.
- {cite:t}`akcayHybridPostNewtonianEffectiveonebody2021`: the PN
  spin-precession equations TEOBResumS integrates, and so this module.

(higher_order_modes)=
# Higher order modes

```{note}
The algebra in these notes has been cross-checked against the primary
references (in particular {cite:t}`ajithDataFormatsNumerical2011` for the
conventions and {cite:t}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020`,
section II A, equations 2.1--2.12, for the frequency-domain
mode-to-polarization map). Conventions
still vary between sources, so double-check signs and complex-conjugation
choices against the reference relevant to your waveform model.
```

This section will give some mathematical notes on how 
a generic spherical harmonic decomposition for a frequency-domain 
GW works.

A time-domain wave $h(t)$ is expressed as {cite:p}`ajithDataFormatsNumerical2011`:

$$ h(t) = h_+(t) - i h_\times (t)
= \frac{M}{d_L} \sum_{\ell \geq 2} \sum_{|m| \leq \ell} H_{\ell m}(t) {}^{(-2)}Y_{\ell m}
$$

Here, we are using $G = c = 1$ units (and thus neglecting a $G/c^2$ factor
multiplying $M/d_L$ to make it dimensionless).

The polarizations $h_+(t)$ and $h_\times(t)$ are both real-valued, 
so knowing $h(t)$ allows us to recover them both.

The reason for this parametrization is that it relates to the 
asymptotic Weyl scalar $\Psi _4$ by 

$$ \Psi _4 = \ddot{h}_+ - i \ddot{h}_ \times \,;
$$

for more details see {cite:t}`ajithDataFormatsNumerical2011`.
The expansion coefficients $H_{\ell m}$ are defined by an integral in the form 

$$ H_{\ell m} = \frac{d_L}{M} \int \mathrm{d}\Omega {}^{(-2)}Y^*_{\ell m} (h_+ - i h_ \times )
$$

One can see from this that separating out the mass dependence in the definition 
of $H_{\ell m}$ is an **arbitrary choice**, we could just as well have defined 
$h \sim d_L^{-1} \sum_{\ell m} \widetilde{H} Y$. 

Another arbitrary choice is setting $h = h_+ - i h_\times$ as opposed to 
$h = h_+ + i h_\times$ --- switching between the two is equivalent to changing
the sign of the phase.

## Spherical harmonics

The spin-weighted spherical harmonics are functions of the orientation 
of the source, parametrized as $\iota$ (inclination angle, between the 
observation direction and the source angular momentum) and $\phi_0$ (initial phase 
of the source's rotation).
The formulas given here are again from 
{cite:t}`ajithDataFormatsNumerical2011` (their equations II.7 and II.8); the
spin-weighted harmonics were originally introduced by
{cite:t}`goldbergSpinsSphericalHarmonics1967`, and
{cite:t}`kidderUsingFullInformation2008` and section 9 of
{cite:t}`blanchetGravitationalRadiationPostNewtonian2014` collect the conventions
and mode symmetries used in the compact-binary literature. If anything does not
make sense check there.

Explicitly, they are given as:

$${}^{(-s)}Y_{\ell m} = (-1)^s \sqrt{\frac{2 \ell + 1}{4 \pi }} d^\ell_{m, s} (\iota ) e^{im \phi_0 }
$$

where $d^\ell_{m, s} (\iota )$ is called a Wigner $d$-function, and is given by

$$ d^\ell_{m, s} (\iota ) = \sum _{k=k_1 }^{k_2 }
\frac{(-1)^k \sqrt{(\ell+m)! (\ell-m)! (\ell+s)! (\ell-s)!}}{(\ell+m-k)! (\ell-s-k)! k! (k+s-m)!} 
\left(\cos (\iota / 2)\right)^{2 \ell + m -s - 2k}
\left(\sin (\iota / 2)\right)^{2 k + s - m}\,,
$$

where $k_1 = \max(0, m-s)$ and $k_2 = \min (\ell + m, \ell - s)$.

Note that in the GW case the label for the harmonics $Y$ is $-2$, but the parameter $s$ in the $d$-function is 
equal to $+2$. 
For reference, we give the two most useful harmonics: 

$$ {}^{(-2)}Y_{2 \pm 2} = \sqrt{ \frac{5}{64 \pi }} (1 \pm \cos \iota )^2 e^{\pm 2i \phi }\,.
$$

### Identities

These harmonics are an orthonormal basis. Note the complex conjugation on one
factor: it is required because the ${}^{(-2)}Y_{\ell m}$ are complex-valued, and
it is what makes the projection integral for $H_{\ell m}$ given above consistent
with the mode sum.

$$ \int \mathrm{d}\Omega \; {}^{(-2)}Y_{\ell m} \, {}^{(-2)}Y^*_{\ell' m'} = \delta _{\ell \ell'} \delta _{m m'}
$$

Also, they satisfy the conjugation identity (the spin weight flips sign on the
right-hand side):

$$ {}^{s}Y_{\ell m} = (-1)^{s+m} \, {}^{-s}Y^*_{\ell, -m}
$$


## The time-domain 22 wave

We derive the time-domain expression for the $\ell=2$, $|m| = 2$ harmonic, 
which is the most commonly used approximation for a gravitational wave.

According to the expression we gave earlier, we will have

$$ \begin{align}
h(t) &= h_+(t) - i h_\times (t)
= \frac{M}{d_L} \sum_{\ell = 2} \sum_{|m| = 2} H_{\ell m}(t) {}^{(-2)}Y_{\ell m}  \\
&= \frac{M}{d_L} \left( H_{22}(t) {}^{(-2)}Y_{22} + H_{2-2}(t) {}^{(-2)}Y_{2-2}\right)  \\
&= \frac{M}{d_L} \sqrt{ \frac{5}{64 \pi }} 
\left( H_{22}(t) (1 + \cos \iota )^2 e^{2 i \phi_0 } + H_{2-2}(t) (1 - \cos \iota )^2 e^{-2 i \phi_0 }\right)  
\end{align}
$$

Now, we can make use of the fact that, thanks to symmetry under reflection across the orbital plane, 
we have $H_{\ell m} = (-1)^\ell H_{\ell -m}^*$, which in this case reduces to 
$H_{22} = H_{2-2}^*$ (see section II.D in {cite:p}`ossokineMultipolarEffectiveOneBodyWaveforms2020`,
equation 2.3 in {cite:p}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020`, or
{cite:p}`kidderUsingFullInformation2008`). 
Therefore, if we define $\widetilde{H}_{22} = H_{22} e^{2i \phi_0 }$, we 
will have 

$$ 
\begin{align}
h(t) &= \frac{M}{d_L} \sqrt{ \frac{5}{64 \pi }} 
\left(
    \widetilde{H}_{22} (t) (1 + \cos \iota )^2 +
    \widetilde{H}_{22}^* (t) (1 - \cos \iota )^2
\right)  \\
&= \frac{M}{d_L} \sqrt{ \frac{5}{64 \pi }} 
\left(
    2 \Re \widetilde{H}_{22} (t) (1 + \cos^2 \iota ) +
    4 i \Im \widetilde{H}_{22} (t) \cos \iota 
\right) \\
&= h_+ - i h_\times
\end{align}
$$

At this point, we can identify the real and imaginary components, as well as expressing 
$\widetilde{H}_{22} = H_{22} e^{2 i \phi_0 } = A_{22}(t) e^{i \phi_{22}(t) + 2i \phi_0 }$: 

$$ \begin{align}
h_+ &= \frac{4 M}{d_L} \sqrt{ \frac{5}{64 \pi }} A_{22}(t) \frac{1 + \cos^2 \iota }{2} \cos(\phi_{22}(t) + 2 \phi_0) \\
h_\times &= - \frac{4 M}{d_L} \sqrt{ \frac{5}{64 \pi }} A_{22}(t) \cos \iota \sin(\phi_{22}(t) + 2 \phi_0 )
\end{align}
$$

This is the same expression we get in the quadrupole, Newtonian approximation ---
see, for example, equations 4.3 in {cite:t}`maggioreGravitationalWavesVolume2007`,
as long as we reabsorb the coefficients into the amplitude.

## Frequency-domain waves

If we wish to work in the frequency domain, things get slightly more 
complicated since we cannot assume that $h_+$ and $h_\times$ are real-valued 
anymore --- their Fourier transforms will not be. 

Most discussions about how to go from the frequency-domain modes $H_{\ell m} (f)$
to the polarizations $\widetilde{h}_{+, \times } (f)$ 
({cite:t}`khanIncludingHigherOrder2020`; section II A of
{cite:t}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020`) 
also discuss the issue of performing 
a time-dependent rotation to move from the precessing case to the non-precessing one. 

We shall write the expressions without the rotation matrices, one may refer
to those papers for the general case.

Since $h = h_+ - i h_\times$ with $h_+, h_\times$ real, we have
$h_+ = (h + h^*)/2$ and $h_\times = i (h - h^*)/2$ (equivalently $h_\times = -\Im h$).
Using the linearity of the Fourier transform together with
$\text{FT}[h(t)^*]\,(f) = \text{FT}[h(t)]^* (-f)$, this gives

$$ \widetilde{h}_+ (f) = \frac{1}{2} \left( \widetilde{h}(f) + \widetilde{h}^*(-f)\right)
$$

$$ \widetilde{h}_\times (f) = \frac{i}{2} \left( \widetilde{h}(f) - \widetilde{h}^*(-f)\right)
$$

Fourier transforming the time-domain equatorial-symmetry relation
$H_{\ell m}(t) = (-1)^\ell H^*_{\ell -m}(t)$ gives its frequency-domain form
$\widetilde{H}_{\ell m} (f) = (-1)^\ell \widetilde{H}^*_{\ell -m} (-f)$
(equation 2.5 in {cite:t}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020`),
which lets us simplify the summation we get when substituting $\widetilde{h}(f)$ and
$\widetilde{h}^*(-f)$ with their expansions in terms of the Fourier-domain modes
$\widetilde{H}_{\ell m}$. The one subtlety is that conjugating $\widetilde{h}(-f)$
also conjugates the spherical harmonics, so the second sum carries
${}^{(-2)}Y^*_{\ell m}$:

$$ \begin{align}
\widetilde{h}_+ (f) &= \frac{1}{2} \left( \widetilde{h}(f) + \widetilde{h}^*(-f)\right)  \\
&= \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{|m|\leq \ell} \left(
    \widetilde{H}_{\ell m} (f) \, {}^{(-2)}Y_{\ell m} +
    \widetilde{H}^*_{\ell m} (-f) \, {}^{(-2)}Y^*_{\ell m}
\right)  \\
&\approx \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{0 < m \leq \ell} \left(
    \widetilde{H}_{\ell m} (f) \, {}^{(-2)}Y_{\ell m} +
    \widetilde{H}^*_{\ell m} (-f) \, {}^{(-2)}Y^*_{\ell m} +
    \widetilde{H}_{\ell -m} (f) \, {}^{(-2)}Y_{\ell -m} +
    \widetilde{H}^*_{\ell -m} (-f) \, {}^{(-2)}Y^*_{\ell - m}
\right)  \\
&= \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{0 < m \leq \ell} \left(
    \widetilde{H}_{\ell m} (f) \, {}^{(-2)}Y_{\ell m} +
    (-1)^\ell\widetilde{H}_{\ell -m} (f) \, {}^{(-2)}Y^*_{\ell m} +
    \widetilde{H}_{\ell -m} (f) \, {}^{(-2)}Y_{\ell -m} +
    (-1)^\ell \widetilde{H}_{\ell m} (f) \, {}^{(-2)}Y^*_{\ell - m}
\right)  \\
&= \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{0 < m \leq \ell} \left(
    \widetilde{H}_{\ell m} (f) 
    \left(
        {}^{(-2)}Y_{\ell m} +
        (-1)^\ell {}^{(-2)}Y^*_{\ell - m}
    \right)
    +
    \widetilde{H}_{\ell -m} (f) 
    \left(
        {}^{(-2)}Y_{\ell -m} +
        (-1)^\ell {}^{(-2)}Y^*_{\ell m}
    \right)
\right)  \\
&\approx \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{0 < m \leq \ell}
    \widetilde{H}_{\ell m} (f) 
    \left(
        {}^{(-2)}Y_{\ell m} +
        (-1)^\ell {}^{(-2)}Y^*_{\ell - m}
    \right)
\end{align}
$$

Going to the third line we split the sum over $|m| \leq \ell$ into its $\pm m$
halves (dropping $m = 0$); going to the fourth we used
$\widetilde{H}^*_{\ell m}(-f) = (-1)^\ell \widetilde{H}_{\ell -m}(f)$ and
$\widetilde{H}^*_{\ell -m}(-f) = (-1)^\ell \widetilde{H}_{\ell m}(f)$; the fifth just
regroups by mode. In the last step we approximated the contribution of the
$\widetilde{H}_{\ell -m}$ modes as zero.
This is because we are working with positive frequencies, and in this convention
a mode with negative $m$ evaluated at positive frequency is negligible: the
$\ell m$ and $\ell,-m$ modes are supported on opposite frequency ranges, as
follows from the stationary-phase approximation (see e.g.
{cite:t}`damourFrequencydomainPapproximantFilters2000`, and the discussion around
equations 2.5--2.8 in {cite:t}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020`).

We have also dropped the $m = 0$ terms, which carry the (non-oscillatory)
gravitational-wave memory and are negligible for the quasi-circular inspiral
signal considered here; higher-mode frequency-domain models such as
{cite:t}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020` and
{cite:t}`khanIncludingHigherOrder2020` likewise restrict to $m \neq 0$.

An identical computation for $\widetilde{h}_\times(f) = \frac{i}{2}(\widetilde{h}(f) - \widetilde{h}^*(-f))$
leads to 

$$ \widetilde{h}_\times(f) = \frac{i}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{0<m \leq \ell}
\widetilde{H}_{\ell m} (f) \left(
    {}^{(-2)}Y_{\ell m} -
    (-1)^\ell {}^{(-2)}Y^*_{\ell - m}
\right)
$$

These match equations 2.7 and 2.8 of
{cite:t}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020` (there written for
the retained mode $\widetilde{h}_{\ell -m}$, i.e. with the roles of $m$ and $-m$
swapped relative to the frequency convention used here).

These are the final expressions we need, since they express the frequency-domain
polarizations $h_+(f)$ and $h_\times(f)$ as a function of the frequency-domain modes
$\widetilde{H}_{\ell m} (f)$.
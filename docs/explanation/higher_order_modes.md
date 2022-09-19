(higher_order_modes)=
# Higher order modes

```{danger} 
These notes are unreviewed at the moment --- some wrong stuff might 
have made it in! 
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
{cite:t}`ajithDataFormatsNumerical2011`, if anything does not make sense check there.

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

These harmonics are an orthogonal basis: 

$$ \int \mathrm{d}\Omega {}^{(-2)}Y_{\ell m} {}^{(-2)}Y_{\ell' m'} = \delta _{\ell \ell'} \delta _{m m'}
$$

Also, they satisfy:

$$ {}^sY_{\ell m} = (-1)^{s+m} Y^*_{\ell -m}
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
$H_{22} = H_{2-2}^*$ (see section II.D in {cite:p}`ossokineMultipolarEffectiveOneBodyWaveforms2020`). 
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
({cite:t}`khanIncludingHigherOrder2020`, appendix E in 
{cite:t}`garcia-quirosIMRPhenomXHMMultimodeFrequencydomain2020`) 
also discuss the issue of performing 
a time-dependent rotation to move from the precessing case to the non-precessing one. 

We shall write the expressions without the rotation matrices, one may refer
to those papers for the general case.

The frequency-domain polarizations are the Fourier transforms of the real and imaginary
parts of the waveform $h(t)$: 

$$ \widetilde{h}_+ (f) = \text{FT}[\Re h(t)] = \frac{1}{2} \left( \widetilde{h}(f) + \widetilde{h}^*(-f)\right)
$$

$$ \widetilde{h}_\times (f) = \text{FT}[\Im h(t)] = \frac{i}{2} \left( \widetilde{h}(f) - \widetilde{h}^*(-f)\right)
$$

where we used the facts that $\Re h = (h + h^*) / 2$, $\Im h = (h - h^*) / 2$, 
and $\text{FT}[h(t)^*](f) = \text{FT}[h(t)]^* (-f)$.

The aforementioned relation $\widetilde{h}_{\ell m} =(-1)^\ell \widetilde{h}^* _{\ell -m} (-f)$ 
allows us to simplify the summation we get when substuting $h(f)$ and $h^*(-f)$ with their 
expression in terms of the Fourier transforms of the modes, $\widetilde{H}_{\ell m}$:

$$ \begin{align}
\widetilde{h}_+ (f) &= \frac{1}{2} \left( \widetilde{h}(f) + \widetilde{h}^*(-f)\right)  \\
&= \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{|m|\leq \ell} \left(
    \widetilde{H}_{\ell m} (f) {}^{(-2)}Y_{\ell m} +
    \widetilde{H}^*_{\ell m} (-f) {}^{(-2)}Y_{\ell m}
\right)  \\
&\approx \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{1 < m \leq \ell} \left(
    \widetilde{H}_{\ell m} (f) {}^{(-2)}Y_{\ell m} +
    \widetilde{H}^*_{\ell m} (-f) {}^{(-2)}Y_{\ell m} +
    \widetilde{H}_{\ell -m} (f) {}^{(-2)}Y_{\ell -m} +
    \widetilde{H}^*_{\ell -m} (-f) {}^{(-2)}Y_{\ell - m}
\right)  \\
&= \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{1 < m \leq \ell} \left(
    \widetilde{H}_{\ell m} (f) {}^{(-2)}Y_{\ell m} +
    (-1)^\ell\widetilde{H}_{\ell -m} (f) {}^{(-2)}Y_{\ell m} +
    \widetilde{H}_{\ell -m} (f) {}^{(-2)}Y_{\ell -m} +
    (-1)^\ell \widetilde{H}_{\ell m} (f) {}^{(-2)}Y_{\ell - m}
\right)  \\
&= \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{1 < m \leq \ell} \left(
    \widetilde{H}_{\ell m} (f) 
    \left(
        {}^{(-2)}Y_{\ell m} +
        (-1)^\ell {}^{(-2)}Y_{\ell - m}
    \right)
    +
    \widetilde{H}_{\ell -m} (f) 
    \left(
        (-1)^\ell {}^{(-2)}Y_{\ell m} +
        {}^{(-2)}Y_{\ell -m}
    \right)
\right)  \\
&\approx \frac{1}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{1 < m \leq \ell} \left(
    \widetilde{H}_{\ell m} (f) 
    \left(
        {}^{(-2)}Y_{\ell m} +
        (-1)^\ell {}^{(-2)}Y_{\ell - m}
    \right)
\right) 
\end{align}
$$

In the last step, we have approximated the contribution of the $\widetilde{H}_{\ell -m}$
modes as zero. This is because we are working with positive frequencies, and modes with 
negative (positive) $m$ computed at positive (negative) frequency are negligible.

Also, we have been neglecting the $m = 0$ term, which is also typically small.

```{danger}
Find references for these statements! 
```

A similar computation leads to 

$$ h_\times(f) = - \frac{i}{2} \frac{M}{d_L} \sum _{\ell \geq 2} \sum _{0<m \leq \ell}
\widetilde{H}_{\ell m} (f) \left(
    {}^{(-2)}Y_{\ell m} -
    (-1)^\ell {}^{(-2)}Y_{\ell - m}
\right)
$$

These are the final expressions we need, since they express the frequency-domain
polarizations $h_+(f)$ and $h_\times(f)$ as a function of the frequency-domain modes
$\widetilde{H}_{\ell m} (f)$.
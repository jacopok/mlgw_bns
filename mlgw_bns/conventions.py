r"""The LALSuite and TEOBResumS conventions, and the switch between them.

The predictors of :class:`~mlgw_bns.model.Model` and
:class:`~mlgw_bns.precessing_model.PrecessingModel` take a ``convention``
argument, ``"lal"`` (the default) or ``"teob"``. It sets:

* the Fourier transform of the returned polarizations:
  :math:`\tilde{h}(f) = \int h(t) e^{-2\pi i f t} \mathrm{d}t` for
  ``"lal"``, :math:`e^{+2\pi i f t}` (TEOBResumS' frequency-domain output)
  for ``"teob"``, i.e. the complex conjugate;
* the meaning of the azimuthal angle of the line of sight: LALSuite's
  :math:`\Phi` in :math:`h_+ - i h_\times = \sum h_{\ell m}\,
  {}_{-2}Y_{\ell m}(\iota, \Phi - \phi_{\rm ref})` (:math:`\Phi = \pi/2`
  in ``SimInspiralChooseFDWaveform``), or TEOBResumS' ``coalescence_angle``
  :math:`c`; the same binary has :math:`c = 3\pi/2 - \Phi`;
* the sign of the third Euler angle, through
  :meth:`EulerAngles.in_convention
  <mlgw_bns.precessing_model.EulerAngles.in_convention>`: LALSuite's
  :math:`\gamma`, of the usual :math:`D^\ell_{m m'}(\alpha, \beta, \gamma)
  = e^{-i m \alpha} d^\ell_{m m'}(\beta) e^{-i m' \gamma}` and the
  minimal rotation :math:`\dot\gamma = -\dot\alpha \cos\beta`
  (``XLALSimInspiralPrecessionRotateModes``), is minus TEOBResumS'
  :math:`\gamma`, which is IMRPhenomXPHM's :math:`\epsilon`.

The sign of :math:`h_\times` relative to :math:`h_+` is the same in both. ``docs/explanation/conventions.md``
has the full comparison.

Internally ``mlgw_bns`` works in one fixed convention, that of the
low-level functions (:func:`mlgw_bns.batched.mode_polarizations`,
:func:`mlgw_bns.batched_precession.precessing_mode_components` and their
``azimuth``): the co-precessing multipole :math:`m` is rotated by
:math:`e^{i m (a + \phi)}`, :math:`a` the internal azimuth and :math:`\phi`
the orbital phase, and the polarizations are in LALSuite's Fourier
convention. The functions here convert to and from it.
"""

from __future__ import annotations

import math

import numpy as np

#: The conventions a predictor accepts.
CONVENTIONS = ("lal", "teob")

#: The default convention.
DEFAULT_CONVENTION = "lal"

#: LALSuite's azimuth of the line of sight in the frame of the spins at the
#: reference frequency, with the orbital phase given separately
#: (``SimInspiralChooseFDWaveform``'s :math:`\pi / 2 - \phi_{\rm ref}`).
LAL_AZIMUTH = math.pi / 2


def check_convention(convention: str) -> str:
    """``convention`` if it is one of :data:`CONVENTIONS`, else a ``ValueError``."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, not {convention!r}")
    return convention


def internal_azimuth(azimuth, convention: str = DEFAULT_CONVENTION):
    r"""The internal azimuth :math:`a` of the line of sight given in ``convention``.

    ``"lal"``: :math:`a = \pi - \Phi`, with :math:`\Phi` LALSuite's
    (:data:`LAL_AZIMUTH` for its waveforms); ``"teob"``: :math:`a =
    c - \pi / 2`, with :math:`c` TEOBResumS' ``coalescence_angle``. Works on
    arrays (numpy or JAX).
    """
    if check_convention(convention) == "lal":
        return math.pi - azimuth
    return azimuth - math.pi / 2


def external_azimuth(azimuth, convention: str = DEFAULT_CONVENTION):
    """The inverse of :func:`internal_azimuth`."""
    if check_convention(convention) == "lal":
        return math.pi - azimuth
    return azimuth + math.pi / 2


def aligned_internal_phase(phase, convention: str = DEFAULT_CONVENTION):
    r"""The internal rotation :math:`a + \phi` of an aligned-spin waveform.

    ``"lal"``: ``phase`` is LALSuite's :math:`\phi_{\rm ref}`, with the line
    of sight at :data:`LAL_AZIMUTH`; ``"teob"``: TEOBResumS'
    ``coalescence_angle``. Where :math:`\phi_{\rm ref}` is held (the merger,
    or ``time_reference_frequency_hz``) is set separately.
    """
    if check_convention(convention) == "lal":
        return internal_azimuth(LAL_AZIMUTH, convention) + phase
    return internal_azimuth(phase, convention)


def fourier(h, convention: str = DEFAULT_CONVENTION, xp=np):
    """A frequency-domain array of the internal (LALSuite's) Fourier
    convention, in ``convention``."""
    if check_convention(convention) == "lal":
        return h
    return xp.conj(h)


def third_euler_angle(gamma, convention: str = DEFAULT_CONVENTION):
    r"""The internal (TEOBResumS') :math:`\gamma`, in ``convention``:
    LALSuite's :math:`-\gamma` for ``"lal"`` (see the module docstring;
    IMRPhenomXPHM's :math:`\epsilon` is TEOBResumS' :math:`\gamma`)."""
    if check_convention(convention) == "lal":
        return -gamma
    return gamma

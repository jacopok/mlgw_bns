r"""Physical constants.

The solar mass enters the waveforms only as :math:`G M_\odot / c^3`, which
sets the map between physical and geometric (natural) frequencies. A
relative error :math:`\delta` in it is a relative error :math:`\delta` in
every mass; for third-generation detectors this matters at the
:math:`10^{-9}` level (a BNS from a few Hz at SNR ~1000: a mismatch of
:math:`\sim 3\times 10^{-7}` for :math:`\delta = 1.6\times 10^{-8}`,
growing as :math:`\delta^2`).
"""

#: :math:`G M_\odot / c^3` (s): the IAU 2015 nominal solar mass parameter
#: (Resolution B3), :math:`\mathcal{GM}_\odot = 1.3271244 \times 10^{20}`
#: m\ :sup:`3` s\ :sup:`-2`, over :math:`c^3`; the same as LAL's
#: ``MTSUN_SI``, astropy and bilby.
SUN_MASS_SECONDS: float = 4.925490947641267e-6

#: TEOBResumS' ``MSUN_S`` (``TEOBResumS.h``), which it uses to convert a
#: total mass in solar masses to seconds with ``use_geometric_units = "no"``:
#: :math:`1.58 \times 10^{-8}` above :data:`SUN_MASS_SECONDS`.
TEOBRESUMS_SUN_MASS_SECONDS: float = 4.925491025543575903411922162094833998e-6

#: A total mass :math:`M` (solar masses) goes to TEOBResumS in physical units
#: (``use_geometric_units = "no"``) as ``M * TEOBRESUMS_MASS_FACTOR``, so
#: that its ``M * MSUN_S`` is :math:`M` in seconds with the IAU value. In
#: geometric units (as the training data are generated) TEOBResumS' constant
#: never enters: the frequencies are converted with :data:`SUN_MASS_SECONDS`.
TEOBRESUMS_MASS_FACTOR: float = SUN_MASS_SECONDS / TEOBRESUMS_SUN_MASS_SECONDS

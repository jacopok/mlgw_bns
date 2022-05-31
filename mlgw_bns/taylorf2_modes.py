r"""Post-Newtonian modes. 

Appendix E of http://arxiv.org/abs/2001.10914

A good approximation for the phases is (eq. 4.8)
:math:`\phi_{\ell m} (f) \approx \frac{m}{2} \phi_{22} (2f / m)`

The convensions are defined in http://arxiv.org/abs/1601.05588:
we need 

:math:`\delta = \frac{m_1 - m_2}{M} = \frac{q-1}{q+1}`

as well as 

:math:`\chi_a^z = \frac{1}{2} (\vec{chi_1} - \vec{\chi_2}) \cdot \hat{L}_N = \frac{1}{2} (\chi_1 - \chi_2)`

and similarly for the symmetric component :math:`\chi_s^z`, with a plus sign.
"""

import numpy as np
from numba import njit  # type: ignore

def H_22():
    
    eta = 1
    v = 1
    delta = 1
    chi_a_z = 1
    chi_s_z = 1
    chi_a = 1
    chi_s = 1

    
    v2_coefficient = (
        451 * eta / 168 - 323 / 224
    )
    
    v3_coefficient = (
        27 * delta * chi_a_z / 8
        - 11 * eta * chi_s_z / 6
        + 27 * chi_s_z / 8
    )
    
    v4_coefficient = (
        - 49 * delta * chi_a_z * chi_s_z / 16
        + 105271 * eta**2 / 24192
        + 6 * eta * chi_a_z**2 
        + eta * chi_s_z**2 / 8
        - 1975055 * eta / 338688
        - 49 * chi_a_z **2 / 32 
        - 49 * chi_s_z **2 / 32 
        - 27312085 / 8128512
    )
    
    # v6_coefficient = (
    #     107291 * delta * eta * chi_a * chi_s / 2688 
    #     - 875047 * delta * chi_a * chi_s / 32256
    #     + 31 * np.pi * delta * chi_a / 12
    #     + 34473079 * eta**3 / 6386688
    #     + 491 * eta**2 * chi_a**2 / 84
    #     - 51329 * eta**2 * chi_s**2 / 4032
    #     - 3248849057 * eta**2 / 178827264
    #     + 129367 * eta * chi_a**2 / 2304
    #     + 8517 * eta * chi_s**2 / 224
    #     - 7 * np.pi * eta * chi_s / 3
    #     - 205 * np.pi**2 * eta / 48
    #     + 545384828789 * eta / 5007163392
    #     - 875047 * chi_a**2 / 64512
    #     - 875047 * chi_s**2 / 64512 
    #     + 31 * np.pi * chi_s / 12
    #     + 428 * 1j * np.pi / 105 
    #     - 177520268561 / 8583708672
    # )
    
def amp_22_SPA():
    
    return np.pi * np.sqrt(2 * eta / 3) * v**(-7/2) * H_22()
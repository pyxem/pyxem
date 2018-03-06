import scipy.constants as sc
import fpd_data_processing.diffraction_tools as dt


def beta_to_bst(beam_deflection, acceleration_voltage, mrad=True):
    """Calculate Bs * t values from beam deflection (beta).

    Parameters
    ----------
    beam_deflection : NumPy array
        In milli radians, unless mrad is set to False
    acceleration_voltage : float
        In Volts
    mrad : bool
        If data is in milli radians, set this to True. If in radians
        set to False. Default True.

    Returns
    -------
    Bst : NumPy array

    Examples
    --------
    >>> import numpy as np
    >>> import fpd_data_processing.dpc_tools as dpct
    >>> data = np.random.random((100, 100))
    >>> bst = dpct.beta_to_bst(data, 200000)

    """
    wavelength = dt.acceleration_voltage_to_wavelength(acceleration_voltage)
    if mrad:
        beam_deflection /= 1000
    beta = beam_deflection
    e = sc.elementary_charge
    h = sc.Planck
    Bst = beta*h/(wavelength*e)
    return Bst

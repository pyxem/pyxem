import scipy.constants as sc
import fpd_data_processing.diffraction_tools as dt


def bst_to_beta(bst, acceleration_voltage):
    """Calculate beam deflection (beta) values from Bs * t.

    Parameters
    ----------
    bst : NumPy array
        Saturation induction Bs times thickness t of the sample. In Tesla*meter
    acceleration_voltage : float
        In Volts

    Returns
    -------
    Bst : NumPy array
        In radians

    Examples
    --------
    >>> import numpy as np
    >>> import fpd_data_processing.dpc_tools as dpct
    >>> data = np.random.random((100, 100))  # In Tesla*meter
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> beta = dpct.bst_to_beta(data, 200000)

    """
    wavelength = dt.acceleration_voltage_to_wavelength(acceleration_voltage)
    e = sc.elementary_charge
    h = sc.Planck
    beta = e*wavelength*bst/h
    return beta


def beta_to_bst(beam_deflection, acceleration_voltage):
    """Calculate Bs * t values from beam deflection (beta).

    Parameters
    ----------
    beam_deflection : NumPy array
        In radians
    acceleration_voltage : float
        In Volts

    Returns
    -------
    Bst : NumPy array
        In Tesla * meter

    Examples
    --------
    >>> import numpy as np
    >>> import fpd_data_processing.dpc_tools as dpct
    >>> data = np.random.random((100, 100))  # In radians
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> bst = dpct.beta_to_bst(data, 200000)

    """
    wavelength = dt.acceleration_voltage_to_wavelength(acceleration_voltage)
    beta = beam_deflection
    e = sc.elementary_charge
    h = sc.Planck
    Bst = beta*h/(wavelength*e)
    return Bst


def tesla_to_am(data):
    """Convert data from Tesla to A/m

    Parameters
    ----------
    data : NumPy array
        Data in Tesla

    Returns
    -------
    output_data : NumPy array
        In A/m

    >>> import numpy as np
    >>> import fpd_data_processing.dpc_tools as dpct
    >>> data_T = np.random.random((100, 100))  # In tesla
    >>> data_am = dpct.tesla_to_am(data_T)

    """
    return data/sc.mu_0

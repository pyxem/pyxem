import scipy.constants as sc
import pixstem.diffraction_tools as dt


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
    beta : NumPy array
        Beam deflection in radians

    Examples
    --------
    >>> import numpy as np
    >>> import pixstem.dpc_tools as dpct
    >>> data = np.random.random((100, 100))  # In Tesla*meter
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> beta = dpct.bst_to_beta(data, acceleration_voltage)

    """
    av = acceleration_voltage
    wavelength = dt.acceleration_voltage_to_wavelength(av)
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
    >>> import pixstem.dpc_tools as dpct
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

    Examples
    --------
    >>> import numpy as np
    >>> import pixstem.dpc_tools as dpct
    >>> data_T = np.random.random((100, 100))  # In tesla
    >>> data_am = dpct.tesla_to_am(data_T)

    """
    return data/sc.mu_0


def acceleration_voltage_to_velocity(acceleration_voltage):
    """Get relativistic velocity of electron from acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float
        In Volt

    Returns
    -------
    v : float
        In m/s

    Example
    -------
    >>> import pixstem.dpc_tools as dpct
    >>> v = dpct.acceleration_voltage_to_velocity(200000) # 200 kV
    >>> round(v)
    208450035

    """
    c = sc.speed_of_light
    av = acceleration_voltage
    e = sc.elementary_charge
    me = sc.electron_mass
    part1 = (1 + (av * e)/(me * c**2))**2
    v = c * (1 - (1/part1))**0.5
    return v


def acceleration_voltage_to_relativistic_mass(acceleration_voltage):
    """Get relativistic mass of electron as function of acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float
        In Volt

    Returns
    -------
    mr : float
        Relativistic electron mass

    Example
    -------
    >>> import pixstem.dpc_tools as dpct
    >>> mr = dpct.acceleration_voltage_to_relativistic_mass(200000) # 200 kV

    """
    av = acceleration_voltage
    c = sc.speed_of_light
    v = acceleration_voltage_to_velocity(av)
    me = sc.electron_mass
    part1 = (1 - (v**2)/(c**2))
    mr = me / (part1)**0.5
    return mr


def et_to_beta(et, acceleration_voltage):
    """Calculate beam deflection (beta) values from E * t.

    Parameters
    ----------
    et : NumPy array
        Electric field times thickness t of the sample.
    acceleration_voltage : float
        In Volts

    Returns
    -------
    beta: NumPy array
        Beam deflection in radians

    Examples
    --------
    >>> import numpy as np
    >>> import pixstem.dpc_tools as dpct
    >>> data = np.random.random((100, 100))
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> beta = dpct.et_to_beta(data, acceleration_voltage)

    """
    av = acceleration_voltage
    e = sc.elementary_charge
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    m = acceleration_voltage_to_relativistic_mass(av)
    h = sc.Planck

    wavelength2 = wavelength**2
    h2 = h**2

    beta = e*wavelength2*m*et/h2
    return beta

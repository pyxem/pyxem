import hyperspy.api as hs

def _set_gaussian_initial_values(gaussian, gaussian_range):
    """Set initial values for Gaussian used for fitting 
    Laue diffraction ring
    
    Parameters
    ----------
    gaussian : HyperSpy Gaussian parameter
        Gaussian parameter using to fit the Laue Zone
        peak in a radially integrated STEM diffraction image.
    gaussian_range : tuple
        Range for fitting the Gaussian
    """
    gaussian.centre.value = (gaussian_range[0]+gaussian_range[1])*0.5
    gaussian.centre.assign_current_value_to_all()
    gaussian.A.value = 100
    gaussian.A.assign_current_value_to_all()

def model_radial_profile_data_with_gaussian(
        signal_radial, 
        background_range,
        gaussian_range,
        sigma_b=None,
        centre_b=None):
    """Generic function for fitting a powerlaw background and a
    Gaussian to a Laue zone peak in a radially integrated
    pixelated STEM dataset.

    Parameters:
    signal_radial : HyperSpy spectrum
        Radially integrated fpd-dataset.
    background_range : (float, float)
        Which range to fit the powerlaw background, in mrad.
    gaussian_range : (float, float)
        Which range to fit the Gaussian, in mrad.
    sigma_b : (float, float), optional
        Bounds for Gaussian sigma parameter, mrad.
    centre_b : (float, float), optional
        Bounds for Gaussian centre parameter, mrad.
    """

    m = signal_radial.create_model()
    powerlaw = hs.model.components1D.PowerLaw()
    m.append(powerlaw)
    m.set_signal_range(background_range[0], background_range[1])
    m.multifit()
    m.reset_signal_range()
    powerlaw.set_parameters_not_free()

    gaussian = hs.model.components1D.Gaussian()
    m.append(gaussian)
    _set_gaussian_initial_values(gaussian, gaussian_range)

    m.fit_component(
            gaussian, 
            signal_range=(gaussian_range[0], gaussian_range[1]),
            only_current=False)
    gaussian.A.assign_current_value_to_all()
    gaussian.centre.assign_current_value_to_all()
    gaussian.sigma.assign_current_value_to_all()
    
    if not (sigma_b == None):
        gaussian.sigma.bmin = sigma_b[0]
        gaussian.sigma.bmax = sigma_b[1]
    if not (centre_b == None):
        gaussian.centre.bmin = centre_b[0]
        gaussian.centre.bmax = centre_b[1]
    m.set_signal_range(gaussian_range[0], gaussian_range[1])
    m.multifit(fitter='mpfit', bounded=True)
    m.reset_signal_range()
    return(m)

def model_lfo_with_one_gaussian(
        signal_radial,
        background_range=(48.,59.5),
        gaussian_range=(60., 73.5),
        sigma_b=(1.0,3.0),
        centre_b=(60.,75.)):
    """Utility function for fitting a powerlaw background and a
    Gaussian to a LFO Laue zone peak in a radially integrated
    pixelated STEM dataset. Basically calls 
    model_radial_profile_data_with_gaussian with range values
    for the LFO Laue zone peak.

    Parameters:
    signal_radial : HyperSpy spectrum
        Radially integrated fpd-dataset.
    background_range : (float, float), optional
        Which range to fit the powerlaw background, in mrad.
    gaussian_range : (float, float), optional
        Which range to fit the Gaussian, in mrad.
    sigma_b : (float, float), optional
        Bounds for Gaussian sigma parameter, mrad.
    centre_b : (float, float), optional
        Bounds for Gaussian centre parameter, mrad.
    """
    m_lfo = model_radial_profile_data_with_gaussian(
            signal_radial=signal_radial,
            background_range=background_range,
            gaussian_range=gaussian_range,
            sigma_b=sigma_b,
            centre_b=centre_b)
    return(m_lfo)

def model_sto(
        signal_radial,
        background_range=(77.,88.),
        gaussian_range=(91.5,98.),
        sigma_b=(1.0,3.0),
        centre_b=(91.5, 98.)):
    """Utility function for fitting a powerlaw background and a
    Gaussian to a STO Laue zone peak in a radially integrated
    pixelated STEM dataset. Basically calls 
    model_radial_profile_data_with_gaussian with range values
    for the STO Laue zone peak.

    Parameters:
    signal_radial : HyperSpy spectrum
        Radially integrated fpd-dataset.
    background_range : (float, float), optional
        Which range to fit the powerlaw background, in mrad.
    gaussian_range : (float, float), optional
        Which range to fit the Gaussian, in mrad.
    sigma_b : (float, float), optional
        Bounds for Gaussian sigma parameter, mrad.
    centre_b : (float, float), optional
        Bounds for Gaussian centre parameter, mrad.
    """
    m_sto = model_radial_profile_data_with_gaussian(
            signal_radial=signal_radial,
            background_range=background_range,
            gaussian_range=gaussian_range,
            sigma_b=sigma_b,
            centre_b=centre_b)
    return(m_sto)

def model_lfo_with_two_gaussians(
        signal_radial,
        background_range=(48.,59.5),
        gaussian_range0=(61.7, 65.1),
        gaussian_range1=(65.9, 69.8),
        sigma_b1=(1.0,3.0),
        centre_b1=(60.,75.)):
    """Utility function for fitting a powerlaw background and 
    two Gaussians to a LFO Laue zone peak in a radially integrated
    pixelated STEM dataset. Basically calls 
    model_radial_profile_data_with_gaussian with range values
    for the LFO Laue zone peak. 
    
    Currently doesn't work very well: the Gaussians needs some
    more constraints.

    Parameters:
    signal_radial : HyperSpy spectrum
        Radially integrated fpd-dataset.
    background_range : (float, float), optional
        Which range to fit the powerlaw background, in mrad.
    gaussian_range0 : (float, float), optional
        Which range to fit the first Gaussian, in mrad.
    gaussian_range1 : (float, float), optional
        Which range to fit the second Gaussian, in mrad.
    sigma_b1 : (float, float), optional
        Bounds for second Gaussian sigma parameter, mrad.
    centre_b1 : (float, float), optional
        Bounds for second Gaussian centre parameter, mrad.
    """
    m_lfo = model_radial_profile_data_with_gaussian(
            signal_radial=signal_radial,
            background_range=background_range,
            gaussian_range=gaussian_range1,
            sigma_b=sigma_b1,
            centre_b=centre_b1)
    m_lfo.set_parameters_not_free()
    gaussian0 = hs.model.components1D.Gaussian()
    gaussian0.name = "Gaussian0"
    
    m_lfo.append(gaussian0)
    m_lfo.fit_component(
            gaussian0, 
            signal_range=(gaussian_range0[0], gaussian_range0[1]),
            only_current=False)
    gaussian0.A.assign_current_value_to_all()
    gaussian0.centre.assign_current_value_to_all()
    gaussian0.sigma.assign_current_value_to_all()
    m_lfo.set_signal_range(gaussian_range0[0], gaussian_range0[1])
    m_lfo.multifit()
    m_lfo.reset_signal_range()
    return(m_lfo)

# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""Electron diffraction pattern calibration.

"""

class DiffractionGenerator(object):
    """Computes electron diffraction patterns for a crystal structure.

    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`

    3. The intensity of each reflection is then given in the kinematic
       approximation as the modulus square of the structure factor.
       :math:`I_{hkl} = F_{hkl}F_{hkl}^*`

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage of the microscope in kV.
    max_excitation_error : float
        The maximum extent of the relrods in reciprocal angstroms. Typically
        equal to 1/{specimen thickness}.
    debye_waller_factors : dict of str : float
        Maps element names to their temperature-dependent Debye-Waller factors.

    """
    # TODO: Refactor the excitation error to a structure property.

    def __init__(self,
                 accelerating_voltage,
                 max_excitation_error,
                 debye_waller_factors=None,
                 scattering_params='lobato'):
        self.wavelength = get_electron_wavelength(accelerating_voltage)
        self.max_excitation_error = max_excitation_error
        self.debye_waller_factors = debye_waller_factors or {}

        scattering_params_dict = {
            'lobato': 'lobato',
            'xtables': 'xtables'
        }
        if scattering_params in scattering_params_dict:
            self.scattering_params = scattering_params_dict[scattering_params]
        else:
            raise NotImplementedError("The scattering parameters `{}` is not implemented. "
                                      "See documentation for available "
                                      "implementations.".format(scattering_params))

    def fit_ring_pattern(self, mask_radius, scale=100, amplitude=1000, spread=2,
                         direct_beam_amplitude=500, asymmetry=1, rotation=0):
        """Determine diffraction pattern calibration and distortions from by
        fitting a polycrystalline gold diffraction pattern to a set of rings.
        It is suggested that the function generate_ring_pattern is used to
        find initial values (initial guess) for the parameters used in the fit.

        This function is written expecting a single 2D diffraction pattern
        with equal dimensions (e.g. 256x256).

        Parameters
        ----------
        mask_radius : int
            The radius in pixels for a mask over the direct beam disc
            (the direct beam disc within given radius will be excluded
            from the fit)
        scale : float
            An initial guess for the diffraction calibration
            in 1/Angstrom units
        amplitude : float
            An initial guess for the amplitude of the polycrystalline rings
            in arbitrary units
        spread : float
            An initial guess for the spread within each ring (Gaussian width)
        direct_beam_amplitude : float
            An initial guess for the background intensity from the direct
            beam disc in arbitrary units
        asymmetry : float
            An initial guess for any elliptical asymmetry in the
            pattern (for a perfectly circular pattern asymmetry=1)
        rotation : float
            An initial guess for the rotation of the (elliptical) pattern
            in radians.

        Returns
        ----------
        params : np.array()
            Array of fitting parameters.
           [scale, amplitude, spread, direct_beam_amplitude, asymmetry, rotation].

        """
        image_size = self.data.shape[0]
        xi = np.linspace(0, image_size - 1, image_size)
        yi = np.linspace(0, image_size - 1, image_size)
        x, y = np.meshgrid(xi, yi)

        mask = calc_radius_with_distortion(x, y, (image_size - 1) / 2,
                                           (image_size - 1) / 2, 1, 0)
        mask[mask > mask_radius] = 0
        self.data[mask > 0] *= 0

        ref = self.data[self.data > 0]
        ref = ref.ravel()

        pts = np.array([x[self.data > 0].ravel(),
                        y[self.data > 0].ravel()]).ravel()
        xcentre = (image_size - 1) / 2
        ycentre = (image_size - 1) / 2

        x0 = [scale, amplitude, spread, direct_beam_amplitude, asymmetry, rotation]
        xf, cov = curve_fit(call_ring_pattern(xcentre, ycentre), pts, ref, p0=x0)

        return xf

    def generate_ring_pattern(self, mask=False, mask_radius=10, scale=100,
                              amplitude=1000, spread=2, direct_beam_amplitude=500,
                              asymmetry=1, rotation=0):
        """Calculate a set of rings to model a polycrystalline gold diffraction
        pattern for use in fitting for diffraction pattern calibration.
        It is suggested that the function generate_ring_pattern is used to
        find initial values (initial guess) for the parameters used in
        the function fit_ring_pattern.

        This function is written expecting a single 2D diffraction pattern
        with equal dimensions (e.g. 256x256).

        Parameters
        ----------
        mask : bool
            Choice of whether to use mask or not (mask=True will return a
            specified circular mask setting a region around
            the direct beam to zero)
        mask_radius : int
            The radius in pixels for a mask over the direct beam disc
            (the direct beam disc within given radius will be excluded
            from the fit)
        scale : float
            An initial guess for the diffraction calibration
            in 1/Angstrom units
        amplitude : float
            An initial guess for the amplitude of the polycrystalline rings
            in arbitrary units
        spread : float
            An initial guess for the spread within each ring (Gaussian width)
        direct_beam_amplitude : float
            An initial guess for the background intensity from the
            direct beam disc in arbitrary units
        asymmetry : float
            An initial guess for any elliptical asymmetry in the pattern
            (for a perfectly circular pattern asymmetry=1)
        rotation : float
            An initial guess for the rotation of the (elliptical) pattern
            in radians.

        Returns
        -------
        image : np.array()
            Simulated ring pattern with the same dimensions as self.data

        """
        image_size = self.data.shape[0]
        xi = np.linspace(0, image_size - 1, image_size)
        yi = np.linspace(0, image_size - 1, image_size)
        x, y = np.meshgrid(xi, yi)

        pts = np.array([x.ravel(), y.ravel()]).ravel()
        xcentre = (image_size - 1) / 2
        ycentre = (image_size - 1) / 2

        ring_pattern = call_ring_pattern(xcentre, ycentre)
        generated_pattern = ring_pattern(pts, scale, amplitude, spread,
                                         direct_beam_amplitude, asymmetry,
                                         rotation)
        generated_pattern = np.reshape(generated_pattern,
                                       (image_size, image_size))

        if mask == True:
            maskROI = calc_radius_with_distortion(x, y, (image_size - 1) / 2,
                                                  (image_size - 1) / 2, 1, 0)
            maskROI[maskROI > mask_radius] = 0
            generated_pattern[maskROI > 0] *= 0

        return generated_pattern

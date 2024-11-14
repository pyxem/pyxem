# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

import numpy as np
from scipy.optimize import curve_fit

from hyperspy._signals.lazy import LazySignal
import hyperspy.api as hs
from diffsims.utils.sim_utils import get_electron_wavelength
from diffsims.simulations.simulation1d import Simulation1D

from pyxem.signals.diffraction1d import Diffraction1D


class ElectronDiffraction1D(Diffraction1D):
    """Signal class for Electron Diffraction radial profiles.

    Parameters
    ----------
    *args
        See :class:`hyperspy.api.signals.Signal1D`.
    **kwargs
        See :class:`hyperspy.api.signals.Signal1D`
    """

    _signal_type = "electron_diffraction"

    from scipy.optimize import curve_fit

    def model_simulation1d(
        self,
        simulation,
        power_law_background=True,
        starting_scale=None,
        auto_limit_peaks=True,
        center_lim=0.05,
        fit=True,
    ):
        """Creates a model for fitting diffraction peaks for a ring pattern using a series of Gaussians.

        This is primarily useful for calibration but could be potentially used for other things like
        phase segmentation from 1D diffraction patterns.  The lack of a parallel option for model fitting
        in hyperspy limits that usefulness for 4D STEM.  If that is something you are interested in doing,
         raise an issue for more discussion.

        Parameters
        ----------
        simulation : diffsims.simulations.Simulation1D
            The simulation of a 1D "powder" diffraction pattern.
        power_law_background : bool
            Whether to include a power law background in the model.
        starting_scale : float
            The starting scale for the model.  If None, the maximum peak intensity in the simulation is used.
            Note that removing the zero peak is generally a good idea to help with fitting.
        auto_limit_peaks : bool
            Whether to automatically guess some reasonable parameters for the Gaussian peaks and bounds.
        fit : bool
            Whether to fit the model.  If False, the model is returned unfitted.

        Returns
        -------
        model : hyperspy.models.Model1D
            The model for fitting the diffraction peaks to a 1D diffraction pattern.

        Notes
        -----
        This functionality is still under development and may change slightly in the future.

        """
        if starting_scale == None:
            max_peak_inten_sim = simulation.reciprocal_spacing[
                np.argmax(simulation.intensities)
            ]
            max_peak_data = self.axes_manager.signal_axes[0].axis[np.argmax(self.data)]
            starting_scale = max_peak_inten_sim / max_peak_data
        model = self.create_model()

        if power_law_background:
            model.append(hs.model.components1D.PowerLaw())
        for r in simulation.reciprocal_spacing:
            g = hs.model.components1D.Gaussian(centre=r / starting_scale)
            if (
                auto_limit_peaks
            ):  # guessing some reasonable parameters for "typical" diffraction rings
                g.centre.bmin = r / starting_scale - center_lim / starting_scale
                g.centre.bmax = r / starting_scale + center_lim / starting_scale
                g.sigma.bmax = 0.03 / starting_scale
                g.sigma.bmin = 0.01 / starting_scale
            model.append(g)
        if fit:
            model.fit(bounded=True)
        return model

    def model2theta_scale(self, simulation, model, beam_energy):
        """Find the theta calibration scale for a 1d diffraction pattern based
        on a :class:`diffsims.simulations.Simulation1D`, a :class:`hyperspy.models.Model1D`
        and the beam energy.

        Parameters
        ----------
        simulation : diffsims.simulations.Simulation1D
            The simulation of a 1D "powder" diffraction pattern.
        model : hyperspy.models.Model1D
            The model for fitting the diffraction peaks to a 1D diffraction pattern.
        beam_energy : float
            The beam energy in keV.

        Notes
        -----
        This functionality is still under development and may change slightly in the future.
        """

        def f(x, m):
            return x * m

        centers = np.array(
            [
                c.centre.value
                for c in model
                if isinstance(c, hs.model.components1D.Gaussian)
            ]
        )
        wavelength = get_electron_wavelength(beam_energy)
        angles = np.arctan2(np.sort(simulation.reciprocal_spacing), 1 / wavelength)
        m, pcov = curve_fit(f, np.sort(centers), angles)
        scale = m[0]
        return scale

    def model2camera_length(self, simulation, model, beam_energy, physical_pixel_size):
        """Calculate the camera length from a model and simulation.

        This is useful for calibrating the camera length very exactly.

        Parameters
        ----------
        simulation : diffsims.simulations.Simulation1D
            The simulation of a 1D "powder" diffraction pattern.
        model : hyperspy.models.Model1D
            The model for fitting the diffraction peaks to a 1D diffraction pattern.
        beam_energy : float
            The beam energy in keV.
        physical_pixel_size : float
            The physical pixel size in meters. Note the Merlin has a pixel size of 55 um
            or 5.5E-5 m and the Celeritas/ Celeritas XS has a 15 um pixel size or 1.5E-5 m.

        Returns
        -------
        camera_length : float
            The camera length in meters for the experiment.  Note that this is often
            slightly different from the micrscope setting, however this calibrated
            value is more accurate than the calculated value from the micrscope.

        Notes
        -----
        This functionality is still under development and may change slightly in the future.
        """
        theta_scale = self.model2theta_scale(simulation, model, beam_energy)
        camera_length = physical_pixel_size / np.tan(theta_scale)
        return camera_length

    def set_experimental_parameters(
        self,
        accelerating_voltage=None,
        camera_length=None,
        scan_rotation=None,
        convergence_angle=None,
        rocking_angle=None,
        rocking_frequency=None,
        exposure_time=None,
    ):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        accelerating_voltage : float
            Accelerating voltage in kV
        camera_length: float
            Camera length in cm
        scan_rotation : float
            Scan rotation in degrees
        convergence_angle : float
            Convergence angle in mrad
        rocking_angle : float
            Beam rocking angle in mrad
        rocking_frequency : float
            Beam rocking frequency in Hz
        exposure_time : float
            Exposure time in ms.
        """
        md = self.metadata

        if accelerating_voltage is not None:
            md.set_item(
                "Acquisition_instrument.TEM.accelerating_voltage", accelerating_voltage
            )
        if camera_length is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.camera_length",
                camera_length,
            )
        if scan_rotation is not None:
            md.set_item("Acquisition_instrument.TEM.scan_rotation", scan_rotation)
        if convergence_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.convergence_angle", convergence_angle
            )
        if rocking_angle is not None:
            md.set_item("Acquisition_instrument.TEM.rocking_angle", rocking_angle)
        if rocking_frequency is not None:
            md.set_item(
                "Acquisition_instrument.TEM.rocking_frequency", rocking_frequency
            )
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.Diffraction.exposure_time",
                exposure_time,
            )

    def set_diffraction_calibration(self, calibration):
        """Set diffraction profile channel size in reciprocal Angstroms.

        Parameters
        ----------
        calibration : float
            Diffraction profile calibration in reciprocal Angstroms per pixel.
        """
        dx = self.axes_manager.signal_axes[0]

        dx.name = "k"
        dx.scale = calibration
        dx.units = "$A^{-1}$"

    def set_scan_calibration(self, calibration):
        """Set scan pixel size in nanometres.

        Parameters
        ----------
        calibration: float
            Scan calibration in nanometres per pixel.
        """
        x = self.axes_manager.navigation_axes[0]
        y = self.axes_manager.navigation_axes[1]

        x.name = "x"
        x.scale = calibration
        x.units = "nm"

        y.name = "y"
        y.scale = calibration
        y.units = "nm"


class LazyElectronDiffraction1D(LazySignal, ElectronDiffraction1D):
    pass

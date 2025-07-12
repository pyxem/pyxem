# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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

"""Utils for calibrating Diffraction Patterns."""

import numpy as np
import pint

from diffsims.utils.sim_utils import get_electron_wavelength
import hyperspy.api as hs
from hyperspy.axes import UniformDataAxis
from hyperspy.misc.utils import DictionaryTreeBrowser

from pyxem.utils._azimuthal_integrations import _get_control_points, _get_factors
from pyxem.utils._deprecated import deprecated
from typing import Union

Number = Union[int, float]

ureg = pint.UnitRegistry()

unit_equivalents = {
    "nm^-1": "nm$^{-1}$",
    "A^-1": r"$\AA^{-1}$",
    "$A^{-1}$": r"$\AA^{-1}$",
    "k_nm^-1": "nm$^{-1}$",
    "k_A^-1": r"$\AA^{-1}$",
    "nm-1": "nm$^{-1}$",
    "A-1": r"$\AA^{-1}$",
}


def value2standard_units(
    value: Union[str, pint.Quantity, Number],
    standard_unit: Union[str, pint.Unit] = None,
) -> pint.Quantity:
    """
    Convert a string representation of units to a standard unit.

    This takes a string like "40 mm" and will convert it to SI units 4E-3 m.
    If the string is not recognized, it will raise a ValueError.

    Parameters
    ----------
    value: str
        The string representation of the units, e.g. "40 mm", "1 cm", "100 um".
    standard_unit: str,
        The standard unit to convert to. If not provided, it will use the
        unit from the value if it is a pint.Quantity, or raise an error if
        the value is a number.

    Returns
    -------
    pint.Quantity
        The value converted to the standard unit.
    """
    if isinstance(value, pint.Quantity):
        return value.to(standard_unit) if standard_unit else value
    elif isinstance(value, (int, float)):
        if standard_unit is None:
            raise ValueError("If value is a number, standard_unit must be provided.")
        return ureg.Quantity(value, standard_unit)
    elif isinstance(value, str):
        quantity = ureg.Quantity(value)
        if standard_unit is not None:
            return quantity.to(standard_unit)
        return quantity


def value2node(
    value: Union[str, pint.Quantity, Number],
    metadata: DictionaryTreeBrowser,
    metadata_key: str,
    standard_unit: Union[str, pint.Unit] = None,
) -> pint.Quantity:
    """Convert a value to a pint.Quantity and store it in the metadata along with
    a unit.

    Parameters
    ----------
    value:
        The value to convert. This can be a string, pint.Quantity, or a number.
    metadata: hs.metadata.Metadata
        The metadata object to store the value in.
    metadata_key:
        The metadata key to store the value in.
    standard_unit:
        The standard unit to convert to. If not provided, it will use the unit from the value
        if it is a pint.Quantity, or raise an error if the value is a number.
    """

    value = value2standard_units(value, standard_unit)
    metadata.set_item(metadata_key, value.m)
    metadata_key_units = f"{metadata_key}_units"
    metadata.set_item(metadata_key_units, str(value.units))
    return value.m


class Calibration:
    """
    This is a class to hold the calibration parameters for some dataset
    and the methods for calibrating the dataset.

    It is designed to hold information about the affine transformation
    and the mask to apply to the data for something like a beam stop.

    There are 2 ways to set the calibration:
    1. You can set the calibration with a known reciprocal space pixel size.  This will assume
       a flat ewald sphere.

    2. You can set the calibration directly with a known real pixel size, beam energy/wavelength,
       and detector distance. This is the most accurate method but requires detailed calibration
       information that is not trivial to acquire.  In most cases option 1 is sufficient.

    If you set the pixel size with `hyperSpy.axes_manager.signal_axes[0].scale = 0.1`
    it will default to the first method. The underlying data will be stored in the metadata
    of the signal to be used later.
    """

    def __init__(
        self,
        signal,
    ):
        """A class to hold the parameters for an Azimuthal Integrator.

        Parameters
        ----------
        affine: (3x3)
            The affine transformation to apply to the data
        mask:
            A boolean array to be added to the integrator.
        """
        self.signal = signal
        self._mrad_scale = None

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _repr_html_(self):
        """Return an HTML representation of the Calibration object."""

        pixel_size = np.array(getattr(self, "physical_pixel_size", "N/A"))
        if isinstance(pixel_size, np.ndarray):
            pixel_size = f"{pixel_size * 1e6}"

        camera_length = getattr(self, "camera_length", 0)
        if camera_length is None:
            camera_length = 0
        camera_length = f"{camera_length:.2f}" if camera_length > 0 else "N/A"

        html = f"""
            <table style="border-collapse: collapse; width: 50%;">
            <caption style="font-weight: bold; font-size: 1.2em;">Calibration</caption>
            <tr><th style="border: 1px solid #ccc; padding: 4px;">Attribute</th>
                <th style="border: 1px solid #ccc; padding: 4px;">Value</th></tr>
            <tr><td style="border: 1px solid #ccc; padding: 4px;">Scale</td>
                <td style="border: 1px solid #ccc; padding: 4px;">{self.scale}</td></tr>
            <tr><td style="border: 1px solid #ccc; padding: 4px;">Units</td>
                <td style="border: 1px solid #ccc; padding: 4px;">{self.units}</td></tr>
            <tr><td style="border: 1px solid #ccc; padding: 4px;">Center</td>
                <td style="border: 1px solid #ccc; padding: 4px;">{self.center}</td></tr>
            <tr><td style="border: 1px solid #ccc; padding: 4px;">Beam Energy (kV)</td>
                <td style="border: 1px solid #ccc; padding: 4px;">{getattr(self, 'beam_energy', 'N/A')}</td></tr>
            <tr><td style="border: 1px solid #ccc; padding: 4px;">Pixel Size (um)</td>
                <td style="border: 1px solid #ccc; padding: 4px;">{pixel_size}</td></tr>
            <tr><td style="border: 1px solid #ccc; padding: 4px;">Camera Length (mm)</td>
                <td style="border: 1px solid #ccc; padding: 4px;">{camera_length}</td></tr>
        </table>
        """
        return html

    def convert_signal_units(self, new_unit):
        """Change the units of the signal axes.

        Parameters
        ----------
        unit: str
            The new unit to set for the signal axes. Options are:
            - 'px': pixels
            - 'mrad': milliradians
            - 'nm^-1': nanometers^-1
            - 'A^-1': Angstroms^-1
        """
        if new_unit in unit_equivalents.keys():
            new_unit = unit_equivalents[new_unit]

        u = self.units[0]
        if u in unit_equivalents.keys():
            u = unit_equivalents[u]
        if u in ["mrad", "nm$^{-1}$", r"$\AA^{-1}$"]:
            # If the units are already in mrad, nm^-1, or A^-1, we need to set the mrad scale
            self.set_mrad_scale_from_calibration()

        if new_unit == "px":
            # If the new unit is pixels, we need to set the scale to 1
            scale = 1
            offset = np.array(self.center) * -1
        elif new_unit == "mrad":
            scale = np.array(self._mrad_scale)
            offset = np.array(self.center) * -scale
        elif new_unit == "nm$^{-1}$":
            scale = np.tan(self._mrad_scale / 1000) * (1 / self.wavelength) * 10
            offset = np.array(self.center) * -scale
        elif new_unit == r"$\AA^{-1}$":
            scale = np.tan(self._mrad_scale / 1000) * (1 / self.wavelength)
            offset = np.array(self.center) * -scale
        else:
            raise ValueError(
                f"Cannot change units to {new_unit}. "
                "Must be one of 'px', 'mrad', 'nm$^{-1}$', or r'$\\AA^{-1}$'."
            )
        # Set the new units and scale
        self.signal.axes_manager.signal_axes.set(
            units=new_unit, scale=scale, offset=offset
        )

    def set_mrad_scale_from_calibration(self):
        """
        Set the mrad scale from the calibration parameters.
        """
        scales = []
        for u, s in zip(self.units, self.scale):
            if u in unit_equivalents.keys():
                u = unit_equivalents[u]
            if u == "mrad":
                scales.append(s / 1000)  # convert to rad
            elif u == "nm$^{-1}$":
                angle_one_pix = np.arctan2(s / 10, 1 / self.wavelength)
                scales.append(np.tan(angle_one_pix))
            elif u == r"$\AA^{-1}$":
                angle_one_pix = np.arctan2(s, 1 / self.wavelength)
                scales.append(np.tan(angle_one_pix))
            else:
                scales = self._mrad_scale / 1000
                if scales is None:
                    raise ValueError(
                        f"Cannot set mrad scale from calibration unless already set if the units are {u}. "
                        "Must be one of 'mrad', 'nm$^{-1}$', or r'$\\AA^{-1}$'."
                    )
        self._mrad_scale = np.array(scales) * 1000

    def set_camera_length_from_calibration(self):
        """Sets the camera length from the calibration parameters.

        Many times the camera length is not exactly known.  Values from the
        microscope are often not exactly accurate.  This method will set the
        camera length based on the units and the scale currently set for
        some signal. This does require that the detector pixel size and the
        beam energy are set as well.
        """
        if not self.flat_ewald:
            raise ValueError(
                "Cannot set camera length from calibration if the ewald sphere is not flat."
            )
        units = self.units[0]
        if units in unit_equivalents.keys():
            units = unit_equivalents[units]
        if units == "mrad":
            camera_length = self.physical_pixel_size / np.tan(self.scale[0] / 1000)
            self.camera_length = camera_length
        elif units == "nm$^{-1}$":
            angle_one_pix = np.arctan2(self.scale[0] / 10, 1 / self.wavelength)
            camera_length = self.physical_pixel_size / np.tan(angle_one_pix)
            self.camera_length = camera_length
        elif units == r"$\AA^{-1}$":
            angle_one_pix = np.arctan2(self.scale[0], 1 / self.wavelength)
            camera_length = self.physical_pixel_size / np.tan(angle_one_pix)
            self.camera_length = camera_length
        elif units == "px":
            pass
        else:
            raise ValueError(
                f"Cannot set camera length from calibration if the units are {self.units[0]}"
            )

    @property
    def affine(self):
        """Set the affine transformation to apply to the data."""
        return self.signal.metadata.get_item("General.affine_transformation")

    @affine.setter
    def affine(self, affine):
        self.signal.metadata.set_item("General.affine_transformation", affine)

    @property
    def mask(self):
        """Set the mask to apply to the data."""
        return self.signal.metadata.get_item("General.mask")

    @mask.setter
    def mask(self, mask):
        self.signal.metadata.set_item("General.mask", mask)

    @property
    def scale(self):
        """Return the scale in units of the axes."""
        if self.flat_ewald:
            return [s.scale for s in self.signal.axes_manager.signal_axes]
        else:
            return None

    @scale.setter
    def scale(self, scale):
        """Set the scale in units of the axes."""
        if not self.flat_ewald:
            raise ValueError("Scale can only be set if the ewald sphere is flat")
        center = self.center  # save the center
        for ax, c in zip(self.signal.axes_manager.signal_axes, center):
            ax.scale = scale
            ax.offset = -c * scale

    @property
    def units(self):
        return [s.units for s in self.signal.axes_manager.signal_axes]

    @units.setter
    def units(self, units):
        # maybe consider converting the scales as well?
        for ax in self.signal.axes_manager.signal_axes:
            ax.units = units

    @property
    def camera_length(self):
        """
        The camera length in meters.
        This is the "distance" from the sample to the detector.
        """
        return self.signal.metadata.get_item("Acquisition_instrument.TEM.camera_length")

    @camera_length.setter
    def camera_length(self, camera_length):
        """Set the camera length in meters."""
        if camera_length is None:
            self.signal.metadata.remove_item("Acquisition_instrument.TEM.camera_length")
            return
        value2node(
            camera_length,
            self.signal.metadata,
            "Acquisition_instrument.TEM.camera_length",
            standard_unit="m",
        )

    @property
    def physical_pixel_size(self):
        """
        The physical pixel size of the detector in m
        """
        return self.signal.metadata.get_item(
            "Acquisition_instrument.TEM.physical_pixel_size"
        )

    @physical_pixel_size.setter
    def physical_pixel_size(self, pixel_size):
        """Set the physical pixel size of the detector in m."""
        if pixel_size is None:
            self.signal.metadata.remove_item(
                "Acquisition_instrument.TEM.physical_pixel_size"
            )
            return
        value2node(
            pixel_size,
            self.signal.metadata,
            "Acquisition_instrument.TEM.physical_pixel_size",
            standard_unit="m",
        )

    @property
    def convergence_angle(self):
        """
        The convergence angle of the beam in mrad.
        """
        return self.signal.metadata.get_item(
            "Acquisition_instrument.TEM.convergence_angle"
        )

    @convergence_angle.setter
    def convergence_angle(self, convergence_angle):
        """Set the convergence angle of the beam in mrad."""
        if convergence_angle is None:
            self.signal.metadata.remove_item(
                "Acquisition_instrument.TEM.convergence_angle"
            )
            return
        value2node(
            convergence_angle,
            self.signal.metadata,
            "Acquisition_instrument.TEM.convergence_angle",
            standard_unit="mrad",
        )

    @property
    def wavelength(self):
        """The electron wavelength in A^-1."""
        wavelength = self.signal.metadata.get_item(
            "Acquisition_instrument.TEM.wavelength"
        )
        if isinstance(wavelength, pint.Quantity):
            return wavelength.m
        else:
            return wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        """Set the electron wavelength in A^-1."""
        if wavelength is None:
            self.signal.metadata.remove_item("Acquisition_instrument.TEM.wavelength")
            return
        wavelength = value2standard_units(wavelength, standard_unit="angstrom^-1")
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.wavelength", wavelength
        )

    @property
    def beam_energy(self):
        """The beam energy in kV."""
        return self.signal.metadata.get_item("Acquisition_instrument.TEM.beam_energy")

    @beam_energy.setter
    def beam_energy(self, beam_energy):
        """Set the beam energy in kV."""
        energy = value2node(
            beam_energy,
            self.signal.metadata,
            "Acquisition_instrument.TEM.beam_energy",
            standard_unit="kV",
        )

        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.wavelength",
            get_electron_wavelength(energy),
        )

    @property
    def detector_gain(self):
        return self.signal.metadata.get_item("Acquisition_instrument.TEM.detector_gain")

    @detector_gain.setter
    def detector_gain(self, gain):
        """
        Calibrate the signal with a known detector gain. This works by dividing the signal the average
        number of counts per electron.  It also sets the appropriate metadata for setting the color
        bar when plotting the signal.
        Parameters
        ----------
        gain: float
            The gain of the detector
        """
        if gain != 1:  # ignore for counted detectors
            self.signal.data = self.signal.data / gain
        self.signal.metadata.set_item(
            "Signal.quantity", "e$^-$"
        )  # set the quantity to be electrons
        self.signal.metadata.set_item("Acquisition_instrument.TEM.detector_gain", gain)

    def detector(
        self,
        pixel_size,
        detector_distance,
        beam_energy=None,
        wavelength=None,
        center=None,
        units="k_nm^-1",
    ):
        """
        Calibrate the signal with a known pixel size, detector distance, and beam energy/wavelength.


        This sets the signal axes to be a `~hyperspy.axes.DataAxis` instead of a UniformAxis to
        account for the Ewald sphere curvature. This is the most accurate method of calibration
        but requires a known beam energy/wavelength and detector distance or to calibration the
        experimental configuration with a known standard.

        Parameters
        ----------
        pixel_size: float
            The pixel size in meter.
        detector_distance: float
            The detector distance in meter.
        beam_energy: float
            The beam energy in kV.
        wavelength: float
            The beam wavelength in A^-1.
        center: list
            The center of the signal in pixels.
        unit: str
            The unit to calculate the radial extent with.
        """

        if beam_energy is None and wavelength is None:
            wavelength = self.wavelength
            if wavelength is None:
                raise ValueError("Must provide either beam_energy or wavelength")
        elif beam_energy is not None:
            wavelength = get_electron_wavelength(beam_energy)
            self.beam_energy = beam_energy
        self.wavelength = wavelength
        # We need these values to properly set the axes
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.physical_pixel_size", pixel_size
        )
        self.signal.metadata.set_item(
            "Acquisition_instrument.TEM.detector_distance", detector_distance
        )

        unit_factors = {
            "k_nm^-1": 10,
            "k_A^-1": 1,
            "q_nm^-1": 10 / (2 * np.pi),
            "q_A^-1": 1 / (2 * np.pi),
        }
        if units not in unit_factors.keys():
            raise ValueError(
                f"Unit {units} not recognized. Must be one of {unit_factors.keys()}"
            )

        if center is None:
            center = [(ax.size - 1) / 2 for ax in self.axes]

        def translate_pixel_coords(px: np.ndarray) -> np.ndarray:
            coord = pixel_size * px
            angle = np.arctan(coord / detector_distance)
            return np.sin(angle) * (1 / wavelength) * unit_factors[units]

        x_pixels = np.arange(self.shape[0]) - center[0]
        y_pixels = np.arange(self.shape[1]) - center[1]

        x_axes = translate_pixel_coords(x_pixels)
        y_axes = translate_pixel_coords(y_pixels)

        for ax, axis in zip(self.signal.axes_manager.signal_axes, [x_axes, y_axes]):
            if isinstance(ax, UniformDataAxis):
                ax.convert_to_non_uniform_axis()
            ax.axis = axis
        self.units = units

    @property
    def shape(self):
        return self.signal.axes_manager.signal_shape[::-1]

    @property
    def flat_ewald(self):
        """If the ewald sphere is flat return True"""
        return isinstance(self.signal.axes_manager.signal_axes[0], UniformDataAxis)

    def __repr__(self):
        rep_str = f"Calibration for {self.signal}, Ewald sphere: "
        if self.flat_ewald:
            rep_str += "flat"
        else:
            rep_str += f"curved"
        rep_str += f", shape: {self.shape}, affine: {self.affine is not None},"
        rep_str += f" mask: {self.mask is not None}"
        return rep_str

    @property
    def axes(self):
        return [ax.axis for ax in self.signal.axes_manager.signal_axes][::-1]

    @property
    def pixel_extent(self):
        """Return an array with axes [x/y, left/right, pixel_extent], as follows:
        [
            # x axis
            [
                # left
                [boundary 1, boundary 2 ...],
                # right
                [boundary 1, boundary 2 ...],
            ],
            # y axis
            [
                # left
                [boundary 1, boundary 2 ...],
                # right
                [boundary 1, boundary 2 ...],
            ],
        ]
        """
        if self.flat_ewald:
            left_scales = self.scale
            right_scales = self.scale
        else:
            x_scales = self.axes[0][1:] - self.axes[0][:-1]
            x_scales = np.pad(x_scales, 1, mode="edge")
            y_scales = self.axes[1][1:] - self.axes[1][:-1]
            y_scales = np.pad(y_scales, 1, mode="edge")
            left_scales = [
                x_scales[:-1],
                y_scales[:-1],
            ]
            right_scales = [
                x_scales[1:],
                y_scales[1:],
            ]

        extents = []
        for ax, left_scale, right_scale in zip(self.axes, left_scales, right_scales):
            left = ax - left_scale / 2
            right = ax + right_scale / 2
            extent = np.stack((left, right))
            extents.append(extent)
        return extents

    def get_slices2d(self, npt, npt_azim, radial_range=None, azimuthal_range=None):
        """Get the slices and factors for some image that can be used to
        slice the image for 2d integration.

        Parameters
        ----------
        npt: int
            The number of radial points
        npt_azim:
            The number of azimuthal points
        """
        radial_range = self._get_radial_range(radial_range)
        if azimuthal_range is None:
            azimuthal_range = (-np.pi, np.pi)
        # Get the slices and factors for the integration
        slices, factors, factors_slice = self._get_slices_and_factors(
            npt, npt_azim, radial_range, azimuthal_range
        )
        return slices, factors, factors_slice, radial_range

    def _get_radial_range(self, radial_range=None):
        if radial_range is None:
            from itertools import combinations

            edges = np.reshape(
                [
                    [ax_ext[0][0] ** 2, ax_ext[1][-1] ** 2]
                    for ax_ext in self.pixel_extent
                ],
                -1,
            )
            max_range = np.max(
                np.power(np.sum(list(combinations(edges, 2)), axis=1), 0.5)
            )
            radial_range = (0, max_range)
        return radial_range

    def get_slices1d(self, npt, radial_range=None):
        """Get the slices and factors for some image that can be used to
        slice the image for 1d integration.

        Parameters
        ----------
        npt: int
            The number of radial points
        radial_range: tuple
            The range of the radial extent

        Returns
        -------
        indexes: np.ndarray (n, 2)
            The indexes of the pixels to integrate flattened
        factors: np.ndarray (n)
            The factors(representing the pixel fraction) to multiply each pixel value by
        factor_slices: np.ndarray (npt+1)
            The start and end index of the factors for each slice such that for some
            slice i, factors[factor_slices[i]:factor_slices[i+1]] is the factors for that radial slice
        radial_range: tuple
            The range of the radial extent used for the integration

        """
        # reuse the 2d method as it is actually fairly fast
        # and can be much faster if we spend time with numba
        npt_azim = 360  # approximate a circle with a 360-gon... Using a circle is harder/ not much better

        slices, factors, factors_slice, radial_range = self.get_slices2d(
            npt, npt_azim, radial_range
        )

        # convert into 1d slices
        indexes = []
        facts = []
        for i in range(npt):
            test = np.zeros(self.shape)
            for j in range(npt_azim):
                ind = i * npt_azim + j
                sl = slices[ind]
                test[sl[0] : sl[2], sl[1] : sl[3]] += factors[
                    factors_slice[ind][0] : factors_slice[ind][1]
                ].reshape((sl[2] - sl[0], sl[3] - sl[1]))
            inds = np.argwhere(test)
            indexes.append(inds)
            facts.append(test[inds[:, 0], inds[:, 1]])
        factor_slices = np.cumsum(
            [
                0,
            ]
            + [len(f) for f in facts]
        )
        indexes, facts = np.vstack(indexes), np.hstack(facts)
        return indexes, facts, factor_slices, radial_range

    def _get_slices_and_factors(self, npt, npt_azim, radial_range, azimuthal_range):
        # In `_get_control_points`, positive x-direction is downwards, and positive y is to the right.
        # This is 90 degrees off from the pyxem definition:
        # https://pyxem.readthedocs.io/en/stable/tutorials/pyxem-demos/13%20Conventions.html
        # As the azimuthal integration performed with a `Calibration`-object should align with
        # the $X_L$ / $Y_L$-definitions in pyxem, we add pi/2 to the azimuthal range.
        # This aligns the azimuthal angle 0 to $X_L$.
        azimuthal_range = (
            azimuthal_range[0] + np.pi / 2,
            azimuthal_range[1] + np.pi / 2,
        )

        # get the points which bound each azimuthal pixel
        control_points = _get_control_points(
            npt, npt_azim, radial_range, azimuthal_range, self.affine
        )

        # get the min and max indices for each control point using the
        pixel_ext_x, pixel_ext_y = self.pixel_extent
        min_x = (
            np.min(
                np.searchsorted(
                    pixel_ext_x[1, :], control_points[:, :, 0], side="left"
                ),
                axis=1,
            ).astype(int)
            - 1
        )
        min_y = (
            np.min(
                np.searchsorted(
                    pixel_ext_y[1, :], control_points[:, :, 1], side="left"
                ),
                axis=1,
            ).astype(int)
            - 1
        )

        max_x = np.max(
            np.searchsorted(pixel_ext_x[0, :], control_points[:, :, 0], side="right"),
            axis=1,
        ).astype(int)
        max_y = np.max(
            np.searchsorted(pixel_ext_y[0, :], control_points[:, :, 1], side="right"),
            axis=1,
        ).astype(int)
        # Note that if a point is outside the range of the axes it will be set to the
        # maximum value of the axis+1 and 0 if it is below the minimum value of the axis.
        slices = np.array(
            [[mx, my, mxx, myy] for mx, my, mxx, myy in zip(min_x, min_y, max_x, max_y)]
        )
        max_y_ind = len(self.axes[1])
        max_x_ind = len(self.axes[0])

        # set the slices to be within the range of the axes.  If the entire slice is outside
        # the range of the axes then set the slice to be the maximum value of the axis
        slices[slices[:, 0] > max_x_ind, 0] = max_x_ind
        slices[slices[:, 1] > max_y_ind, 1] = max_y_ind
        slices[slices[:, 2] > max_x_ind, 2] = max_x_ind
        slices[slices[:, 3] > max_y_ind, 3] = max_y_ind

        slices[slices[:, 0] < 0, 0] = 0
        slices[slices[:, 1] < 0, 1] = 0
        slices[slices[:, 2] < 0, 2] = 0
        slices[slices[:, 3] < 0, 3] = 0

        factors, factors_slice = _get_factors(control_points, slices, self.pixel_extent)
        return slices, factors, factors_slice

    @property
    def center(self):
        """Return the center in pixels"""
        if self.flat_ewald:
            return [-s.offset / s.scale for s in self.signal.axes_manager.signal_axes]
        else:
            return [
                np.argmin(np.abs(s.axis)) for s in self.signal.axes_manager.signal_axes
            ]

    @center.setter
    def center(self, center=None):
        """Set the center in pixels"""
        if not self.flat_ewald:
            raise ValueError(
                "To set the center of a curved ewald sphere "
                "use the s.calibration.detector method"
            )
        if center is None:
            for ax in self.signal.axes_manager.signal_axes:
                ax.offset = -ax.scale * ((ax.size - 1) / 2)
        else:
            for ax, off in zip(self.signal.axes_manager.signal_axes, center):
                ax.offset = -off * ax.scale


@deprecated(
    since="0.18.0",
    removal="1.0.0",
    alternative="pyxem.signals.Diffraction2D.calibration",
)
def find_diffraction_calibration(
    patterns,
    calibration_guess,
    library_phases,
    lib_gen,
    size,
    max_excitation_error=0.01,
    **kwargs,
):
    """Finds the diffraction calibration for a pattern or set of patterns by maximizing correlation scores.

    Parameters
    ----------
    patterns : hyperspy.signals.Signal2D
        Diffration patterns to be iteratively matched to find maximum correlation scores.
    calibration_guess : float
        Inital value for the diffraction calibration in inverse Angstoms per pixel
    library_phases : diffsims.libraries.StructureLibrary
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    lib_gen : diffsims.generators.DiffractionLibraryGenerator
        Computes a library of electron diffraction patterns for specified atomic
        structures and orientations.  Used to create the DiffractionLibrary.
    size : integer
        How many different steps to test for the first two iterations. These steps have a size of 1% of the calibration guess.
    max_excitation_error : float
        Maximum exacitation error.  Default is 0.01.
    kwargs
        Keyword arguments passed to :meth:`pyxem.utils.indexation_utils.index_dataset_with_template_rotation``.

    Returns
    -------
    mean_cal : float
        Mean of calibrations found for each pattern.
    full_corrlines : numpy.ndarray
        Gives the explicit correlation vs calibration values. Shape:(size*2 + 20, 2 , number of patterns)
    found_cals : numpy.ndarray
        List of optimal calibration values for each pattern. Shape:(number of patterns)
    """

    images = patterns

    num_patterns = images.data.shape[0]
    found_cals = np.full((num_patterns,), calibration_guess)
    full_corrlines = np.zeros((0, 2, num_patterns))

    stepsize = 0.01 * calibration_guess
    # first set of checks
    corrlines = _calibration_iteration(
        images,
        calibration_guess,
        library_phases,
        lib_gen,
        stepsize,
        size,
        num_patterns,
        max_excitation_error,
        **kwargs,
    )
    full_corrlines = np.append(full_corrlines, corrlines, axis=0)

    # refined calibration checks
    calibration_guess = full_corrlines[
        full_corrlines[:, 1, :].argmax(axis=0), 0, 0
    ].mean()
    corrlines = _calibration_iteration(
        images,
        calibration_guess,
        library_phases,
        lib_gen,
        stepsize,
        size,
        num_patterns,
        max_excitation_error,
        **kwargs,
    )
    full_corrlines = np.append(full_corrlines, corrlines, axis=0)

    # more refined calibration checks with smaller step
    stepsize = 0.001 * calibration_guess
    size = 20
    calibration_guess = full_corrlines[
        full_corrlines[:, 1, :].argmax(axis=0), 0, 0
    ].mean()

    corrlines = _calibration_iteration(
        images,
        calibration_guess,
        library_phases,
        lib_gen,
        stepsize,
        size,
        num_patterns,
        max_excitation_error,
        **kwargs,
    )
    full_corrlines = np.append(full_corrlines, corrlines, axis=0)
    found_cals = full_corrlines[full_corrlines[:, 1, :].argmax(axis=0), 0, 0]

    mean_cal = found_cals.mean()
    return mean_cal, full_corrlines, found_cals


def _calibration_iteration(
    images,
    calibration_guess,
    library_phases,
    lib_gen,
    stepsize,
    size,
    num_patterns,
    max_excitation_error,
    **kwargs,
):
    """For use in find_diffraction_calibration.  Controls the iteration of _create_check_diflib over a set of steps.

    Parameters
    ----------
    images : hyperspy.signals.Signal2D
        Diffration patterns to be iteratively matched to find maximum correlation scores.
    calibration_guess : float
        Inital value for the diffraction calibration in inverse Angstoms per pixel
    library_phases : diffsims.libraries.StructureLibrary
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    lib_gen : diffsims.generators.DiffractionLibraryGenerator
        Computes a library of electron diffraction patterns for specified atomic
        structures and orientations.  Used to create the DiffractionLibrary.
    stepsize : float
        Stepsize of iteration.
    size : integer
        How many different steps to test.
    num_patterns : integer
        Number of patterns.
    max_excitation_error : float
        Maximum exacitation error.  Default is 0.01.
    kwargs
        Keyword arguments passed to :meth:`pyxem.utils.indexation_utils.index_dataset_with_template_rotation`.

    Returns
    -------
    corrlines : numpy.ndarray
    """
    corrlines = np.zeros((0, 2, num_patterns))
    temp_line = np.zeros((1, 2, num_patterns))
    cal_guess_greater = calibration_guess
    cal_guess_lower = calibration_guess
    for i in range(size // 2):
        temp_line[0, 0, :] = cal_guess_lower
        temp_line[0, 1, :] = _create_check_diflib(
            images,
            cal_guess_lower,
            library_phases,
            lib_gen,
            max_excitation_error,
            **kwargs,
        )
        corrlines = np.append(corrlines, temp_line, axis=0)

        temp_line[0, 0, :] = cal_guess_greater
        temp_line[0, 1, :] = _create_check_diflib(
            images,
            cal_guess_greater,
            library_phases,
            lib_gen,
            max_excitation_error,
            **kwargs,
        )
        corrlines = np.append(corrlines, temp_line, axis=0)

        cal_guess_lower = cal_guess_lower - stepsize
        cal_guess_greater = cal_guess_greater + stepsize

    return corrlines


def _create_check_diflib(
    images, calibration_guess, library_phases, lib_gen, max_excitation_error, **kwargs
):
    """For use in find_diffraction_calibration via _calibration_iteration.  Creates a new DiffractionLibrary from the inputs and then matches it the images.

    Parameters
    ----------
    images : hyperspy.signals.Signal2D
        Diffration patterns to be iteratively matched to find maximum correlation scores.
    calibration_guess : float
        Inital value for the diffraction calibration in inverse Angstoms per pixel
    library_phases : diffsims.libraries.StructureLibrary
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    lib_gen : diffsims.generators.DiffractionLibraryGenerator
        Computes a library of electron diffraction patterns for specified atomic
        structures and orientations.  Used to create the DiffractionLibrary.
    max_excitation_error : float
        Maximum exacitation error.  Default is 0.01.
    kwargs
        Keyword arguments passed to :meth:`pyxem.utils.indexation_utils.index_dataset_with_template_rotation`.

    Returns
    -------
    correlations : numpy.ndarray
    """
    # lazily import index_dataset_with_template_rotation
    from pyxem.utils.indexation_utils import index_dataset_with_template_rotation

    half_shape = (images.data.shape[-2] // 2, images.data.shape[-1] // 2)
    reciprocal_r = np.sqrt(half_shape[0] ** 2 + half_shape[1] ** 2) * calibration_guess
    diff_lib = lib_gen.get_diffraction_library(
        library_phases,
        calibration=calibration_guess,
        reciprocal_radius=reciprocal_r,
        half_shape=half_shape,
        with_direct_beam=False,
        max_excitation_error=max_excitation_error,
    )

    result, phasedict = index_dataset_with_template_rotation(images, diff_lib, **kwargs)
    correlations = result["correlation"][:, :, 0].flatten()
    return correlations

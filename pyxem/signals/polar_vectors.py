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

from hyperspy._signals.lazy import LazySignal

from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.vectors import (
    to_cart_three_angles,
    polar_to_cartesian,
    get_three_angles,
)


class PolarVectors(DiffractionVectors):
    """
    Class for representing polar vectors in reciprocal space. The first
    two columns are the polar coordinates of the vectors, (r, theta).
    """

    _signal_dimension = 0
    _signal_type = "polar_vectors"

    def get_angles(
        self,
        intensity_index=2,
        intensity_threshold=None,
        accept_threshold=0.05,
        min_k=0.05,
        min_angle=None,
        **kwargs
    ):
        """Calculate the angles between pairs of 3 diffraction vectors.

        Parameters
        ----------
        intensity_index : int, optional
            The column index of the intensity column.  The default is 2.
        intensity_threshold : float, optional
            The minimum intensity for a vector to be considered.  The default is None.
        accept_threshold : float, optional
            The maximum difference between the inscribed angle and the
            reduced angle for the inscribed angle to be accepted.  The default is 0.05.
        min_k : float, optional
            The minimum average delta k between the three vectors
             for a angle to be considered.  The default is 0.05.
        min_angle : float, optional
            The minimum inscribed angle for an angle to be considered.  The default is None.
            Which means that two vectors even with a small angle between them will be
            considered.
        **kwargs:
            Keyword arguments to :meth:`hyperspy.api.signals.BaseSignal.map`.

        Returns
        -------
        angles : BaseSignal
            A signal with navigation dimensions as the original diffraction
            vectors containg an array of inscribed angles at each
            navigation position.

        """
        angles = self.map(
            get_three_angles,
            k_index=0,
            angle_index=1,
            intensity_index=intensity_index,
            intensity_threshold=intensity_threshold,
            accept_threshold=accept_threshold,
            min_k=min_k,
            min_angle=min_angle,
            inplace=False,
            ragged=True,
            **kwargs,
        )
        # set the column names
        col_names = ["k", "delta phi", "min-angle", "intensity", "reduced-angle"]

        # Maybe we should pass any other vector columns through?
        angles.column_names = col_names
        angles.units = ["nm^-1", "rad", "rad", "a.u.", "rad"]
        return angles

    @property
    def has_delta_phi(self):
        """If the polar vector has the delta phi column.

        If True, the vectors are in the form (r, delta phi, min-angle) as a result of the
         ``get_angles method``.  If False, the vectors are in the form (r, theta)."""
        return self.column_names[1] == "delta phi"

    def to_cartesian(self, has_delta_angle=None):
        """Convert the vectors to cartesian coordinates.

        Parameters
        ----------
        has_delta_angle : bool
            If True, the vectors are in the form (r, delta phi, min-angle).  If False,
            the vectors are in the form (r, theta).

        Returns
        -------
        DiffractionVectors
            The vectors in cartesian coordinates.
        """
        if has_delta_angle is None:
            has_delta_angle = self.has_delta_phi
        if has_delta_angle:
            cartesian_vectors = self.map(
                to_cart_three_angles,
                inplace=False,
                ragged=True,
            )
        else:
            cartesian_vectors = self.map(
                polar_to_cartesian,
                inplace=False,
                ragged=True,
            )
        cartesian_vectors.column_names = ["x", "y"]
        cartesian_vectors.units = [self.units[0], self.units[0]]
        cartesian_vectors.set_signal_type("diffraction_vectors")
        return cartesian_vectors

    def to_markers(self, has_delta_angle=None, cartesian=True, **kwargs):
        """Convert the vectors to markers to be plotted on a signal.

        Parameters
        ----------
        has_delta_angle : bool, optional
            If the vectors are polar in the form (r, theta), then this parameter should
            be set to False.  If the vectors are in the form (r, delta phi, min-angle), then
            this parameter should be set to True.  The default is None which will infer
            the format from the column names.
        cartesian : bool, optional
            If True, the vectors will be converted to cartesian coordinates before plotting.
            The default is True.
        **kwargs :
            Keyword arguments to be passed to the :class:`hyperspy.api.plot.markers.Point` class.

        Returns
        -------
        :class:`hyperspy.api.plot.markers.Point`
            A Point object containing the markers.
        """
        if cartesian:
            vectors = self.to_cartesian(has_delta_angle=has_delta_angle)
        else:
            vectors = self
        return vectors.to_markers(**kwargs)


class LazyPolarVectors(LazySignal, PolarVectors):
    pass

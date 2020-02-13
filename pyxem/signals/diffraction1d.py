# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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
"""Signal class for Electron Diffraction radial profiles

"""

import numpy as np

from hyperspy.api import interactive
from hyperspy.signals import Signal1D, BaseSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.roi import SpanROI

from pyxem.signals import push_metadata_through


class Diffraction1D(Signal1D):
    _signal_type = "diffraction1d"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def plot_interactive_virtual_image(self, left, right, **kwargs):
        """Plots an interactive virtual image formed by integrating scatterered
        intensity over a specified range.

        Parameters
        ----------
        left : float
            Lower bound of the data range to be plotted.
        right : float
            Upper bound of the data range to be plotted.
        **kwargs:
            Keyword arguments to be passed to `Diffraction1D.plot`

        Examples
        --------
        .. code-block:: python

            rp.plot_interactive_virtual_image(left=0.5, right=0.7)

        """
        # Define ROI
        roi = SpanROI(left=left, right=right)
        # Plot signal
        self.plot(**kwargs)
        # Add the ROI to the appropriate signal axes.
        roi.add_widget(self, axes=self.axes_manager.signal_axes)
        # Create an output signal for the virtual dark-field calculation.
        dark_field = roi.interactive(self, navigation_signal='same')
        dark_field_placeholder = \
            BaseSignal(np.zeros(self.axes_manager.navigation_shape[::-1]))
        # Create an interactive signal
        dark_field_sum = interactive(
            # Formed from the sum of the pixels in the dark-field signal
            dark_field.sum,
            # That updates whenever the widget is moved
            event=dark_field.axes_manager.events.any_axis_changed,
            axis=dark_field.axes_manager.signal_axes,
            # And outputs into the prepared placeholder.
            out=dark_field_placeholder,
        )
        # Set the parameters
        dark_field_sum.axes_manager.update_axes_attributes_from(
            self.axes_manager.navigation_axes,
            ['scale', 'offset', 'units', 'name'])
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        # Plot the result
        dark_field_sum.plot()

    def get_virtual_image(self, left, right):
        """Obtains a virtual image associated with a specified scattering range.

        Parameters
        ----------
        left : float
            Lower bound of the data range to be plotted.
        right : float
            Upper bound of the data range to be plotted.

        Returns
        -------
        dark_field_sum : :obj:`hyperspy.signals.Signal2D`
            The virtual image signal associated with the specified scattering
            range.

        Examples
        --------
        .. code-block:: python

            rp.get_virtual_image(left=0.5, right=0.7)

        """
        # Define ROI
        roi = SpanROI(left=left, right=right)
        dark_field = roi(self, axes=self.axes_manager.signal_axes)
        dark_field_sum = dark_field.sum(
            axis=dark_field.axes_manager.signal_axes
        )
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        vdfim = dark_field_sum.as_signal2D((0, 1))

        return vdfim

    def as_lazy(self, *args, **kwargs):
        """Create a copy of the Diffraction1D object as a
        :py:class:`~pyxem.signals.diffraction1d.LazyDiffraction1D`.

        Parameters
        ----------
        copy_variance : bool
            If True variance from the original Diffraction1D object is copied to
            the new LazyDiffraction1D object.

        Returns
        -------
        res : :py:class:`~pyxem.signals.diffraction1d.LazyDiffraction1D`.
            The lazy signal.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyDiffraction1D
        res.__init__(**res._to_dictionary())
        return res

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = Diffraction1D


class LazyDiffraction1D(LazySignal, Diffraction1D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):
        super().compute(*args, **kwargs)
        self.__class__ = Diffraction1D
        self.__init__(**self._to_dictionary())

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = LazyDiffraction1D

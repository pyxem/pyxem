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

"""
Signal base class for two-dimensional diffraction data.
"""

import numpy as np
from warnings import warn

from hyperspy.api import interactive
from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from hyperspy._signals.lazy import LazySignal

from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals import push_metadata_through

from pyxem.utils.expt_utils import _index_coords, _cart2polar, _polar2cart, \
    radial_average, gain_normalise, remove_dead,\
    regional_filter, subtract_background_dog, subtract_background_median, \
    subtract_reference, circular_mask, find_beam_offset_cross_correlation, \
    peaks_as_gvectors, convert_affine_to_transform, apply_transformation, \
    find_beam_center_blur, find_beam_center_interpolate

from pyxem.utils.peakfinders2D import find_peaks_zaefferer, find_peaks_stat, \
    find_peaks_dog, find_peaks_log, find_peaks_xc

from pyxem.utils import peakfinder2D_gui

from skimage import filters
from skimage import transform as tf
from skimage.morphology import square


class PolarDiffraction2D(Signal2D):
    _signal_type = "polar_diffraction2d"

    def __init__(self, *args, **kwargs):
        """
        Create a PolarDiffraction2D object from a hs.Signal2D or np.array.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            either a numpy.ndarray or a Signal2D
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def as_lazy(self, *args, **kwargs):
        """Create a copy of the PolarDiffraction2D object as a
        :py:class:`~pyxem.signals.polar_diffraction2d.LazyPolarDiffraction2D`.

        Parameters
        ----------
        copy_variance : bool
            If True variance from the original PolarDiffraction2D object is
            copied to the new LazyPolarDiffraction2D object.

        Returns
        -------
        res : :py:class:`~pyxem.signals.polar_diffraction2d.LazyPolarDiffraction2D`.
            The lazy signal.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyPolarDiffraction2D
        res.__init__(**res._to_dictionary())
        return res

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = PolarDiffraction2D


class LazyPolarDiffraction2D(LazySignal, PolarDiffraction2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):
        super().compute(*args, **kwargs)
        self.__class__ = PolarDiffraction2D
        self.__init__(**self._to_dictionary())

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = LazyPolarDiffraction2D

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
"""Signal class for two-dimensional electron diffraction data.
"""

from hyperspy._signals.lazy import LazySignal

from pyxem.signals import push_metadata_through
from pyxem.signals.diffraction2d import Diffraction2D
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D


class RawDiffraction2D(Diffraction2D):
    _signal_type = "raw_diffraction2d"

    def __init__(self, *args, **kwargs):
        """
        Create a RawDiffraction2D object from a hs.Signal2D or np.array.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Diffraction2D. The first arg should be
            either a numpy.ndarray or a Signal2D
        **kwargs :
            Passed to the __init__ of Diffraction2D
        """
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        # Set default attributes
        if 'Acquisition_instrument' in self.metadata.as_dictionary():
            if 'SEM' in self.metadata.as_dictionary()['Acquisition_instrument']:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def to_electron_diffraction2d(self,
                                  beam_energy,
                                  camera_length,
                                  detector,
                                  pixel_size,
                                  step_size
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
        pass

    def as_lazy(self, *args, **kwargs):
        """Create a copy of the ElectronDiffraction2D object as a
        :py:class:`~pyxem.signals.electron_diffraction2d.LazyElectronDiffraction2D`.

        Parameters
        ----------
        copy_variance : bool
            If True variance from the original ElectronDiffraction2D object is
            copied to the new LazyElectronDiffraction2D object.

        Returns
        -------
        res : :py:class:`~pyxem.signals.electron_diffraction2d.LazyElectronDiffraction2D`.
            The lazy signal.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyRawDiffraction2D
        res.__init__(**res._to_dictionary())
        return res

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = RawDiffraction2D


class LazyRawDiffraction2D(LazySignal, RawDiffraction2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):
        super().compute(*args, **kwargs)
        self.__class__ = RawDiffraction2D
        self.__init__(**self._to_dictionary())

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = LazyElectronDiffraction2D

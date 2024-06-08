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
from hyperspy.signals import Signal1D
from pyxem.signals import DiffractionVectors
from hyperspy.utils.plot import plot_images


class DiffractionVectors1D(DiffractionVectors, Signal1D):
    """A Collection of Diffraction Vectors with a defined 1D set of vectors.

    For every navigation position there is specifically one vector that represents
    the dataset.

    Examples of DiffractionVectors1D Signals:
        - STEM_DPC
        - STRAIN Maps
        - Diffraction shifts/ Centers

    Attributes
    ----------
    column_scale : np.array()
        The scale for each column in the signal.  For converting the real values
        to pixel values in some image.

    column_offsets : np.array()
        The offsets for each column in the signal.  For converting the real values
        to pixel values in some image.



    """

    _signal_dimension = 1
    _signal_type = "diffraction_vectors"

    def plot(self, tight_layout=True, **kwargs):
        """
        Plot the beam shifts, utilizing HyperSpy's :func:`hyperspy.api.plot.plot_images`
        function. Each Vector is plotted as a separate image with the column names as labels.

        Parameters
        ----------
        tight_layout : bool, optional
            Whether to use tight layout in the plot. The default is True.
        **kwargs : dict
            Keyword arguments to pass to :func:`hyperspy.api.plot.plot_images`.

        """
        if self._lazy:
            raise ValueError(
                "plot is not implemented for lazy signals, " "run compute() first"
            )
        vectors = self.T

        if "suptitle" not in kwargs:
            kwargs["label"] = [c if c is not None else "" for c in self.column_names]
        axes_list = plot_images(
            vectors,
            tight_layout=tight_layout,
            **kwargs,
        )
        return axes_list

    def flatten_diffraction_vectors(self, **kwargs):
        raise NotImplementedError(
            "flatten_diffraction_vectors is not implemented yet for DiffractionVectors1D."
        )

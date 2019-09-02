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

import pickle
import numpy as np
from hyperspy.misc.utils import stack as stack_method


class CalibrationDataLibrary(dict):
    """Maps crystal structure (phase) to diffraction vectors.

    Attributes
    ----------
    au_x_grating_dp : ElectronDiffraction2D
        A ring diffraction pattern obtained from an Au X-grating standard.
    au_x_grating_im : :obj:`hyperspy.signals.Signal2D`
        An image of an Au X-grating standard.

    """

    def __init__(self, au_x_grating_dp=None, au_x_grating_im=None,
                 moo3_dp=None, moo3_im=None):
        self.au_x_grating_dp = au_x_grating_dp
        self.au_x_grating_im = au_x_grating_im
        self.moo3_dp = moo3_dp
        self.moo3_im = moo3_im

    def plot_calibration_data(self, data_to_plot, roi=None,
                              *args, **kwargs):
        """ Plot the calibration data.

        Parameters
        ----------
        data_to_plot : string
            Specify the calibration data to be plotted. Valid options are:
            {'au_x_grating_dp', 'au_x_grating_im'}
        roi : :obj:`hyperspy.roi.BaseInteractiveROI`
            An optional ROI object, as detailed in HyperSpy, to be added as a
            widget to the calibration data plot.
        """
        # Construct object containing user defined data to plot
        if data_to_plot == 'au_x_grating_dp':
            data = self.au_x_grating_dp
        elif data_to_plot == 'au_x_grating_im':
            data = self.au_x_grating_im
        elif data_to_plot == 'moo3_dp':
            data = self.moo3_dp
        elif data_to_plot == 'moo3_im':
            data = self.moo3_im
        else:
            raise ValueError("Please specify valid data_to_plot.")
        # Plot the data
        data.plot(*args, **kwargs)
        if roi:
            roi.add_widget(data, axes=data.axes_manager.signal_axes)

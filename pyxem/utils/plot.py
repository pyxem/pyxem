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

import numpy as np
import itertools

def _find_max_length_peaks(peaks):
    """Worker function for generate_marker_inputs_from_peaks.

    Parameters
    ----------
    peaks : :class:`pyxem.diffraction_vectors.DiffractionVectors`
        Identified peaks in a diffraction signal.

    Returns
    -------
    longest_length : int
        The length of the longest peak list.

    """
    x_size, y_size = peaks.axes_manager.navigation_shape[0], peaks.axes_manager.navigation_shape[1]
    length_of_longest_peaks_list = 0
    for x in np.arange(0, x_size):
        for y in np.arange(0, y_size):
            if peaks.data[y, x].shape[0] > length_of_longest_peaks_list:
                length_of_longest_peaks_list = peaks.data[y, x].shape[0]
    return length_of_longest_peaks_list


def generate_marker_inputs_from_peaks(peaks):
    """Takes a peaks (defined in 2D) object from a STEM (more than 1 image) scan
    and returns markers.

    Parameters
    ----------
    peaks : :class:`pyxem.diffraction_vectors.DiffractionVectors`
        Identifies peaks in a diffraction signal.

    Example
    -------
    How to get these onto images::

        mmx,mmy = generate_marker_inputs_from_peaks(found_peaks)
        dp.plot(cmap='viridis')
        for mx,my in zip(mmx,mmy):
            m = hs.markers.point(x=mx,y=my,color='red',marker='x')
            dp.add_marker(m,plot_marker=True,permanent=False)

    """
    max_peak_len = _find_max_length_peaks(peaks)
    pad = np.array(list(itertools.zip_longest(*np.concatenate(peaks.data), fillvalue=[np.nan, np.nan])))
    pad = pad.reshape((max_peak_len), peaks.data.shape[0], peaks.data.shape[1], 2)
    xy_cords = np.transpose(pad, [3, 0, 1, 2])  # move the x,y pairs to the front
    x = xy_cords[0]
    y = xy_cords[1]

    return x, y

class IndexTracker(object):
    """
    Class for keeping track of indices for the function plot_templates_on_1D_signal, using
    matplotlib.figure.canvas.mpl_connect()
    """

    def __init__(self, ax1, ax2, ax3, signal, storage, kwargs_for_signal = {}, kwargs_for_template_scatter = {}):
        """
        Set initial parameters for IndexTracker class.

        Parameters
        ----------
        ax1,ax2,ax3 : matplotlib.axes._subplots.AxesSubplot
            Image axes for figure which will be updated by the IndexTracker class

        signal : pyxem.signals.electron_diffraction2d.ElectronDiffraction2D
            1D ElectronDiffraction2D object. The hyperspy generator for ElectronDiffraction1D does not
            produce the expected result for np.arrays of dimensions (30,512,512) therefore ElectronDiffraction2D
            is chosen.

        storage : list
            List containing phase information, angle, correlation score, template coordinates and template intensities
            for every match result.

        kwargs_for_signal : dict
            Arguments passed on to ax.imshow() used to plot the signal.

        kwargs_for_template_scatter : dict
            Arguments passed on to ax.scatter() used to plot template data.
        """
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3

        self.signal = signal

        self.slices = signal.shape[0]
        self.ind = self.slices//2
        self.rank = 0
        self.max_rank = len(storage[0])
        self.storage = storage

        coordinates = self.storage[self.ind][self.rank][3]
        intensities = self.storage[self.ind][self.rank][4]
        x,y = zip(*coordinates)

        self.im = ax1.imshow(self.signal[self.ind], **kwargs_for_signal)
        self.im2 = ax2.imshow(self.signal[self.ind], **kwargs_for_signal)
        self.im3 = ax3.imshow(self.signal[self.ind], **kwargs_for_signal)
        self.line1 = ax2.scatter(x = x, y = y, c = intensities, **kwargs_for_template_scatter)
        self.line2 = ax3.scatter(x = x, y = y, c = 'r', **kwargs_for_template_scatter)
        self.update()

    def onscroll(self, event):
        """
        Updates the index of the tracker object upon detecting a mouse scroll event. Plot is updated when
        an event is detected.
        """
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def click(self, event):
        """
        Updates the rank index of the tracker object upon detecting a mouse click event. Plot is updated when
        an event is detected.
        """
        self.rank = (self.rank + 1) % self.max_rank
        self.update()

    def update(self):
        """
        Updates the current plot after detecting an event.
        """
        self.im.set_data(self.signal[self.ind,:,:])
        self.ax1.set_ylabel('slice %s' % self.ind)
        self.ax1.set_title('Signal')
        self.im.axes.figure.canvas.draw()

        phase = self.storage[self.ind][self.rank][0]
        angle = self.storage[self.ind][self.rank][1]
        correlation_score = self.storage[self.ind][self.rank][2]
        coordinates = self.storage[self.ind][self.rank][3]
        intensities = self.storage[self.ind][self.rank][4]

        self.im2.set_data(self.signal[self.ind,:,:])
        self.line1.set_offsets(coordinates)
        self.line1.set_array(intensities)
        self.ax2.set_title('Euler Angle: ({0:.0f},{1:.0f},{2:.0f}) Phase: {3}'.format(angle[0],
                                                                                     angle[1],
                                                                                     angle[2],
                                                                                     phase))

        self.im2.axes.figure.canvas.draw()

        self.im3.set_data(self.signal[self.ind,:,:])
        self.line2.set_offsets(coordinates)
        self.ax3.set_title('Rank: {0}, Correlation score: {1:.3f}'.format(self.rank,
                                                                    correlation_score))
        self.im3.axes.figure.canvas.draw()


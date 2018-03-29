# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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


import sys
from unittest import mock


import numpy.testing as npt
import numpy as np
import scipy as sp
from scipy.misc import face, ascent
from scipy.ndimage import fourier_shift

import hyperspy.api as hs
from pyxem import ElectronDiffraction
from hyperspy.decorators import lazifyTestClass


class TestFindPeaks2D:

    def setUp(self):
        coefficients = np.array(
            [350949.04890400 + 0.j, -22003.98742841 + 51494.56650429j,
             37292.52741553 + 38067.97686711j, 37292.52741553 - 38067.97686711j,
             -22003.98742841 - 51494.56650429j]
        )
        coordinates = np.array([[0, 26, 30, 994, 998],
                                [0, 1003, 39, 985, 21]]
        )
        dense = np.zeros((1024, 1024), dtype=complex)
        dense[coordinates[0], coordinates[1]] = coefficients
        dense = ElectronDiffraction(np.real(np.fft.ifft2(dense)))
        self.dense = dense.isig[500:550, 500:550]

        coefficients = np.array(
            [10, 5, 86, 221, 6, 95, 70, 12, 255, 5, 255, 3, 23,
             24, 77, 255, 11, 255, 8, 35, 195, 165, 27, 255, 8, 14,
             255, 21, 53, 107, 255, 18, 255, 4, 26, 255, 39, 27, 255,
             6, 255, 7, 13, 37, 35, 9, 83]
        )
        coordinates = np.array(
            [[3, 40],    [3, 138],  [9, 67],   [14, 95],   [20, 23],
             [20, 122],  [26, 51],  [26, 100], [31, 78],   [31, 128],
             [37, 107],  [38, 7],   [43, 34],  [43, 84],   [43, 134],
             [49, 62],   [49, 112], [54, 90],  [60, 17],   [60, 67],
             [60, 118],  [66, 45],  [66, 96],  [72, 73],   [72, 124],
             [77, 51],   [77, 101], [83, 28],  [83, 79],   [83, 130],
             [89, 57],   [89, 107], [95, 85],  [101, 12],  [101, 62],
             [101, 113], [106, 40], [107, 91], [112, 68],  [113, 119],
             [119, 97],  [124, 23], [124, 74], [124, 125], [130, 51],
             [130, 103], [136, 80]])
        sparse = np.zeros((144, 144))
        xs, ys = np.ogrid[:144, :144]
        for (x0, y0), a in zip(coordinates, coefficients):
            sparse += a * sp.stats.norm.pdf(xs, x0)*sp.stats.norm.pdf(ys, y0)
        sparse = sparse[50:100, 50:100]
        self.sparse0d = ElectronDiffraction(sparse)
        self.sparse1d = ElectronDiffraction(np.array([sparse for i in range(2)]))
        self.sparse2d = ElectronDiffraction(np.array([[sparse for i in range(2)] for j in range(2)]))
        xref, yref = 72, 72
        ref = np.zeros((144, 144))
        ref += 100 * sp.stats.norm.pdf(xs, xref)*sp.stats.norm.pdf(ys, yref)

        self.ref = ElectronDiffraction(ref)
        ans = np.empty((1,), dtype=object)
        ans[0] = np.array([[xref, yref]])
        self.ans = ans

        self.methods = ['skimage', 'max', 'minmax', 'zaefferer', 'stat', 'laplacian_of_gaussians', 'difference_of_gaussians']
        self.datasets = {'dense': self.dense, 'sparse0d': self.sparse0d, 'sparse1d': self.sparse1d, 'sparse2d': self.sparse2d}
        self.thetests = {'creates_array': self.creates_array, 'peaks_match_input': self.peaks_match_input, 'peaks_are_coordinates': self.peaks_are_coordinates}

    def test_properties(self):
        self.setUp()  # Test generation does not explicitly call setUp()
        for method in self.methods:
            for dataset in self.datasets:
                for each_test in self.thetests:
                    yield self.thetests[each_test], method, self.datasets[dataset]

    def test_answers(self):
        self.setUp()  # Test generation does not explicitly call setUp()
        for method in self.methods:
            yield self.gets_right_answer, method, self.ref, self.ans

    def creates_array(self, method, dataset):
        peaks = dataset.find_peaks(method=method)
        nt.assert_is_instance(peaks, np.ndarray)

    def peaks_match_input(self, method, dataset):
        peaks = dataset.find_peaks(method=method)
        signal_shape = dataset.axes_manager.navigation_shape[::-1] if dataset.axes_manager.navigation_size > 0 else (1,)
        nt.assert_equal(peaks.shape, signal_shape)

    def peaks_are_coordinates(self, method, dataset):
        peaks = dataset.find_peaks(method=method)
        peak_shapes = np.array([peak.shape for peak in peaks.flatten()])
        nt.assert_true(np.all(peak_shapes[:, 1] == 2))

    def gets_right_answer(self, method, dataset, answer):
        peaks = dataset.find_peaks()
        nt.assert_true(np.all(peaks[0] == answer[0]))

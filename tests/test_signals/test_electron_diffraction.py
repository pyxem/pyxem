# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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
import pytest
from hyperspy.signals import Signal1D, Signal2D
from pyxem.signals.electron_diffraction import ElectronDiffraction


@pytest.fixture(params=[
    np.array([[[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 2., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 2., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 2.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 2., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 2., 0., 0., 0.],
               [0., 0., 0., 2., 2., 2., 0., 0.],
               [0., 0., 0., 0., 2., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]]])
])
def diffraction_pattern(request):
    return ElectronDiffraction(request.param)


@pytest.fixture(params=[
    np.array([[[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 2., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 2., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 2.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 2., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]],
              [[0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 2., 0., 0., 0.],
               [0., 0., 0., 2., 2., 2., 0., 0.],
               [0., 0., 0., 0., 2., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]]]).reshape(2,2,8,8)
])

def diffraction_pattern_SED(request):
    return ElectronDiffraction(request.param)


@pytest.mark.skip(reason='Defaults not implemented in pyXem')
def test_default_params(diffraction_pattern):
    a = diffraction_pattern.metadata.Acquisition_instrument.TEM.rocking_angle
    pass


@pytest.mark.parametrize('calibration, center', [
    (1, (4, 4),),
    (0.017, (3, 3)),
    (0.5, None,),
])
def test_set_diffraction_calibration(diffraction_pattern, calibration, center):
    calibrated_center = calibration * np.array(center) if center is not None else center
    diffraction_pattern.set_diffraction_calibration(calibration, center=calibrated_center)
    dx, dy = diffraction_pattern.axes_manager.signal_axes
    assert dx.scale == calibration and dy.scale == calibration
    if center is not None:
        assert np.all(diffraction_pattern.isig[0., 0.].data == diffraction_pattern.isig[center[0], center[1]].data)


@pytest.mark.parametrize('dark_reference, bright_reference', [
    (-1, 1),
    (0, 1),
    (0, 256),
])
def test_apply_gain_normalisation(diffraction_pattern: ElectronDiffraction,
                                  dark_reference, bright_reference):
    diffraction_pattern.apply_gain_normalisation(
        dark_reference=dark_reference, bright_reference=bright_reference)
    assert diffraction_pattern.max() == bright_reference
    assert diffraction_pattern.min() == dark_reference


class TestDirectBeamMethods:

    @pytest.mark.parametrize('mask_expected', [
        (np.array([
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False,  True,  True, False, False, False],
            [False, False,  True,  True,  True,  True, False, False],
            [False, False,  True,  True,  True,  True, False, False],
            [False, False, False,  True,  True, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]]),),
                  ])
    def test_get_direct_beam_mask(self, diffraction_pattern, mask_expected):
        mask_calculated = diffraction_pattern.get_direct_beam_mask(2)
        assert isinstance(mask_calculated, Signal2D)
        assert np.equal(mask_calculated, mask_expected)


class TestRadialProfile:

    @pytest.fixture
    def diffraction_pattern(self):
        dp = ElectronDiffraction(np.zeros((2, 8, 8)))
        dp.data[0] = np.array([[0., 0., 2., 2., 2., 2., 0., 0.],
                               [0., 2., 3., 3., 3., 3., 2., 0.],
                               [2., 3., 3., 4., 4., 3., 3., 2.],
                               [2., 3., 4., 5., 5., 4., 3., 2.],
                               [2., 3., 4., 5., 5., 4., 3., 2.],
                               [2., 3., 3., 4., 4., 3., 3., 2.],
                               [0., 2., 3., 3., 3., 3., 2., 0.],
                               [0., 0., 2., 2., 2., 2., 0., 0.]])

        dp.data[1] = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 1., 0., 0., 0.],
                               [0., 0., 0., 1., 1., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.]])

        return dp

    def test_radial_profile_signal_type(self, diffraction_pattern):
        rp = diffraction_pattern.get_radial_profile()
        assert isinstance(rp, Signal1D)

    @pytest.mark.parametrize('expected',[
        (np.array(
            [[5., 4., 3., 2., 0.],
             [1., 0., 0., 0., 0.]]
        ))])

    def test_radial_profile(self, diffraction_pattern,expected):
        rp = diffraction_pattern.get_radial_profile()
        assert np.allclose(rp.data, expected, atol=1e-3)


class TestApplyAffineTransformation:

    def test_affine_transformation_signal_type(self, diffraction_pattern):
        diffraction_pattern.apply_affine_transformation(
            D=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]]))
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    @pytest.mark.parametrize('diffraction_pattern, transformation, expected', [
        (
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ], dtype=float), np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=float), np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ], dtype=float),
        ),
        (
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ], dtype=float),
            np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ], dtype=float),
            np.array([
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ], dtype=float),
        )
    ], indirect=['diffraction_pattern'])
    def test_geometric_distortion(self, diffraction_pattern, transformation, expected):
        diffraction_pattern.apply_affine_transformation(D=transformation)
        assert np.allclose(diffraction_pattern.data, expected)


class TestBackgroundMethods:

    @pytest.mark.parametrize('method, kwargs', [
        ('h-dome', {'h': 1}),
        ('gaussian_difference', {'sigma_min': 0.5, 'sigma_max': 1, }),
        ('median', {'footprint': 4, })
    ])
    def test_remove_background(self, diffraction_pattern: ElectronDiffraction,
                               method, kwargs):
        bgr = diffraction_pattern.remove_background(method=method, **kwargs)
        assert bgr.data.shape == diffraction_pattern.data.shape
        assert bgr.max() <= diffraction_pattern.max()

@pytest.mark.skip(reason="Diffraction Simulation not yet fixed")
class TestPeakFinding:
    #This isn't testing the finding, that is done in test_peakfinders2D
    @pytest.fixture
    def single_peak(self):
        pattern = np.zeros((2,2,128,128))
        pattern[:,:,40,45] = 1  #single point peak
        return ElectronDiffraction(pattern)

    @pytest.fixture
    def ragged_peak(self):
        pattern = np.zeros((2,2,128,128))
        pattern[:,:,40,45] = 1
        pattern[1,0,71,21] = 1
        return ElectronDiffraction(pattern)

    methods = ['skimage', 'zaefferer','laplacian_of_gaussians', 'difference_of_gaussians','stat']
    def test_argless_run(self,single_peak):
        single_peak.find_peaks()
        pass

    @pytest.mark.parametrize('method', methods)
    def test_findpeaks_single(self,single_peak,method):
        output = (single_peak.find_peaks(method)).inav[0,0] #should be <2,2|2,1>
        assert output.isig[1] == 2        #  correct number of dims
        assert output.isig[0] == 1        #  correct number of peaks
        assert output.isig[0] == (40-(128/2)) #x
        assert output.isig[1] == (45-(128/2)) #y

    def test_findpeaks_ragged(self,ragged_peak,method):
        output = (ragged_peak.find_peaks(method))
        # as before at 0,0
        assert output.inav[0,0].isig[1] == 2        #  correct number of dims
        assert output.inav[0,0].isig[0] == 1        #  correct number of peaks
        assert output.inav[0,0].isig[0] == (40-(128/2)) #x
        assert output.inav[0,0].isig[1] == (45-(128/2)) #y
        #but at
        assert np.sum(output.inav[0,1].data.shape) == 4 # 2+2

class TestSimpleMaps:
    #This class simply confirms that maps run without error.
    # These tests are not suitable for objects that may return ragged arrays

    def test_get_direct_beam_postion(self,diffraction_pattern_SED):
        shifts = diffraction_pattern_SED.get_direct_beam_position(radius_start=1,radius_finish=3)

    def test_center_direct_beam(self,diffraction_pattern_SED):
        assert isinstance(diffraction_pattern_SED,ElectronDiffraction) #before inplace transform applied
        diffraction_pattern_SED.center_direct_beam(radius_start=1,radius_finish=3)
        assert isinstance(diffraction_pattern_SED,ElectronDiffraction) #after inplace transform applied

    methods = ['threshold']
    @pytest.mark.parametrize('method', methods)
    @pytest.mark.xfail(raises=NotImplementedError)
    def test_get_no_diffraction_mask(self, diffraction_pattern,method):
        mask = diffraction_pattern.get_no_diffraction_mask(method = method)

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
from hyperspy.roi import CircleROI
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
               [0., 0., 0., 0., 0., 0., 0., 0.]]]).reshape(2,2,8,8)
])

def diffraction_pattern(request):
    return ElectronDiffraction(request.param)


class TestSimpleMaps:
    #Confirms that maps run without error.

    def test_get_direct_beam_postion(self,diffraction_pattern):
        shifts = diffraction_pattern.get_direct_beam_position(radius_start=1,radius_finish=3)

    def test_center_direct_beam(self,diffraction_pattern):
        assert isinstance(diffraction_pattern,ElectronDiffraction) #before inplace transform applied
        diffraction_pattern.center_direct_beam(radius_start=1,radius_finish=3)
        assert isinstance(diffraction_pattern,ElectronDiffraction) #after inplace transform applied

    def test_center_direct_beam_in_small_region(self,diffraction_pattern):
        assert isinstance(diffraction_pattern,ElectronDiffraction)
        diffraction_pattern.center_direct_beam(radius_start=1,radius_finish=3,square_width=3)
        assert isinstance(diffraction_pattern,ElectronDiffraction)

    def test_apply_affine_transformation(self, diffraction_pattern):
        diffraction_pattern.apply_affine_transformation(
                                                        D=np.array([[1., 0., 0.],
                                                                    [0., 1., 0.],
                                                                    [0., 0., 1.]]))
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    methods = ['average','nan']
    @pytest.mark.parametrize('method', methods)
    #@pytest.mark.skip(reason="currently crashes via a tqdm issue")
    def test_remove_dead_pixels(self,diffraction_pattern,method):
        diffraction_pattern.remove_deadpixels([[1,2],[5,6]],method,progress_bar=False)
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_remove_dead_pixels_failing(self,diffraction_pattern):
        diffraction_pattern.remove_deadpixels([[1,2],[5,6]],'fake_method',progress_bar=False)



class TestSimpleHyperspy:
    # Tests functions that assign to hyperspy metadata

    def test_set_experimental_parameters(self,diffraction_pattern):
        diffraction_pattern.set_experimental_parameters(accelerating_voltage=3,
                                                             camera_length=3,
                                                             scan_rotation=1,
                                                             convergence_angle=1,
                                                             rocking_angle=1,
                                                             rocking_frequency=1,
                                                             exposure_time=1)
        assert isinstance(diffraction_pattern,ElectronDiffraction)

    def test_set_scan_calibration(self,diffraction_pattern):
        diffraction_pattern.set_scan_calibration(19)
        assert isinstance(diffraction_pattern,ElectronDiffraction)

    @pytest.mark.parametrize('calibration, center', [
                                (1, (4, 4),),
                                (0.017, (3, 3)),
                                (0.5, None,),])

    def test_set_diffraction_calibration(self,diffraction_pattern, calibration, center):
        calibrated_center = calibration * np.array(center) if center is not None else center
        diffraction_pattern.set_diffraction_calibration(calibration, center=calibrated_center)
        dx, dy = diffraction_pattern.axes_manager.signal_axes
        assert dx.scale == calibration and dy.scale == calibration
        if center is not None:
            assert np.all(diffraction_pattern.isig[0., 0.].data == diffraction_pattern.isig[center[0], center[1]].data)

class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    def test_plot_interactive_virtual_image(self,diffraction_pattern):
        roi = CircleROI(3,3,5)
        diffraction_pattern.plot_interactive_virtual_image(roi)

    def test_get_virtual_image(self,diffraction_pattern):
        roi = CircleROI(3,3,5)
        diffraction_pattern.get_virtual_image(roi)



class TestGainNormalisation:

    @pytest.mark.parametrize('dark_reference, bright_reference', [
                                                                    (-1, 1),
                                                                    (0, 1),
                                                                    (0, 256),
                                                                ])
    def test_apply_gain_normalisation(self, diffraction_pattern,
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
    def diffraction_pattern_for_radial(self):
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

    def test_radial_profile_signal_type(self, diffraction_pattern_for_radial):
        rp = diffraction_pattern_for_radial.get_radial_profile()
        assert isinstance(rp, Signal1D)

    @pytest.mark.parametrize('expected',[
        (np.array(
            [[5., 4., 3., 2., 0.],
             [1., 0., 0., 0., 0.]]
        ))])

    def test_radial_profile(self, diffraction_pattern_for_radial,expected):
        rp = diffraction_pattern_for_radial.get_radial_profile()
        assert np.allclose(rp.data, expected, atol=1e-3)

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

class TestPeakFinding:
    #This isn't testing the finding, that is done in test_peakfinders2D

    @pytest.fixture
    def ragged_peak(self):
        pattern = np.zeros((2,2,128,128))
        pattern[:,:,40:42,45] = 1
        pattern[:,:,110,30:32] = 1
        pattern[1,0,71:73,21:23] = 1
        return ElectronDiffraction(pattern)

    @pytest.fixture
    def nonragged_peak(self):
            pattern = np.zeros((2,2,128,128))
            pattern[:,:,40:42,45] = 1
            pattern[:,:,110,30:32] = 1
            return ElectronDiffraction(pattern)


    methods = ['zaefferer','laplacian_of_gaussians', 'difference_of_gaussians','stat']

    @pytest.mark.parametrize('method', methods)
    @pytest.mark.parametrize('peak',[ragged_peak,nonragged_peak])
    def test_findpeaks_ragged(self,peak,method):
        output = peak(self).find_peaks(method=method)
        if method != 'difference_of_gaussians':
            # three methods return the expect peak
            assert output.inav[0,0].isig[1] == 2        #  correct number of dims (boring square)
            assert output.inav[0,0].isig[0] == 1        #   """ peaks """
            if peak(self).data[1,0,72,22] == 1: # 3 peaks
                assert output.inav[0,1].data.shape == (3,2)
            else: #2 peaks
                assert output.data.shape == (2,2,2,2)
        else:
            # DoG doesn't find the correct peaks, but runs without error
            if peak(self).data[1,0,72,22] == 1: # 3 peaks
                assert output.data.shape == (2,2) # tests we have a sensible ragged array


    @pytest.mark.xfail(raises=NotImplementedError)
    def test_failing_run(self,ragged_peak):
        ragged_peak.find_peaks(method='no_such_method_exists')

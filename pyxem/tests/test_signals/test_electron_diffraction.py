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

import pytest
import numpy as np

from hyperspy.signals import Signal1D, Signal2D
from hyperspy.roi import CircleROI

from pyxem.signals.electron_diffraction import ElectronDiffraction


def test_init():
    z = np.zeros((2, 2, 2, 2))
    dp = ElectronDiffraction(z, metadata={'Acquisition_instrument': {'SEM': 'Expensive-SEM'}})


class TestSimpleMaps:
    # Confirms that maps run without error.

    def test_get_direct_beam_postion(self, diffraction_pattern):
        shifts = diffraction_pattern.get_direct_beam_position(radius_start=1,
                                                              radius_finish=3)

    def test_center_direct_beam(self, diffraction_pattern):
        # before inplace transform applied
        assert isinstance(diffraction_pattern, ElectronDiffraction)
        diffraction_pattern.center_direct_beam(radius_start=1, radius_finish=3)
        # after inplace transform applied
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    def test_center_direct_beam_in_small_region(self, diffraction_pattern):
        assert isinstance(diffraction_pattern, ElectronDiffraction)
        diffraction_pattern.center_direct_beam(radius_start=1,
                                               radius_finish=3,
                                               square_width=3)
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    def test_apply_affine_transformation(self, diffraction_pattern):
        diffraction_pattern.apply_affine_transformation(
            D=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]]))
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    methods = ['average', 'nan']

    @pytest.mark.parametrize('method', methods)
    def test_remove_dead_pixels(self, diffraction_pattern, method):
        dpr = diffraction_pattern.remove_deadpixels([[1, 2], [5, 6]], method,
                                                    inplace=False)
        assert isinstance(dpr, ElectronDiffraction)


class TestSimpleHyperspy:
    # Tests functions that assign to hyperspy metadata

    def test_set_experimental_parameters(self, diffraction_pattern):
        diffraction_pattern.set_experimental_parameters(accelerating_voltage=3,
                                                        camera_length=3,
                                                        scan_rotation=1,
                                                        convergence_angle=1,
                                                        rocking_angle=1,
                                                        rocking_frequency=1,
                                                        exposure_time=1)
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    def test_set_scan_calibration(self, diffraction_pattern):
        diffraction_pattern.set_scan_calibration(19)
        assert isinstance(diffraction_pattern, ElectronDiffraction)

    @pytest.mark.parametrize('calibration, center', [
        (1, (4, 4),),
        (0.017, (3, 3)),
        (0.5, None,), ])
    def test_set_diffraction_calibration(self,
                                         diffraction_pattern,
                                         calibration, center):
        calibrated_center = calibration * np.array(center) if center is not None else center
        diffraction_pattern.set_diffraction_calibration(calibration, center=calibrated_center)
        dx, dy = diffraction_pattern.axes_manager.signal_axes
        assert dx.scale == calibration and dy.scale == calibration
        if center is not None:
            assert np.all(diffraction_pattern.isig[0., 0.].data ==
                          diffraction_pattern.isig[center[0], center[1]].data)


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    def test_plot_interactive_virtual_image(self, diffraction_pattern):
        roi = CircleROI(3, 3, 5)
        diffraction_pattern.plot_interactive_virtual_image(roi)

    def test_get_virtual_image(self, diffraction_pattern):
        roi = CircleROI(3, 3, 5)
        diffraction_pattern.get_virtual_image(roi)


class TestGainNormalisation:

    @pytest.mark.parametrize('dark_reference, bright_reference', [
        (-1, 1),
        (0, 1),
        (0, 256),
    ])
    def test_apply_gain_normalisation(self, diffraction_pattern,
                                      dark_reference, bright_reference):
        dpr = diffraction_pattern.apply_gain_normalisation(
            dark_reference=dark_reference, bright_reference=bright_reference,
            inplace=False)
        assert dpr.max() == bright_reference
        assert dpr.min() == dark_reference


class TestDirectBeamMethods:

    @pytest.mark.parametrize('mask_expected', [
        (np.array([
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, True, True, False, False, False],
            [False, False, True, True, True, True, False, False],
            [False, False, True, True, True, True, False, False],
            [False, False, False, True, True, False, False, False],
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
        """
        Two diffraction patterns with easy to see radial profiles, wrapped
        in ElectronDiffraction  <2|8,8>
        """
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
                               [1., 1., 1., 1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1., 1., 1., 1.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.]])

        return dp

    @pytest.fixture
    def mask_for_radial(self):
        """
        An 8x8 mask array, to test that part of the radial average code.
        """
        mask = np.array([[0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [1., 0., 1., 1., 1., 1., 1., 1.],
                         [1., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.],
                         [0., 0., 1., 1., 1., 1., 1., 1.]])

        return mask

    def test_radial_profile_signal_type(self, diffraction_pattern_for_radial,
                                        mask_for_radial):
        rp = diffraction_pattern_for_radial.get_radial_profile()
        rp_mask = diffraction_pattern_for_radial.get_radial_profile(
            mask_array=mask_for_radial)

        assert isinstance(rp, Signal1D)
        assert isinstance(rp_mask, Signal1D)

    @pytest.fixture
    def axes_test_dp(self):
        dp_data = np.random.randint(0, 10, (2, 2, 10, 10))
        dp = ElectronDiffraction(dp_data)
        return dp

    def test_radial_profile_axes(self, axes_test_dp):
        n_scale = 0.5
        axes_test_dp.axes_manager.navigation_axes[0].scale = n_scale
        axes_test_dp.axes_manager.navigation_axes[1].scale = 2 * n_scale
        name = 'real_space'
        axes_test_dp.axes_manager.navigation_axes[0].name = name
        axes_test_dp.axes_manager.navigation_axes[1].units = name
        units = 'um'
        axes_test_dp.axes_manager.navigation_axes[1].name = units
        axes_test_dp.axes_manager.navigation_axes[0].units = units

        rp = axes_test_dp.get_radial_profile()
        rp_scale_x = rp.axes_manager.navigation_axes[0].scale
        rp_scale_y = rp.axes_manager.navigation_axes[1].scale
        rp_units_x = rp.axes_manager.navigation_axes[0].units
        rp_name_x = rp.axes_manager.navigation_axes[0].name
        rp_units_y = rp.axes_manager.navigation_axes[1].units
        rp_name_y = rp.axes_manager.navigation_axes[1].name

        assert n_scale == rp_scale_x
        assert 2 * n_scale == rp_scale_y
        assert units == rp_units_x
        assert name == rp_name_x
        assert name == rp_units_y
        assert units == rp_name_y

    @pytest.mark.parametrize('expected', [
        (np.array(
            [[5., 4., 3., 2., 0.],
             [1., 0.5, 0.2, 0.2, 0.]]
        ))])
    @pytest.mark.parametrize('expected_mask', [
        (np.array(
            [[5., 4., 3., 2., 0.],
             [1., 0.5, 0.125, 0.25, 0.]]
        ))])
    def test_radial_profile(self, diffraction_pattern_for_radial, expected,
                            mask_for_radial, expected_mask):
        rp = diffraction_pattern_for_radial.get_radial_profile()
        rp_mask = diffraction_pattern_for_radial.get_radial_profile(
            mask_array=mask_for_radial)
        assert np.allclose(rp.data, expected, atol=1e-3)
        assert np.allclose(rp_mask.data, expected_mask, atol=1e-3)


class TestBackgroundMethods:

    @pytest.mark.parametrize('method, kwargs', [
        ('h-dome', {'h': 1, }),
        ('gaussian_difference', {'sigma_min': 0.5, 'sigma_max': 1, }),
        ('median', {'footprint': 4, }),
        ('median', {'footprint': 4, 'implementation': 'skimage'}),
        ('reference_pattern', {'bg': np.ones((8, 8)), })
    ])
    # skimage being warned by numpy, not for us
    @pytest.mark.filterwarnings('ignore::FutureWarning')
    # we don't care about precision loss here
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_remove_background(self, diffraction_pattern,
                               method, kwargs):
        bgr = diffraction_pattern.remove_background(method=method, **kwargs)
        assert bgr.data.shape == diffraction_pattern.data.shape
        assert bgr.max() <= diffraction_pattern.max()


class TestPeakFinding:
    # This is assertion free testing

    @pytest.fixture
    def ragged_peak(self):
        """
        A small selection of peaks in an ElectronDiffraction, to allow flexibilty
        of test building here.
        """
        pattern = np.zeros((2, 2, 128, 128))
        pattern[:, :, 40:42, 45] = 1
        pattern[:, :, 110, 30:32] = 1
        pattern[1, 0, 71:73, 21:23] = 1
        dp = ElectronDiffraction(pattern)
        dp.set_diffraction_calibration(1)
        return dp

    methods = ['zaefferer', 'laplacian_of_gaussians', 'difference_of_gaussians', 'stat', 'xc']

    @pytest.mark.parametrize('method', methods)
    # skimage internals
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_findpeaks_ragged(self, ragged_peak, method):
        if method == 'xc':
            disc = np.ones((2, 2))
            output = ragged_peak.find_peaks(method='xc',
                                            disc_image=disc,
                                            min_distance=3)
        else:
            output = ragged_peak.find_peaks(method=method,
                                            show_progressbar=False)


class TestsAssertionless:
    def test_decomposition(self, diffraction_pattern):
        storage = diffraction_pattern.decomposition()

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    # we don't want to use xc in this bit
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_find_peaks_interactive(self, diffraction_pattern):
        from matplotlib import pyplot as plt
        plt.ion()  # to make plotting non-blocking
        diffraction_pattern.find_peaks_interactive()
        plt.close('all')


@pytest.mark.xfail(raises=NotImplementedError)
class TestNotImplemented():
    def test_failing_run(self, diffraction_pattern):
        diffraction_pattern.find_peaks(method='no_such_method_exists')

    def test_remove_dead_pixels_failing(self, diffraction_pattern):
        dpr = diffraction_pattern.remove_deadpixels(
            [[1, 2], [5, 6]], 'fake_method', inplace=False, progress_bar=False)

    def test_remove_background_fake_method(self, diffraction_pattern):
        bgr = diffraction_pattern.remove_background(method='fake_method')

    def test_remove_background_fake_implementation(self, diffraction_pattern):
        bgr = diffraction_pattern.remove_background(
            method='median', implementation='fake_implementation')

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
               [0., 0., 0., 0., 0., 0., 0., 0.]]]),
])
def diffraction_pattern(request):
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


def test_reproject_as_polar(diffraction_pattern: ElectronDiffraction):
    shape_cartesian = diffraction_pattern.axes_manager.signal_shape
    diffraction_pattern.reproject_as_polar()
    assert isinstance(diffraction_pattern, Signal2D)
    shape_polar = diffraction_pattern.axes_manager.signal_shape
    assert shape_polar[0] == max(shape_cartesian)
    assert shape_polar[1] > np.sqrt(2) * shape_cartesian[0] / 2


def test_get_diffraction_variance(diffraction_pattern: ElectronDiffraction):
    dv = diffraction_pattern.get_diffraction_variance()
    assert dv.axes_manager.navigation_shape == (3,)
    assert dv.axes_manager.signal_shape == diffraction_pattern.axes_manager.signal_shape


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

    @pytest.mark.parametrize('closing, opening, mask_expected', [
        (False, False, np.array([True, True, False, True])),
        (True, False, np.array([True, True, True, True])),
        (False, True, np.array([True, True, False, False])),
    ])
    def test_get_vacuum_mask(self, diffraction_pattern, closing, opening, mask_expected):
        mask_calculated = diffraction_pattern.get_vacuum_mask(
            radius=3, threshold=1, closing=closing, opening=opening)
        assert np.allclose(mask_calculated, mask_expected)


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

    @pytest.mark.parametrize('saturation_radius', [
        1,
        2,
    ])
    def test_get_background_model(
            self, diffraction_pattern: ElectronDiffraction, saturation_radius):
        bgm = diffraction_pattern.get_background_model(saturation_radius)
        assert bgm.axes_manager.signal_shape == diffraction_pattern.axes_manager.signal_shape

    @pytest.mark.parametrize('method, kwargs', [
        ('h-dome', {'h': 1}),
        ('model', {'saturation_radius': 2, }),
        ('gaussian_difference', {'sigma_min': 0.5, 'sigma_max': 1, }),
        ('median', {'footprint': 4, })
    ])
    def test_remove_background(self, diffraction_pattern: ElectronDiffraction,
                               method, kwargs):
        bgr = diffraction_pattern.remove_background(method=method, **kwargs)
        assert bgr.data.shape == diffraction_pattern.data.shape
        assert bgr.max() <= diffraction_pattern.max()

# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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


from pyxem.utils import indexation_utils as iutls
from pyxem.utils import polar_transform_utils as ptutls
import numpy as np
import pytest
from unittest.mock import Mock
import dask.array as da
import sys
from pyxem.utils.cuda_utils import is_cupy_array


try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False
    cp = np

skip_cupy = pytest.mark.skipif(not CUPY_INSTALLED, reason="cupy is required")


@pytest.fixture()
def simulations():
    mock_sim_1 = Mock()
    mock_sim_1.calibrated_coordinates = np.array(
        [
            [3, 4, 0],  # 5
            [5, 12, 0],  # 13
            [8, 15, 0],  # 17
            [-8, 15, 0],  # 17
        ]
    )
    mock_sim_1.intensities = np.array([2, 3, 4, 2])
    mock_sim_2 = Mock()
    mock_sim_2.calibrated_coordinates = np.array(
        [[5, 12, 0], [8, 15, 0], [7, 24, 0]]  # 13  # 17  # 25
    )
    mock_sim_2.intensities = np.array([1, 2, 10])
    mock_sim_3 = Mock()
    mock_sim_3.calibrated_coordinates = np.array(
        [
            [3, 4, 0],  # 5
            [5, 12, 0],  # 13
        ]
    )
    mock_sim_3.intensities = np.array([8, 1])
    simlist = [mock_sim_1, mock_sim_2, mock_sim_3]
    return simlist


@pytest.fixture()
def mock_sim():
    xx = np.linspace(-10, 11, 5)
    yy = np.linspace(-10, 11, 3)
    XX, YY = np.meshgrid(xx, yy)
    Z = np.zeros(XX.ravel().shape)
    mock_sim = Mock()
    mock_sim.calibrated_coordinates = np.vstack([XX.ravel(), YY.ravel(), Z]).T
    mock_sim.intensities = np.ones(Z.shape)
    return mock_sim


@pytest.mark.parametrize(
    "rot, dr, dt, nim, nt",
    [
        (20, 1, 1, True, True),
        (50, 0.1, 1, False, True),
        (180, 0.1, 0.1, True, False),
        (270, 0.1, 0.1, False, False),
    ],
)
def test_match_image_to_template(mock_sim, rot, dr, dt, nim, nt):
    # serves to actually validate the correctness
    x, y, i = ptutls.get_template_cartesian_coordinates(
        mock_sim, in_plane_angle=rot, center=(20, 20)
    )
    # generate an image from the template
    tim = np.zeros((40, 40))
    tim[np.rint(y).astype(np.int32), np.rint(x).astype(np.int32)] = 1.0
    # compare image and template and retrieve the same angle
    a, c = iutls.get_in_plane_rotation_correlation(
        tim,
        mock_sim,
        find_direct_beam=False,
        delta_r=dr,
        delta_theta=dt,
        normalize_image=nim,
        normalize_template=nt,
    )
    assert abs(a[np.argmax(c)] - rot) % 180 < 2.0


@pytest.mark.parametrize(
    "rot, dr, dt, nim, nt",
    [
        (20, 1, 1, True, True),
        (50, 0.1, 1, False, True),
        (180, 0.1, 0.1, True, False),
        (270, 0.1, 0.1, False, False),
    ],
)
@skip_cupy
def test_match_image_to_template_gpu(mock_sim, rot, dr, dt, nim, nt):
    # serves to actually validate the correctness
    x, y, i = ptutls.get_template_cartesian_coordinates(
        mock_sim, in_plane_angle=rot, center=(20, 20)
    )
    # generate an image from the template
    tim = cp.zeros((40, 40))
    tim[np.rint(y).astype(np.int32), np.rint(x).astype(np.int32)] = 1.0
    # compare image and template and retrieve the same angle
    a, c = iutls.get_in_plane_rotation_correlation(
        tim,
        mock_sim,
        find_direct_beam=False,
        delta_r=dr,
        delta_theta=dt,
        normalize_image=nim,
        normalize_template=nt,
    )
    assert abs(a[cp.argmax(c)] - rot) % 180 < 2.0


@pytest.mark.parametrize(
    "max_radius, expected_shapes, test_pos, test_int",
    [
        (None, ((3, 2, 4), (3, 4)), np.array((-8, 15)), 2),
        (15, ((3, 2, 4), (3, 4)), np.array((0, 0)), 0),
    ],
)
def test_simulations_to_arrays(
    simulations, max_radius, expected_shapes, test_pos, test_int
):
    positions, intensities = iutls._simulations_to_arrays(
        simulations, max_radius=max_radius
    )
    np.testing.assert_array_equal(positions.shape, expected_shapes[0])
    np.testing.assert_array_equal(intensities.shape, expected_shapes[1])
    np.testing.assert_array_equal(positions[0, :, -1], test_pos)
    np.testing.assert_array_equal(intensities[0, -1], test_int)


def test_match_polar_to_polar_template():
    image = np.ones((123, 50))
    r = np.linspace(2, 40, 30, dtype=np.int32)
    theta = np.linspace(10, 110, 30, dtype=np.int32)
    intensities = np.ones(30, dtype=np.float64)
    cor = iutls._match_polar_to_polar_template(image, r, theta, intensities)
    np.testing.assert_array_almost_equal(cor, np.ones(123, dtype=np.float64) * 30)


@skip_cupy
def test_match_polar_to_polar_template_gpu():
    image = cp.ones((123, 50))
    r = cp.linspace(2, 40, 30, dtype=np.int32)
    theta = cp.linspace(10, 110, 30, dtype=np.int32)
    intensities = cp.ones(30, dtype=np.float64)
    cor = iutls._match_polar_to_polar_template(image, r, theta, intensities)
    cp.testing.assert_array_almost_equal(cor, cp.ones(123, dtype=np.float64) * 30)


@pytest.mark.parametrize(
    "itf, norim, nortemp",
    [
        (None, True, False),
        (np.sqrt, False, True),
    ],
)
def test_get_in_plane_rotation_correlation(simulations, itf, norim, nortemp):
    image = np.ones((123, 50))
    simulation = simulations[0]
    ang, cor = iutls.get_in_plane_rotation_correlation(
        image,
        simulation,
        intensity_transform_function=itf,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_template=nortemp,
    )
    assert ang.shape[0] == 277
    assert cor.shape[0] == 277


@skip_cupy
@pytest.mark.parametrize(
    "itf, norim, nortemp",
    [
        (None, True, False),
        (cp.sqrt, False, True),
    ],
)
def test_get_in_plane_rotation_correlation_gpu(simulations, itf, norim, nortemp):
    image = cp.ones((123, 50))
    simulation = simulations[0]
    ang, cor = iutls.get_in_plane_rotation_correlation(
        image,
        simulation,
        intensity_transform_function=itf,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_template=nortemp,
    )
    assert ang.shape[0] == 277
    assert cor.shape[0] == 277


@pytest.mark.parametrize(
    "N, n_keep, frac_keep, res",
    [
        (100, None, None, 100),
        (100, None, 0.392, 39),
        (100, 20, None, 20),
        (100, 20, 0.5, 20),
    ],
)
def test_get_max_n(N, n_keep, frac_keep, res):
    np.testing.assert_almost_equal(res, iutls._get_max_n(N, n_keep, frac_keep))


@pytest.mark.parametrize(
    "dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
def test_match_polar_to_polar_library(dtype):
    image = np.ones((123, 50), dtype=dtype)
    r = np.linspace(2, 40, 30, dtype=np.int32)
    r = np.repeat(r[np.newaxis, ...], 4, axis=0)
    theta = np.linspace(10, 110, 30, dtype=np.int32)
    theta = np.repeat(theta[np.newaxis, ...], 4, axis=0)
    intensities = np.ones(30, dtype=dtype)
    intensities = np.repeat(intensities[np.newaxis, ...], 4, axis=0)
    angles, cor, angles_m, cor_m = iutls._get_full_correlations(
        image,
        r,
        theta,
        intensities,
    )
    assert cor.shape[0] == 4
    assert angles.shape[0] == 4
    assert cor_m.shape[0] == 4
    assert angles_m.shape[0] == 4
    assert cor.dtype == dtype
    assert angles.dtype == np.int32
    assert cor_m.dtype == dtype
    assert angles_m.dtype == np.int32


@pytest.mark.parametrize(
    "norim, nortemp",
    [
        (True, False),
        (False, True),
    ],
)
def test_correlate_library_to_pattern(simulations, norim, nortemp):
    image = np.ones((123, 50))
    index, ang, cor, ang_m, cor_m = iutls.correlate_library_to_pattern(
        image,
        simulations,
        delta_r=2.6,
        delta_theta=1.3,
        frac_keep=1.0,
        normalize_image=norim,
        normalize_templates=nortemp,
    )
    assert ang.shape[0] == 3
    assert cor.shape[0] == 3
    assert ang_m.shape[0] == 3
    assert cor_m.shape[0] == 3
    assert index.dtype == np.int32
    assert ang.dtype == image.dtype
    assert cor.dtype == image.dtype
    assert ang_m.dtype == image.dtype
    assert cor_m.dtype == image.dtype


@pytest.mark.parametrize(
    "normed",
    [True, False],
)
def test_get_integrated_templates(normed):
    rmax = 50
    r = np.linspace(2, 40, 30, dtype=np.uint64)
    r = np.repeat(r[np.newaxis, ...], 4, axis=0)
    intensities = np.ones(30, dtype=np.float64)
    intensities = np.repeat(intensities[np.newaxis, ...], 4, axis=0)
    integrated = iutls._get_integrated_polar_templates(rmax, r, intensities, normed)
    assert integrated.shape[0] == 4
    assert integrated.shape[1] == 50


# @pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_match_library_to_polar_fast():
    polar_sum = np.ones(50, dtype=np.float64)
    integrated_temp = np.arange(0, 50, dtype=np.float64)
    integrated_temp = np.repeat(integrated_temp[np.newaxis, ...], 5, axis=0)
    coors = iutls._match_library_to_polar_fast(
        polar_sum,
        integrated_temp,
    )
    assert coors.shape[0] == 5
    assert coors.ndim == 1


# @pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
@pytest.mark.parametrize(
    "norim, nortemp",
    [
        (True, False),
        (False, True),
    ],
)
def test_correlate_library_to_pattern_fast(simulations, norim, nortemp):
    image = np.ones((123, 50))
    cor = iutls.correlate_library_to_pattern_fast(
        image,
        simulations,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_templates=nortemp,
    )
    assert cor.shape[0] == 3


@pytest.mark.parametrize(
    "norim, nortemp",
    [
        (True, False),
        (False, True),
    ],
)
@skip_cupy
def test_correlate_library_to_pattern_fast_gpu(simulations, norim, nortemp):
    image = cp.ones((123, 50))
    cor = iutls.correlate_library_to_pattern_fast(
        image,
        simulations,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_templates=nortemp,
    )
    assert cor.shape[0] == 3
    assert is_cupy_array(cor)


# @pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
@pytest.mark.parametrize("intensity_transform_function", [np.sqrt, None])
def test_prep_image_and_templates(simulations, intensity_transform_function):
    image = np.ones((123, 51))
    pim, r, t, i = iutls._prepare_image_and_templates(
        image,
        simulations,
        2.1,
        3.2,
        33.3,
        intensity_transform_function,
        True,
        None,
        True,
        True,
    )
    assert pim.shape[0] == 112
    assert pim.shape[1] == 16
    assert r.max() <= 15
    assert t.min() >= 0
    assert t.max() < 112
    assert i.shape[0] == 3
    assert i.shape[1] == 4


@skip_cupy
@pytest.mark.parametrize("intensity_transform_function", [cp.sqrt, None])
def test_prep_image_and_templates_gpu(simulations, intensity_transform_function):
    image = cp.ones((123, 51))
    pim, r, t, i = iutls._prepare_image_and_templates(
        image,
        simulations,
        2.1,
        3.2,
        33.3,
        intensity_transform_function,
        True,
        None,
        True,
        True,
    )
    assert pim.shape[0] == 112
    assert pim.shape[1] == 16
    assert r.max() <= 15
    assert t.min() >= 0
    assert t.max() < 112
    assert i.shape[0] == 3
    assert i.shape[1] == 4


# @pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
@pytest.mark.parametrize(
    "nbest",
    [1, 2],
)
@pytest.mark.slow
def test_mixed_matching_lib_to_polar(nbest):
    image = np.ones((123, 51), dtype=np.float64)
    polar_norm = 1.0
    polar_sum = np.sum(image, axis=0)
    polar_sum_norm = 2.3
    integrated_templates = np.ones((5, 51), dtype=np.float64)
    integrated_template_norms = np.sum(integrated_templates, axis=1)
    r = np.linspace(2, 40, 30, dtype=np.int64)
    r = np.repeat(r[np.newaxis, ...], 5, axis=0)
    theta = np.linspace(10, 110, 30, dtype=np.int64)
    theta = np.repeat(theta[np.newaxis, ...], 5, axis=0)
    intensities = np.ones(30, dtype=np.float64)
    intensities = np.repeat(intensities[np.newaxis, ...], 5, axis=0)
    template_norms = np.ones(5, dtype=np.float64) * 3
    fraction = 0.6
    answer = iutls._mixed_matching_lib_to_polar(
        image,
        integrated_templates,
        r,
        theta,
        intensities,
        None,
        1.0,
        nbest,
    )
    assert answer.shape[0] == nbest
    assert answer.shape[1] == 4


# @pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
@pytest.mark.parametrize(
    "nbest, frk, norim, nortemp",
    [
        (3, None, True, False),
        (1, 0.5, False, True),
    ],
)
@pytest.mark.slow
def test_get_n_best_matches(simulations, nbest, frk, norim, nortemp):
    image = np.ones((123, 50))
    indx, angs, cor, signs = iutls.get_n_best_matches(
        image,
        simulations,
        n_best=nbest,
        frac_keep=frk,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_templates=nortemp,
    )
    assert cor.shape[0] == indx.shape[0] == angs.shape[0] == nbest


@skip_cupy
@pytest.mark.parametrize(
    "nbest, frk, norim, nortemp",
    [
        (3, None, True, False),
        (1, 0.5, False, True),
    ],
)
@pytest.mark.slow
def test_get_n_best_matches_gpu(simulations, nbest, frk, norim, nortemp):
    image = cp.ones((123, 50))
    indx, angs, cor, signs = iutls.get_n_best_matches(
        image,
        simulations,
        n_best=nbest,
        frac_keep=frk,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_templates=nortemp,
    )
    assert cor.shape[0] == indx.shape[0] == angs.shape[0] == nbest


def create_dataset(shape):
    dataset = Mock()
    data = np.ones(shape)
    dataset.data = da.from_array(data)
    dataset._lazy = True
    dataset.axes_manager.navigation_dimension = len(shape) - 2
    dataset.axes_manager.signal_indices_in_array = (len(shape) - 2, len(shape) - 1)
    return dataset


@pytest.fixture()
def library():
    library = {}
    mock_sim_1 = Mock()
    mock_sim_1.calibrated_coordinates = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [1, 2, 0],
            [-1, 2, 0],
        ]
    )
    mock_sim_1.intensities = np.array([2, 3, 4, 2])
    mock_sim_2 = Mock()
    mock_sim_2.calibrated_coordinates = np.array([[-1, -2, 0], [1, 2, 0], [-2, 1, 0]])
    mock_sim_2.intensities = np.array([1, 2, 10])
    simlist = [mock_sim_1, mock_sim_2]
    orientations = np.array(
        [
            [1, 2, 3],
            [3, 4, 5],
        ]
    )
    library["dummyphase"] = {"simulations": simlist, "orientations": orientations}
    return library


# @pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
@pytest.mark.parametrize(
    "sigdim, n_best, frac_keep, norim, nort, chu",
    [
        ((2, 3), 1, 0.5, True, False, None),
        ((2, 3, 4), 1, 0.5, False, True, "auto"),
        ((2, 3, 4, 5), 2, 0.5, False, False, {0: "auto", 1: "auto", 2: None, 3: None}),
        ((2, 3, 4, 5), 9, 0.5, False, False, {0: "auto", 1: "auto", 2: None, 3: None}),
    ],
)
@pytest.mark.slow
def test_index_dataset_with_template_rotation(
    library, sigdim, n_best, frac_keep, norim, nort, chu
):
    signal = create_dataset(sigdim)
    result = iutls.index_dataset_with_template_rotation(
        signal,
        library,
        n_best=n_best,
        frac_keep=frac_keep,
        delta_r=0.5,
        delta_theta=36,
        max_r=None,
        intensity_transform_function=np.sqrt,
        normalize_images=norim,
        normalize_templates=nort,
        chunks=chu,
    )


# @pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
@pytest.mark.slow
def test_fail_index_dataset_with_template_rot(library):
    signal = create_dataset((3, 2, 4, 1, 2))
    with pytest.raises(ValueError):
        result = iutls.index_dataset_with_template_rotation(
            signal,
            library,
        )


@pytest.mark.parametrize(
    "sigdim, maxr, n_best, frac_keep, norim, nort, chu",
    [
        (
            (2, 3, 4, 5),
            None,
            1,
            1,
            False,
            False,
            {0: "auto", 1: "auto", 2: None, 3: None},
        ),
        ((2, 3, 4, 5), 3, 2, 1, False, False, {0: "auto", 1: "auto", 2: None, 3: None}),
        (
            (2, 3, 4, 5),
            9,
            3,
            0.5,
            False,
            False,
            {0: "auto", 1: "auto", 2: None, 3: None},
        ),
    ],
)
@pytest.mark.slow
@skip_cupy
def test_index_dataset_with_template_rotation_gpu(
    library, sigdim, maxr, n_best, frac_keep, norim, nort, chu
):
    signal = create_dataset(sigdim)
    result, dicto = iutls.index_dataset_with_template_rotation(
        signal,
        library,
        n_best=n_best,
        frac_keep=frac_keep,
        delta_r=0.5,
        delta_theta=36,
        max_r=maxr,
        intensity_transform_function=np.sqrt,
        normalize_images=norim,
        normalize_templates=nort,
        chunks=chu,
        target="gpu",
    )


def run_index_chunk(di, n_best, frac_keep):
    max_r = 4
    max_theta = 5
    di.random.seed(1001)
    data = di.random.random((2, 3, 10, 12))
    lib_n = 4
    lib_spots = 5
    r = di.random.randint(0, max_r, size=(lib_n, lib_spots))
    theta = di.random.randint(0, max_theta, size=(lib_n, lib_spots))
    intensities = di.random.random((lib_n, lib_spots))
    integrated_templates = di.random.random((lib_n, max_r))
    center = (7, 4)
    max_radius = max_r
    precision = np.float32
    answer = iutls._index_chunk(
        data,
        center,
        max_radius,
        (5, 4),
        precision,
        integrated_templates,
        r,
        theta,
        intensities,
        None,
        frac_keep,
        n_best,
        True,
    )
    assert answer.shape == (2, 3, n_best, 4)


@pytest.mark.parametrize(
    "n_best, fraction",
    [
        (1, 1),
        (2, 0),
        (1, 0.5),
    ],
)
def test_index_chunk(n_best, fraction):
    run_index_chunk(np, n_best, fraction)


@pytest.mark.parametrize(
    "n_best, fraction",
    [
        (1, 1),
        (2, 0),
        (1, 0.5),
    ],
)
@skip_cupy
def test_index_chunk_gpu(n_best, fraction):
    run_index_chunk(cp, n_best, fraction)

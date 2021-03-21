from pyxem.utils import indexation_utils as iutls
import numpy as np
import pytest
from unittest.mock import Mock
import dask.array as da
import sys

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

@pytest.mark.parametrize(
    "max_radius, expected_shapes, test_pos, test_int",
    [
        (None, ((3, 2, 4), (3, 4)), np.array((-8, 15)), 2),
        (15, ((3, 2, 4), (3, 4)), np.array((0, 0)), 0),
    ],
)
def test_simulations_to_arrays(simulations,max_radius, expected_shapes, test_pos, test_int):
    positions, intensities = iutls._simulations_to_arrays(
        simulations, max_radius=max_radius
    )
    np.testing.assert_array_equal(positions.shape, expected_shapes[0])
    np.testing.assert_array_equal(intensities.shape, expected_shapes[1])
    np.testing.assert_array_equal(positions[0, :, -1], test_pos)
    np.testing.assert_array_equal(intensities[0, -1], test_int)

def test_match_polar_to_polar_template():
    image = np.ones((123, 50))
    r = np.linspace(2, 40, 30, dtype=np.uint64)
    theta = np.linspace(10, 110, 30, dtype=np.uint64)
    intensities = np.ones(30, dtype=np.float64)
    image_norm = 1.0
    template_norm = 1.0
    cor = iutls._match_polar_to_polar_template(
        image, r, theta, intensities, image_norm, template_norm
    )
    np.testing.assert_array_almost_equal(cor, np.ones(123, dtype=np.float64) * 30)


@pytest.mark.parametrize(
    "itf, norim, nortemp",
    [
        (None, True, False),
        (np.sqrt, False, True),
    ],
)
def test_get_in_plane_rotation_correlation(simulations,itf, norim, nortemp):
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


def test_match_polar_to_polar_library():
    image = np.ones((123, 50))
    r = np.linspace(2, 40, 30, dtype=np.int64)
    r = np.repeat(r[np.newaxis, ...], 4, axis=0)
    theta = np.linspace(10, 110, 30, dtype=np.int64)
    theta = np.repeat(theta[np.newaxis, ...], 4, axis=0)
    intensities = np.ones(30, dtype=np.float64)
    intensities = np.repeat(intensities[np.newaxis, ...], 4, axis=0)
    image_norm = 1.0
    template_norms = np.ones(4, dtype=np.float64)
    cor, angles, cor_m, angles_m = iutls._match_polar_to_polar_library(
        image, r, theta, intensities, image_norm, template_norms
    )
    assert cor.shape[0] == 4
    assert angles.shape[0] == 4
    assert cor_m.shape[0] == 4
    assert angles_m.shape[0] == 4


@pytest.mark.parametrize(
    "norim, nortemp",
    [
        (True, False),
        (False, True),
    ],
)
def test_correlate_library_to_pattern(simulations,norim, nortemp):
    image = np.ones((123, 50))
    ang, cor, ang_m, cor_m = iutls.correlate_library_to_pattern(
        image,
        simulations,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_templates=nortemp,
    )
    assert ang.shape[0] == 3
    assert cor.shape[0] == 3
    assert ang_m.shape[0] == 3
    assert cor_m.shape[0] == 3


def test_correlation_at_angle():
    image = np.ones((123, 50))
    r = np.linspace(2, 40, 30, dtype=np.uint64)
    r = np.repeat(r[np.newaxis, ...], 4, axis=0)
    theta = np.linspace(10, 110, 30, dtype=np.uint64)
    theta = np.repeat(theta[np.newaxis, ...], 4, axis=0)
    intensities = np.ones(30, dtype=np.float64)
    intensities = np.repeat(intensities[np.newaxis, ...], 4, axis=0)
    angles = np.empty(4, dtype=np.uint64)
    angles.fill(40)
    image_norm = 1.0
    template_norms = np.ones(4, dtype=np.float64)
    cor = iutls._get_correlation_at_angle(
        image, r, theta, intensities, angles, image_norm, template_norms
    )
    assert cor.shape[0] == 4


def test_get_integrated_templates():
    rmax = 50
    r = np.linspace(2, 40, 30, dtype=np.uint64)
    r = np.repeat(r[np.newaxis, ...], 4, axis=0)
    intensities = np.ones(30, dtype=np.float64)
    intensities = np.repeat(intensities[np.newaxis, ...], 4, axis=0)
    integrated = iutls._get_integrated_polar_templates(rmax, r, intensities)
    assert integrated.shape[0] == 4
    assert integrated.shape[1] == 50

@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_match_library_to_polar_fast():
    polar_sum = np.ones(50, dtype=np.float64)
    integrated_temp = np.arange(0, 50, dtype=np.float64)
    integrated_temp = np.repeat(integrated_temp[np.newaxis, ...], 5, axis=0)
    template_norms = np.ones(5, dtype=np.float64)
    polar_norm = 1
    coors = iutls._match_library_to_polar_fast(
        polar_sum, integrated_temp, polar_norm, template_norms
    )
    assert coors.shape[0] == 5
    assert coors.ndim == 1


@pytest.mark.parametrize(
    "norim, nortemp",
    [
        (True, False),
        (False, True),
    ],
)
@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_correlate_library_to_pattern_fast(simulations,norim, nortemp):
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


@pytest.mark.parametrize("intensity_transform_function", [np.sqrt, None])
@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_prep_image_and_templates(simulations,intensity_transform_function):
    image = np.ones((123, 51))
    pim, r, t, i = iutls._prepare_image_and_templates(
        image, simulations, 2.1, 3.2, 33.3, intensity_transform_function, False, None
    )
    assert pim.shape[0] == 112
    assert pim.shape[1] == 15
    assert r.max() <= 15
    assert t.min() >= 0
    assert t.max() < 112
    assert i.shape[0] == 3
    assert i.shape[1] == 4


@pytest.mark.parametrize(
    "nbest",
    [1, 2],
)
@pytest.mark.slow
@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
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
        polar_norm,
        polar_sum,
        polar_sum_norm,
        integrated_templates,
        integrated_template_norms,
        r,
        theta,
        intensities,
        template_norms,
        fraction,
        nbest,
    )
    assert answer.shape[0] == nbest
    assert answer.shape[1] == 4


@pytest.mark.parametrize(
    "frk, norim, nortemp",
    [
        (None, True, False),
        (0.5, False, True),
    ],
)
@pytest.mark.slow
@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_correlate_library_to_pattern_partial(simulations,frk, norim, nortemp):
    image = np.ones((123, 50))
    indx, angs, cor, angs_m, cors_m = iutls.correlate_library_to_pattern_partial(
        image,
        simulations,
        frac_keep=frk,
        delta_r=2.6,
        delta_theta=1.3,
        normalize_image=norim,
        normalize_templates=nortemp,
    )
    assert cor.shape[0] == indx.shape[0] == angs.shape[0] == angs_m.shape[0] == cors_m.shape[0]


@pytest.mark.parametrize(
    "nbest, frk, norim, nortemp",
    [
        (3, None, True, False),
        (1, 0.5, False, True),
    ],
)
@pytest.mark.slow
@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_get_n_best_matches(simulations,nbest, frk, norim, nortemp):
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


@pytest.mark.parametrize(
    "sigdim, norim, nort, chu",
    [
        ((2, 3), True, False, None),
        ((2, 3, 4), False, True, "auto"),
        ((2, 3, 4, 5), False, False, {0: "auto", 1: "auto", 2: None, 3: None}),
    ],
)
@pytest.mark.slow
@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_index_dataset_with_template_rotation(library,sigdim, norim, nort, chu):
    signal = create_dataset(sigdim)
    result = iutls.index_dataset_with_template_rotation(
        signal,
        library,
        frac_keep=0.5,
        delta_r=0.5,
        delta_theta=36,
        max_r=None,
        intensity_transform_function=np.sqrt,
        normalize_images=norim,
        normalize_templates=nort,
        chunks=chu,
    )
@pytest.mark.slow
@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
def test_fail_index_dataset_with_template_rot(library):
    signal = create_dataset((3, 2, 4, 1, 2))
    with pytest.raises(ValueError):
        result = iutls.index_dataset_with_template_rotation(
        signal,
        library,
        )

from pyxem.utils import plotting_utils as plu
import matplotlib.pyplot as plt
import pytest
import numpy as np
from unittest.mock import Mock
from pyxem.signals import ElectronDiffraction2D

_, ax = plt.subplots()


@pytest.fixture()
def mock_simulation():
    mock_sim = Mock()
    mock_sim.calibrated_coordinates = np.array(
        [[3, 4, 0], [5, 12, 0], [8, 15, 0], [7, 24, 0]]  # 5  # 13  # 17
    )  # 25
    mock_sim.intensities = np.array([1, 2, 3, 4])
    return mock_sim


@pytest.mark.parametrize(
    "axis, find_direct_beam, direct_beam_position, coordinate_system",
    [
        (None, False, None, "polar"),
        (ax, True, None, "cartesian"),
        (ax, False, (3, 5), "cartesian"),
    ],
)
def test_plot_sim_over_pattern(
    mock_simulation, axis, find_direct_beam, direct_beam_position, coordinate_system
):
    pattern = np.ones((15, 20))
    plu.plot_template_over_pattern(
        pattern,
        mock_simulation,
        axis,
        find_direct_beam=find_direct_beam,
        direct_beam_position=direct_beam_position,
        coordinate_system=coordinate_system,
    )


def test_plot_sim_over_pattern_fail(mock_simulation):
    pattern = np.ones((15, 20))
    with pytest.raises(NotImplementedError):
        plu.plot_template_over_pattern(
            pattern, mock_simulation, coordinate_system="dracula"
        )


@pytest.fixture
def shape():
    return (4, 4, 4, 4)


@pytest.fixture
def mock_electron_diffraction_data(shape):
    data = np.arange(np.prod(shape)).reshape(*shape)
    edd = ElectronDiffraction2D(data)
    edd.set_diffraction_calibration(0.1)
    return edd


def create_mock_templatematching_results(shape, n_best):
    shape = shape[2:] + (n_best,)
    return {
        "phase_index": np.zeros(shape, dtype=int),
        "template_index": np.ones(shape, dtype=int),
        "orientation": np.zeros((*shape, 3), dtype=float),
        "correlation": np.ones(shape, dtype=float),
        "mirrored_template": np.zeros(shape, dtype=bool),
    }


@pytest.fixture
def mock_phase_key_dict():
    return {0: "dummyphase"}


@pytest.fixture()
def mock_library(mock_phase_key_dict):
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
    library[mock_phase_key_dict[0]] = {
        "simulations": simlist,
        "orientations": orientations,
    }
    return library


@pytest.mark.parametrize(
    "n_best_sim, n_best, find_direct_beam, direct_beam_position, marker_colors",
    [
        (1, 1, False, None, None),
        (2, 2, False, None, None),
        (2, 1, False, None, None),
        (1, 1, False, (0, 0), None),
        (1, 1, False, (1, 0), None),
        (1, 1, True, None, None),
        (2, 2, False, None, ["red"]), # Prints a warning, but is still passes as colors will just loop
        (2, 1, False, None, ["red"]*4),
        (2, 2, False, None, ["red"]*2),
    ],
)
@pytest.mark.slow
def test_plot_templates_over_signal(
    mock_electron_diffraction_data,
    mock_library,
    shape,
    mock_phase_key_dict,
    n_best_sim,
    n_best,
    find_direct_beam,
    direct_beam_position,
    marker_colors,
):
    mock_result = create_mock_templatematching_results(shape, n_best_sim)
    plu.plot_templates_over_signal(
        mock_electron_diffraction_data,
        mock_library,
        mock_result,
        mock_phase_key_dict,
        n_best,
        find_direct_beam,
        direct_beam_position,
        marker_colors,
        verbose=False,
    )

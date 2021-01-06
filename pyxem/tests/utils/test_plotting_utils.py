from pyxem.utils import plotting_utils as plu
import matplotlib.pyplot as plt
import pytest
import numpy as np
from unittest.mock import Mock

_, ax = plt.subplots()

def mock_simulation():
    mock_sim = Mock()
    mock_sim.calibrated_coordinates = np.array([[3, 4, 0],  # 5
                                                [5, 12, 0],  # 13
                                                [8, 15, 0],  # 17
                                                [7, 24, 0]])  # 25
    mock_sim.intensities = np.array([1, 2, 3, 4])
    return mock_sim


@pytest.mark.parametrize(
        "axis, find_direct_beam, direct_beam_position, coordinate_system",
        [
            (None, False, None, "polar"),
            (ax, True, None, "cartesian"),
            (ax, False, (3, 5), "cartesian"),
        ]
        )
def test_plot_sim_over_pattern(axis, find_direct_beam,
        direct_beam_position, coordinate_system):
    pattern = np.ones((15, 20))
    simulation = mock_simulation()
    plu.plot_template_over_pattern(pattern, simulation, axis, 
            find_direct_beam=find_direct_beam, direct_beam_position=direct_beam_position,
            coordinate_system=coordinate_system)


@pytest.mark.xfail(raises=NotImplementedError)
def test_plot_sim_over_pattern_fail():
    pattern = np.ones((15, 20))
    simulation = mock_simulation()
    plu.plot_template_over_pattern(pattern, simulation,
            coordinate_system="dracula")

from pyxem.utils import plotting_utils as plu
import matplotlib.pyplot as plt
import pytest
import numpy as np
from unittest.mock import Mock

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
        plu.plot_template_over_pattern(pattern, mock_simulation, coordinate_system="dracula")

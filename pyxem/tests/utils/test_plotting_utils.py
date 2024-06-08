from pyxem.utils import plotting as plu
from pyxem.signals import BeamShift
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
        plu.plot_template_over_pattern(
            pattern, mock_simulation, coordinate_system="dracula"
        )


def test_plot_beam_shift_color():
    s = BeamShift(np.random.random(size=(100, 100, 2)))
    plu.plot_beam_shift_color(s)
    plu.plot_beam_shift_color(
        s,
        phase_rotation=45,
        indicator_rotation=10,
        autolim=True,
        autolim_sigma=1,
        scalebar_size=10,
    )
    plu.plot_beam_shift_color(s, only_phase=True)
    plu.plot_beam_shift_color(s, autolim=False, magnitude_limits=(0, 0.5))
    fig, ax = plt.subplots()
    plu.plot_beam_shift_color(s, ax=ax)

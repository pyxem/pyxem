from pyxem.utils.subpixel_utils import (get_experimental_square,
                                        get_simulated_disc,_conventional_xc
                                        )
import numpy as np
import pytest

class TestSubpixelUtils:
    @pytest.fixture
    def gaussian_peak(self):
        x = np.arange(0, 10, 1, float)
        y = x[:, np.newaxis]
        fwhm = 3
        x0,y0 = (5,5)
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2

    def test_get_experimental_square(self):
        square = get_experimental_square(np.arange(25).reshape(5, 5), [2, 2], 4)
        assert square.shape == (4, 4)

    def test_conventional_xc(self, gaussian_peak):
        _conventional_xc(gaussian_peak, gaussian_peak)
        xc = _conventional_xc(x, y)
        assert xc == 0
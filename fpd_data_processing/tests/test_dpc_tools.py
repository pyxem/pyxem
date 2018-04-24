from pytest import approx
import numpy as np
import fpd_data_processing.dpc_tools as dpct


class TestBetaToBst:

    def test_zero(self):
        data = np.zeros((100, 100))
        bst = dpct.beta_to_bst(data, 200000)
        assert data.shape == bst.shape
        assert (data == 0.).all()

    def test_ones(self):
        data = np.ones((100, 100))*10
        bst = dpct.beta_to_bst(data, 200000)
        assert data.shape == bst.shape
        assert (data != 0.).all()

    def test_beta_to_bst_to_beta(self):
        beta = 2e-6
        output = dpct.bst_to_beta(dpct.beta_to_bst(beta, 200000), 200000)
        assert beta == output

    def test_known_value(self):
        # From https://dx.doi.org/10.1016/j.ultramic.2016.03.006
        bst = 10e-9 * 1  # 10 nm, 1 Tesla
        av = 200000  # 200 kV
        beta = dpct.bst_to_beta(bst, av)
        assert approx(beta, rel=1e-4) == 6.064e-6


class TestBstToBeta:

    def test_zero(self):
        data = np.zeros((100, 100))
        bst = dpct.bst_to_beta(data, 200000)
        assert data.shape == bst.shape
        assert (data == 0.).all()

    def test_ones(self):
        data = np.ones((100, 100))*10
        bst = dpct.bst_to_beta(data, 200000)
        assert data.shape == bst.shape
        assert (data != 0.).all()

    def test_bst_to_beta_to_bst(self):
        bst = 10e-6
        output = dpct.beta_to_bst(dpct.bst_to_beta(bst, 200000), 200000)
        assert bst == output

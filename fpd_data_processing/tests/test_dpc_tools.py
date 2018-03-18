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

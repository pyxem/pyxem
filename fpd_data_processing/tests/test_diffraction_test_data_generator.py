import unittest
import fpd_data_processing.diffraction_test_data_generator as TestData


class test_diffraction_test_data_genrator(unittest.TestCase):

    def test_init(self):
        test_data_1 = TestData.TestData()
        test_data_2 = TestData.TestData(size_x=1,size_y=10,scale=0.05)

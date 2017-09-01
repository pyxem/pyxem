import unittest
import fpd_data_processing.diffraction_test_data_generator as TestData


class test_diffraction_test_data_genrator(unittest.TestCase):

    def test_init(self):
        test_data_1 = TestData.TestData()
        test_data_2 = TestData.TestData(size_x=1,size_y=10,scale=0.05)
        
    def test_zero_signal(self):
        test_data_1 = TestData.TestData(default=False)
        self.assertTrue((test_data_1.signal.data == 0.).all())

        test_data_2 = TestData.TestData()
        test_data_2.set_signal_zero()
        self.assertTrue((test_data_2.signal.data == 0.).all())
        
    def test_large_disk(self):
        test_data_1 = TestData.TestData(size_x=10,size_y=10,scale=0.01,default=False)
        test_data_1.add_disk(x0=5,y0=5,r=20,I=100)
        self.assertTrue((test_data_1.signal.data > 0.).all())

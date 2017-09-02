import unittest
import fpd_data_processing.diffraction_test_data_generator as TestData
import numpy as np

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
        
    def test_radius(self):
        scale, r = 0.1, 2
        x0, y0 = 0, 0
        
        ring_1 = TestData.TestData(size_x=10, size_y=10, scale=scale, default=False, blur=False)
        ring_1.add_ring(x0=x0, y0=y0, r=r, I=10)
        slice_y = ring_1.signal.data[:,0]
        slice_x = ring_1.signal.data[0,:]
        r_x_idx = np.where(slice_x>0)[0][-1]
        r_y_idx = np.where(slice_y>0)[0][-1]
        self.assertTrue(r_x_idx == int(r/scale))
        self.assertTrue(r_y_idx == int(r/scale))
        
        x0, y0 = 10, 10
        ring_2 = TestData.TestData(size_x=10, size_y=10, scale=scale, default=False, blur=False)
        ring_2.add_ring(x0=x0, y0=y0, r=r, I=10)
        slice_y_2 = ring_2.signal.data[:,-1]
        slice_x_2 = ring_2.signal.data[-1,:]
        r_x_idx_2 = np.where(slice_x_2 == 10)[0][0]
        r_y_idx_2 = np.where(slice_y_2 == 10)[0][0]
        self.assertTrue(r_x_idx_2 == int((10-r)/scale))
        self.assertTrue(r_y_idx_2 == int((10-r)/scale))


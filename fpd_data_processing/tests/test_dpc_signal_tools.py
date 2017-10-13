import unittest
import numpy as np
import fpd_data_processing.pixelated_stem_tools as pst


class test_get_rgb_phase_magnitude_array(unittest.TestCase):

    def test_simple(self):
        phase = np.zeros((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        self.assertTrue((rgb_array == 0.).all())

    def test_magnitude_zero(self):
        phase = np.random.random((50, 50))
        magnitude = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        self.assertTrue((rgb_array == 0.).all())

    def test_all_same(self):
        phase = np.ones((50, 50))
        magnitude = np.ones((50, 50))
        rgb_array = pst._get_rgb_phase_magnitude_array(phase, magnitude)
        self.assertTrue((rgb_array == rgb_array[0][0]).all())


class test_get_rgb_phase_array(unittest.TestCase):

    def test_all_same0(self):
        phase = np.zeros((50, 50))
        rgb_array = pst._get_rgb_phase_array(phase)
        self.assertTrue((rgb_array == rgb_array[0][0]).all())

    def test_all_same1(self):
        phase = np.ones((50, 50))
        rgb_array = pst._get_rgb_phase_array(phase)
        self.assertTrue((rgb_array == rgb_array[0][0]).all())


class test_find_phase(unittest.TestCase):

    def test_simple(self):
        phase = np.zeros((50, 50))
        phase0 = pst._find_phase(phase)
        self.assertTrue((phase0 == 0.).all())
    
    def test_rotation(self):
        phase = np.zeros((50, 50))
        phase0 = pst._find_phase(phase, rotation=90)
        self.assertTrue((phase0 == np.pi/2).all())
        phase1 = pst._find_phase(phase, rotation=45)
        self.assertTrue((phase1 == np.pi/4).all())
        phase2 = pst._find_phase(phase, rotation=180)
        self.assertTrue((phase2 == np.pi).all())
        phase3 = pst._find_phase(phase, rotation=360)
        self.assertTrue((phase3 == 0).all())
        phase4 = pst._find_phase(phase, rotation=-90)
        self.assertTrue((phase4 == 3*np.pi/2).all())

    def test_max_phase(self):
        phase = (np.ones((50, 50))*np.pi*0.5) + np.pi
        phase0 = pst._find_phase(phase, max_phase=np.pi)
        self.assertTrue((phase0 == np.pi/2).all())

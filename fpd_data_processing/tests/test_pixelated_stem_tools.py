import unittest
import fpd_data_processing.pixelated_stem_tools as pst
import numpy as np
from hyperspy.signals import Signal2D


class test_pixelated_tools(unittest.TestCase):

    def test_find_longest_distance_manual(self):
        # These values are tested manually, against knowns results,
        # to make sure everything works fine.
        imX, imY = 10, 10
        centre_list = ((0, 0), (imX, 0), (0, imY), (imX, imY))
        distance = 14
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            self.assertEqual(dist, distance)

        centre_list = ((1, 1), (imX-1, 1), (1, imY-1), (imX-1, imY-1))
        distance = 12
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            self.assertEqual(dist, distance)

        imX, imY = 10, 5
        centre_list = ((0, 0), (imX, 0), (0, imY), (imX, imY))
        distance = 11
        for cX, cY in centre_list:
            dist = pst._find_longest_distance(imX, imY, cX, cY, cX, cY)
            self.assertEqual(dist, distance)

        imX, imY = 10, 10
        cX_min, cX_max, cY_min, cY_max = 1, 2, 2, 3
        distance = 12
        dist = pst._find_longest_distance(
                imX, imY, cX_min, cX_max, cY_min, cY_max)
        self.assertEqual(dist, distance)

    def test_find_longest_distance_all(self):
        imX, imY = 100, 100
        x_array, y_array = np.mgrid[0:10, 0:10]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int(((imX-x)**2+(imY-y)**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)

        x_array, y_array = np.mgrid[90:100, 90:100]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int((x**2+y**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)

        x_array, y_array = np.mgrid[0:10, 90:100]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int(((imX-x)**2+y**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)

        x_array, y_array = np.mgrid[90:100, 0:10]
        for x, y in zip(x_array.flatten(), y_array.flatten()):
            distance = int((x**2+(imY-y)**2)**0.5)
            dist = pst._find_longest_distance(
                    imX, imY, x, y, x, y)
            self.assertEqual(dist, distance)


class test_dpcsignal_tools(unittest.TestCase):

    def test_get_corner_value(self):
        corner_size = 0.1
        image_size = 100
        s = Signal2D(np.ones(shape=(image_size, image_size)))
        corner_list = pst._get_corner_value(s, corner_size=0.1)
        corner0, corner1 = corner_list[:, 0], corner_list[:, 1]
        corner2, corner3 = corner_list[:, 2], corner_list[:, 3]

        pos = image_size*corner_size*0.5
        high_value = s.axes_manager[0].high_value
        self.assertTrue(((pos, pos, 1) == corner0).all())
        self.assertTrue(((pos, high_value-pos, 1) == corner1).all())
        self.assertTrue(((high_value-pos, pos, 1) == corner2).all())
        self.assertTrue(((high_value-pos, high_value-pos, 1) == corner3).all())

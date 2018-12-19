import numpy as np
import pixstem.api as ps
import pixstem.marker_tools as mt


class TestGet4DMarkerList:

    def test_simple(self):
        peak_array = np.empty((2, 3), dtype=np.object)
        peak_array[0, 0] = [[2, 4]]
        peak_array[0, 1] = [[8, 2]]
        peak_array[0, 2] = [[1, 8]]
        peak_array[1, 0] = [[3, 1]]
        peak_array[1, 1] = [[9, 1]]
        peak_array[1, 2] = [[6, 3]]
        s = ps.PixelatedSTEM(np.zeros(shape=(2, 3, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
                peak_array, s.axes_manager.signal_axes, color='red')
        mt._add_permanent_markers_to_signal(s, marker_list)
        assert len(marker_list) == 1
        marker = marker_list[0]
        assert marker.marker_properties['color'] == 'red'
        s.plot()
        for iy, ix in np.ndindex(peak_array.shape[:2]):
            peak = peak_array[iy, ix]
            s.axes_manager.indices = (ix, iy)
            print(peak, (iy, ix), marker)
            assert marker.get_data_position('x1') == peak[0][1]
            assert marker.get_data_position('y1') == peak[0][0]

    def test_color(self):
        color = 'blue'
        peak_array = np.zeros(shape=(3, 2, 1, 2))
        s = ps.PixelatedSTEM(np.zeros(shape=(3, 2, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
                peak_array, s.axes_manager.signal_axes, color=color)
        assert marker_list[0].marker_properties['color'] == 'blue'

    def test_size(self):
        size = 12
        peak_array = np.zeros(shape=(3, 2, 1, 2))
        s = ps.PixelatedSTEM(np.zeros(shape=(3, 2, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
                peak_array, s.axes_manager.signal_axes, size=size)
        assert marker_list[0].get_data_position('size') == size

    def test_several_markers(self):
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = ps.PixelatedSTEM(np.zeros(shape=(3, 2, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
                peak_array, s.axes_manager.signal_axes)
        assert len(marker_list) == 3

    def test_several_markers_different_peak_array_size(self):
        peak_array = np.empty((2, 3), dtype=np.object)
        peak_array[0, 0] = [[2, 4], [1, 9]]
        peak_array[0, 1] = [[8, 2]]
        s = ps.PixelatedSTEM(np.zeros(shape=(2, 3, 10, 10)))
        marker_list = mt._get_4d_points_marker_list(
                peak_array, s.axes_manager.signal_axes, color='red')
        assert len(marker_list) == 2


class TestAddPeakArrayToSignalAsMarkers:

    def test_simple(self):
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = ps.PixelatedSTEM(np.zeros(shape=(3, 2, 10, 10)))
        mt.add_peak_array_to_signal_as_markers(s, peak_array)
        assert len(s.metadata.Markers) == 3

    def test_color(self):
        color = 'blue'
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = ps.PixelatedSTEM(np.zeros(shape=(3, 2, 10, 10)))
        mt.add_peak_array_to_signal_as_markers(s, peak_array, color=color)
        marker = list(s.metadata.Markers)[0][1]
        assert marker.marker_properties['color'] == color

    def test_size(self):
        size = 17
        peak_array = np.zeros(shape=(3, 2, 3, 2))
        s = ps.PixelatedSTEM(np.zeros(shape=(3, 2, 10, 10)))
        mt.add_peak_array_to_signal_as_markers(s, peak_array, size=size)
        marker = list(s.metadata.Markers)[0][1]
        assert marker.get_data_position('size') == size


def test_peak_finding_to_marker():
    data = np.zeros(shape=(3, 2, 10, 12))
    data[0, 0, 2, 7] = 1
    data[0, 1, 7, 3] = 1
    data[1, 0, 4, 6] = 1
    data[1, 1, 2, 3] = 1
    data[2, 0, 3, 6] = 1
    data[2, 1, 2, 2] = 1
    s = ps.PixelatedSTEM(data)
    peak_array = s.find_peaks(min_sigma=0.1, max_sigma=2,
                              threshold=0.01, lazy_result=False)
    marker_list = mt._get_4d_points_marker_list(
            peak_array, s.axes_manager.signal_axes)
    assert len(marker_list) == 1
    marker = marker_list[0]
    mt._add_permanent_markers_to_signal(s, marker_list)
    s.plot()
    for ix, iy in s.axes_manager:
        px, py = marker.get_data_position('x1'), marker.get_data_position('y1')
        value = s.inav[ix, iy].isig[int(px), int(py)].data[0]
        assert value == 1.

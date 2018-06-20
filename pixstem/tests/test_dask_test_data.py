import pixstem.dask_test_data as dtd


def test_get_hot_pixel_test_data_2d():
    data = dtd._get_hot_pixel_test_data_2d()
    assert len(data.shape) == 2


def test_get_hot_pixel_test_data_3d():
    data = dtd._get_hot_pixel_test_data_3d()
    assert len(data.shape) == 3


def test_get_hot_pixel_test_data_4d():
    data = dtd._get_hot_pixel_test_data_4d()
    assert len(data.shape) == 4


def test_get_dead_pixel_test_data_2d():
    data = dtd._get_dead_pixel_test_data_2d()
    assert len(data.shape) == 2


def test_get_dead_pixel_test_data_3d():
    data = dtd._get_dead_pixel_test_data_3d()
    assert len(data.shape) == 3


def test_get_dead_pixel_test_data_4d():
    data = dtd._get_dead_pixel_test_data_4d()
    assert len(data.shape) == 4

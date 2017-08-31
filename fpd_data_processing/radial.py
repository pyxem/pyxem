import numpy as np
from hyperspy._components.gaussian import Gaussian
from hyperspy.signals import Signal2D


def centre_comparison(
        s, centre_x_list, centre_y_list,
        crop_radial_signal=None, angleN=8):
    """
    Compare how the centre position affects the radial integration.
    Useful for finding on optimal centre position if one has some
    round feature.

    Currently only works with signals with 0 navigation dimensions.

    crop_radial_signal : tuple, optional
    """
    if s.axes_manager.navigation_dimension != 0:
        raise ValueError(
                "centre_comparison only works for pixelatedSTEM "
                "signals with 0 navigation dimensions")
    s_list = []
    for centre_x, centre_y in zip(centre_x_list, centre_y_list):
        s_angle = s.angular_slice_radial_integration(
                angleN=angleN,
                centre_x_array=np.array([centre_x]),
                centre_y_array=np.array([centre_y]))
        if crop_radial_signal is not None:
            s_angle = s_angle.isig[crop_radial_signal[0]:crop_radial_signal[1]]
        s_angle.metadata.add_node("Angle_slice_processing")
        s_angle.metadata.Angle_slice_processing.centre_x = centre_x
        s_angle.metadata.Angle_slice_processing.centre_y = centre_y
        s_list.append(s_angle)
    return(s_list)


def get_optimal_centre_position(
        s, centre_x_list, centre_y_list,
        radial_signal_span, angleN=8):

    s_list = centre_comparison(s, centre_x_list, centre_y_list,
            crop_radial_signal=radial_signal_span, angleN=angleN)

    m_list = []
    for temp_s in s_list:
        temp_s.change_dtype('float64')
        temp_s = temp_s - temp_s.data.min()
        temp_s = temp_s/temp_s.data.max()
        am = temp_s.axes_manager
        g_centre = temp_s.inav[0].data.argmax()*am[-1].scale+am[-1].offset
        g_sigma = 3
        g_A = temp_s.data.max()*2*g_sigma

        m = temp_s.create_model()
        g = Gaussian(
                A=g_A,
                centre=g_centre,
                sigma=g_sigma)
        m.append(g)
        m.multifit()
        m_list.append(m)
    s_centre_std_array = _get_offset_image(m_list)
    return(s_centre_std_array)


def _get_offset_image(model_list):
    cX_list, cY_list = [], []
    for model in model_list:
        cX_list.append(model.signal.metadata.Angle_slice_processing.centre_x)
        cY_list.append(model.signal.metadata.Angle_slice_processing.centre_y)
    cX_list, cY_list = np.array(cX_list), np.array(cY_list)
    centre_std_array = np.zeros((cY_list.max()-cY_list.min()+1, cX_list.max()-cX_list.min()+1))
    for model in model_list:
        md = model.signal.metadata
        x = md.Angle_slice_processing.centre_x-cX_list.min()
        y = md.Angle_slice_processing.centre_y-cY_list.min()
        centre_std_array[x, y] = model.components.Gaussian.centre.as_signal().data.std()
    s = Signal2D(centre_std_array)
    s.axes_manager[0].offset = cX_list.min()
    s.axes_manager[1].offset = cY_list.min()
    return(s)

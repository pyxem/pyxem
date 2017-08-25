import numpy as np

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
        s_list.append(s_angle)
    return(s_list)

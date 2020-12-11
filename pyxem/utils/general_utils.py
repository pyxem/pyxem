from skimage.filters import gaussian
import numpy as np


def get_direct_beam_center(image, window=60, window_shift=(0, 0), **kwargs):
    """Find the direct beam position in an image with a rough gaussian
    smoothing of the center"""
    center_y = image.shape[0]//2+window_shift[1]
    center_x = image.shape[1]//2+window_shift[0]
    start_x = max(0, center_x - window//2)
    start_y = max(0, center_y - window//2)
    end_x = min(image.shape[1], center_x + window//2)
    end_y = min(image.shape[0], center_y + window//2)
    image_window = image[start_y:end_y, start_x:end_x]
    filtered = gaussian(image_window, **kwargs)
    maximum = np.max(filtered)
    max_y, max_x = np.where(filtered == maximum)
    peak_x = start_x + max_x[0]
    peak_y = start_y + max_y[0]
    return peak_x, peak_y

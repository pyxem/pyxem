import numpy as np


def get_displacements(center, shape, affine):
    """ Gets the displacements for a set of points based on some affine transformation
    about some center point.

    Parameters
    -------------
    center: (tuple)
        The center to preform the affine transformation around
    shape: (tuple)
        The shape of the array
    affine: 3x3 array
        The affine transformation to apply to the image

    Returns
    ----------
    dx: np.array
        The displacement in the x direction of shape = shape
    dy: np.array
        The displacement in the y direction of shape = shape
    """
    difference=np.subtract(shape,center)
    xx,yy = np.mgrid[center[0]:difference[0],center[1]:difference[1]]  # all x and y coordinates on the grid
    coord = np.array([xx.flatten(), yy.flatten(), np.ones(shape[0]*shape[1])])
    corrected = np.matmul(coord.T, affine)
    dx = xx - corrected[:, 0]
    dy = yy - corrected[:, 1]
    return dx,dy

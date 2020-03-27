import numpy as np

def get_displacements(affine, center, shape):
    np.subtract(shape,center)
    np.ogrid()
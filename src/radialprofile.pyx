import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt, ceil

# cython implementation from:
# http://stackoverflow.com/q/21242011

DTYPE_IMG = np.int
ctypedef np.int_t DTYPE_IMG_t
DTYPE = np.int
ctypedef np.int_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void cython_radial_profile(DTYPE_IMG_t [:, :] img_view, DTYPE_t [:] r_profile_view, int xs, int ys, int x0, int y0) nogil:

    cdef int x, y, r, tmp

    for x in prange(xs):
        for y in range(ys):
            r =<int>(sqrt((x - x0)**2 + (y - y0)**2))
            tmp = img_view[x, y]
            r_profile_view[r] +=  tmp 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def radialprofile(np.ndarray img, np.ndarray center):


    cdef int xs, ys, r_max
    xs, ys = img.shape[0], img.shape[1]
    cdef int centerX, centerY
    centerX, centerY = center[0], center[1]

    cdef int topLeft, topRight, botLeft, botRight

    topLeft = <int> ceil(sqrt(centerX**2 + centerY**2))
    topRight = <int> ceil(sqrt((xs - centerX)**2 + (centerY)**2))
    botLeft = <int> ceil(sqrt(centerX**2 + (ys-centerY)**2))
    botRight = <int> ceil(sqrt((xs-centerX)**2 + (ys-centerY)**2))

    r_max = max(topLeft, topRight, botLeft, botRight)

    cdef np.ndarray[DTYPE_t, ndim=1] r_profile = np.zeros([r_max], dtype=DTYPE)
    cdef DTYPE_t [:] r_profile_view = r_profile
    cdef DTYPE_IMG_t [:, :] img_view = img

    with nogil:
        cython_radial_profile(img_view, r_profile_view, xs, ys, centerX, centerY)
    return r_profile
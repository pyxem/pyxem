import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale
from skimage import draw

from pyxem.utils.expt_utils import normalize_template_match


def get_experimental_square(z, vector, square_size):
    """Defines a square region around a given diffraction vector and returns.

    Parameters
    ----------
    z : np.array()
        Single diffraction pattern
    vector : np.array()
        Single vector in pixels (int) [x,y] with top left as [0,0]
    square_size : int
        The length of one side of the bounding square (must be even)

    Returns
    -------
    square : np.array()
        Of size (L,L) where L = square_size

    """
    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    # Pad the image with zeros to avoid edge effects
    z = np.pad(z, square_size, mode="reflect")
    cx, cy, half_ss = vector[0], vector[1], int(square_size / 2)

    _z = z[cy: cy + half_ss*2, cx: cx + half_ss*2]
    return _z


def get_simulated_disc(square_size, disc_radius):
    """Create a uniform disc for correlating with the experimental square.

    Parameters
    ----------
    square size : int
        (even) - size of the bounding box
    disc_radius : int
        radius of the disc

    Returns
    -------
    arr: np.array()
        Upsampled copy of the simulated disc as a numpy array

    """

    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    ss = int(square_size)
    arr = np.zeros((ss, ss))
    rr, cc = draw.disk(
        (int(ss / 2), int(ss / 2)), radius=disc_radius, shape=arr.shape
    )  # is the thin disc a good idea
    arr[rr, cc] = 1
    return arr


def _center_of_mass_hs(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention

    Parameters
    ----------
    z : np.array

    Returns
    -------
    (x,y) : tuple of floats
        The x and y locations of the center of mass of the parsed square
    """

    s = np.sum(z)
    if s != 0:
        z *= 1 / s
    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))
    return cx, cy


def _com_experimental_square(z, vector, square_size):
    """Wrapper for get_experimental_square that makes the non-zero
    elements symmetrical around the 'unsubpixeled' peak by zeroing a
    'spare' row and column (top and left).

    Parameters
    ----------
    z : np.array

    vector : np.array([x,y])

    square_size : int (even)

    Returns
    -------
    z_adpt : np.array
        z, but with row and column zero set to 0
    """
    # Copy to make sure we don't change the dp
    z_adpt = np.copy(get_experimental_square(z, vector=vector, square_size=square_size))
    z_adpt[:, 0] = 0
    z_adpt[0, :] = 0
    return z_adpt

def _conventional_xc(slic, kernel, upsample_factor):
    """Takes two images of disc and finds the shift between them using
    conventional cross correlation.
    """
    slic = rescale(slic, upsample_factor, order=1,
            mode='reflect')
    kernel = rescale(kernel, upsample_factor, order=1,
            mode='reflect')

    temp = normalize_template_match(slic, kernel)
    max = np.unravel_index(np.argmax(temp), temp.shape)
    shifts = np.array(max) - np.array(kernel.shape) / 2
    shifts = np.flip(shifts)/upsample_factor  # to comply with hyperspy conventions - see issue#490
    shifts = shifts+1
    return shifts
#####################################################
# Methods for subpixel refinement on a set of vectors
#####################################################


def _center_of_mass_map(dp, vectors, square_size, offsets, scales):
    if vectors.shape == (2,):
        vectors = [
            vectors,
        ]
    shifts = np.zeros_like(vectors, dtype=np.float64)
    for i, vector in enumerate(vectors):
        expt_disc = _com_experimental_square(dp, vector, square_size)
        shifts[i] = [a - square_size / 2 for a in _center_of_mass_hs(expt_disc)]
    return (vectors + shifts) * scales + offsets


def _conventional_xc_map(
        dp, vectors, kernel, square_size, upsample_factor, offsets, scales
):
    shifts = np.zeros_like(vectors, dtype=np.float64)
    for i, vector in enumerate(vectors):
        expt_disc = get_experimental_square(dp, vector, square_size)
        shifts[i] = _conventional_xc(expt_disc, kernel, upsample_factor)
    return (vectors + shifts) * scales + offsets


def _reference_xc_map(dp, vectors, kernel, square_size, upsample_factor, offsets, scales):
    shifts = np.zeros_like(vectors, dtype=np.float64)
    for i, vector in enumerate(vectors):
        expt_disc = get_experimental_square(dp, vector, square_size)
        shifts[i] = _conventional_xc(expt_disc, kernel, upsample_factor)
    return (vectors + shifts) * scales + offsets
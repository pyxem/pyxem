from numba import cuda
import numpy as np

try:
    import cupy as cp
    CUPY_INSTALLED=True
except ImportError:
    CUPY_INSTALLED=False


def dask_array_to_gpu(dask_array):
    """
    Copy a dask array to the GPU chunk by chunk
    """
    if not CUPY_INSTALLED:
        raise BaseException("cupy is required")
    return dask_array.map_blocks(cp.asarray)


def dask_array_from_gpu(dask_array):
    """
    Copy a dask array from the GPU chunk by chunk
    """
    if not CUPY_INSTALLED:
        raise BaseException("cupy is required")
    return dask_array.map_blocks(cp.asnumpy)


def to_numpy(array):
    """
    Returns the array as an numpy array
    Parameters
    ----------
    array : numpy or cupy array
        Array to determine whether numpy or cupy should be used
    Returns
    -------
    array : numpy.ndarray
    """
    if is_cupy_array(array):
        import cupy as cp
        array = cp.asnumpy(array)
    return array


def get_array_module(array):
    """
    Returns the array module for the given array
    Parameters
    ----------
    array : numpy or cupy array
        Array to determine whether numpy or cupy should be used
    Returns
    -------
    module : module
    """
    module = np
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            module = cp
    except ImportError:
        pass
    return module


def is_cupy_array(array):
    """
    Convenience function to determine if an array is a cupy array

    Parameters
    ----------
    array : array
        The array to determine whether it is a cupy array or not.
    Returns
    -------
    bool
        True if it is cupy array, False otherwise.
    """
    try:
        import cupy as cp
        return isinstance(array, cp.ndarray)
    except ImportError:
        return False


@cuda.jit
def _correlate_polar_image_to_library_gpu(polar_image, sim_r, sim_t, sim_i, image_norm, template_norms, correlation):
    """
    Custom cuda kernel for calculating the correlation for each template
    in a library at all in-plane angles with a polar image

    Parameters
    ----------
    polar_image: 2D numpy.ndarray or cupy.ndarray
        The image in polar coordinates of shape (theta, R)
    sim_r: 2D numpy.ndarray or cupy.ndarray
        The r coordinates of all spots in all templates in the library.
        Spot number is the column, template is the row
    sim_t: 2D numpy.ndarray or cupy.ndarray
        The theta coordinates of all spots in all templates in the library.
        Spot number is the column, template is the row
    sim_i: 2D numpy.ndarray or cupy.ndarray
        The intensity of all spots in all templates in the library.
        Spot number is the column, template is the row
    image_norm: float
        The norm of the image (square root of the sum of all pixel intensities
        squared)
    template_norms: 1D numpy.ndarray or cupy.ndarray
        The norm of each template, same length as number of rows in sim arrays
    correlation: 2D numpy.ndarray or cupy.ndarray
        The output correlation matrix of shape (template number, theta), giving
        the correlation of each template at each in-plane angle
    """
    # grid and stride for the output so that each cuda thread deals with more than one element at a time
    start_template, start_shift = cuda.grid(2)
    stride_template, stride_shift = cuda.gridsize(2)
    # don't calculate for grid positions outside
    if start_template >= sim_r.shape[0] or start_shift >= polar_image.shape[0]:
        return
    # loop over all templates
    for template in range(start_template, sim_r.shape[0], stride_template):
        # loop over all in-plane angles
        for shift in range(start_shift, polar_image.shape[0], stride_shift):
            tmp = 0
            # add up all contributions to the correlation from spots
            for spot in range(sim_r.shape[1]):
                tmp += (polar_image[(sim_t[template, spot] + shift) % polar_image.shape[0],
                                    sim_r[template, spot]] * sim_i[template, spot])
            correlation[template, shift] = tmp / (image_norm * template_norms[template])


def _match_polar_to_polar_library_gpu(
    pattern,
    r_templates,
    theta_templates,
    intensities_templates,
    polar_norm,
    template_norms,
    blockspergrid,
    threadsperblock,
):
    correlation = cp.empty((r_templates.shape[0], pattern.shape[0]), dtype=cp.float32)
    _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
        pattern,
        r_templates,
        theta_templates,
        intensities_templates,
        polar_norm,
        template_norms,
        correlation,
    )
    correlation_m = cp.empty((r_templates.shape[0], pattern.shape[0]), dtype=cp.float32)
    _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
        pattern,
        r_templates,
        (pattern.shape[0] - theta_templates) % pattern.shape[0],
        intensities_templates,
        polar_norm,
        template_norms,
        correlation_m,
    )
    return correlation, correlation_m


def _index_block_gpu(
    polar_images,
    r_templates,
    theta_templates,
    intensities_templates,
    template_norms,
    n_best,
    norm_images,
    threadsperblock=(16, 16),
):
    """
    Helper function to perform indexation on the GPU

    Parameters
    ----------
    polar_images: cupy.ndarray (y, x, theta, r)
        images in a chunk in polar coordinates on the gpu
    r_templates: cupy.ndarray (T, S)
        r coordinates of all S spots in all T patterns
    theta_templates: cupy.ndarray (T, S)
        theta coordinates of all S spots in all T patterns
    intensities_templates: cupy.ndarray (T, S)
        intensities of all S spots in all T templates
    n_best: int
        number of best results in the templates to return in the result
    norm_images: bool
        whether to normalize the images in the correlation calculation
    threadsperblock: 2-tuple of ints
        number of threads in each block (setting for evaluating cuda kernel)

    Returns
    -------
    result: numpy.ndarray (y, x, n_best, 4)
        n_best results for each pattern of the form:
        (template_index, correlation, theta_shift, sign)

    Notes
    -----
    sign in the result is 1 if the positive template is best matched, -1 if
    the mirrored template is best matched. theta_shift is expressed in pixels
    of the polar_template, it should still be multiplied by delta_theta to
    express it in units of degrees.
    """
    indexation_result_chunk = cp.empty(
        (polar_images.shape[0], polar_images.shape[1], n_best, 4), dtype=cp.float32
    )
    blockspergrid_x = int(np.ceil(polar_images.shape[2] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(polar_images.shape[3] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)    
    pattern_norms = cp.ones(polar_images.shape[:2])
    if norm_images:
        pattern_norms = cp.linalg.norm(polar_images, axis = (2, 3))
    for index in np.ndindex(polar_images.shape[:2]):
        pattern = polar_images[index]
        polar_norm = float(pattern_norms[index])
        correlation = cp.empty((r_templates.shape[0], pattern.shape[0]), dtype=cp.float32)
        _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
            pattern,
            r_templates,
            theta_templates,
            intensities_templates,
            polar_norm,
            template_norms,
            correlation,
        )
        correlation_m = cp.empty((r_templates.shape[0], pattern.shape[0]), dtype=cp.float32)
        _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
            pattern,
            r_templates,
            (polar_images.shape[0] - theta_templates)%pattern.shape[0],
            intensities_templates,
            polar_norm,
            template_norms,
            correlation_m,
        )
        best_in_plane_corr = cp.amax(correlation, axis=1)
        best_in_plane_corr_m = cp.amax(correlation_m, axis=1)
        best_in_plane_shift = cp.argmax(correlation, axis=1)
        best_in_plane_shift_m = cp.argmax(correlation_m, axis=1)
        # compare positive and negative templates and combine
        positive_is_best = best_in_plane_corr > best_in_plane_corr_m
        best_sign = positive_is_best*1 + np.invert(positive_is_best)*(-1)
        best_cors = positive_is_best*best_in_plane_corr + np.invert(positive_is_best)*best_in_plane_corr_m
        best_angles = positive_is_best*best_in_plane_shift + np.invert(positive_is_best)*best_in_plane_shift_m
        answer = cp.empty((n_best, 4), dtype=cp.float32)
        if n_best == 1:
            best_index = cp.argmax(best_cors)
            answer[0,0] = best_index
            answer[0,1] = best_cors[best_index]
            answer[0,2] = best_angles[best_index]
            answer[0,3] = best_sign[best_index]
        else:
            indices_sorted = cp.argsort(-best_cors)
            n_best_indices = indices_sorted[:n_best]
            for i in range(n_best):
                j = n_best_indices[i]
                answer[i, 0] = j
                answer[i, 1] = best_cors[j]
                answer[i, 2] = best_angles[j]
                answer[i, 3] = best_sign[j]
        indexation_result_chunk[index] = answer
    return cp.asnumpy(indexation_result_chunk)

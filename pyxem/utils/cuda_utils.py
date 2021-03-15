from numba import cuda
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndigpu


def warp_polar_gpu(image, center=None, radius=None, output_shape=None, **kwargs):
    """
    Function to emulate warp_polar in skimage.transform on the GPU. Not all
    parameters are supported
    
    Parameters
    ----------
    image: cupy.ndarray
        Input image. Only 2-D arrays are accepted.         
    center: tuple (row, col), optional
        Point in image that represents the center of the transformation
        (i.e., the origin in cartesian space). Values can be of type float.
        If no value is given, the center is assumed to be the center point of the image.
    radius: float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape: tuple (row, col), optional

    Returns
    -------
    polar: cupy.ndarray
        polar image

    Notes
    -----
    Speed gains on the GPU will depend on the size of your problem.
    On a Tesla V100 a 10x speed up was achieved with a 256x256 image:
    from 5 ms to 400 microseconds. For a 4000x4000 image a 1000x speed up
    was achieved: from 180 ms to 400 microseconds. However, this does not
    count the time to transfer data from the CPU to the GPU and back.
    """
    if radius is None:
        radius = int(np.ceil(np.sqrt((image.shape[0] / 2)**2 + (image.shape[1] / 2)**2)))
    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    if center is not None:
        cx, cy = center
    if output_shape is None:
        output_shape = (360, radius)
    delta_theta = 360 / output_shape[0]
    delta_r = radius / output_shape[1]
    t = cp.arange(output_shape[0])
    r = cp.arange(output_shape[1])
    R, T = cp.meshgrid(r, t)
    X = R * delta_r * cp.cos(cp.deg2rad(T * delta_theta)) + cx
    Y = R * delta_r * cp.sin(cp.deg2rad(T * delta_theta)) + cy
    coordinates = cp.stack([Y, X])
    polar = ndigpu.map_coordinates(image, coordinates, order=1)
    return polar


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

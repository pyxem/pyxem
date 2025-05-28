"""
Strain Mapping
==============

Strain mapping in pyxem is done by fitting a :class:`~.signals.tensor_field.DisplacementGradientMap` to the data.
This can be thought of as image distortion around some central point.
"""

from pyxem.data import simulated_strain
import hyperspy.api as hs


# %%
# In this example we will create a simulated strain map using the :meth:`~.data.simulated_strain` function.
# This just creates a simulated diffraction pattern and applies a simple "strain" to it. In this
# case using simulated data is slightly easier for demonstration purposes. If you want to use
# real data the :meth:`~.data.zrnb_precipitate` dataset is a good example of strain from a precipitate.

strained_signal = simulated_strain(
    navigation_shape=(32, 32),
    signal_shape=(512, 512),
    disk_radius=20,
    num_electrons=1e5,
    strain_matrix=None,
    lazy=True,
)

# %%
# The first thing we want to do is to find peaks within the diffraction pattern. I'd recommend
# using the :meth:`~.signals.diffraction2d.get_diffraction_vectors` method

strained_signal.calibration.center = (
    None  # First set the center to be (256, 256) or the center of the signal
)
template_matched = strained_signal.template_match_disk(disk_r=20, subtract_min=False)
template_matched.plot(vmin=0.4)
# %%
# Plotting the template matched signal and setting ``vmin`` is a good way to see what threshold you
# should use for the :meth:`~.signals.diffraction2d.get_diffraction_vectors` method.

diffraction_vectors = template_matched.get_diffraction_vectors(
    threshold_abs=0.4, min_distance=5
)

markers = diffraction_vectors.to_markers(color="w", sizes=5, alpha=0.5)
strained_signal.plot()
strained_signal.add_marker(markers)

# %%
# Determining the Strain
# ----------------------
# We can just use the first ring of the diffraction pattern to determine the strain. We can do this by
# using the :meth:`~.signals.DiffractionVectors.filter_magnitude` method. You can also look at the
# :ref:`filtering vectors <_sphx_glr_examples_vectors_masking_vectors.py>` example to see
# how to select which vectors you want to use more generally. You can also just manually input the un-strained
# vectors or use simulated/ rotated vectors as well.

first_ring_vectors = diffraction_vectors.filter_magnitude(
    min_magnitude=0.1,
    max_magnitude=1,
)
unstrained_vectors = first_ring_vectors.inav[0, 0]

strain_maps = first_ring_vectors.get_strain_maps(
    unstrained_vectors=unstrained_vectors, return_residuals=False
)

strain_maps.plot()

# %%
# Some final notes about strain mapping. In general, you want to use as many pixels as possible. 512 x 512 is a good
# place to start.  You can do strain mapping with fewer pixels, but the results will be less accurate. Precession
# also helps improve the results as does having a thinner sample both of which reduce the effects of dynamical
# diffraction.

# %%
# sphinx_gallery_thumbnail_number = 5

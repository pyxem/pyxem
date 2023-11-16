"""
This can be used to segment a 4-D STEM dataset into different clusters based on the diffraction
pattern at each real space position.
"""

import pyxem as pxm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Getting the vectors for some dataset
s = pxm.data.mgo_nanocrystals()
s.data[s.data < 120] = 1
s.filter(gaussian_filter, sigma=(0.5, 0.5, 0, 0), inplace=True)  # only in real space
s.template_match_disk(disk_r=3, subtract_min=False, inplace=True)
pks = s.find_peaks(
    interactive=False, threshold_abs=0.5, min_distance=3, get_intensity=True
)
vectors = pxm.signals.DiffractionVectors.from_peaks(
    pks
)  # calibration is automatically set

# Now we can convert the vectors into a 2D array of rows/columns
flat_vectors = (
    vectors.flatten_diffraction_vectors()
)  # flatten the vectors into a 2D array
scan = DBSCAN(eps=1.0, min_samples=2)
# It is very important that we first normalize the real and reciprocal space distances
# The column scale factors map the real space and reciprocal space distances to the same scale
# Here this means that the clustering algorithm operates on 6 nm in real space and .1 nm^-1 in
# reciprocal space based on the units for the vectors.
clustered = flat_vectors.cluster(
    scan, column_scale_factors=[10, 10, 0.05, 0.05], min_vectors=40
)
m, p = clustered.to_markers(s, alpha=0.8, get_polygons=True)
s.plot()
s.add_marker(m)
s.add_marker(p, plot_on_signal=False)

# %%

vect = clustered.map_vectors(
    pxm.utils.labeled_vector_utils.column_mean,
    columns=[0, 1],
    label_index=-1,
    dtype=float,
    shape=(2,),
)
plt.figure()
plt.scatter(vect[:, 1], vect[:, 0])

# %%

clusterer = DBSCAN(min_samples=2, eps=20)

clustered2 = clustered.cluster_labeled_vectors(method=clusterer)

# This clustering is decent.  It shows that there might be some small tilt boundaries in the data
# which segment some of the nano-crystals into different clusters.  It also shows the effect of using
# a phosphor screen which has some pretty severe after glow.  This results in a smearing of the
# features and elongated clusters along the scan direction.

s.plot()
s.add_marker(m)
s.add_marker(p, plot_on_signal=False)
# %%

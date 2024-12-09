"""
==================
Clustering Vectors
==================

This can be used to segment a 4-D STEM dataset into different clusters based on the diffraction
pattern at each real space position.
"""

import pyxem as pxm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Getting the vectors for some dataset
# s = pxm.data.mgo_nanocrystals(allow_download=True)
s = pxm.data.simulated_overlap()
s.filter(gaussian_filter, sigma=(0.5, 0.5, 0, 0), inplace=True)  # only in real space
s.template_match_disk(disk_r=5, subtract_min=False, inplace=True)
vectors = s.get_diffraction_vectors(threshold_abs=0.5, min_distance=3)

# Now we can convert the vectors into a 2D array of rows/columns
flat_vectors = (
    vectors.flatten_diffraction_vectors()
)  # flatten the vectors into a 2D array
scan = DBSCAN(eps=1.0, min_samples=2)
# %%
# Clustering the Vectors
# ======================
# It is very important that we first normalize the real and reciprocal space distances
# The column scale factors map the real space and reciprocal space distances to the same scale
# Here this means that the clustering algorithm operates on 4 nm in real space and .1 nm^-1 in
# reciprocal space based on the units for the vectors.
clustered = flat_vectors.cluster(
    scan,
    column_scale_factors=[2, 2, 0.05, 0.05],
    columns=[0, 1, 2, 3],
    min_vectors=40,
)
m, p = clustered.to_markers(s, alpha=0.8, get_polygons=True)
s.plot()
s.add_marker(m)
s.add_marker(p, plot_on_signal=False)

# %%

vect = clustered.map_vectors(
    pxm.utils.vectors.column_mean,
    columns=[0, 1],
    label_index=-1,
    dtype=float,
    shape=(2,),
)
plt.figure()
plt.scatter(vect[:, 1], vect[:, 0])

# %%

clusterer = DBSCAN(min_samples=2, eps=4)

clustered2 = clustered.cluster_labeled_vectors(method=clusterer)
m, p = clustered2.to_markers(s, alpha=0.8, get_polygons=True)

# %%
# Visualizing the Clustering
# ==========================
#
# This clustering is works pretty good after the second step.  We can interact with the results as well in order
# to see regions where the clustering doesn't work quite as well!

# %%

m, p = clustered2.to_markers(
    s, alpha=0.8, get_polygons=True, facecolor="none", sizes=(30,), lw=5
)
s.axes_manager.indices = (45, 45)
s.plot()
s.add_marker(m)
s.add_marker(p, plot_on_signal=False)
# sphinx_gallery_thumbnail_number = 5

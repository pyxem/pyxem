"""
Clustering Vectors
==================

This example shows some basic clustering algorithms which work quite well for clustering vectors.

This can be used to segment a 4-D STEM dataset into different clusters based on the diffraction
pattern at each real space position.
"""

import pyxem as pxm
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

# Getting the vectors for some dataset
s = pxm.data.mgo_nanocrystals()
s.filter(gaussian_filter, sigma=(1, 1, 0, 0), inplace=True)  # only in real space
s.template_match_disk(disk_r=5, subtract_min=False, inplace=True)
pks = s.find_peaks(threshold_abs=0.6, min_distance=5, get_intensity=True, inplace=True)
vectors = pxm.signals.DiffractionVectors.from_peaks(pks)  # calibration is automatically set

# Now we can convert the vectors into a 2D array of rows/columns
flat_vectors = vectors.flatten()  # flatten the vectors into a 2D array
scan = DBSCAN(eps=0.5, min_samples=2)

# It is very important that we first normalize the real and reciprocal space distances
# The column scale factors map the real space and reciprocal space distances to the same scale
# Here this means that the clustering algorithm operates on 1 nm in real space and .1 nm^-1 in
# reciprocal space based on the units for the vectors.
clustered = flat_vectors.cluster(scan, column_scale_factors=[1, 1, .1, .1])

# We can also cluster in

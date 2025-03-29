import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sng_dbscan import SNG_DBSCAN

n = 1000

x, y = make_blobs(n_samples=n, centers=4, center_box=(-20, 20))  # type: ignore

assert isinstance(x, np.ndarray)

epsilon = 2
min_pts = 3

sampling_rate = 20 * np.log(n) / n

sng = SNG_DBSCAN(sampling_rate=sampling_rate, max_dist=epsilon, min_points=min_pts)
sng_labels = sng.fit_predict(x)

dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
dbscan_labels = dbscan.fit_predict(x)

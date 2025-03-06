import numpy as np
from sklearn.datasets import make_blobs

from sng_dbscan.sng_dbscan import SNG_DBSCAN

x, y = make_blobs()  # type: ignore

assert isinstance(x, np.ndarray)

SNG_DBSCAN(x, 0.05, 0.2, 3)

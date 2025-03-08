import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

from sng_dbscan.sng_dbscan import SNG_DBSCAN

x, y = make_blobs()  # type: ignore

assert isinstance(x, np.ndarray)

epsilon = 2
min_pts = 3

n = len(x)

sng = SNG_DBSCAN(sampling_rate=np.log(n) / n, max_dist=epsilon, min_points=min_pts)
sng_labels = sng.fit(x)

dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
dbscan_labels = dbscan.fit_predict(x)

df = pd.DataFrame(
    x,
    columns=["x", "y"],
)

df.loc[:, "true"] = y.astype(str)
df.loc[:, "sng"] = sng_labels.astype(str)
df.loc[:, "dbscan"] = dbscan_labels.astype(str)

df = df.melt(
    id_vars=["x", "y"],
    value_vars=["true", "sng", "dbscan"],
    value_name="Label",
    var_name="Cluster Type",
)

px.scatter(df, x="x", y="y", color="Label", facet_row="Cluster Type").write_html(
    "out.html", include_plotlyjs="cdn"
)

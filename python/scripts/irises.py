import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sng_dbscan.sng_dbscan import SNG_DBSCAN

X, y = load_iris(return_X_y=True)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = X[:, [0, 1]]

assert isinstance(X, np.ndarray)

epsilon = 0.5
min_pts = 2

n = len(X)

sng = SNG_DBSCAN(sampling_rate=20 * np.log(n) / n, max_dist=epsilon, min_points=min_pts)
sng_labels = sng.fit(X)

dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
dbscan_labels = dbscan.fit_predict(X)

df = pd.DataFrame(
    X,
    columns=["X", "y"],
)

df.loc[:, "true"] = y.astype(str)
df.loc[:, "sng"] = sng_labels.astype(str)
df.loc[:, "dbscan"] = dbscan_labels.astype(str)

df = df.melt(
    id_vars=["X", "y"],
    value_vars=["true", "sng", "dbscan"],
    value_name="Label",
    var_name="Cluster Type",
)

px.scatter(df, x="X", y="y", color="Label", facet_col="Cluster Type").write_html(
    "out.html", include_plotlyjs="cdn"
)

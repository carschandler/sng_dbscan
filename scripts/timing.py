# %%
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

from sng_dbscan.sng_dbscan import SNG_DBSCAN

# %%
x, y = make_blobs(n_samples=100, centers=4, center_box=(-20, 20))  # type: ignore

epsilon = 2
min_pts = 3

n = len(x)

sampling_rate = 20 * np.log(n) / n

sng = SNG_DBSCAN(sampling_rate=sampling_rate, max_dist=epsilon, min_points=min_pts)
dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)

# %%
%%timeit
sng_labels = sng.fit_predict(x)

# %%
sng.kdtree = True

# %%
%%timeit
sng_labels = sng.fit_predict(x)

# %%
%%timeit
dbscan_labels = dbscan.fit_predict(x)

# %%
%%timeit
dbscan.algorithm = "brute"
dbscan_labels = dbscan.fit_predict(x)

# %%
df = pd.DataFrame(
    x,
    columns=["x", "y"],
)

df.loc[:, "true"] = y.astype(str)
df.loc[:, "sng"] = sng_labels.astype(str)
df.loc[:, "dbscan"] = dbscan_labels.astype(str)

df = df.melt,(
    id_vars=["x", "y"],
    value_vars=["true", "sng", "dbscan"],
    value_name="Label",
    var_name="Cluster Type",
)

fig = px.scatter(df, x="x", y="y", color="Label", facet_col="Cluster Type")

fig.write_html("fig.html", include_plotlyjs="cdn")

fig.write_json("fig.json")

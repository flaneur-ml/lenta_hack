from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
import pandas as pd
import numpy as np

# PRESERVING ORDER IS IMPORTANT


def split_to_groups(df):
    groups = []
    columns = list(df.columns)
    columns.pop("group")
    for group_id in range(5):
        groups.append(df[df["group"] == group_id][columns])
    return groups


def preprocess(dsv):  # dsv = dataset vectorized
    return dsv


def project(dsv, seed=1):
    temp = TSNE(2, 30, random_state=seed).fit_transform(dsv)
    return temp.T[0], temp.T[1]
    

def plot_clusters(projection, clustering):
    try:
        clustering = clustering.labels_
    except AttributeError:
        pass
    plt.plot(*projection, c=clustering)
    plt.show()
    


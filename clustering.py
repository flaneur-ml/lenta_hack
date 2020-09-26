from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import scale
from misc import cached
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
import pandas as pd
import numpy as np

# PRESERVEING DATASET ORDER IS IMPORTANT

# 0. split intital table by groups, then for each group...
# 1. Choose adequate subset of features and adequate preprocessing based on projections
# 2. Choose adequate clustering based on colored projection
# 3. Assign cluster labels to intitial table

@cached
def split_to_groups(df):
    groups = []
    columns = list(df.columns)
    columns.pop("group")
    for group_id in range(5):
        groups.append(df[df["group"] == group_id][columns])
    return groups

def preprocess(dsv): # dsv = dataset vectorized
    return dsv

def project(dsv, seed=1):
    temp = TSNE(2, 30, random_state=seed).fit_transform(dsv)
    return temp.T[0], temp.T[1]
    

def plot_clusters(projection, clustering): # cluster = None for simply plotting projection
    try:
        clustering = clsutering.labels_
    except AttributeError:
        pass
    plt.plot(*projection, c=clustering)
    plt.show()
    
def complement_customers(df, a_clusterings, b_clusterings):
    groups = split_to_groups()
    try:
        for i in range(len(a_clusterings)):
            a_clusterings[i] = a_clustterings[i].labels_
            b_clusterings[i] = b_clustterings[i].labels_
    except AttributeError:
        pass
    final = []
    for group, a_clustering, b_clustering in zip(groups, a_clusterings, b_clusterings):
        group.insert(-1, "A-Cluster", a_clustering)
        group.insert(-1, "B-Cluster", b_clustering)
        final.append(group)
    return pd.concat(final)


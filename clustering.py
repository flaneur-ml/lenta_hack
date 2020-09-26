from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import scale as skl_scale
from misc import cached
from data_management import save_data, load_data
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

group_names = ["M-MSC", "F-MSC", "M-SPB", "F-SPB", "M-O", "F-O"]

@cached(threshold=1)
def split_to_groups(df):
    groups = []
    columns = list(df.columns)
    columns.pop("group")
    for group_id in range(5):
        groups.append(df[df["group"] == group_id][columns])
    return groups

def __signature__project(dsv, seed, perplexity):
    return hash(dsv.tostring()), seed, perplexity
    
def project(dsv, seed=1, perplexity=30):
    signature = __signature__project(dsv, seed, perplexity)
    if  signature in project.cache:
        return project.cache[signature]
    temp = TSNE(2, perplexity, random_state=seed).fit_transform(dsv)
    project.cache[signature] = (temp.T[0], temp.T[1])
    return temp.T[0], temp.T[1]
project.cache = {}
    

def plot_clusters(projection, clustering): # cluster = None for simply plotting projection
    try:
        clustering = clsutering.labels_
    except AttributeError:
        pass
    plt.scatter(*projection, c=clustering)
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


def inspect(df, preprocessing=None, seed=1):
    if preprocessing is None:
        print("Assembling preprocessing pipeline...")
        def preprocessing(dsv):
            for function in inspect.pipeline:
                dsv = function(dsv)
            return dsv
    print("Splitting dataset...")
    groups = split_to_groups(df)
    fig, axs = plt.subplots(2, 3)
    for group, ax, group_name in zip(groups, axs, group_names):
        print("Preprocessing group %s..." % group_name)
        columns = list(group.columns)
        columns.pop('client_id')
        dsv = group[columns].to_numpy()
        dsv = preprocessing(dsv)
        print("Projecting group %s..." % group_name)
        projection = project(dsv, seed=1)
        print("Plotting group %s projection..." % group_name)
        ax.scatter(*projection)
        ax.set_title(group_name)
    plt.show()
inspect.pipeline = []

def test_model(preprocessing=None, seed=1):
    if preprocessing is None:
        def preprocessing(dsv):
            for function in inspect.pipeline:
                dsv = function(dsv)
            return dsv
    
test_model.pipeline = []



#_________________________PREPROCESSING__PIPELINE__SEGMENTS________________________
#__________________________________________________________________________________
def scale(dsv):
    return skl_scale(dsv, with_mean=False)

def center(dsv):
    return skl_scale(dsv, with_std=False)

def random_subsample(seed, indecies, min_remaining, max_remaining):
    pass
#__________________________________________________________________________________



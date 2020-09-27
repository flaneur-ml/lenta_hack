print("Loading imports...")
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import scale as skl_scale
from misc import cached
from data_management import save_data, load_data
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams['figure.dpi'] = 100
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

print("Done loading imports.")
# PRESERVEING DATASET ORDER IS IMPORTANT

# 0. split intital table by groups, then for each group...
# 1. Choose adequate subset of features and adequate preprocessing based on projections
# 2. Choose adequate clustering based on colored projection
# 3. Assign cluster labels to intitial table

group_names = ["Male-Other", "Female-Other", "Female-Moscow", "Male-Moscow", "Male-SPB", "Female-SPB"]


def split_to_groups(df):
    if split_to_groups.cache:
        return split_to_groups.cache[0]
    groups = []
    columns = list(df.columns)
    columns.pop(columns.index("gender_city_group"))
    columns.pop(columns.index("gender"))
    columns.pop(columns.index("city"))
    columns.pop(columns.index("loyalty_period"))
    for group_id in range(6):
        groups.append(df[df["gender_city_group"] == group_id][columns])
    split_to_groups.cache.append(groups)
    return groups
split_to_groups.cache=[]


def __signature__project(dsv, seed, perplexity):
    return hash(dsv[:5].tostring()), seed, perplexity
    
def project(dsv, seed=1, perplexity=30):
    for item in project.cache.items():
        if item[1][0].size == dsv.shape[0]:
            return item[1]
    temp = TSNE(2, perplexity, random_state=seed).fit_transform(dsv)
    project.cache[signature] = (temp.T[0], temp.T[1])
    return project.cache[signature]
project.cache = load_data("projection_cache")
    

def plot_clusters(projection, clustering): # cluster = None for simply plotting projection
    try:
        clustering = clustering.labels_
    except AttributeError:
        pass
    plt.scatter(*projection, c=clustering)
    plt.show()
    
    
def complement_customers(df, a_clusterings, b_clusterings):
    groups = split_to_groups(df)
    try:
        for i in range(len(a_clusterings)):
            a_clusterings[i] = a_clusterings[i].labels_
            b_clusterings[i] = b_clusterings[i].labels_
    except AttributeError:
        pass
    for i in range(1, len(a_clusterings)):
        a_clusterings[i] += 1 + np.max(a_clusterings[i - 1])
        b_clusterings[i] += 1 + np.max(b_clusterings[i - 1])
    final = []
    for group, a_clustering, b_clustering in zip(groups, a_clusterings, b_clusterings):
        group.insert(1, "A-Cluster", a_clustering)
        group.insert(1, "B-Cluster", b_clustering)
        final.append(group)
    return pd.concat(final)


def get_group(group_name, df=None):
    if df is None:
        df = get_group.df
    i = group_names.index(group_name)
    groups = split_to_groups(df)
    group = groups[i]
    columns = list(group.columns)
    columns.pop(columns.index('client_id'))
    dsv = group[columns].to_numpy()
    return dsv
get_group.df = None


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
    for group, ax, group_name in zip(groups, axs.flatten(), group_names):
        print("Preprocessing group %s..." % group_name)
        columns = list(group.columns)
        columns.pop(columns.index('client_id'))
        dsv = group[columns].to_numpy()
        dsv = preprocessing(dsv)
        print("Projecting group %s..." % group_name)
        projection = project(dsv, seed=seed)
        print("Plotting group %s projection..." % group_name)
        ax.scatter(*projection)
        ax.set_title(group_name)
    plt.show()
inspect.pipeline = []


def inspect_mp(df, preprocessing=None, seed=1):
    if preprocessing is None:
        print("Assembling preprocessing pipeline...")
        def preprocessing(dsv):
            for function in inspect.pipeline:
                dsv = function(dsv)
            return dsv
    print("Splitting dataset...")
    groups = split_to_groups(df)
    dsvs = []
    for group, group_name in zip(groups, group_names):
        print("Preprocessing group %s..." % group_name)
        columns = list(group.columns)
        columns.pop(columns.index('client_id'))
        dsv = group[columns].to_numpy()
        dsv = preprocessing(dsv)
        dsvs.append(dsv)
    print("Projecting...")
    pool = mp.Pool(6)
    projections = pool.map(partial(project, seed=seed), dsvs)
    pool.close()
    print("Plotting...")
    fig, axs = plt.subplots(2, 3)
    for projection, group_name, ax in zip(projections, group_names, axs.flatten()):
        ax.scatter(*projection)
        ax.set_title(group_name) 
    plt.show()
    cache = []
    for projection, dsv in zip(projections, dsvs):
        cache.append((__signature__project(dsv, seed, 30), projection))
    save_data(dict(cache), "projection_cache")



def test_model(model, dsv, preprocessing=None, seed=1):
    if preprocessing is None:
        def preprocessing(dsv):
            for function in inspect.pipeline:
                dsv = function(dsv)
            return dsv
    dsv = preprocessing(dsv)
    model.fit(dsv)
    projection = project(dsv, seed=seed)
    plot_clusters(projection, model)
    return model.labels_ - np.min(model.labels_)
test_model.pipeline = []



#_________________________PREPROCESSING__PIPELINE__SEGMENTS________________________
#__________________________________________________________________________________
def anti_nan(dsv):
    dsv[dsv != dsv] = 0
    return dsv

def share(dsv):
    first_index, last_index = 0, 0 #!!!
    subdsv = dsv.T[first_index:last_index + 1].T
    totals = np.apply_along_axis(np.sum, 1, subdsv)
    subdsv = np.apply_along_axis(lambda x: x / totals, 0, subdsv)
    dsv.T[first_index:last_index + 1] = subdsv.T
    return dsv

def scale(dsv):
    return skl_scale(dsv, with_mean=False)

def center(dsv):
    return skl_scale(dsv, with_std=False)

def Random_subsampling(seed, indices, min_remaining, max_remaining):
    a = list(indices)
    def subsampling(dsv):
        np.random.seed(seed)
        remaining = np.random.randint(min_remaining, max_remaining + 1)
        b = np.sort(np.random.choice(a, size=remaining))
        return dsv.T[b].T
    return subsampling

def Random_magnification(seed):
    def magnification(dsv):
        m = 2 ** (np.random.random(dsv.shape[1]) * 3 - 1)
        return np.apply_along_axis(lambda x: x * m, 1, dsv)
    return magnification
#__________________________________________________________________________________



#_____________________________PREPROCESSING__PIPELINES_____________________________
#__________________________________________________________________________________

seed = 1
preference_indices = list(range(63))[2:-1]

pipeline_1 = [share, scale]
pipeline_2 = [share, center, scale]
pipeline_3 = [share, center, scale, Random_subsampling(seed, preference_indices, 5, 5)]
pipeline_4 = [share, center, scale, Random_subsampling(seed, preference_indices, 5, 5), Random_magnification(1)]
pipeline_5 = [share, scale, Random_subsampling(seed, preference_indices, 5, 5)]

inspect.pipeline = pipeline_5
test_model.pipeline = pipeline_5
#__________________________________________________________________________________


print("Loading data...")
df = load_data("data")
print("Done loading data.")
get_group.df = df

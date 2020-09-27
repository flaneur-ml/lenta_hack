from misc import cached
import scipy.stats as stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


@cached
def get_cluster_stats(data, cluster_type, cluster_id, features=None, plot=True):

    """
    :param plot: plot flag
    :param data: pandas data frame
    :param cluster_type: A or B
    :param cluster_id: A1 ... An
    :param features: age, spending ...
    :return:
    """

    spec_data = data.loc[data[cluster_type] == cluster_id]
    mean = {}
    var = {}

    if features is None:

        for feature in spec_data:
            mean[feature] = np.mean(spec_data[feature])
            var[feature] = np.var(spec_data[feature])
        cov = spec_data.cov()

        if plot:  # Correlation matrix
            sns.set()
            sns.heatmap(spec_data.corr(), annot=False)
            plt.show()

    else:
        for feature in features:
            mean[feature] = np.mean(spec_data[feature])
            var[feature] = np.var(spec_data[feature])
        cov = spec_data[features].cov()  # Covariance with specified features

        if plot:  # Correlation matrix
            sns.set()
            sns.heatmap(spec_data[features].corr(), annot=False)
            plt.show()

    print(f"Mean values per features: {mean}")
    print(f"Variance values per features: {var}")
    print(f"Covariance matrix: {cov}")

    return mean, var, cov


def get_heat_matrix(data, category, cluster_type, feature):
    sns.set()
    sns.heatmap(data[[category, cluster_type, feature]], annot=True)
    plt.show()


def get_preference_matrix(data, categories, transactions, category_name, subcategories, cluster_type):
    sns.set()
    spec_data = data.loc[data[categories] == category_name]

    total = sum(spec_data[transactions])
    sum_per_subcategory_df = spec_data.groupby(subcategories)[transactions].sum().reset_index()
    sum_per_subcategory_df["average"] = sum_per_subcategory_df[transactions]/total

    sns.heatmap(sum_per_subcategory_df[[subcategories, cluster_type, "average"]], annot=True)
    plt.show()


def get_yield_matrix(data, category, closest_subcategories, cluster_type):
    sns.set()


    sns.heatmap(data[[closest_subcategories, cluster_type]], annot=True)
    plt.show()
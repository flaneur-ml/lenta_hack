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


def get_preference_matrix(data, category, cluster_type):
    sns.set()
    sns.heatmap(data[[cluster_type, category]], annot=False)
    plt.show()


def get_yield_matrix(data, category, cluster_type):
    sns.set()
    sns.heatmap(data[[cluster_type, category]], annot=False)
    plt.show()
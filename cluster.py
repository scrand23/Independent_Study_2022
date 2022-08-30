#!/usr/bin/env python
# coding: utf-8

"""
the kmeans algorithm in scipy uses random initialization of centroids. The best way to initialize cluster centroids is through kmeans++. The plus_plus function from https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html creates a numpy array of initial centroids which can then be passed into the kmeans method of scipy. 

Kmeans++ initialization method works by choosing the location of one centroid randomly, then maximizing the distance between this centroid and the other k centroids. This ensures an even dispersion of centroids, whereas randomly initialized centroids can leave gaps in the data.


"""

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import datetime

def select_time(df, hour):
    selectdf = df[df.hour==hour]
    return(selectdf)

def scale_and_convert_to_numpy(selectdf, included_cols):
    data = selectdf.loc[:, selectdf.columns.isin(list(included_cols))].to_numpy()
    scaled_data = whiten(data)
    return data, scaled_data

def cluster_data(init_centroids, scaled_data):
    centroids, _  = kmeans(scaled_data,init_centroids,iter=100)
    idx, _ = vq(scaled_data,centroids)
    idx=idx+1
    return centroids, idx

def plus_plus(ds, k, random_state=42):
    """
    Create cluster centroids using the k-means++ algorithm.
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """

    np.random.seed(random_state)
    centroids = [ds[0]]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centroids.append(ds[i])

    return np.array(centroids)    

# unscale Centroids
def get_centroids(scaledCentroids, included_cols, data):
    centroidsScaledDF = pd.DataFrame(scaledCentroids, columns = included_cols)
    centroidsDF = pd.DataFrame(data, columns = included_cols).describe().loc["std"]*centroidsScaledDF
    centroidsDF.index = centroidsDF.index+1
    centroidsDF.index.name = "cluster"
    centroidsDF.reset_index(inplace = True)
    return centroidsDF
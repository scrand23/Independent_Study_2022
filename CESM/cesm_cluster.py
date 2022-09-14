'''
Author: Sydney Crandall
Last Revised: July 14th, 2022

cluster.py includes various functions for k-means clustering on a pandas dataframe. These include:

    select_time: select the hour at which you would like to cluster the data. This function is designed to represent hourly data on a more daily time-scale.
    
    scale_and_convert_to_numpy: normalize each feature by dividing by its standard deviation to minimize white noise, then convert the resullting pandas dataframe into a numpy array. This is a necessary function before clustering.
    
    cluster_data: runs the kmeans method from scipy.cluster.vq to cluster the scaled numpy array from scale_and_convert_to_numpy. 
    
    plus_plus: imitializes cluster centroids as the maximum distance from a random point in the cluster dataset. This is done to improve cluster accuracy, however, the scipy.cluster.vq.kmeans method does not run multiple iterations if initial centroids are supplied. To support this functionality, a new kmeans method would need to be constructed (potenital for further development).
    
    get_centroids: descales centroid points from cluster_data. This is done for plotting purposes, and is not a necessary step.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import datetime

'''
function to select an hour from hourly data

 inputs:
     dataframe to select from (pandas DataFrame)
     hour to select (int)
     
 returns:
     a dataframe at the desired hour (pandas DataFrame)
'''
def select_time(df, hour):
    selectdf = df[df.hour==hour]
    return(selectdf)


'''
function to scale data to be clustered and convert to a numpy array

 inputs:
     dataframe of data to cluster (pandas DataFrame)
     columns to include in the numpy array (list of strings)
 returns:
     unscaled data (numpy array)
     scaled data (numpy array)
'''
def scale_and_convert_to_numpy(selectdf, included_cols):
    data = selectdf.loc[:, selectdf.columns.isin(list(included_cols))].to_numpy()
    scaled_data = whiten(data)
    return data, scaled_data


'''
function to cluster data using kmeans method from scipy.cluster.vq

 inputs:
     initial centroids or number of centroids (list of numpy arrays or int)
     scaled data to cluster (numpy array)
     
 returns:
     centroid locations (numpy arrays)
     cluster assignment index (numpy array)
'''
def cluster_data(init_centroids, scaled_data):
    centroids, _  = kmeans(scaled_data,init_centroids,iter=100)
    idx, _ = vq(scaled_data,centroids)
    idx=idx+1
    return centroids, idx


"""
**NOT ORIGINAL**
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
def plus_plus(ds, k, random_state=42):
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


'''
function to descale cluster centroids and convert to a pandas DataFrame

 inputs: 
     scaled centroid locations (numpy array)
     column names included in the clustering (list of strings)
     unscaled data chosen for clustering (numpy array)
     
 returns:
     dataframe of centriod locations (pandas DataFrame)
'''
def get_centroids(scaledCentroids, included_cols, data):
    centroidsScaledDF = pd.DataFrame(scaledCentroids, columns = included_cols)
    centroidsDF = pd.DataFrame(data, columns = included_cols).describe().loc["std"]*centroidsScaledDF
    centroidsDF.index = centroidsDF.index+1
    centroidsDF.index.name = "cluster"
    centroidsDF.reset_index(inplace = True)
    return centroidsDF
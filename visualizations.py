#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import datetime as dt

def seasonal_assignment_by_year(city, cluster_label_list, NO_CLUSTERS, selectdf, idx):
    ### find the julian day and year for each point in the dataset
    
    fool=selectdf['datetime'].values
    fool = np.datetime_as_string(fool, unit = 's')
    
    jday = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%j')) for date in fool]
    #print(jday)
    year = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%Y')) for date in fool]
    #print(year)

    ### find the minimum and maximum index of the array for each year
    idx_min=[]
    idx_max=[]
    for yearSEL in np.arange(2012,2022,1):
        #print(yearSEL)
        yearidx=[index for index,value in enumerate(year) if value==yearSEL]
        idx_min.append(min(yearidx))
        idx_max.append(max(yearidx))

    ### plot all years
    plt.figure(figsize = (10,6))

    for y in np.arange(0,10,1):
        plt.plot(jday[np.min(idx_min[y]):np.max(idx_max[y])],idx[np.min(idx_min[y]):np.max(idx_max[y])]+0.08*y,'.',label=str(y+2012))

    plt.xlabel('Day of the year');
    plt.legend(bbox_to_anchor=(1, 0.75), loc='upper left', ncol=1);
    plt.yticks(np.arange(1.25,NO_CLUSTERS+1.25),labels=cluster_label_list)
    plt.title("Seasonal cycle of cluster assignment by year for "+city)
    
    
def variable_scatter_plots(var1, var2, included_cols, NO_CLUSTERS, idx, data,centroids):
    vars2plot = [var1,var2]
    data2plot = [data[:,included_cols.index(var)] for var in vars2plot]

    ## find the integer index of the variable to plot
    varidx2plot=np.zeros(2,dtype="int")
    for i in np.arange(0,2):
      #print(vars2plot[i])
      varidx2plot[i]=included_cols.index(vars2plot[i])
    #print(varidx2plot)

    ### Next plot these variables as the original values with colors to identify the associated cluster
    # (red=1, blue=2, grey=3, orange=4)
    cols = ['','red','blue','grey','orange']
    plt.figure(figsize=(8,5))
    plt.title('K-means classification with ' + str(NO_CLUSTERS) + ' Clusters',fontsize=22)
    for (ind,val) in enumerate(np.transpose(data2plot)):
        plt.plot(val[0],val[1],".", color=cols[idx[ind]], markersize=10, markerfacecolor = 'none')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(vars2plot[0],fontsize=18);
    plt.ylabel(vars2plot[1],fontsize=18);
    plt.scatter(centroids[:, varidx2plot[0]], centroids[:, varidx2plot[1]],color='black',marker='*',s=1000, zorder = 5)
    plt.show()
    
    
def two_cluster_histograms(variable, maximum, minimum, ylim, cluster1, cluster2, bins):
    fig, ax = plt.subplots(2, figsize = (10,7))
    fig.suptitle(variable+" distribution",font = "Times New Roman", fontsize = 22)
    fig.supxlabel(variable, font = "Times New Roman", fontsize = 20)
    fig.supylabel("Count", font = "Times New Roman", fontsize = 20)
    
    ax[0].hist(cluster1[variable], bins = 20)
    ax[1].hist(cluster2[variable], bins = 20)

    ax[0].set_title("cluster1")
    ax[1].set_title("cluster2")

    ax[0].set_xlim(minimum-5,maximum+5)
    ax[1].set_xlim(minimum-5,maximum+5)

    ax[0].set_ylim(0,ylim)
    ax[1].set_ylim(0,ylim)

# In[ ]:





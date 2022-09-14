'''
Author: Sydney Crandall
Last Revised: July 14th, 2022

visualizations.py includes various functions for plotting k-means clustering outputs. These include:

    seasonal_assignment_by_year: plots the cluster assignment for each day, split by year. Axes are designated number of clusters (y) and the day of year (x).
    
    variable_scatter_plots: plots two features (and their centroids) against each other, split by cluster assignment. This is more of a check if the clustering is accurate.
    
    two_cluster_histograms: only applicable for 2 cluster representations of data. Since the best fit for k in this case is 3, this function is not used.
    
    variable_location_scatter: plots the centroid value for one feature by latitude. Standard deviation is portrayed as horizontal lines on either side of the centroid (length +- 1 std), the relative size of the centroid marker the size of the represented cluster (number of days in that cluster), and the mean value of the feature for each location as a black star. Salient characteristics of the represented clusters are denoted by the marker color.
    
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import datetime as dt
import seaborn as sb
import matplotlib as mpl

'''

'''
def seasonal_assignment_by_year(city, cluster_label_list, NO_CLUSTERS, selectdf, idx, minYear, maxYear, save = False, figname = 'None'):
    fool=selectdf['datetime'].values
    fool = np.datetime_as_string(fool, unit = 's')
    
    jday = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%j')) for date in fool]
    #print(jday)
    year = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%Y')) for date in fool]
    #print(year)

    ### find the minimum and maximum index of the array for each year
    idx_min=[]
    idx_max=[]
    for yearSEL in np.arange(minYear,maxYear,1):
        #print(yearSEL)
        yearidx=[index for index,value in enumerate(year) if value==yearSEL]
        idx_min.append(min(yearidx))
        idx_max.append(max(yearidx))

    ### plot all years
    plt.figure(figsize = (10,6))

    for y in np.arange(0,maxYear-minYear,1):
        plt.plot(jday[np.min(idx_min[y]):np.max(idx_max[y])],idx[np.min(idx_min[y]):np.max(idx_max[y])]+0.03*y,'.')

    plt.xlabel('Day of the year');
    #Splt.legend(bbox_to_anchor=(1, 0.75), loc='upper left', ncol=1);
    plt.yticks(np.arange(1.25,NO_CLUSTERS+1.25),labels=cluster_label_list)
    plt.title('Seasonal cycle of cluster assignment by year for {}: {}-{}'.format(city, minYear, maxYear));
    
    if save is True:
        plt.savefig(figname)
    
    
def variable_scatter_plots(var1, var2, included_cols, NO_CLUSTERS, idx, data, centroids, save = False, figname = 'None'):
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
    plt.scatter([centroids[var1][0:]],[centroids[var2][0:]],color='black',marker='*',s=1000, zorder = 5)
    plt.show()
    
    if save is True:
        plt.savefig(figname)
    
    
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
    
    

def variable_location_scatter(variable_name, df, means, groups, alldfGroups, save = False, figname = 'none'):
    offset = [-0.2,0,0.2]
    palette = {"High temp + wet":"firebrick",
           "High temp + dry": "tomato",
           "Low temp + dry":"cornflowerblue", 
           "Low temp + wet":"mediumblue",
           "Low temp": 'cornflowerblue',
           "High temp": 'tomato'}
    
    plt.figure(figsize = (7,6))
    sb.scatterplot(x = variable_name, y = 'latitude', data = df, sizes = (60,200), hue = 'informed_cluster',
                  palette = palette, edgecolor = 'black', legend = 'brief', size = 'count')
    sb.scatterplot(x = variable_name, y = 'latitude', data = means, s = 200, color = 'black', marker = '*', 
                  edgecolor = 'black', linewidth = 0.5)
    
    for group in range(0,len(groups)):
        std = alldfGroups.get_group(groups[group]).describe().loc['std', variable_name]
        mean = alldfGroups.get_group(groups[group]).describe().loc['mean', variable_name]
        latitude = alldfGroups.get_group(groups[group]).describe().loc['mean','latitude']
        plt.plot([mean+std, mean-std],[latitude+offset[group%3], latitude+offset[group%3]],
                color = palette[groups[group][1]], linewidth = 2);
    plt.grid();
    plt.title('Centroid +- 1std for each cluster and location of the cross section: {}'.format(variable_name));
    plt.legend(shadow = True, fontsize = 9, bbox_to_anchor = (1,1));
    
    if save == True:
        plt.savefig(figname)

def variable_location_scatter_compare(variable_name, pdf, fdf, pmeans, fmeans, save = False, figname = 'none'):
    palette = {"High temp + wet":"firebrick",
               "High temp + dry": "tomato",
               "Low temp + dry":"cornflowerblue", 
               "Low temp + wet":"mediumblue",
               "Low temp": 'cornflowerblue',
               "High temp": 'tomato'}

    # to get the open circle markerstyle with outline in palette color
    pnts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pnts) / 2, -np.cos(pnts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    plt.figure(figsize = (7,6))
    sb.scatterplot(x = variable_name, y = 'latitude', data = fdf, sizes = (60,200), hue = 'informed_cluster',
                  palette = palette, edgecolor = 'black', legend = 'brief', size = 'count')
    sb.scatterplot(x = variable_name, y = 'latitude', data = fmeans, s = 200, color = 'black', marker = '*', 
                  edgecolor = 'black', linewidth = 0.5)
    sb.scatterplot(x = variable_name, y = 'latitude', data = pdf, sizes = (60,200), hue = 'informed_cluster',
               palette = palette, facecolors = 'none', size = 'count', marker = open_circle, legend = False, linewidth = 0.25)
    sb.scatterplot(x = variable_name, y = 'latitude', data = pmeans, s = 200, marker = '*',
               edgecolor = 'black', linewidth = 1, facecolor = 'none')

    plt.grid();
    plt.title('Centroid for each cluster and location of the cross section: {}'.format(variable_name));
    plt.legend(shadow = True, fontsize = 9, bbox_to_anchor = (1,1));
    
    if save == True:
        plt.savefig(figname, dpi = 300, bbox_inches = 'tight')


def seasonal_assignment_by_year_compare(pselectdf, pminYear, pmaxYear, pidx, fselectdf, fminYear, fmaxYear, fidx, NO_CLUSTERS, city, cluster_label_list, save = False, figname = 'none'):
    fool=pselectdf['datetime'].values
    fool = np.datetime_as_string(fool, unit = 's')
    
    jday = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%j')) for date in fool]
    #print(jday)
    year = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%Y')) for date in fool]
    #print(year)

    ### find the minimum and maximum index of the array for each year
    pidx_min=[]
    pidx_max=[]
    for yearSEL in np.arange(pminYear,pmaxYear,1):
        #print(yearSEL)
        pyearidx=[index for index,value in enumerate(year) if value==yearSEL]
        pidx_min.append(min(pyearidx))
        pidx_max.append(max(pyearidx))

    fool = fselectdf['datetime'].values
    fool = np.datetime_as_string(fool, unit = 's')

    jday = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%j')) for date in fool]
    #print(jday)
    year = [int(dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').strftime('%Y')) for date in fool]

    fidx_min = []
    fidx_max = []
    for yearSEL in np.arange(fminYear,fmaxYear,1):
        fyearidx = [index for index,value in enumerate(year) if value == yearSEL]
        fidx_min.append(min(fyearidx))
        fidx_max.append(max(fyearidx))

    ### plot all years
    fig, axes = plt.subplots(2,1, figsize = (12,16))

    for y in np.arange(0,pmaxYear-pminYear,1):
        axes[0].plot(jday[np.min(pidx_min[y]):np.max(pidx_max[y])],pidx[np.min(pidx_min[y]):np.max(pidx_max[y])]+0.03*y,'.')
        axes[1].plot(jday[np.min(fidx_min[y]):np.max(fidx_max[y])],fidx[np.min(fidx_min[y]):np.max(fidx_max[y])]+0.03*y,'.')

    axes[0].set_xlabel('Day of the year');
    #Splt.legend(bbox_to_anchor=(1, 0.75), loc='upper left', ncol=1);
    axes[0].set_yticks(np.arange(1.25,NO_CLUSTERS+1.25))
    axes[0].set_yticklabels(cluster_label_list)
    axes[0].set_title('{}-{}'.format(pminYear, pmaxYear));
    axes[1].set_xlabel('Day of the year');
    axes[1].set_yticks(np.arange(1.25,NO_CLUSTERS+1.25))
    axes[1].set_yticklabels(cluster_label_list)
    axes[1].set_title('{}-{}'.format(fminYear, fmaxYear));
    plt.suptitle('Seasonal cycle of cluster assignment by year for {}'.format(city), fontsize = 15)
    

    if save is True:
        plt.savefig(figname)
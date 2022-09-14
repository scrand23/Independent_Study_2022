'''
Author: Sydney Crandall

Last Edited: July 7th, 2022

includes 3 functions: one to pull data from an xarray of cesm data, one to plot the mean of a variable as a map, and one to convert multiple xarray's of cesm data into one pandas dataframe, where locations are selected to match closely to the 7 locations selected from observational LCD data for clustering.
'''


from pathlib import Path 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import pickle
import cartopy.crs as crt
from cartopy.util import add_cyclic_point
from matplotlib.gridspec import GridSpec
import datetime
import pandas as pd
import datetime as dt
import cftime
from matplotlib import ticker

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

'''
function to open and rename CESM datasets into a list of datasets

 inputs:
     location of the files (string)
     short variable names (list of strings)
     end of file name (string)
     
 returns:
     list of xarrays (each index as a variable xarray)
'''
def read_xarray_data(fileroot, variableList, timeSeries):
    dataDict = {}
    for i in range(0,len(variableList)):
        sameVariableList = []
        for j in range(0,3):
            variableNameLong = variableList[i]+'_LON_240e_to_270e_LAT_20n_to_60n'
            sameVariableList.append(xr.open_dataset(fileroot+variableNameLong+timeSeries[j]).rename({'LON_240e_to_270e':'lon','LAT_20n_to_60n':'lat',variableNameLong:variableList[i]})[variableList[i]])
            if j == 2:
                combined_xarray = xr.combine_nested(sameVariableList, concat_dim = 'time')
                dataDict[combined_xarray.name] = combined_xarray
        
    return dataDict
                    
                        

'''
function to pull out the variables from the variable dataset

 inputs into the function:
     xarray dataframe (data). variable of interest
     ex: data.TS

 function returns:
     mean of variable over time
     the values of the variable, mean over time
     latitude associated with that variable
     longitude associated with that variable
'''
    
def pull_data(data_variable):
    
    variable_mean = data_variable.mean(dim = 'time')
    variable_mean_values = variable_mean.values # select only the values of interest
    lat = variable_mean.lat # get the latitude
    lon = variable_mean.lon # get the longitude
        
    # return variable_mean, variable_values, lat, lon (in this order!)
    return variable_mean, variable_mean_values, lat, lon



'''
function for plotting a section of the globe in a similar way as above

 future development:
     be able to change the colormap (cmap)
     be able to change the map style
     be able to change gridline things
     change location of colorbar from right to below?
    

 inputs:
     variable values (variable_values) array
     latitude (lat) array
     longitude (lon) array
     plot domain (domain) list [minlon, maxlon, minlat, maxlat]
     contour levels (minlevel, maxlevel, step) int
     key for colormap (color) string
     figure size (figsize) tuple
     title (title) string
     save (save) bool
     file name for save (filename) string
    
 returns:
     figure (f)
'''

def map_plot(variable_values, lat, lon, color, figsize, title, save = False, filename = 'none'):
    # create the figure
    colormap = {'temp':plt.cm.RdBu_r, 'pres': plt.cm.PRGn, 'prec':plt.cm.BrBG,'wspd':plt.cm.YlGnBu_r, 'misc':plt.cm.hot_r,'hum':plt.cm.BrBG, 'UV':plt.cm.PRGn_r}
    
    f = plt.figure(figsize = figsize)
    ax = plt.axes(projection = crt.PlateCarree())
    
    # set the plot domain
    #ax.set_extent(domain, crt.PlateCarree())
    
    # set the contour levels
    levels = np.linspace(variable_values.min(),variable_values.max(),30)
    
    # plot the variable
    CS = ax.contourf(lon, lat, variable_values, levels, cmap = colormap[color], transform = crt.PlateCarree(), extend = 'both');
    
    ax.coastlines();
    plt.colorbar(CS);
    plt.title(title);
    
    # add lat and lon gridlines
    gl = ax.gridlines(crs=crt.PlateCarree(), draw_labels=True,
                  linewidth=1, color='k', alpha=0.3, linestyle='--');
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # add state boundries
    ax.add_feature(cfeature.STATES, zorder=1, linewidth=0.5, edgecolor='black')
    
    if save is True:
        f.savefig(filename)
        
    return f



'''
function to plot 6 hourly data, by hour
 
 inputs:
     data to plot (xarray)
     key for colormap (string)

 outputs:
    none
'''

def plots_by_hour(data, color):
    dataByHour = data.groupby('time.hour').mean('time')
    hours = [6,12,18,0]
    colormap = {'temp':plt.cm.RdBu_r, 'pres':plt.cm.Oranges,'prec':plt.cm.BrBG,'wspd':plt.cm.YlGnBu_r,'misc':plt.cm.hot_r,'hum':plt.cm.BrBG, 'UV':plt.cm.PRGn_r}
    
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (10,13), subplot_kw = {'projection':crt.PlateCarree()})
    axs = axs.flatten()
    
    if color == 'UV':
        if abs(dataByHour.min()) < abs(dataByHour.max()):
            levels = np.linspace(-dataByHour.max(),dataByHour.max(),30)
        if abs(dataByHour.min()) > abs(dataByHour.max()):
            levels = np.linspace(dataByHour.min(),abs(dataByHour.min(),30))
    else: 
        levels = np.linspace(dataByHour.min(),dataByHour.max(),30)
    
    for i in range(0,4):
        CS = axs[i].contourf(dataByHour.lon, dataByHour.lat, dataByHour.sel(hour = hours[i]), levels, cmap = colormap[color], transform = crt.PlateCarree(), extend = 'both');
        
        axs[i].set_title('Hour = {}'.format(hours[i]))
        axs[i].coastlines();
        plt.suptitle(data.name);
        
        gl = axs[i].gridlines(crs = crt.PlateCarree(), draw_labels = True, linewidth = 1, color = 'k', alpha = 0.3, linestyle = '--');
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        axs[i].add_feature(cfeature.STATES, linewidth = 0.5, edgecolor = 'black');
    
    fig.subplots_adjust(bottom = 0.25, top = 0.95, left = 0.1, right = 0.9, wspace = 0.12, hspace = 0.12)
    
    cbar_ax = fig.add_axes([0.2,0.2,0.6,0.02])
    cbar = fig.colorbar(CS, cax = cbar_ax, orientation = 'horizontal')


    
'''
function to plot daily data as monthly averages 

 inputs:
     data to plot (xarray)
     key for colormap (string)
     
 outputs:
     none
'''

def plots_by_month(data, color, log = False, save = False, figname = 'none'):
    dataByMonth = data.groupby('time.month').mean('time')
    colormap = {'temp':plt.cm.seismic,'pres':plt.cm.Oranges,'prec':plt.cm.BrBG,'wspd':plt.cm.YlGnBu_r,'misc':plt.cm.hot_r,'hum':plt.cm.BrBG, 'UV':plt.cm.PRGn_r, 'Trange':plt.cm.Reds}
    

    fig,axs = plt.subplots(nrows = 2, ncols = 6, figsize = (16,10), subplot_kw = {'projection': crt.PlateCarree()})
    axs = axs.flatten()

    if log is False:
        
        if color == 'UV':
            if abs(dataByMonth.min()) < abs(dataByMonth.max()):
                levels = np.linspace(-dataByMonth.max(),dataByMonth.max(),30)
            if abs(dataByMonth.min()) > abs(dataByMonth.max()):
                levels = np.linspace(dataByMonth.min(),abs(dataByMonth.min()),30)
            cbarlabels = [-3,0,3]
        else:
            levels = np.linspace(dataByMonth.values.min(),dataByMonth.max(),30)
            cbarlabels = np.linspace(dataByMonth.min(),dataByMonth.max(),5).round(5)
        
        for i in range(1,13):
            CS = axs[i-1].contourf(dataByMonth.lon,dataByMonth.lat,dataByMonth.sel(month = i),levels, cmap = colormap[color], transform = crt.PlateCarree(), extend = 'both');
            axs[i-1].set_title('Month = {}'.format(i))
            axs[i-1].coastlines();
            axs[i-1].add_feature(cfeature.OCEAN, zorder=100, edgecolor='k');
            plt.suptitle(dataByMonth.name);
    
            gl = axs[i-1].gridlines(crs = crt.PlateCarree(), draw_labels = False, linewidth = 1, color = 'k', alpha = 0.3, linestyle = '--');
    
            axs[i-1].add_feature(cfeature.STATES, linewidth = 0.5, edgecolor = 'black');
    else:
        
        levels  = np.logspace(dataByMonth.min(),dataByMonth.min()+0.75,num = 30)
        for i in range(1,13):
            CS = axs[i-1].contourf(dataByMonth.lon,dataByMonth.lat,dataByMonth.sel(month = i),levels, cmap = colormap[color], transform = crt.PlateCarree(), extend = 'both', locator = ticker.LogLocator());
            axs[i-1].set_title('Month = {}'.format(i))
            axs[i-1].coastlines();
            axs[i-1].add_feature(cfeature.OCEAN, zorder=100, edgecolor='k');
            plt.suptitle(dataByMonth.name);
    
            gl = axs[i-1].gridlines(crs = crt.PlateCarree(), draw_labels = False, linewidth = 1, color = 'k', alpha = 0.3, linestyle = '--');
    
            axs[i-1].add_feature(cfeature.STATES, linewidth = 0.5, edgecolor = 'black');
        cbarlabels = [1,1.1,1.5,3,5]

    fig.subplots_adjust(bottom = 0.25, top = 0.95, left = 0.1, right = 0.9, wspace = 0.05, hspace = 0.05)

    cbar_ax = fig.add_axes([0.2,0.2,0.6,0.05])
    cbar = fig.colorbar(CS,cax = cbar_ax, orientation = 'horizontal', label = data.attrs['units'])
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    
    if save is True:
        fig.savefig(figname)

'''
function to create a DataFrame from CESM-LE for the locations clustered

 inputs:
     a list of xarray's that contain the data from CESM-LE runs

 returns:
     a DataFrame of variables, time, and location (lat,lon)
'''
    
def extract_cluster_data(dataList):
    locationList = ['elp','alb','trd','bou','cas','bil','gls']
    minLatList = [31,35,37,40,42,45,48]
    maxLatList = [32,36,38,40.5,43,46,49]
    minLonList = [253,253,254,254,253,251,253]
    maxLonList = [254,254,256,256,254,252,254]
    locationDfList = []
    dfList = [] 
    columns_to_drop = []
    for k in range(0,len(keyList)):
        print(keyList[k])
        column_name = keyList[k]+'_'+dataDict[keyList[k]].attrs['units']
        for i in range(0,len(locationList)):
            #select the correct location
            location_data = dataDict[keyList[k]].sel(lat = slice(minLatList[i],maxLatList[i])).sel(lon = slice(minLonList[i],maxLonList[i]))
            # convert xarray to dataframe
            location_df = location_data.to_dataframe()
            # rename the column to something unique
            location_df.rename({keyList[k]:column_name+str(i)}, axis = 1, inplace = True)
            # append the new dataframe to dfList
            locationDfList.append(location_df)
        # combine the locations into one dataframe, all one variable
        variable_df = pd.concat(locationDfList, axis = 1)
        # create a column for the variable, populated with NaN
        variable_df[column_name] = np.NaN
        # iterate through the locations
        for j in range(0,len(locationList)):
            # fill the NaN values in the new column with the values from the location column
            variable_df[column_name].fillna(variable_df[column_name+str(j)], inplace = True)
            columns_to_drop.append(column_name+str(j))
            # add variable_df to the dfList
        dfList.append(variable_df)
    
    # combine all variable_df's in the dfList
    df = pd.concat(dfList, axis = 1)
    df.drop(columns_to_drop, inplace  = True, axis = 1)
    
    return df



'''
function to calculate wind direction and wind speed from U and V components, then adjust wind direction to match meteorological convention

inputs:
    a dataframe including U and V components (pandas DataFrame)

outputs:
    a dataframe including wind speed and direction
'''
    
def wind_speed_direction(df):
    df['W'] = ((df['UBOT']**2)+(df['VBOT']**2))**0.5
    df['WDIR'] = 90 - np.arctan(-df['VBOT']/-df['UBOT'])*180/np.pi
    
    # adjust the wind direction to match meteorological convention
    df['WDIR'].loc[(df['UBOT'] > 0) & (df['VBOT'] > 0)] = df['WDIR']
    df['WDIR'].loc[(df['UBOT'] < 0) & (df['VBOT'] > 0)] = 180 + df['WDIR']
    df['WDIR'].loc[(df['UBOT'] < 0) & (df['VBOT'] < 0)] = 180 + df['WDIR']
    df['WDIR'].loc[(df['UBOT'] > 0) & (df['VBOT'] < 0)] = df['WDIR']
    
    return df



'''
function to alter units on the dataframe for clustering

inputs:
    a dataframe for clustering (pandas DataFrame)
    
outputs:
    a dataframe for clustering (pandas DataFrame)
    
 temperature: K to C (or F) 
 wind: m/s to mph
 pressure: Pa to hPa
 specific humidity: kg/kg to g/kg
 precipitation: m/s to mm/day
'''

def change_units(df):
    df['PRECL'] = df['PRECL']*1000*86400
    df['PRECT'] = df['PRECT']*1000*86400
    
    df['PSL'] = df['PSL']/100
    
    df['QBOT'] = df['QBOT']*1000
    
    df['TREFHT'] = df['TREFHT'] - 273.15
    df['TREFHTMX'] = df['TREFHTMX'] - 273.15
    df['TREFHTMN'] = df['TREFHTMN'] - 273.15
    
    df['WSPDSRFAV'] = df['WSPDSRFAV']*3600/1609.344
    df['WSPDSRFMX'] = df['WSPDSRFMX']*3600/1609.344
    df['W'] = df['W']*3600/1609.344
    df['UBOT'] = df['UBOT']*3600/1609.344
    df['VBOT'] = df['VBOT']*3600/1609.344
    
    df.rename({'FSNS':'FSNS_W_m2','PRECL':'PRECL_mm_day','PRECT':'PRECT_mm_day','PSL':'PSL_hPa','QBOT':'QBOT_g_kg','TMQ':'TMQ_kg_m2','TREFHT':'TREFHT_C','TREFHTMX':'TREFHTMX_C',
               'TREFHTMN':'TREFHTMN_C','WSPDSRFAV':'WSPDSRFAV_mph','WSPDSRFMX':'WSPDSRFMX_mph','W':'W_mph','WDIR':'WDIR_degree','UBOT':'UBOT_mph','VBOT':'VBOT_mph'},
              inplace = True, axis =1)
    return df



'''
function to convert cftime.DatetimeNoLeap to datetime

inputs:
    a cftime.DatetimeNoLeap or another datetime-like object to convert
    
outputs:
    a datetime object with the value of the input variable
'''

def to_datetime(d):

    if isinstance(d, dt.datetime):
        return d
    if isinstance(d, cftime.DatetimeNoLeap):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.DatetimeGregorian):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, str):
        errors = []
        for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.datetime.strptime(d, fmt)
            except ValueError as e:
                errors.append(e)
                continue
        raise Exception(errors)
    elif isinstance(d, np.datetime64):
        return d.astype(dt.datetime)
    else:
        raise Exception("Unknown value: {} type: {}".format(d, type(d))) 
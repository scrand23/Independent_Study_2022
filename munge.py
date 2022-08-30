#!/usr/bin/env python
# coding: utf-8

# In[1]:

"""
to_daily changes to implement:
- change the pressure categories to a daily pressure change
    - this needs to include positive for increasing pressure or negative for decreasing pressure
    - call it daily pressure tendency
- change the dewpoint column to a daily average
    - dewpoint doesn't really change drastically over the course of a day
"""

# import packages
import numpy as np
import pandas as pd
import cluster
from metpy.units import units
import math

# formatting raw lcd file into a usable dataframe
def from_raw(filename):
    # read in the raw lcd file
    # may want to pass use_cols into function in the future, if other columns are needed
    df = pd.read_csv(filename, usecols = [1,43,44,45,49,52,54,55,56,57], low_memory = False)
    
    # convert DATE column into a datetime object
    df['datetime'] = pd.to_datetime(df['DATE'], format='%Y-%m-%dT%H:%M:%S')
    
    ## rename the columns, shorter and include units
    df.rename(columns={'HourlyDryBulbTemperature': 'drytemp_F', 'HourlyDewPointTemperature': 'dewtemp_F',
                       'HourlyPrecipitation': 'prec_inch', 'HourlyRelativeHumidity': 'RH',
                       'HourlyStationPressure': 'pres_Hg', 'HourlyWindDirection': 'wdir',
                       'HourlyWindSpeed': 'wspd_mph', 'HourlyWindGustSpeed': 'wgust_mph',
                       "HourlyWetBulbTemperature": "wetbulb_F"}, inplace=True)
    
    # drop the DATE column
    df.drop("DATE",axis = 1, inplace = True)
    
    # create a column for hour (may be able to delete this later)
    df["hour"] = df["datetime"].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # change the columns from strings to floats
    df["drytemp_F"] = pd.to_numeric(df["drytemp_F"], downcast="float",errors='coerce')
    df["dewtemp_F"] = pd.to_numeric(df["dewtemp_F"], downcast="float",errors='coerce')
    df["prec_inch"] = pd.to_numeric(df["prec_inch"], downcast="float",errors='coerce')
    df["RH"] = pd.to_numeric(df["RH"], downcast="float",errors='coerce')
    df["wdir"] = pd.to_numeric(df["wdir"], downcast="float",errors='coerce')
    df["wspd_mph"] = pd.to_numeric(df["wspd_mph"], downcast="float",errors='coerce')
    df["wgust_mph"] = pd.to_numeric(df["wgust_mph"], downcast="float",errors='coerce')
    df["pres_Hg"] = pd.to_numeric(df["pres_Hg"], downcast="float",errors='coerce')
    df["wetbulb_F"] = pd.to_numeric(df["wetbulb_F"], downcast = "float", errors = 'coerce')

    # convert inches Hg pressure to mbar pressure
    df['pres_Hg']=(df.pres_Hg.values * units('inHg')).to('mbar')

    # rename the pressure column
    df.rename(columns={'pres_Hg': 'pres_mbar'}, inplace=True)
    
    # return the dataframe
    return df

# relate season to month
def season(ser):
    # month-to-season dictionary
    seasons = {(1, 12, 2): 1, (3, 4, 5): 2, (6, 7, 8): 3, (9, 10, 11): 4}
    
    for k in seasons.keys():
        if ser in k:
            return seasons[k]
        
# add season column
def add_season_rearrange(df):
    
    df['tilt_season']= [season(month) for month in pd.DatetimeIndex(df['datetime']).month]
    
    # rearrange the columns 
    df = df[["datetime", "drytemp_F", "dewtemp_F",'wetbulb_F', "RH", 'pres_mbar', 'wdir', 'wspd_mph', 'wgust_mph', 'prec_inch',
             'hour','month', 'tilt_season']]
    
    # return the dataframe
    return df


# replace missing values by month and hour
def replace_month_hour(df):
    # replace missing values with the average of the value for its season and hour
    df['wgust_mph'] = df.groupby(['month', 'hour']).wgust_mph.apply(lambda x: x.fillna(x.mean())).round(decimals = 0)
    df['drytemp_F'] = df.groupby(['month', 'hour']).drytemp_F.apply(lambda x: x.fillna(x.mean())).round(decimals = 0)
    df['dewtemp_F'] = df.groupby(['month', 'hour']).dewtemp_F.apply(lambda x: x.fillna(x.mean())).round(decimals = 0)
    df['RH'] = df.groupby(['month', 'hour']).RH.apply(lambda x: x.fillna(x.mean())).round(decimals = 0)
    df['wdir'] = df.groupby(['month', 'hour']).wdir.apply(lambda x: x.fillna(x.mean())).round(decimals = 0)
    df['wspd_mph'] = df.groupby(['month', 'hour']).wspd_mph.apply(lambda x: x.fillna(x.mean())).round(decimals = 0)
    df['pres_mbar'] = df.groupby(['month', 'hour']).pres_mbar.apply(lambda x: x.fillna(x.mean())).round(decimals = 0)
    df['prec_inch'] = df["prec_inch"].fillna(0) # replace missing precip values with 0
    
    # print the number of missing values in each column to check that it worked
    #print(df.isnull().sum())
    
    # return the dataframe
    return df

# calculate wet bulb based on air temp and rh
def wet_bulb_tw(t, rh):
    a = t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
    b = math.atan(t + rh) - math.atan(rh - 1.676331)
    c = 0.00391838 * rh ** (1.5) * math.atan(0.023101 * rh)
    d = -4.686035
    tw = a + b + c + d
    return tw

# # replace all values in the wetbulb temperature column with this equation 
# Tw = T * arctan[0.151977 * (rh% + 8.313659)^(1/2)] + arctan(T + rh%) - arctan(rh% - 1.676331) + 0.00391838 *(rh%)^(3/2) * arctan(0.023101 * rh%) - 4.686035
# from here: https://jccraig.medium.com/calculate-dangerous-wet-bulb-temperatures-with-python-7f38585b8a81
def wet_bulb(df):
    for row in range(0,len(df)):
        if pd.isnull(df.loc[row,"wetbulb_F"]):
            df.loc[row,"wetbulb_F"] = wet_bulb_tw(df["drytemp_F"][row], df["RH"][row]).round(decimals = 0)
    return df
        
# make the dataset into an hourly dataset
def to_hourly(df):
    # group by date and hour and keep the median values of each day
    dfHourly = df.groupby(pd.Grouper(key = "datetime", axis = 0, freq = "H")).median().reset_index()

        
    # print the number of hours without data
    #print(dfHourly.isnull().sum())
    
    # drop any rows (hours) without data
    dfHourly.fillna(method = 'ffill',inplace = True)
    
    # return the dataframe
    return dfHourly # save this dataframe after return

# make the dataset into a daily dataset
def to_daily(dfHourly):
    # group the data by day and aggregate on the sum of the precip column
    dfPrecip = dfHourly[["prec_inch", "datetime"]].groupby(pd.Grouper(key='datetime', axis=0,
                                                                                 freq='D')).sum()
    dfMax = dfHourly[["drytemp_F", "pres_mbar","RH", "datetime", "tilt_season", 'wspd_mph']].groupby(pd.Grouper(key = "datetime", axis = 0, freq = "D")).max()
    dfMax.rename(columns = {"drytemp_F": "drytemp_max", "pres_mbar":"pres_max", "RH":"RH_max", 'wspd_mph':'wspd_max'}, inplace = True)

    dfMin = dfHourly[["drytemp_F", "pres_mbar","RH", "datetime", 'wspd_mph']].groupby(pd.Grouper(key = "datetime", axis = 0, freq = "D")).min()
    dfMin.rename(columns = {"drytemp_F": "drytemp_min", "pres_mbar":"pres_min" ,"RH":"RH_min", 'wspd_mph':'wspd_min'}, inplace = True)

    dfAvg = dfHourly[["wdir","datetime", "dewtemp_F", 'wspd_mph', 'wetbulb_F']].groupby(pd.Grouper(key = "datetime", axis = 0, freq = "D")).mean().round(decimals = 0)
    dfAvg.rename(columns = {"wdir":"wdir_avg","dewtemp_F":'dewtemp_avg', 'wspd_mph':'wspd_avg', 'wetbulb_F': 'wetbulb_avg'}, inplace = True)
    
    # concatinate boulderMax, boulderMin, boulderAvg, and boulderPrecip into one daily dataframe
    dfDaily = pd.concat([dfMax, dfMin, dfAvg, dfPrecip], axis = 1).reset_index()
    
    # change the order of the variables
    dfDaily = dfDaily[["datetime","tilt_season", "drytemp_max", "drytemp_min", "dewtemp_avg",'wetbulb_avg',"RH_max", "RH_min", "pres_max","pres_min", "wspd_max", 'wspd_min', "wdir_avg", "prec_inch"]]
    
    # print the number of days without data
    #print(dfDaily.isnull().sum())
    
    # drop any rows (days) without data
    dfDaily.fillna(method = "ffill", inplace =True)
    
    # add time data for maxtemp, mintemp, and maxprecip
    maxTemp = dfHourly.loc[dfHourly.groupby(pd.Grouper(key = "datetime", axis = 1, freq= "D"))['drytemp_F'].idxmax()].reset_index(drop = True)
    dfDaily["drytemp_max_hour"] = maxTemp["datetime"].dt.hour
    minTemp = dfHourly.loc[dfHourly.groupby(pd.Grouper(key = "datetime", axis = 1, freq= "D"))['drytemp_F'].idxmin()].reset_index(drop = True)
    dfDaily["drytemp_min_hour"] = minTemp["datetime"].dt.hour
    maxPrecip = dfHourly.loc[dfHourly.groupby(pd.Grouper(key = "datetime", axis = 1, freq = "D"))["prec_inch"].idxmax()].reset_index(drop = True)
    dfDaily["prec_max_hour"] = maxPrecip["datetime"].dt.hour
    
    maxPres = dfHourly.loc[dfHourly.groupby(pd.Grouper(key = "datetime", axis = 1, freq= "D"))['pres_mbar'].idxmax()].reset_index(drop = True)
    dfDaily["pres_max_hour"] = maxPres["datetime"].dt.hour

    minPres = dfHourly.loc[dfHourly.groupby(pd.Grouper(key = "datetime", axis = 1, freq= "D"))['pres_mbar'].idxmin()].reset_index(drop = True)
    dfDaily["pres_min_hour"] = minPres["datetime"].dt.hour
    
    dfDaily["pres_tend"] = 'NaN'
    for row in range(0,len(dfDaily)):
        if dfDaily.loc[row,"pres_min_hour"] > dfDaily.loc[row,"pres_max_hour"]:
            dfDaily.loc[row,"pres_tend"] = int(dfDaily["pres_min"][row]) - int(dfDaily["pres_max"][row])
        elif(dfDaily.loc[row,'pres_min_hour'] <= dfDaily.loc[row,'pres_max_hour']):
            dfDaily.loc[row,'pres_tend'] = int(dfDaily.loc[row,"pres_max"]) - int(dfDaily.loc[row,"pres_min"])
            
    # convert pres_tend from int to float
    dfDaily["pres_tend"] = pd.to_numeric(dfDaily["pres_tend"], downcast = "float", errors = 'coerce')

    # return the dataframe
    return dfDaily # save this dataframe after return


# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:51:25 2022

@author: nelsonng
"""

import numpy as np 

from datetime import date 
from scipy import stats 

# ----------------------------------------------------------------------------------------------

# Subfunction of ctry_a30_df func and print_ctry_a30 func. 
def ctry_a30_lt(df): 
    
    ''' Identify countries with more than 30 data points.'''
    
    # Create a deepcopy of the dataframe.
    df = df.copy(True)
    
    # Groupby 'Origin' column. 
    df = df.groupby('Origin').agg('count')
    
    # Filter for countries with more than 30 data points in df.
    df = df[df['Species'] >= 30]
    
    # Reset index for cfpq_gb_a30_df. 
    df.reset_index(inplace=True)

    # Create a list of countries with more than 30 data points. 
    ctry_a30_lt = df['Origin'].to_list()

    return ctry_a30_lt 

def ctry_a30_df(df, ctry_a30_lt):
    
    '''Create a dataframe with country with more than 30 data points and without ethopia.'''
    
    # Create a deep copy of the dataframe.
    df = df.copy(True) 
    
    # Reset index of df. 
    df.reset_index(inplace=True) 
    
    ind_lt = []
    
    # Append index of rows that do not have country with more than 30 data points and ethopia. 
    for i, j in enumerate(df['Origin']):
        if j not in ctry_a30_lt:
            ind_lt.append(i) 
            
    # Drop index of rows that do not have country with more than 30 data points and ethopia. 
    df.drop(index=ind_lt, inplace=True)
    
    # Reset index of df.
    df.reset_index(inplace=True)
    
    # Drop columns.
    df.drop(columns=[df.columns[0], df.columns[1]], inplace=True)
    
    return df 

# ----------------------------------------------------------------------------------------------

def print_ctry_a30(ctry_a30_lt): 
    
    '''Print countries that have >= 30 data points.'''
    
    # Print bold text.
    print('\033[1m' + 'Countries that have >= 30 data points excluding Ethopia:' + '\033[0m')

    # Print the countries. 
    for i in ctry_a30_lt:
        print(i) 

# ----------------------------------------------------------------------------------------------

# Subfunction of start_end_dates_tp. 
def cnvt_datetime_to_str(dt_datetime):
    
    '''Convert date from datetime to str.'''
    
    dt_str = np.datetime_as_string(dt_datetime, unit='D') 
    
    return dt_str

def start_end_dates_tp(df):
    
    '''Create a tuple of the start and end dates of the dataframe.'''
    
    # Find the date from the first row of 'Date' column. 
    start_date = df['Date'].head(1).values[0]
    
    # Find the date from the last row of 'Date' column. 
    end_date = df['Date'].tail(1).values[0]
    
    # Convert date from datetime to str. 
    start_date = cnvt_datetime_to_str(start_date)
    end_date = cnvt_datetime_to_str(end_date)
    
    return start_date, end_date 

# ----------------------------------------------------------------------------------------------

# Subfunction of cnvt_dt_str_to_datetime. 
def year_mth_day_tp(dt_str):
    
    '''Create a tuple made of year, month and day.'''
    
    # Iterate after splitting based on '-'. 
    for i, j in enumerate(dt_str.split('-')):

        # Sort for year. 
        if i == 0:
            year = int(j)

        # Sort for month.
        elif i == 1:
            mth = int(j)

        # Sort for day.
        elif i == 2:
            day = int(j)
    
    return year, mth, day  

# Subfunction of dt_str_fmt. 
def cnvt_dt_str_to_datetime(dt_str): 
    
    '''Convert date from str to datetime format.'''
    
    # Create a tuple made of year, mth, day. 
    year, mth, day = year_mth_day_tp(dt_str)
    
    # Create date into datetime format. 
    return date(year, mth, day) 

def dt_str_fmt(dt_str):
    
    '''Rearrange datetime str format.'''
    
    # Convert datetime str format to datetime. 
    dt_str = cnvt_dt_str_to_datetime(dt_str)
    
    # Convert datetime back to a new str format. 
    dt_str_fmt = dt_str.strftime("%d %B, %Y")
     
    return dt_str_fmt 

# ----------------------------------------------------------------------------------------------

def thold_inliner_outlier_tp(df, col, thold): 
    
    '''Create a tuple of inliner index, outlier index, number of inliners, and number of outliers.'''
    
    # Find the z score for each data point based on column.
    z = np.abs(stats.zscore(df[col]))

    # Index to keep based on threshold.
    in_ind = np.where(z < thold)

    # Index of outliers based on threshold.
    out_ind = np.where(z > thold)

    # Number of datapoints to keep based on threshold.
    in_no = len(in_ind[0])

    # Number of outliers to remove based on threshold.
    out_no = len(out_ind[0])
    
    return in_ind, out_ind, in_no, out_no  


def yr_lt(df, col):
    
    '''Create a list of year.'''
    
    yr_lt = []
    
    # Iterate over col of df. 
    for i in df[col].values:
        
        # Convert format to str.
        i = str(i)
        
        # Convert format to int and append into yr_lt. 
        yr_lt.append(int(i[:4]))
    
    return yr_lt


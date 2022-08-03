#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:56:30 2022

@author: nelsonng
"""

import matplotlib.pyplot as plt 
import missingno as msno 

# Import my packages. 
import my_Functions_cf_graph as myfcf_gra 

# ----------------------------------------------------------------------------------------------

def null_barchart(df, category):

    '''Display bar chart to identify missing null values.'''
    
    # Reset layout to default. 
    myfcf_gra.set_rcparams() 
    
    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height # 8, 3 
    
    # Plot bar chart. 
    df.isna().sum().plot(kind="bar", color='red')
    
    # Title of the bar chart. 
    plt.title('Bar Chart of Null values\n(' + str(category) + ')', size=20)  
    
    # Display the bar chart. 
    plt.show() 


def print_null_col(df):
    
    '''Print null values of columns.'''
    
    # Null values per column
    print('\033[1m' + 'Column              No. of Null Values' + '\033[0m') 
    print(df.isnull().sum())     


def null_msno(df, title_size):
    
    '''Display null values.'''

    # Plot msno.matrix
    g = msno.matrix(df)
    
    # Set title for the missingno. 
    g.set_title('Missingno of Non-null and Null Values', size=title_size)
    
    # Display msno.matrix.
    plt.show() 


def remove_null_row(df, col):
    
    '''Remove null row of dataframe.'''
    
    # Identify the index of the null values.
    labels=[x for x in df[df[col].isnull()].index]

    # Drop null rows of df basing on identified index.
    df.drop(labels=labels, axis=0, inplace=True) 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:48:16 2022

@author: nelsonng
"""

# Import libraries.  
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy.stats as stats 
import seaborn as sns 

from scipy.stats import boxcox 
from scipy.stats import kruskal 
from scipy.stats import normaltest
from scipy.stats import shapiro 
from scipy.stats import skew  
from statsmodels.stats.weightstats import ztest 

# Import my packages. 
import my_Functions_cf_wrangle as myfcf_wra 
import my_Functions_cf_graph as myfcf_gra 

# ----------------------------------------------------------------------------------------------

def h_boxplot(df, col, start_date, end_date):
    
    '''Create a horizontal boxplot.'''
    
    # Set graph layout. 
    myfcf_gra.set_sns_font_dpi()
    myfcf_gra.set_sns_large_white()
    
    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height

    # Create boxplot. 
    sns.boxplot(x = df[col], orient = 'h', showmeans=True)
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date) 
    
    # Set title. 
    plt.title('Box Plot\n(' + col + ': ' + str(start_date) + ' - ' + str(end_date) + ')', size=20)
    
    # Display plot. 
    plt.show() 

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_shap_wilk_test func. 
def shap_wilk_test(df, col, alpha, start_date, end_date):
    
    '''Perform normality test using shapiro-wilk.''' 
    
    # Perform a normal test using shapiro-wilk.
    stat, p = shapiro(df[col]) 
    
    # Convert alpha into percentage. 
    alpha_pct = alpha*100  
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Print out the result for the normality check.
    print('\033[1m' + 'Shapiro–Wilk Test for ' + str(col) + ' (' + str(start_date) + ' - ' + str(end_date) + ')' + '\033[0m')
    print('stat=%.2f, p=%.3f' % (stat, p))
    
    if p < alpha:
        text1 = '\033[3;37;41m' + 'Reject Ho and conclude that there is significant evidence, at ' + str(alpha_pct) + '% level of significance, that the population is not normally-distributed.' + '\033[0;30;0m' 
        print(text1)

    else:
        text2 = '\033[3;37;42m' + 'Do not reject Ho and conclude that there is no significant evidence, at ' + str(alpha_pct) + '% level of significance, that the population is normally-distributed.' + '\033[0;30;0m' 
        print(text2)

def print_mult_shap_wilk_test(df, col, alpha_lt, start_date, end_date):
    
    '''Print a list of shapiro-wilk test results from alpha_lt.'''
    
    # Iterate over alpha_lt.
    for i in alpha_lt:

        # Assign i to alpha. 
        alpha = i

        # Convert alpha into percentage. 
        alpha_pct = alpha*100  

        # Print level of significance text. 
        sig_txt = '\033[1;30;45m' + 'Shapiro–Wilk Test at ' + str(alpha_pct) + '%' + ' level of significance,'  + '\033[0;30;0m'
        print(sig_txt)

        # Print shaprio-wilk test result. 
        shap_wilk_test(df, col, alpha, start_date, end_date)
        print('\n') 

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_normal_test func.   
def normal_test(df, col, alpha, start_date, end_date):
    
    '''Perform normality test based on d’agostino and pearson’s test.''' 
    
    # Perform a normal test using normality test based on d’agostino and pearson’s test. 
    stat, p = stats.normaltest(df[col])
    
    # Convert alpha into percentage. 
    alpha_pct = alpha*100  
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Print out the result for the normality check. 
    print('\033[1m' + "D'Agostino-Pearson Test for " + str(col) + ' (' + str(start_date) + ' - ' + str(end_date) + ')' + '\033[0m')
    print('stat=%.2f, p=%.3f' % (stat, p)) 
    
    if p < alpha:
        text1 = '\033[3;37;41m' + 'Reject Ho and conclude that there is significant evidence, at ' + str(alpha_pct) + '% level of significance, that the population is not normally-distributed.' + '\033[0;30;0m' 
        print(text1)

    else:
        text2 = '\033[3;37;42m' + 'Do not reject Ho and conclude that there is no significant evidence, at ' + str(alpha_pct) + '% level of significance, that the population is normally-distributed.' + '\033[0;30;0m' 
        print(text2) 

def print_mult_normal_test(df, col, alpha_lt, start_date, end_date):
    
    '''Print a list of normality test based on d’agostino and pearson’s test results from alpha_lt.'''
    
    # Iterate over alpha_lt.
    for i in alpha_lt:

        # Assign i to alpha. 
        alpha = i

        # Convert alpha into percentage. 
        alpha_pct = alpha*100  

        # Print level of significance text. 
        sig_txt = '\033[1;30;45m' + "D'Agostino-Pearson Test at " + str(alpha_pct) + '%' + ' level of significance,'  + '\033[0;30;0m'
        print(sig_txt) 

        # Print normality test based on d’agostino and pearson’s test result. 
        normal_test(df, col, alpha, start_date, end_date)
        print('\n') 

# ----------------------------------------------------------------------------------------------

def prob_plot(df, col, start_date, end_date):
    
    '''Create a probability plot for normal distribution.'''
    
    # Set graph layout. 
    myfcf_gra.set_sns_font_dpi()
    myfcf_gra.set_sns_large_white()

    # Set fig and ax. 
    fig = plt.figure()
    ax = fig.add_subplot(111) 

    # Plot probability plot for normal distribution. 
    res = stats.probplot(df[col], dist=stats.norm, plot=ax) # Ignore the sypder error warning: code is correct. 
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date) 
    end_date = myfcf_wra.dt_str_fmt(end_date) 
    
    # Set title.
    ax.set_title('Probability Plot for Normal Distribution\n(' + col + ': ' + str(start_date) + ' - ' + str(end_date) + ')', size=20) 

    # Display plot.  
    plt.show() 

def kde_norm_distplot(df, col, start_date, end_date):

    '''Plot gaussian kernel density estimate and normal distribution.'''
    
    # Set graph layout. 
    myfcf_gra.set_sns_font_dpi()
    myfcf_gra.set_sns_large_white() 

    # Add normal distribution and KDE.
    sns.distplot(df[col], fit=stats.norm, kde=True)
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date) 
    
    # Set title. 
    plt.title('Gaussian Kernel Density Estimate\n(' + col + ': ' + str(start_date) + ' - ' + str(end_date) + ')', size=20)

    # Set xlabel.
    plt.xlabel(col)

    # Display plot. 
    plt.show()    

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_krus_wall_test_9df func. 
def print_ctry_a30_krus_wall_test_9df(df1, df2, df3, df4, df5, df6, df7, df8, df9, x, y, alpha, start_date, end_date):
    
    '''Print kruskal-wallis h test results of countries (>= 30 data points) for 9 df.'''
    
    # Perform kruskal-wallis h test.
    stat, p = kruskal(df1[x], df2[x], df3[x], 
                      df4[x], df5[x], df6[x], 
                      df7[x], df8[x], df9[x])
    
    # Convert alpha into percentage. 
    alpha_pct = alpha*100  
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)

    # Print result for total quality score. 
    if y == 'Total Quality Score': 
        
        print('\033[1;30;1m' + 'Kruskal-Wallis H Test for ' + y + ' and ' + x + ' (' + str(start_date) + ' - ' + str(end_date) + ')' + '\033[0;30;0m')
        print('stat=%.2f, p=%.3f' % (stat, p))
        
        if p < alpha:
                text1 = '\033[3;37;41m' + 'Reject Ho and conclude that there is significant evidence, at ' + str(alpha_pct) + '% level of significance, that total quality score is systematically higher or different in some countries than in others.' + '\033[0;30;0m' 
                print(text1)

        else:
            text2 = '\033[3;37;42m' + 'Do not reject Ho and conclude that there is no significant evidence, at ' + str(alpha_pct) + '% level of significance, that total quality score has the same distribution in all countries.' + '\033[0;30;0m' 
            print(text2) 
  
def print_mult_krus_wall_test_9df(df1, df2, df3, df4, df5, df6, df7, df8, df9, x, y, alpha_lt, start_date, end_date):
    
    '''Print a list of kruskal-wallis H test results from alpha_lt.'''
    
    # Iterate over alpha_lt. 
    for i in alpha_lt:

        # Assign i to alpha. 
        alpha = i

        # Convert alpha into percentage. 
        alpha_pct = alpha*100  

        # Print level of significance text. 
        sig_txt = '\033[1;30;45m' + 'Kruskal-Wallis H Test at ' + str(alpha_pct) + '%' + ' level of significance,'  + '\033[0;30;0m'
        print(sig_txt)

        # Print kruskal-wallis h test results. 
        print_ctry_a30_krus_wall_test_9df(df1, df2, df3, df4, df5, df6, df7, df8, df9, x, y, alpha, start_date, end_date)
        print('\n')      

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_mann_whit_test func. 
def print_mann_whit_test(df, x, y, alpha, start_date, end_date):
    
    '''Print mann-whitney u test two-sided result.'''
    
    # Convert alpha into percentage. 
    alpha_pct = alpha*100  
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Create a list of unique values from selected column. 
    x_lt = [i for i in df[x].unique()] 
    i_j_lt = [] 
    
    # Nested list.  
    for i in x_lt:
        for j in x_lt:

            # Ensure that i is not equal to j and (j, i) is not inside the list (this done to prevent duplicate result).
            if i != j and (j, i) not in i_j_lt: 
                
                # Run mann-whitney u test.
                (stat, p) = stats.mannwhitneyu(df[df[x] == i][y],
                                               df[df[x] == j][y],
                                               alternative='two-sided')
                
                # Append i, j and p_val into i_j_p_lt. 
                i_j_lt.append((i, j))
                i_j_lt.append((j, i)) # Append (j, i) into i_j_lt to prevent duplicate result. 
                
                # Print mann-whitney u test result. 
                print('\033[1;30;1m' + 'Mann-Whitney U Test for ' + y + ' and ' + x + ' (' + str(i) + ' and ' + str(j) + ')' + ' (' + str(start_date) + ' - ' + str(end_date) + ')' + '\033[0;30;0m')
                print('stat=%.2f, p=%.3f' % (stat, p))
                
                if p < alpha:
                    text1 = '\033[3;37;41m' + 'Reject Ho and conclude that there is significant evidence, at ' + str(alpha_pct) + '% level of significance, that mean rank of ' + str(y) + ' for ' + str(i) + ' and mean rank of ' + str(y) + ' for ' + str(j) + ' is not equal.' + '\033[0;30;0m' 
                    print(text1)
                    
                else: 
                    text2 = '\033[3;37;42m' + 'Do not reject Ho and conclude that there is no significant evidence, at ' + str(alpha_pct) + '% level of significance, that mean rank of ' + str(y) + ' for ' + str(i) + ' and mean rank of ' + str(y) + ' for ' + str(j) + ' is equal.' + '\033[0;30;0m' 
                    print(text2) 

def print_mult_mann_whit_test(df, x, y, alpha_lt, start_date, end_date):
    
    '''Print a list of mann-whitney u test two-sided results from alpha_lt.'''
    
    # Iterate over alpha_lt. 
    for i in alpha_lt: 

        # Assign i to alpha. 
        alpha = i

        # Convert alpha into percentage. 
        alpha_pct = alpha*100  

        # Print level of significance text. 
        sig_txt = '\033[1;30;45m' + 'Mann-Whitney U Test at ' + str(alpha_pct) + '%' + ' level of significance,'  + '\033[0;30;0m'
        print(sig_txt)

        # Print mann-whitney u test results. 
        print_mann_whit_test(df, x, y, alpha, start_date, end_date)
        print('\n') 

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_krus_wall_test_3df func. 
def print_ctry_a30_krus_wall_test_3df(df1, df2, df3, x, y, alpha, start_date, end_date):
    
    '''Print kruskal-wallis h test results of countries (>= 30 data points) for 3 df.'''
    
    # Perform kruskal-wallis h test.
    stat, p = kruskal(df1[x], df2[x], df3[x])
    
    # Convert alpha into percentage. 
    alpha_pct = alpha*100  
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Print result for total quality score classification. 
    if x == 'Total Quality Score Classification':
        
        print('\033[1;30;1m' + 'Kruskal-Wallis H Test for ' + y + ' and ' + x + ' (' + str(start_date) + ' - ' + str(end_date) + ')' + '\033[0;30;0m')
        print('stat=%.2f, p=%.3f' % (stat, p))
        
        if p < alpha:
                text1 = '\033[3;37;41m' + 'Reject Ho and conclude that there is significant evidence, at ' + str(alpha_pct) + '% level of significance, that coffee price (USD) is systematically higher or different in some total quality score classification than in others.' + '\033[0;30;0m' 
                print(text1)

        else:
            text2 = '\033[3;37;42m' + 'Do not reject Ho and conclude that there is no significant evidence, at ' + str(alpha_pct) + '% level of significance, that coffee price (USD) has the same distribution in all total quality score classification.' + '\033[0;30;0m' 
            print(text2)

def print_mult_krus_wall_test_3df(df1, df2, df3, x, y, alpha_lt, start_date, end_date):
    
    '''Print a list of kruskal-wallis H test results from alpha_lt.'''
    
    # Iterate over alpha_lt. 
    for i in alpha_lt:

        # Assign i to alpha. 
        alpha = i

        # Convert alpha into percentage. 
        alpha_pct = alpha*100  
        
        # Print level of significance text. 
        sig_txt = '\033[1;30;45m' + 'Kruskal-Wallis H Test at ' + str(alpha_pct) + '%' + ' level of significance,'  + '\033[0;30;0m'
        print(sig_txt)

        # Print kruskal-wallis h test results. 
        print_ctry_a30_krus_wall_test_3df(df1, df2, df3, x, y, alpha, start_date, end_date)
        print('\n')

# ----------------------------------------------------------------------------------------------

# Subfunction of box_cox_df func. 
def bc_strt_dt_p_lambda_tp(df, col):
    
    '''Create a tuple of lists: start date, p-value, lambda, box-cox, skew.'''
    
    strt_dt_lt = []
    p_lt = []
    lambda_lt = []
    box_cox_lt = [] 
    box_cox_skew_lt = []
    
    # Run box-cox transformation.
    try: 
        # Iterate over dataframe. 
        for i, j in enumerate(df['Date']):

            # Create start date in str format. 
            j = str(j)[:10] 

            # Sort dataframe based on selected date. 
            df = df.loc[df['Date'] >= j]  

            # Append date into strt_dt_lt.
            strt_dt_lt.append(j)

            # Perform box-cox transformation. 
            box_cox, best_Lambda_maxlog = boxcox(df[col])
            
            # Perform d'agostino-pearson test. 
            stat, p = normaltest(box_cox)  
            
            # Find skew after performing box-cox transformation. 
            box_cox_skew = np.round(skew(box_cox),2)

            # Append p-value and lamba into p_lt and lambda_lt, box_cox_lt, and box_cox_skew_lt. 
            p_lt.append(p)
            lambda_lt.append(best_Lambda_maxlog)
            box_cox_lt.append(box_cox)
            box_cox_skew_lt.append(box_cox_skew)
    
    # Unable to run box-cox transformation.
    except:
         
        # Remove the last index of strt_dt_lt. 
        strt_dt_lt.pop(-1)

    return strt_dt_lt, p_lt, lambda_lt, box_cox_lt, box_cox_skew_lt 

def box_cox_df(df, col):
    
    '''Create dataframe of with columns: start date, p-value, lambda, box-cox, skew.'''
    
    # Create lists of start date, p-value and lamba, skew.
    strt_dt_lt, p_lt, lambda_lt, box_cox_lt, box_cox_skew_lt = bc_strt_dt_p_lambda_tp(df, col) 
    
    # Create a dataframe for date and lambda. 
    box_cox_df = pd.DataFrame(p_lt, strt_dt_lt)

    # Reset index of dataframe.
    box_cox_df.reset_index(inplace=True)
 
    # Add columns. 
    box_cox_df[1] = lambda_lt
    box_cox_df[2] = box_cox_lt
    box_cox_df[3] = box_cox_skew_lt

    # Rename columns of dataframe. 
    box_cox_df.rename(columns={box_cox_df.columns[0]: 'Start Date', 
                               box_cox_df.columns[1]: 'p-value',
                               box_cox_df.columns[2]: 'Lambda',
                               box_cox_df.columns[3]: 'Box-Cox',
                               box_cox_df.columns[4]: 'Skew'}, inplace=True) 
    
    return box_cox_df  

# ---------------------------------------------------------------------------------------------- 

def new_cfp_df(df, strt_dt):
    
    '''Filter dataframe for transformation.'''
    
    # Slice dataframe based on start date. 
    df = df.loc[df['Date'] >= strt_dt]

    return df  

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_one_sample_ztest func. 
def box_cox_pop_samp_mean_tp(pop_df, samp_df, col, trans_cat): 
    
    '''Tuple of box-cox population mean and box-cox sample mean.'''
    
    if trans_cat == 'Box-Cox': 
        trans_cat = 'Box-Cox Transformation of ' + str(col)
    
    # Calculate box-cox sample mean. 
    box_cox_samp_mean = samp_df[trans_cat].mean()

    # Find best_Lambda_maxlog from sample mean.
    box_cox, best_Lambda_maxlog = boxcox(samp_df[trans_cat]) 

    # Calculate box-cox population mean. 
    box_cox_pop_mean = boxcox(pop_df['Coffee Price (USD)'], best_Lambda_maxlog).mean()
    
    return box_cox_pop_mean, box_cox_samp_mean 

# Subfunction of print_mult_one_sample_ztest func. 
def print_one_sample_ztest(samp_df, col, pop_mean, sign, alpha, start_date, end_date): 
    
    '''Perform one sample z-test result.'''
    
    # Convert alpha into percentage. 
    alpha_pct = alpha*100  
    
    if 'Transform' in col:
        trans_txt = 'transformed '
    else:
        trans_txt = ''
    
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Perform one-tailed z-test.
    stat, p = ztest(samp_df[col], value=pop_mean, 
                    alternative=sign, ddof=1.0) 
    
    # Print one sample z-test result. 
    print('\033[1;30;1m' + '1-Sample Z-Test for ' + col + ' (' + str(start_date) + ' - ' + str(end_date) + ')' + '\033[0;30;0m')
    print('stat=%.2f, p=%.3f' % (stat, p))
    
    # Segregated based on sign. 
    if sign == 'smaller' or sign == 'larger': 
        
        if p < alpha:
            text1 = '\033[3;37;41m' + 'Reject Ho and conclude that there is significant evidence, at ' + str(alpha_pct) + '% level of significance, that ' + trans_txt + 'sample mean is ' + sign + ' than the ' + trans_txt + 'population mean.' + '\033[0;30;0m'
            print(text1)

        else: 
            text2 = '\033[3;37;42m' + 'Do not reject Ho and conclude that there is no significant evidence, at ' + str(alpha_pct) + '% level of significance, that ' + trans_txt + 'sample mean is ' + sign + ' than the ' + trans_txt + 'population mean.' + '\033[0;30;0m'
            print(text2)
    
    # sign == 'two-sided'
    else: 
        
        if p < alpha:
            text1 = '\033[3;37;41m' + 'Reject Ho and conclude that there is significant evidence, at ' + str(alpha_pct) + '% level of significance, that ' + trans_txt + 'sample mean is not equal to the ' + trans_txt + 'population mean.' + '\033[0;30;0m'
            print(text1)

        else: 
            text2 = '\033[3;37;42m' + 'Do not reject Ho and conclude that there is no significant evidence, at ' + str(alpha_pct) + '% level of significance, that ' + trans_txt + 'sample mean is equal to the ' + trans_txt + 'population mean.' + '\033[0;30;0m'
            print(text2)

def print_mult_one_sample_ztest(samp_df, col, pop_mean, sign, alpha_lt, start_date, end_date):
    
    '''Print a list of 1-sample z-test from alpha_lt.'''
   
    # Iterate over alpha_lt. 
    for i in alpha_lt: 

        # Assign i to alpha. 
        alpha = i

        # Convert alpha into percentage. 
        alpha_pct = alpha*100  

        # Print level of significance text. 
        sig_txt = '\033[1;30;45m' + '1-sample Z Test at ' + str(alpha_pct) + '%' + ' level of significance,'  + '\033[0;30;0m'
        print(sig_txt)

        # Run one_sample_ztest func.
        print_one_sample_ztest(samp_df, col, pop_mean, sign, alpha, start_date, end_date)
        print('\n') 

# ----------------------------------------------------------------------------------------------

# Subfunction of trans_hist func. 
def mean_cfp_trans_tp(df): 
    
    '''Tuple with the mean of coffee price and mean transformation value.'''
    
    # Update trans_cat for column selection. 
    trans_cat = 'Box-Cox Transformation of Coffee Price (USD)' 
    
    # Perform inverse box-cox transformation. 
    # Sort dataframe based on the values close to the transformation mean.
    df = df[(df[trans_cat] > df[trans_cat].mean() - 0.0003) &
            (df[trans_cat] < df[trans_cat].mean() + 0.0003)]
    
    # Mean of transformation.
    mean_trans = df[trans_cat].mean()
    
    # Mean of coffee price. 
    mean_cf_price = df['Coffee Price (USD)'].mean() 
    
    return mean_cf_price, mean_trans

def trans_hist(pop_df, samp_df, col, trans_cat, start_date, end_date): 
    
    '''Plot gaussian kernel density estimate and normal distribution.''' 
    
    # Find sample mean of coffee price and sample mean of transformed coffee price. 
    samp_mean_cfp, samp_mean_trans_cfp = mean_cfp_trans_tp(samp_df) 
    
    # Find population mean of coffee price and population mean of transformed coffee price. 
    pop_mean_cfp, pop_mean_trans_cfp = mean_cfp_trans_tp(pop_df) 
    
    # Set figure layout. 
    myfcf_gra.set_sns_font_dpi()
    myfcf_gra.set_sns_large_white()

    # Custom_palette to set color for the lineplot. 
    custom_palette = sns.color_palette("tab10") 

    # Add normal distribution and KDE.
    sns.distplot(pop_df[col], fit=stats.norm, kde=True)

    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Set size of text.
    txt_size = 15
    
    # Set size of title.
    title_size = 20
    
    # Set date text.
    date_txt = str(start_date) + ' - ' + str(end_date) 
    
    # Segregate based by selected column. 
    if col == 'Coffee Price (USD)': 

        # Set vertical line for population mean. 
        plt.axvline(pop_mean_cfp, color=custom_palette[3], 
                    linestyle='dashed', linewidth=1)

        # Set text for population mean. 
        plt.text(pop_mean_cfp+0.01, 1.1, 
                 'Population Mean: USD {}'.format(round(pop_mean_cfp,2)), 
                 fontsize=15, color=custom_palette[3], size=txt_size) 
        
        # Set vertical line for sample mean. 
        plt.axvline(samp_mean_cfp, color=custom_palette[0], 
                    linestyle='dashed', linewidth=1)
        
        # Set text for sample mean. 
        plt.text(samp_mean_cfp+0.01, 1.0,
                 'Sample Mean: USD {}'.format(round(samp_mean_cfp,2)), 
                 fontsize=15, color=custom_palette[0], size=txt_size) 

        # Set title.  
        plt.title('Gaussian Kernel Density Estimate\n(' + col + ': ' + date_txt + ')', size=title_size)

    if col == 'Box-Cox Transformation of Coffee Price (USD)':  

        # Set vertical line for population mean.  
        plt.axvline(pop_mean_trans_cfp, color=custom_palette[3], 
                    linestyle='dashed', linewidth=1)

        # Set text for population mean. 
        plt.text(pop_mean_trans_cfp+0.01, 1.1, 
                 'Transformed Population Mean: {}'.format(round(pop_mean_trans_cfp,3)), 
                 fontsize=15, color=custom_palette[3], size=txt_size) 
        
        # Set vertical line for sample mean. 
        plt.axvline(samp_mean_trans_cfp, color=custom_palette[0], 
                    linestyle='dashed', linewidth=1)
        
        # Set text for sample mean. 
        plt.text(samp_mean_trans_cfp+0.01, 1.0, 
                 'Transformed Sample Mean: {}'.format(round(samp_mean_trans_cfp,3)), 
                 fontsize=15, color=custom_palette[0], size=txt_size) 

        # Set title. 
        plt.title('Gaussian Kernel Density Estimate\n(' + col + ': ' + date_txt + ')', size=title_size) 

# ----------------------------------------------------------------------------------------------

def print_mean_pop_samp(pop_df, samp_df): 
    
    '''Print sample mean and population mean of box-cox transformed coffee price and inverse box-cox transformed coffee price.'''
    
    # Find sample mean of coffee price and sample mean of transformed coffee price. 
    samp_mean_cfp, samp_mean_trans_cfp = mean_cfp_trans_tp(samp_df) 
    
    # Find population mean of coffee price and population mean of transformed coffee price. 
    pop_mean_cfp, pop_mean_trans_cfp = mean_cfp_trans_tp(pop_df) 
    
    # Create text for box-cox transformation. 
    pop_bc_text = 'Population mean of box-cox transformed coffee price (USD): ' + str(round(pop_mean_trans_cfp, 3))
    samp_bc_text = 'Sample mean of box-cox transformed coffee price (USD): ' + str(round(samp_mean_trans_cfp, 3))
    
    # Create text for inverse box-cox transformation. 
    pop_price_text = 'Population mean of coffee price: USD ' + str(round(pop_mean_cfp, 2))
    samp_price_text = 'Sample mean of coffee price: USD ' + str(round(samp_mean_cfp, 2))
    
    # Print text for box-cox transformation. 
    print('\033[1m' + 'Box-Cox Transformation:' + '\033[0m')
    print(pop_bc_text)
    print(samp_bc_text) 
    print('\n')
    
    # Print text for inverse box-cox transformation.
    print('\033[1m' + 'After Inverse Box-Cox Transformation:' + '\033[0m')
    print(pop_price_text)
    print(samp_price_text) 


def print_pct_abv_bel_cfp(df, cfp, cat_cfp, ttl_txt):
    
    '''Print percentage of data points above and below the chosen coffee price.'''
    
    # Filter for data points above the chosen coffee price and perform count next.
    abv_cfp = df[df['Coffee Price (USD)'] > cfp].count()[1]

    # Count the total number of data points. 
    tot_dat = df.count()[1]

    # Calculate the percentage of data points above and below the chosen coffee price.
    pct_abv_cfp = round(abv_cfp/tot_dat*100, 2)
    pct_bel_cfp = 100 - pct_abv_cfp

    # Create text for percentage of data points above and below the chosen coffee price.
    pct_abv_cfp_txt = str(pct_abv_cfp) + '% of the data points are above USD' + str(round(cfp,2)) + ' (' + cat_cfp + ')'
    pct_bel_cfp_txt = str(pct_bel_cfp) + '% of the data points are below USD' + str(round(cfp,2)) + ' (' + cat_cfp + ')'

    # Print out. 
    print('\033[1m' + ttl_txt + '\033[0m') 
    print(pct_abv_cfp_txt)
    print(pct_bel_cfp_txt)

        
        
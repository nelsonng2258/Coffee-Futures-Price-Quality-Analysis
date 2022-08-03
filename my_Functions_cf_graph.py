#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:20:30 2022

@author: nelsonng
""" 

# Import libraries. 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Import my packages. 
import my_Functions_cf_wrangle as myfcf_wra 

# ---------------------------------------------------------------------------------------------- 

# Standard sub functions for plotting graphs. 
def set_rcparams(): 
    
    '''Set rcParams setting for graph layout.'''
    
    # Reset layout to default. 
    plt.rcdefaults()  
    
    # Set rcParams settings.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman' 
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600 


def set_sns_font_dpi(): 
    
    '''Set sns.setting for graph font and dpi.'''
    
    # Reset layout to default. 
    plt.rcdefaults() 
     
    # Set sns.set settings. 
    sns.set_style({'font.serif':'Times New Roman'}) 
    sns.set(rc = {"figure.dpi":600, 'savefig.dpi':600}) # Improve dpi. 


def set_sns_large_white(): 
    
    '''Set sns.setting to create a graph with large and whitegrid.'''
    
    # Reset layout to default. 
    plt.rcdefaults()  

    # Set style to times new roman.
    sns.set(rc = {'figure.figsize':(15, 6)}) # width x height. # 15, 10
    sns.set_style('whitegrid') # Set background. 

# ----------------------------------------------------------------------------------------------

def scatterplot(df, x, y, start_date, end_date): 
    
    # Set graph layout. 
    set_sns_font_dpi()
    set_sns_large_white()

    # Plot lineplot. 
    g = sns.scatterplot(x=x, y=y, data=df)

    # Rearrange date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Create title.
    title = y + '\n(' + start_date + ' - ' + end_date + ')'

    # Set title of lineplot.
    g.set_title(title, size=20)
    
    plt.show() 


def cfg_piechart(df, start_date, end_date):  
    
    '''Plot pie chart for coffee total score quality classification.'''
    
    # Set graph layout setting. 
    set_rcparams()

    # Custom_palette to set color for the lineplot. 
    custom_palette = sns.color_palette('tab10') 

    # Modify figure size.
    plt.figure(figsize=(5,1)) # width x height

    # Check the distribution of grades using pie chart 
    val = df['Origin']

    # Set slices. 
    slices = [val[3],val[2],val[1],val[0]]

    # Set labels. 
    labels = ['Outstanding specialty coffee', 
              'Excellent', 
              'Very good', 
              'Below specialty coffee quality']

    # Set colors. 
    colors = [custom_palette[0], 
              custom_palette[1],
              custom_palette[2],
              custom_palette[3]]

    # Plot pie chart. 
    plt.pie(slices, labels = labels, colors = colors, 
            shadow = False, startangle = 180,
            # show % on each slice
            autopct = '%1.1f%%',
            wedgeprops = {'edgecolor':'black'}, textprops={'fontsize': 4}) 
    
    # Rearrange date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Set title.
    ttl_txt = 'Classification of Coffee Total Quality Score\n' + '(' + start_date + ' - ' + end_date + ')'
    plt.title(ttl_txt, size=6)

    # Display plot. 
    plt.show()  

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_countplot func. 
def hue_countplot(df, x, hue):
    
    '''Plot countplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()   

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 

    if hue:
        
        # Plot countplot.
        g = sns.countplot(data=df, x=x, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot scatterplot. 
        g = sns.countplot(data=df, x=x) 

    # Set title. 
    ttl_txt = 'Countplot\n' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_countplot func.  
def countplot_x_hue_lt(x_lt, hue_lt): 
    
    '''Create a list of tuples of x, y and hue values for print_mult_countplot func.'''

    x_hue_lt = [] 

    # Iterating over nested list. 
    for i in x_lt:
        for j in hue_lt:
            
            # Append x, hue values into x_hue_lt. 
            x_hue_lt.append((i, j))
                    
    return x_hue_lt

def print_mult_countplot(df, x_lt, hue_lt): 
    
    '''Print multiple countplot.'''
    
    # Iterate over catplots_x_y_hue_lt.   
    for i in countplot_x_hue_lt(x_lt, hue_lt):
    
        # Create variables x, hue from i.
        x = i[0] 
        hue = i[1] 
        
        # Print title. 
        ttl_txt = '\033[1m' + x + ' (' + hue + ')'+ '\033[0m'
        print(ttl_txt)
        
        # Plot countplot. 
        hue_countplot(df, x, hue)

        # Display countplot.
        plt.show() 

# ----------------------------------------------------------------------------------------------

# Subfunction of print_mult_catplots func.  
def hue_stripplot(df, x, y, hue):
    
    '''Plot stripplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()   

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 

    if hue:
        
        # Plot stripplot.
        g = sns.stripplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot stripplot.
        g = sns.stripplot(data=df, x=x, y=y, size=2) 

    # Set title. 
    ttl_txt = 'Stripplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()
    
# Subfunction of print_mult_catplots func. 
def hue_swarmplot(df, x, y, hue):
    
    '''Plot swarmplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()  

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height  

    if hue:
        
        # Plot swarmplot.
        g = sns.swarmplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot swarmplot.
        g = sns.swarmplot(data=df, x=x, y=y, size=2) 

    # Set title. 
    ttl_txt = 'Swarmplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func. 
def hue_boxplot(df, x, y, hue):
    
    '''Plot boxplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()  

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 

    if hue:
        
        # Plot boxplot.
        g = sns.boxplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot boxplot.
        g = sns.boxplot(data=df, x=x, y=y) 

    # Set title. 
    ttl_txt = 'Boxplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func. 
def hue_violinplot(df, x, y, hue):
    
    '''Plot violinplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()   

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 

    if hue:
        
        # Plot violinplot.
        g = sns.violinplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot violinplot.
        g = sns.violinplot(data=df, x=x, y=y) 

    # Set title. 
    ttl_txt = 'Violinplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func. 
def hue_boxenplot(df, x, y, hue):
    
    '''Plot boxenplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()  

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 

    if hue:
        
        # Plot boxenplot.
        g = sns.boxenplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot boxenplot.
        g = sns.boxenplot(data=df, x=x, y=y) 

    # Set title. 
    ttl_txt = 'Boxenplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func. 
def catplots_x_y_hue_lt(x_lt, y_lt, hue_lt): 
    
    '''Create a list of tuples of x, y and hue values for print_mult_catplots func.'''

    x_y_hue_lt = [] 

    # Iterating over nested list. 
    for i in x_lt:
        for j in y_lt:
            for k in hue_lt:
                    
                # x and y cannot be equal.  
                if i != j: 
                    
                    # Append x, y, hue values into x_y_hue_lt. 
                    x_y_hue_lt.append((i, j, k))
                    
    return x_y_hue_lt

def print_mult_catplots(df, x_lt, y_lt, hue_lt): 
    
    '''Print multiple stripplot, swarmplot, boxplot, violinplot, boxenplot.'''
    
    # Iterate over catplots_x_y_hue_lt.   
    for i in catplots_x_y_hue_lt(x_lt, y_lt, hue_lt):
    
        # Create variables x, y, hue from i.
        x = i[0] 
        y = i[1]
        hue = i[2] 
        
        # Print title. 
        ttl_txt = '\033[1m' + y + ' VS ' + x + ' (' + hue + ')' + '\033[0m'
        print(ttl_txt)
        
        # Plot stripplot, swarmplot, boxplot, violinplot, boxenplot. 
        hue_stripplot(df, x, y, hue)
        hue_swarmplot(df, x, y, hue)
        hue_boxplot(df, x, y, hue)
        hue_violinplot(df, x, y, hue)
        hue_boxenplot(df, x, y, hue)

        # Display stripplot, swarmplot, boxplot, violinplot, boxenplot.
        plt.show()  

# ----------------------------------------------------------------------------------------------

def cfpq_boxplot(df, x, y, hue, curr_cfp, pop_mean_cfp, samp_mean_cfp, start_date, end_date):
    
    '''Create a boxplot.'''
    
    # Set graph layout setting. 
    set_rcparams()  

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 
    
    # Custom_palette to set color for the lineplot. 
    custom_palette = sns.color_palette("tab10") 
    
    if hue: 
        # Plot boxplot. 
        g = sns.boxplot(x=x, y=y, data=df, showmeans=True, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1, title=hue)
    
    else: 
        # Plot boxplot. 
        sns.boxplot(x=x, y=y, data=df, showmeans=True)
        
    # Set horizontal line. 
    if y == 'Total Quality Score': 
        plt.axhline(y=90, color = 'r') 
        plt.axhline(y=85, color = 'g') 
        plt.axhline(y=80, color = 'b') 
    
    if y == 'Coffee Price (USD)': 
        # Create line for current, mean sample, and population mean of coffee price.
        if curr_cfp: 
            plt.axhline(y = curr_cfp, color = custom_palette[1])  
        
        if samp_mean_cfp:
            plt.axhline(y = samp_mean_cfp, color = custom_palette[2])    
        
        if pop_mean_cfp: 
            plt.axhline(y = pop_mean_cfp, color = custom_palette[3])  
    
    # Create y_text_adj to adjust the text distance above the line. 
    y_text_adj = 0.2 

    # Set text. 
    if y == 'Total Quality Score':
        plt.text(0, 90 + y_text_adj, 'Outstanding', fontsize=10, color ='r') 
        plt.text(0, 85 + y_text_adj, 'Excellent', fontsize=10, color ='g') 
        plt.text(0, 80 + y_text_adj, 'Very Good', fontsize=10, color ='b') 
    
    if y == 'Coffee Price (USD)': 
        # Create text for current, mean sample, and population mean of coffee price.
        if curr_cfp: 
            plt.text(0, curr_cfp+0.03, 'Current', fontsize=15, color = custom_palette[1]) 
        
        if samp_mean_cfp:
            plt.text(0, samp_mean_cfp+0.03, 'Sample Mean', fontsize=15, color = custom_palette[2]) 
        
        if pop_mean_cfp: 
            plt.text(0, pop_mean_cfp+0.03, 'Population Mean', fontsize=15, color = custom_palette[3]) 
   
    # Rearrange date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)
    
    # Set title.  
    plt.title(y + ' VS ' + x + '\n' + '(Countries (>= 30 Data Points): ' + start_date + ' - ' + end_date + ')', size=20)  

    # Set xlabel and ylabel.
    plt.xlabel(x)
    plt.ylabel(y)

    # Rotate xticks. 
    plt.xticks(rotation = 90)

    # Display plot. 
    plt.show()


def ctry_a30_histplot_9df(df1, df2, df3, df4, df5, df6, df7, df8, df9, y, sub_ttl_lt, start_date, end_date):
    
    '''Plot histogram of countries with >=30 datapoints for 9 df.'''
    
    # Set graph layout setting. 
    set_rcparams() 

    # Custom_palette to set color for the lineplot. 
    custom_palette = sns.color_palette('tab10') 

    # Setup subplots.
    fig,((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10,10))

    # Plot histograms. 
    g1 = sns.histplot(df1[y], kde = True, color=custom_palette[1], ax=ax1)
    g2 = sns.histplot(df2[y], kde = True, color=custom_palette[2], ax=ax2) 
    g3 = sns.histplot(df3[y], kde = True, color=custom_palette[3], ax=ax3) 
    g4 = sns.histplot(df4[y], kde = True, color=custom_palette[4], ax=ax4) 
    g5 = sns.histplot(df5[y], kde = True, color=custom_palette[5], ax=ax5) 
    g6 = sns.histplot(df6[y], kde = True, color=custom_palette[6], ax=ax6) 
    g7 = sns.histplot(df7[y], kde = True, color=custom_palette[7], ax=ax7) 
    g8 = sns.histplot(df8[y], kde = True, color=custom_palette[8], ax=ax8) 
    g9 = sns.histplot(df9[y], kde = True, color=custom_palette[9], ax=ax9) 

    # Find the median values. 
    med_df1 = df1[y].median()
    med_df2 = df2[y].median()
    med_df3 = df3[y].median()
    med_df4 = df4[y].median()
    med_df5 = df5[y].median()
    med_df6 = df6[y].median()
    med_df7 = df7[y].median()
    med_df8 = df8[y].median()
    med_df9 = df9[y].median()

    # Find the mean values. 
    mean_df1 = df1[y].mean()
    mean_df2 = df2[y].mean()
    mean_df3 = df3[y].mean()
    mean_df4 = df4[y].mean()
    mean_df5 = df5[y].mean()
    mean_df6 = df6[y].mean()
    mean_df7 = df7[y].mean()
    mean_df8 = df8[y].mean() 
    mean_df9 = df9[y].mean() 

    # Plot median lines on histograms.
    g1.axvline(x=med_df1, color = 'g') 
    g2.axvline(x=med_df2, color = 'g')  
    g3.axvline(x=med_df3, color = 'g')  
    g4.axvline(x=med_df4, color = 'g')  
    g5.axvline(x=med_df5, color = 'g')  
    g6.axvline(x=med_df6, color = 'g')  
    g7.axvline(x=med_df7, color = 'g')  
    g8.axvline(x=med_df8, color = 'g')  
    g9.axvline(x=med_df9, color = 'g')  

    # Plot mean lines on histograms.
    g1.axvline(x=mean_df1, color = 'r') 
    g2.axvline(x=mean_df2, color = 'r')  
    g3.axvline(x=mean_df3, color = 'r')  
    g4.axvline(x=mean_df4, color = 'r')  
    g5.axvline(x=mean_df5, color = 'r')  
    g6.axvline(x=mean_df6, color = 'r')  
    g7.axvline(x=mean_df7, color = 'r')  
    g8.axvline(x=mean_df8, color = 'r')  
    g9.axvline(x=mean_df9, color = 'r')  

    # Set subtitle. 
    ax1.set_title(sub_ttl_lt[0])
    ax2.set_title(sub_ttl_lt[1])
    ax3.set_title(sub_ttl_lt[2])
    ax4.set_title(sub_ttl_lt[3])
    ax5.set_title(sub_ttl_lt[4])
    ax6.set_title(sub_ttl_lt[5]) 
    ax7.set_title(sub_ttl_lt[6])
    ax8.set_title(sub_ttl_lt[7])
    ax9.set_title(sub_ttl_lt[8]) 

    # Rearrange date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)

    # Set title. 
    plt.suptitle('Histogram\n(Countries (>= 30 Data Points): ' + start_date + ' - ' + end_date + ')', size=20)

    # Ensure the graphs do not overlap. 
    plt.tight_layout()

    # Display plot.
    plt.show()

    # Print comment. 
    print('Note: Green line: Median; Red line: Mean')


def ctry_a30_histplot_3df(df1, df2, df3, y, sub_ttl_lt, start_date, end_date):
    
    '''Plot histogram of countries with >=30 datapoints for 3 df.'''
    
    # Set graph layout setting. 
    set_rcparams() 

    # Custom_palette to set color for the lineplot. 
    custom_palette = sns.color_palette('tab10') 

    # Setup subplots.
    fig,((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10,10/3)) # subplots(row, col, ...)
    
    # Plot histograms. 
    g1 = sns.histplot(df1[y], kde = True, color=custom_palette[1], ax=ax1)
    g2 = sns.histplot(df2[y], kde = True, color=custom_palette[2], ax=ax2) 
    g3 = sns.histplot(df3[y], kde = True, color=custom_palette[3], ax=ax3) 

    # Find the median values. 
    med_df1 = df1[y].median()
    med_df2 = df2[y].median()
    med_df3 = df3[y].median()

    # Find the mean values. 
    mean_df1 = df1[y].mean()
    mean_df2 = df2[y].mean()
    mean_df3 = df3[y].mean()

    # Plot median lines on histograms.
    g1.axvline(x=med_df1, color = 'g') 
    g2.axvline(x=med_df2, color = 'g')  
    g3.axvline(x=med_df3, color = 'g')  

    # Plot mean lines on histograms.
    g1.axvline(x=mean_df1, color = 'r') 
    g2.axvline(x=mean_df2, color = 'r')  
    g3.axvline(x=mean_df3, color = 'r')  

    # Set subtitle. 
    ax1.set_title(sub_ttl_lt[0])
    ax2.set_title(sub_ttl_lt[1])
    ax3.set_title(sub_ttl_lt[2])

    # Rearrange date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date)

    # Set title. 
    plt.suptitle('Histogram\n(Countries (>= 30 Data Points): ' + start_date + ' - ' + end_date + ')', size=20)

    # Ensure the graphs do not overlap. 
    plt.tight_layout()

    # Display plot.
    plt.show()

    # Print comment. 
    print('Note: Green line: Median; Red line: Mean') 


def cfp_h_boxplot(df, col, curr_cfp, pop_mean_cfp, samp_mean_cfp, start_date, end_date):
    
    '''Plot boxplot of current, mean sample, and population mean of coffee prices.'''
    
    # Set graph layout. 
    set_sns_font_dpi()
    set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height
    
    # Custom_palette to set color for the lineplot. 
    custom_palette = sns.color_palette("tab10") 
    
    # Create boxplot. 
    sns.boxplot(x = df[col], orient = 'h', showmeans=True)
    
    # Create line for current, mean sample, and population mean of coffee price.
    plt.axvline(x = curr_cfp, color = custom_palette[1])  
    plt.axvline(x = samp_mean_cfp, color = custom_palette[2])    
    plt.axvline(x = pop_mean_cfp, color = custom_palette[3])  
    
    # Create text for current, mean sample, and population mean of coffee price.
    plt.text(curr_cfp+0.03, -0.35, 'Current', fontsize=15, color = custom_palette[1]) 
    plt.text(samp_mean_cfp+0.03, -0.40, 'Sample Mean', fontsize=15, color = custom_palette[2]) 
    plt.text(pop_mean_cfp+0.03, -0.45, 'Population Mean', fontsize=15, color = custom_palette[3]) 
   
    # Rearrange the start and end date str format. 
    start_date = myfcf_wra.dt_str_fmt(start_date)
    end_date = myfcf_wra.dt_str_fmt(end_date) 
    
    # Set title. 
    plt.title('Box Plot\n(' + col + ': ' + str(start_date) + ' - ' + str(end_date) + ')', size=20)
    
    # Display plot. 
    plt.show()  


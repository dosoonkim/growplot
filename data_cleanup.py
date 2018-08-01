import os
import seaborn as sns
import numpy as np
import pandas as pd
from math import log
from scipy import stats

import sys
import argparse

# We now use argparse -- positional arg is filename, then
# --series for the names (i.e., AMW6) of the series to plot.

def process_excel(filename, specified_series = []):
    df_kinetic_total = pd.read_excel(filename, sheet_name = 'Data')

    def get_hrs(time_str):
        h, m, s = time_str.split(':')
        return ((int(h) * 3600 + int(m) * 60 + int(s))/3600)

    def get_minutes(time_str):
        h,m,s = time_str.split(':')
        return (float(h)*60  + float(m) + float(s)/60)
    
    hrs_list = []
    min_list = []
    
    for time in df_kinetic_total['Time']:
        hrs_list.append(get_hrs(str(time)))
        min_list.append(get_minutes(str(time)))
        
    df_kinetic_total['hrs'] = hrs_list
    df_kinetic_total['mins'] = min_list
    
    df_samples = pd.read_excel(filename, sheet_name = 'Layout')
    series_names = list(df_samples['Sample'])
    # filter series if any series to plot are specified
    if len(specified_series) > 0:
        series_names = [f for f in series_names if f in specified_series]
    df_samples.set_index('Sample', inplace=True)

    samples = []
    for series in series_names:
        samples.append(df_samples.loc[series].tolist()[0].split(','))

    dfs = []
    for sample in samples:
        dfs.append(df_kinetic_total.loc[:,sample])

    data_dfs = []
    for sample_df, series in zip(dfs, series_names):
        df = pd.DataFrame()
        df['avg'] = sample_df.mean(axis=1)
        df['std'] = sample_df.std(axis=1)
        df['Hours'] = df_kinetic_total['hrs']
        df['sample'] = series 
        data_dfs.append(df)

    #data_dfs = [WT_df_avg, sample_1_df_avg, sample_2_df_avg, sample_3_df_avg, sample_4_df_avg, sample_5_df_avg]
    data_final = pd.concat(data_dfs)
    data_final['time'] = df_kinetic_total['hrs']

    data_final.to_csv('growth_kinetic.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Break down an Excel file into collections of data series for later plotting in R")
    parser.add_argument('filename', type=str, nargs=1, help="The Excel file to be processed")
    parser.add_argument('--series', type=str, nargs='+', help="The series from the Excel file, as named in the Layout tab, to be plotted")
    args = parser.parse_args()
    process_excel(filename=args.filename[0], specified_series=args.series)

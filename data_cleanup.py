import os
import seaborn as sns
import numpy as np
import pandas as pd
from math import log
from scipy import stats

import sys

# sys.argv => filename
def process_excel(filename):
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
    process_excel(sys.argv[1])

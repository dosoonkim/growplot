from scipy.optimize import curve_fit
from math import log
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    
    return data_dfs

if __name__ == '__main__':
    dfs = process_excel(sys.argv[1])

    def f(x, L, k, x0):
        return L/(1+np.exp(-k*(x-x0)))

    xdata = dfs[0]['Hours']
    ydata = dfs[0]['avg']
    print(xdata)
    print(ydata)
    
    params, pcov = curve_fit(f, xdata, ydata)
    print(params)
    plt.scatter(xdata, ydata)
    plt.plot(xdata, f(xdata, *params), label="fit")
    plt.legend()
    plt.show()


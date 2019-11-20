from scipy.optimize import curve_fit
from math import log
from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
import pandas as pd
import sys
import argparse

#font = {'family' : 'normal',
#    'weight' : 'bold',
#    'size'   : 24}

#matplotlib.rc('font', **font)
plt.style.use('seaborn-talk')

def process_excel(filename, specified_series):
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
        samples.append(filter(lambda it: it != '', map(lambda it: it.strip(), df_samples.loc[series].tolist()[0].split(','))))

    df_dict = {}
    for sample, series in zip(samples, series_names):
        if specified_series is None or (len(specified_series) > 0 and series in specified_series):
            df_dict[series] = df_kinetic_total.loc[:,sample].astype(float)

    return df_dict, df_kinetic_total
    #data_dfs = []
    #for sample_df, series in zip(dfs, series_names):
    #    df = pd.DataFrame()
    #    #df['avg'] = sample_df.mean(axis=1)
    #    df['avg'] = sample_df#.mean(axis=1)
    #    #df['std'] = sample_df.std(axis=1)
    #    df['Hours'] = df_kinetic_total['hrs']
    #    df['sample'] = series
    #    data_dfs.append(df)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Break down an Excel file into collections of data series for later plotting in R")
    parser.add_argument('filename', type=str, nargs=1, help="The Excel file to be processed")
    parser.add_argument('--series', type=str, nargs='+', help="The series from the Excel file, as named in the Layout tab, to be plotted")
    args = parser.parse_args()
    df_dict, t = process_excel(filename=args.filename[0], specified_series=args.series)

    def f(x, L, k, x0):
        return L/(1+np.exp(-k*(x-x0)))

    final_proto_df = []

    for k,df in df_dict.items():
        npdf = np.array(df)
        ydata = npdf[:,0]
        xdata = t['hrs']
        for i in range(1,npdf.shape[1]):
            ydata = np.concatenate((ydata, npdf[:,i]))
            xdata = np.concatenate((xdata, t['hrs']))
        #ydata = np.concatenate((ydata, npdf[:,2]))
        #ydata = np.concatenate((ydata, npdf[:,3]))
        #ydata = np.concatenate((ydata, npdf[:,4]))
        #ydata = np.concatenate((ydata, npdf[:,5]))
        def av(l):
            return sum(l)/len(l)
        yav = np.array([av([npdf[i,k] for k in range(npdf.shape[1])]) for i in range(len(npdf))]) 
        #['avg']
        #print(xdata)
        #print(ydata)
        
        params, pcov = curve_fit(f, xdata, ydata)
        if '_' in k:
            final_proto_df.append({'tag': k, 'seq': k.split('_')[0], 'cond': k.split('_')[1], 'odmax': params[0], 'midpt': params[2], 'maxslope': (params[0]*params[1])/4, 'doubling': np.log(2)/params[1], 'cov': pcov })
        else:    
            final_proto_df.append({'tag': k, 'seq': k, 'cond': 'n/a', 'odmax': params[0], 'midpt': params[2], 'maxslope': (params[0]*params[1])/4, 'doubling': np.log(2)/params[1], 'cov': pcov })

        #print(k, params, pcov)

        stdevs = np.sqrt(np.diag(pcov))
        print(k)
        for name, param, stdev in zip(["Max", "K-param", "Midpoint"], params, stdevs):
            print("{}: {} +/- {}".format(name, param, stdev))
        # lag_time = (mx0 - y0) / m = x0 - y0/m

        # Assuming covariances are close enough to zero.

        half_max = params[0]/2
        half_max_sd = stdevs[0]/2

        max_slope = (params[0]*params[1])/4
        max_slope_sd = ( (stdevs[0]**2 * stdevs[1]**2 + stdevs[0]**2 * params[0]**2 + stdevs[1]**2 * params[1]**2  )**0.5 ) / 4
        
        half_max_over_max_slope = half_max / max_slope
        # minimal consideration of covariance would append -2cov(x,y)/xy to the second term
        half_max_over_max_slope_sd = ( half_max**2 / max_slope**2 * ( half_max_sd**2 / half_max**2 + max_slope_sd**2/max_slope**2 ) )**0.5

        lag_time = params[2] - half_max_over_max_slope
        # minimal consideration of covariance would subtract -2cov(x,y) inside the sqrt
        lag_time_sd = ( stdevs[2]**2 + half_max_over_max_slope_sd**2 ) ** 0.5
        
        print("Max slope: {} +/- {}".format(max_slope, max_slope_sd))
        print("Lag time: {} +/- {}".format(lag_time, lag_time_sd))
    
    finaldf = pd.DataFrame.from_dict(final_proto_df)

    plt.clf()
    sns.lmplot(x='odmax', y='midpt',data=finaldf, fit_reg=False, hue='seq', legend='True')
    plt.savefig('plot_odmax_v_midpt_byseq.svg')
    plt.clf()
    sns.lmplot(x='odmax', y='doubling',data=finaldf, fit_reg=False, hue='seq', legend='True')
    plt.savefig('plot_odmax_v_doubling_byseq.svg')
    #plt.scatter(odmax_CS, midpt_CS, c='b')
    #plt.scatter(odmax_C, midpt_C, c='g')
    plt.clf()
    sns.lmplot(x='odmax', y='midpt',data=finaldf, fit_reg=False, hue='cond', legend='True')
    plt.savefig('plot_odmax_v_midpt_bycond.svg')
    plt.clf()
    sns.lmplot(x='odmax', y='doubling',data=finaldf, fit_reg=False, hue='cond', legend='True')
    plt.savefig('plot_odmax_v_doubling_bycond.svg')



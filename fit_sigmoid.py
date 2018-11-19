from scipy.optimize import curve_fit
from math import log
from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
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
        samples.append(filter(lambda it: it != '', map( lambda it: it.strip(), df_samples.loc[series].tolist()[0].split(','))))

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
    parser.add_argument('--no_individuals', action='store_const', const=True, help="Specify --no_individuals if you don't want to plot each individual trial")
    args = parser.parse_args()
    df_dict, t = process_excel(filename=args.filename[0], specified_series=args.series)

    def f(x, L, k, x0):
        return L/(1+np.exp(-k*(x-x0)))

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
        stdevs = np.sqrt(np.diag(pcov))
        print(k)
        for param, stdev in zip(params, stdevs):
            print("{} +/- {}".format(param, stdev))
        if args.no_individuals:
            plt.scatter(t['hrs'], yav, alpha=0.8, s=4, label=k)
        else:
            plt.scatter(t['hrs'], yav, alpha=0.8, s=4, c='black', label=k)
            plt.scatter(xdata, ydata, alpha=0.3, s=4)
        xdp = sorted(xdata)
        reg_curve, = plt.plot(xdp, f(xdp, *params), label="{}_fit".format(k))

        grid = np.linspace(min(xdata), max(xdata), 100)

        # error bands is EITHER a CI for each y value OR it's upper values for each x and then lowers for each x.
        # generate
        n_stdevs = 1.96
        eight_params_sets = [
                [params[0]-n_stdevs*stdevs[0], params[1]-n_stdevs*stdevs[1], params[2]-n_stdevs*stdevs[2]],
                [params[0]-n_stdevs*stdevs[0], params[1]-n_stdevs*stdevs[1], params[2]+n_stdevs*stdevs[2]],
                [params[0]-n_stdevs*stdevs[0], params[1]+n_stdevs*stdevs[1], params[2]-n_stdevs*stdevs[2]],
                [params[0]-n_stdevs*stdevs[0], params[1]+n_stdevs*stdevs[1], params[2]+n_stdevs*stdevs[2]],
                [params[0]+n_stdevs*stdevs[0], params[1]-n_stdevs*stdevs[1], params[2]-n_stdevs*stdevs[2]],
                [params[0]+n_stdevs*stdevs[0], params[1]-n_stdevs*stdevs[1], params[2]+n_stdevs*stdevs[2]],
                [params[0]+n_stdevs*stdevs[0], params[1]+n_stdevs*stdevs[1], params[2]-n_stdevs*stdevs[2]],
                [params[0]+n_stdevs*stdevs[0], params[1]+n_stdevs*stdevs[1], params[2]+n_stdevs*stdevs[2]],
            ]

        yvals = [f(grid, *param_set) for param_set in eight_params_sets]
        min_yval, max_yval = [], []
        for idx in range(len(grid)):
            min_yval.append(min([yval[idx] for yval in yvals]))
            max_yval.append(max([yval[idx] for yval in yvals]))
        err_bands = [min_yval, max_yval]
        
        ax = plt.gca()
        ax.fill_between(grid, *err_bands, facecolor=reg_curve.get_color(), alpha=.25)#facecolor=fill_color, alpha=.15)


    plt.legend()
    plt.savefig('plot.png')
    plt.rcParams['svg.fonttype']='none'
    plt.savefig('plot.svg')


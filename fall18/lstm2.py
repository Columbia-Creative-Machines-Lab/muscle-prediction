import os
import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib
import matplotlib.pyplot as plt # this is used for the plot the graph 
from matplotlib import patches
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts

import sklearn
if sklearn.__version__ >"0.2.1":
    from sklearn.model_selection import KFold
else:
    from sklearn.cross_validation import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
from scipy.signal import argrelextrema, resample
from datetime import datetime, timedelta

sns.set(style='whitegrid')

def load_dataset(filename, window=500):
    df = pd.DataFrame.from_dict(pd.read_msgpack(os.path.join('data', filename)))
    df.columns = ['f', 'mdia', 'msgtype', 'pwm', 'rc', 'rw', 't', 't0', 'timestamp']
    if (df.iloc[0].timestamp > 0):
        df.timestamp -= df.iloc[0].timestamp
    df.pwm.replace(to_replace=0, value=np.NaN, inplace=True)
    df.f = -df.f
    df.drop(columns=['mdia', 'msgtype', 'rc', 'rw'], inplace=True)
    df['f_ra'] = df.f.rolling(window=window).mean()
    return df.fillna(0)

def get_extrema(df, order=2000, ra=False):
    col = 'f_ra' if ra else 'f'
    maxima = argrelextrema(df[col].values, np.greater, order=order)[0]
    minima = argrelextrema(df[col].values, np.less, order=order)[0]
    cutoff = (df.shape[0] / minima.shape[0])/5
    np.delete(minima, np.where(minima < cutoff))
    np.delete(minima, np.where(minima > df.shape[0]-cutoff))
    n_cycles = minima.shape[0]
    return n_cycles, maxima, minima

def plot_raw(df, n_cycles, savefig=None, figsize=(8, 4)):
    # one figure / two charts
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize,dpi=300,sharex=True)
    ax.flat[0].set_title(u'Raw measurements, %d cycles' % n_cycles)
    x = df.timestamp
    # top chart
    ax.flat[0].plot(x,df.pwm, c='r', alpha=1,lw=0.5, label=u'PWM')
    ax2 = ax.flat[0].twinx()
    ax2.plot(x,df.f, alpha=0.8,lw=0.5, label='Force');
    ax2.set_ylabel(u'Force [N]')
    ax.flat[0].legend(loc='best'); ax2.legend(loc='best')
    ax.flat[0].set_ylabel(u'PWM')
    # bottom chart
    ax.flat[1].plot(x,df.t,lw=0.5, label=u'Muscle temperature (C)');
    ax.flat[1].plot(x,df.t0,lw=0.5, label=u'Environment temperature (C)')
    ax.flat[1].legend(loc='best')
    ax.flat[1].set_ylabel(u'Temperature [°C]')
    ax.flat[1].set_xlabel(u'time [s]')
    plt.gca().set_xlim(left=0)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig,dpi=300)
    plt.show()

DATASET_NAME = '3tubes'
df = load_dataset('data_2019-04-10-00-48-44.msgpack', window=150)
n_cycles, minima, maxima = get_extrema(df)
plot_raw(df, n_cycles=n_cycles, savefig=('plots/raw measurement,%s.png' % DATASET_NAME))


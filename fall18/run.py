import os
import sys 
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import pickle
import matplotlib
import matplotlib.pyplot as plt # this is used for the plot the graph 
from matplotlib import patches
from model import Model
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts

from scipy.stats import randint
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
    path = filename if 'data' in filename else os.path.join('data', filename)
    df = pd.DataFrame.from_dict(pd.read_msgpack(path))
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
    ax.flat[0].set_title('Raw measurements, %d cycles' % n_cycles)
    x = df.timestamp
    # top chart
    ax.flat[0].plot(x,df.pwm, c='r', alpha=1,lw=0.5, label=u'PWM')
    ax2 = ax.flat[0].twinx()
    ax2.plot(x,df.f, alpha=0.8,lw=0.5, label='Force');
    ax2.set_ylabel('Force [N]')
    ax.flat[0].legend(loc='best'); ax2.legend(loc='best')
    ax.flat[0].set_ylabel('PWM')
    # bottom chart
    ax.flat[1].plot(x,df.t,lw=0.5, label='Muscle temperature (C)');
    ax.flat[1].plot(x,df.t0,lw=0.5, label='Environment temperature (C)')
    ax.flat[1].legend(loc='best')
    ax.flat[1].set_ylabel('Temperature (C)')
    ax.flat[1].set_xlabel('time [s]')
    plt.gca().set_xlim(left=0)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def train_val_test_split(df, ratio=0.8):
    if 'cycle' in df.columns:
        data = df.drop(columns=['cycle'], inplace=False)
    else:
        data = df.copy()
    data = np.apply_along_axis(lambda row: np.reshape(row, (1,-1)), 1, data.drop(columns=['timestamp']).values)
    train_cutoff, val_cutoff = int(ratio*ratio * data.shape[0]), int(ratio * data.shape[0])
    train, val, test = data[:train_cutoff,], data[train_cutoff:val_cutoff,], data[val_cutoff:,]
    train_X, train_y = train[:,:,1:], train[:,:,0].flatten()
    val_X, val_y = val[:,:,1:], val[:,:,0].flatten()
    test_X, test_y = test[:,:,1:], test[:,:,0].flatten()
    print("TRAIN_X: ", train_X.shape, " TRAIN_Y: ", train_y.shape)
    print("VAL_X: ", val_X.shape, " VAL_Y: ", val_y.shape)
    print("TEST_X: ", test_X.shape, " TEST_Y: ", test_y.shape)
    return train_X, train_y, val_X, val_y, test_X, test_y


if __name__=='__main__':

    # Accept command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='.msgpack dataset')
    parser.add_argument('--name', type=str, default='', help='name of dataset')
    parser.add_argument('--cuda', type=bool, default=False, help='train using CUDA')
    parser.add_argument('--plot', type=bool, default=True, help='display&save plots')
    parser.add_argument('--train', type=bool, default=False, help='train LSTM')
    args = parser.parse_args()
    assert 'msgpack' in args.dataset

    DATASET_NAME = args.name if len(args.name) else args.dataset.split('/')[-1]
    if '.msgpack' in DATASET_NAME:
        DATASET_NAME = DATASET_NAME[:DATASET_NAME.find('.')]

    print('Using dataset: %s' % args.dataset)
    df = load_dataset(args.dataset, window=150)
    n_cycles, minima, maxima = get_extrema(df)

    ############################## Prediction ##############################

    if not args.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    train_X, train_y, val_X, val_y, test_X, test_y = train_val_test_split(df)
    model = Model(DATASET_NAME, input_shape=(train_X.shape[1], train_X.shape[2]))

    if args.train:
        model.train(train_X, train_y, val_X, val_y, name=('%d' % n_cycles), overwrite=args.train)

    if args.plot:
        plot_raw(df, n_cycles=n_cycles, savefig=('plots/raw measurement,%s.png' % DATASET_NAME))
        model.plot_model()
        model.plot_history()

    val_y_hat = model.predict(val_X, y=val_y, plot=args.plot)
    test_y_hat = model.predict(test_X, y=test_y, plot=args.plot)
    
    print('LOSS:')
    print(model.history.history['loss'])

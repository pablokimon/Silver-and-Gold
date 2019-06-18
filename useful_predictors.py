import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from datetime import timedelta  as td
from datetime import datetime as dt

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import signal
from scipy import stats

start_date = sys.argv[1]
end_date = sys.argv[2]
name = sys.argv[3].capitalize()

def load_precious_metals_data(start_date,end_date,name):

    #This function expects Gold Futures Historical Data.csv and Silver Futures Historical Data.csv 
    #in the ./data/ folder
    
    #date will be used for the index, price as the value of interest. 
    #these can be changed in the variables below
    price = 'Price'
    date = 'Date'
    file_name = './data/{}'.format(''.join([ name, ' Futures Historical Data.csv']))
    #print (file_name)
    df = pd.read_csv(file_name,sep=',', thousands=',')
    #df.sort_values(date,inplace=True)
    #print(df)
    df = df.set_index(pd.to_datetime(df[date]),drop=True)
    df=df.sort_index()
    df=df.loc[start_date:end_date]
    print('Summary statistics for the',name,'data series from',start_date,'to',end_date)
    print (pd.concat([df.describe()[0:4],df.describe()[-1::]]))
    return pd.Series(df[price], df.index)

metal = load_precious_metals_data(start_date,end_date,name)
#print (metal)
#difference the data
metal_diff = metal.diff()[1:]
#print (metal_diff)

'''def ADF_test(x,name):
    alpha = .05
    test = sm.tsa.stattools.adfuller(x.values)
    print(name," ADF p-value: {0:2.2f}".format(test[1]))
    print('A low p-value would indicate the data are stationary')'''
def ADF_test(x,name):
    alpha = .05
    test = sm.tsa.stattools.adfuller(x.values)
    print ("An Augmented Dickey-Fuller test for the",name,'data:')
    print(name," ADF p-value: {0:2.2f}".format(test[1]))
    if test[1] <= alpha:
        print('A p-value lower than {} indicates the {} data are stationary'.format(alpha,name))
    else:
        print('A p-value greater than {} indicates the {} data are not stationary'.format(alpha,name))

ADF_test(metal,name)
name_diff = "differenced " + name
ADF_test(metal_diff,name_diff)


def auto_regressive_process(size, coefs, init=None):
    """Generate an autoregressive process with Gaussian white noise.  The
    implementation is taken from here:
    
      http://numpy-discussion.10968.n7.nabble.com/simulate-AR-td8236.html
      
    Exaclty how lfilter works here takes some pen and paper effort.
    """
    coefs = np.asarray(coefs)
    if init == None:
        init = np.zeros(len(coefs))
    else:
        init = np.asarray(init)
    init = np.append(init, np.random.normal(size=(size - len(init))))
    assert(len(init) == size)
    a = np.append(np.array([1]), -coefs)
    b = np.array([1])
    return pd.Series(signal.lfilter(b, a, init))

def format_list_of_floats(L):
    return ["{0:2.2f}".format(f) for f in L]

def plot_autoregressive_process(ax, size, coefs, init=None):
    ar = auto_regressive_process(size, coefs, init)
    ax.plot(ar.index, ar)

def plot_series_and_difference(axs, series, title):
    diff = series.diff()
    axs[0].plot(series.index, series, marker='.')
    axs[0].set_title("Raw Series: {}".format(title))
    axs[1].plot(series.index, diff, marker='.')
    axs[1].set_title("Series of First Differences: {}".format(title))

#plot the series and the series differenced
fig, axs = plt.subplots(2, figsize=(14, 4))
plot_series_and_difference(axs, metal, "Precious Metals")
plt.tight_layout()
plt.show()

#conduct Augemented Dickey Fuller test
ADF_test(metal,name)
#conduct Augemented Dickey Fuller test of differenced data
metal_diff = metal.diff()[1:]
ADF_test(metal_diff,name)

#auto corelation plot
fig, ax = plt.subplots(1, figsize=(14, 3))
_ = sm.graphics.tsa.plot_acf(metal, lags=25, ax=ax)
#partial auto correlation plot
fig, ax = plt.subplots(1, figsize=(14, 3))
_ = sm.graphics.tsa.plot_pacf(metal_diff, lags=25, ax=ax)
plt.show()

#create ARIMA model for data
metal_model = ARIMA(metal, order=(3, 1, 0)).fit()

print("ARIMA(3, 1, 0) coefficients from metal model:\n  Intercept {0:2.2f}\n  AR {1}".format(
    metal_model.params[0], 
        format_list_of_floats(list(metal_model.params[1:]))
    ))
#display data and simulated data from ARIMA model
fig, ax = plt.subplots(4, figsize=(14, 8))

ax[0].plot(metal_diff.index, metal_diff, marker='.')
ax[0].set_title("First Differences of "+name+" Data")

for i in range(1, 4):
    simulated_data = auto_regressive_process(len(metal_diff), 
                                             np.array(list(metal_model.params)[1:]))
    simulated_data.index = metal_diff.index
    ax[i].plot(simulated_data.index, simulated_data, marker='.')
    ax[i].set_title("Simulated Data from "+name+" Model Fit")
    
plt.tight_layout()
plt.show()

#make projection and compare to real data
#not working year 0
next_day = pd.to_datetime(end_date) + td(days=1)
next_year = pd.to_datetime(end_date,) + td(days=365)
next_day = next_day.date()

metal.reindex(pd.DatetimeIndex(start=start_date, end=next_year.year, freq='D'))

fig, ax = plt.subplots(1, figsize=(14, 4))
ax.plot(metal.index, metal, marker='.')
fig = metal_model.plot_predict(end_date, next_year.year, 
                                  dynamic=True, ax=ax, plot_insample=False)

_ = ax.legend().get_texts()[1].set_text("95% Prediction Interval")
_ = ax.legend(loc="lower left")

_ = ax.set_title(name+" Series Forcasts from ARIMA Model")

plt.show()
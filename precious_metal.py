import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

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
    df = pd.read_csv(file_name,sep=',', thousands=',' )
    df.sort_values(date,inplace=True)
    df = df.set_index(pd.to_datetime(df[date]),drop=True)
    df = df.loc[start_date:end_date]
    print('Summary statistics for the',name,'data series from',start_date,'to',end_date)
    print (pd.concat([df.describe()[0:4],df.describe()[-1::]]))
    return pd.Series(df[price], df.index)

metal = load_precious_metals_data(start_date,end_date,name)
#print (metal)
#difference the data
metal_diff = metal.diff()[1:]
#print (metal_diff)

def ADF_test(x):
    alpha = .05
    test = sm.tsa.stattools.adfuller(x.values)
    print ("An Augmented Dickey-Fuller test for the",name,'data:')
    print(name," ADF p-value: {0:2.2f}".format(test[1]))
    if test[1] <= alpha:
        print('A p-value lower than {} indicates the {} data are stationary'.format(alpha,name))
    else:
        print('A p-value greater than {} indicates the {} data are not stationary'.format(alpha,name))

ADF_test(metal)
name = "differenced " + name
ADF_test(metal_diff)



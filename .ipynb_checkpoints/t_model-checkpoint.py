#!/usr/bin/env python
# coding: utf-8

# ### This is a test project to predict short-term stock returns using streaming data from public API
# 
# Data sources:
# - Alphavantage: daily/intraday data with history of 2 years. Intraday data is delayed by a few days.
# - yfinance: Intraday data is near-realtime, maybe delay of 1 min. Shorter sample window, 60d for itd data.
# 
# 
# for more vendor api options, see:
# https://nordicapis.com/10-real-time-stock-data-apis/
# https://algotrading101.com/learn/yahoo-finance-api-guide/
# https://algotrading101.com/learn/yfinance-guide/
# 
# 
# Another streaming data project may be to use Google Trends data, see a template on Kaggle.

# In[43]:


# !pip install yfinance

import numpy as np
import pandas as pd
import os, time, warnings, random, shap, requests, optuna, datetime
import seaborn as sns
import matplotlib.pyplot as plt
import functools as ft
import yfinance as yf


from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor

pd.set_option('display.max_columns', 100)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore') 

time0 = time.time()

os.chdir('/home/jupyter/projects_gcp_cpu/spx/src')
os.getcwd()


# Datapull using AV APIs:

# In[44]:


# data = yf.download(
#         tickers = "^GSPC ^IXIC ^RUT EEM",
#         period = "60d",
#         interval = "2m",
#         ignore_tz = True,
#         group_by = 'ticker',
#         auto_adjust = True,
#         prepost = False,
#         threads = True,
#         proxy = None
#     )

# display(data.head(2), data.tail(6))


# In[45]:


tickerStrings = ['^GSPC', '^IXIC', '^RUT', 'EEM']
df_list = list()
for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period='60d', interval='30m', prepost=False, auto_adjust=True)
    data['ticker'] = ticker  
    df_list.append(data)

df = pd.concat(df_list)
df


# In[48]:


df = df[['Close', 'ticker']]
df.replace({'^GSPC':'Spx', '^IXIC':'Nasdaq', '^RUT':'Russel'}, inplace=True)
df.info()


# In[49]:


df = (df.pivot_table(index=['Datetime'], columns='ticker', values='Close'))
df


# In[50]:


df.info()


# In[51]:


df.columns = ['EEM', 'Nasdaq', 'Russel', 'Spx']
df['time'] = df.index.time
df['date'] = df.index.date
df


# In[52]:


df = df.fillna(method='ffill')
df.info()


# In[53]:


dayclose = df[df.time==datetime.time(15, 30, 0)]
dayopen = df[df.time==datetime.time(9, 30, 0)]
display(df, dayopen.head(), dayclose.head())


# In[54]:


df['spx_ret'] = 100*(df.Spx/df.Spx.shift(1)-1)
df['s_spx_ret_30m'] = (100*(df.Spx/df.Spx.shift(1)-1)).shift(1)
df['s_spx_ret_1h'] = (100*(df.Spx/df.Spx.shift(2)-1)).shift(1)
df['s_spx_ret_2h'] = (100*(df.Spx/df.Spx.shift(4)-1)).shift(1)
display(df.shape, df.head(7))


# In[55]:


df.loc[df.time < datetime.time(10, 0, 0), 'spx_ret'] = np.nan
df.loc[df.time < datetime.time(10, 30, 0), 's_spx_ret_30m'] = np.nan
df.loc[df.time < datetime.time(11, 0, 0), 's_spx_ret_1h'] = np.nan
df.loc[df.time < datetime.time(12, 0, 0), 's_spx_ret_2h'] = np.nan

# df = df[df.time >= datetime.time(12, 0, 0)]
df


# In[56]:


df.info()


# In[57]:


df


# In[58]:


dayopen.reset_index(drop=True, inplace=True)
dayopen.rename(columns={'Spx':'spx_open'}, inplace=True)
dayopen.head()

dayclose.reset_index(drop=True, inplace=True)
dayclose.sort_values(by='date')
dayclose.rename(columns={'Spx':'spx_close'}, inplace=True)
dayclose_l1 = dayclose.copy()
dayclose_l2 = dayclose.copy()

dayclose_l1['spx_close_l1'] = dayclose_l1.spx_close.shift(1)
dayclose_l2['spx_close_l2'] = dayclose_l2.spx_close.shift(2)

display(dayclose_l1.head(), dayclose_l2.head())


# In[59]:


df = pd.merge(df, dayopen[['date', 'spx_open']], on=['date'], how='left')
df = pd.merge(df, dayclose_l1[['date', 'spx_close_l1']], on=['date'], how='left')
df = pd.merge(df, dayclose_l2[['date', 'spx_close_l2']], on=['date'], how='left')
df


# In[60]:


df['s_spx_ret_open'] = (100*(df.Spx/df.spx_open-1)).shift(1)
df['s_spx_ret_close1'] = (100*(df.Spx/df.spx_close_l1-1)).shift(1)
df['s_spx_ret_close2'] = (100*(df.Spx/df.spx_close_l2-1)).shift(1)

df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# grab dayopen, dayclose, create 3 more signals for spx by joining dfs.
# then wrap up evth into a function or loop and do it for all indices.
# then xgb evth on evth and see what sticks.


# In[ ]:





# In[61]:


t_df = df[['date', 
           'time', 
           'spx_ret', 
           's_spx_ret_30m', 
           's_spx_ret_1h', 
           's_spx_ret_2h', 
           's_spx_ret_open',
           's_spx_ret_close1',
           's_spx_ret_close2']]
t_df.rename(columns={'spx_ret':'target'}, inplace=True)
t_df


# In[62]:


t_df = t_df.dropna()
t_df.info()


# In[65]:


X = t_df[['s_spx_ret_30m',
          's_spx_ret_1h',
          's_spx_ret_2h', 
          's_spx_ret_open',
          's_spx_ret_close1',
          's_spx_ret_close2']]
y = t_df['target']


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(0.2*X.shape[0]))
display(X_train.shape, X_test.shape, y_train.shape, X_train.head())


# In[79]:


xgbm = XGBRegressor(eta=0.015, max_depth=3)
xgbm.fit(X_train, y_train)

# rdm = Ridge()
# rdm.fit(X_train, y_train)



print('In sample, xgb: ', r2_score(y_train, xgbm.predict(X_train)))
print('Out of sample, xgb: ', r2_score(y_test, xgbm.predict(X_test)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = "^GSPC ^IXIC ^RUT EEM",

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "60d",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "2m",

        # Whether to ignore timezone when aligning ticker data from 
        # different timezones. Default is True. False may be useful for 
        # minute/hourly data.
        ignore_tz = True,

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = False,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

display(data.head(2), data.tail(6))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=SPY&interval=5min&slice=year1month1&apikey=KBYUCPQEHAG67WNC&datatype=csv'
#adjusted=true&
spy = pd.read_csv(url)
spy

# at 9:50am monday there is still no data after friday close.
# apparently, there is one calendar day of delay.

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=EEM&interval=1min&slice=year1month1&apikey=KBYUCPQEHAG67WNC&datatype=csv'
eem = pd.read_csv(url)
print(eem.shape)


# In[ ]:





# In[16]:


url = 'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol=ETH&market=USD&interval=5min&apikey=KBYUCPQEHAG67WNC'
eth = pd.read_csv(url)
eth


# In[12]:


eth[0:30]


# In[15]:


eth.iloc[0,0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# can try predicting 10 yield, try smth like xgb 200, 4, 0.04


# In[67]:


# it is hard to get any positive results at all using daily freq major assets
# can try to exploit intraday lead-lag effects
# i.e., pick up intraday major assets (3 indices) and try to predict less liquid assets.
# can try btc or etfs of small stocks. e.g., eem or eems etfs. 


# In[ ]:





# In[88]:


url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=EEM&interval=1min&slice=year1month1&apikey=KBYUCPQEHAG67WNC&datatype=csv'
eem = pd.read_csv(url)
print(eem.shape)

# EEMS is even less liquid, try eema emxc

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=EEMS&interval=1min&slice=year1month1&apikey=KBYUCPQEHAG67WNC&datatype=csv'
eems = pd.read_csv(url)
print(eems.shape)

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=EWX&interval=1min&slice=year1month1&apikey=KBYUCPQEHAG67WNC&datatype=csv'
ewx = pd.read_csv(url)
print(ewx.shape)

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=EEMA&interval=1min&slice=year1month1&apikey=KBYUCPQEHAG67WNC&datatype=csv'
eema = pd.read_csv(url)
print(eema.shape)

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=EMXC&interval=1min&slice=year1month1&apikey=KBYUCPQEHAG67WNC&datatype=csv'
emxc = pd.read_csv(url)
print(emxc.shape)


# In[87]:


display(spy[192:250], eem[:50], emxc[:50])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


# fix date, create returns and rate differences over 1d, 5d, 21d, 63d, 121d, 252d, 504d for all 4 variables
# then create signals by lagging evth by 1d
# then clean evth and fir xgb (around 30 features)


# #### step 1:
# build simple XGB model
# #### step 2:
# deploy this model via Cloud Run and static web app, fixed model
# #### step 3
# deploy model via Cloud Run and Flask with dynamic model, retrained daily

# In[ ]:





# In[ ]:





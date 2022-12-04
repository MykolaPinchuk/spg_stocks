import numpy as np
import pandas as pd
import os, time, warnings, requests, datetime, joblib
import functools as ft
import yfinance as yf

from sklearn.linear_model import ElasticNet
from google.cloud import storage

pd.set_option('display.max_columns', 100)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore') 

project_name = 'GCP-pp2'
project_id = 'valid-heuristic-369117'
regionn = 'us-west1'

time0 = time.time()

os.chdir('/home/jupyter/projects_gcp_cpu/spx/src')

tickerStrings = ['^GSPC', '^IXIC', '^RUT', 'EEM', 'EMXC', 'EEMA', 'VTHR']
df_list = list()
for ticker in tickerStrings:
    data = yf.download(ticker, 
                       group_by="Ticker", 
                       period='4d', 
                       interval='2m', 
                       prepost=False, 
                       auto_adjust=True)
    data['ticker'] = ticker  
    df_list.append(data)

df = pd.concat(df_list)
df = df[['Close', 'ticker']]
df.replace({'^GSPC':'Spx', '^IXIC':'Nasdaq', '^RUT':'Russel'}, inplace=True)
df = (df.pivot_table(index=['Datetime'], columns='ticker', values='Close'))
df.columns = ['EEM', 'EEMA', 'EMXC', 'Nasdaq', 'Russel', 'Spx', 'VTHR']

df['time'] = df.index.time
df['date'] = df.index.date
df['datetime'] = df.index

df = df.fillna(method='ffill')
dayclose = df[df.time==datetime.time(15, 58, 0)]
dayopen = df[df.time==datetime.time(9, 30, 0)]
dayopen.reset_index(drop=True, inplace=True)
dayclose.reset_index(drop=True, inplace=True)
dayclose.sort_values(by='date')
# display(df)

# df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
# df['minute'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.minute

### now i wanna do feature engineering for all assets 

asset_list = ['Spx', 'Nasdaq', 'Russel', 'EMXC', 'EEMA', 'EEM', 'VTHR']

for asset in asset_list:
    
    df[asset + '_ret'] = 100*(df[asset]/df[asset].shift(1)-1)
    df['s_' + asset + '_ret_1prd'] = (100*(df[asset]/df[asset].shift(1)-1)).shift(1)
    df['s_' + asset + '_ret_2prd'] = (100*(df[asset]/df[asset].shift(2)-1)).shift(1)
    df['s_' + asset + '_ret_4prd'] = (100*(df[asset]/df[asset].shift(4)-1)).shift(1)
    # display(df.shape, df.head(5))

    df.loc[df.time < datetime.time(9, 32, 0), 's_' + asset + '_1prd'] = np.nan
    df.loc[df.time < datetime.time(9, 33, 0), 's_' + asset + '_2prd'] = np.nan
    df.loc[df.time < datetime.time(9, 35, 0), 's_' + asset + '_4prd'] = np.nan

    dayopen.rename(columns={asset:asset+'_open'}, inplace=True)
    # dayopen.head()
    dayclose.rename(columns={asset:asset+'_close'}, inplace=True)
    dayclose_l1 = dayclose.copy()
    dayclose_l2 = dayclose.copy()
    dayclose_l1[asset+'_close_l1'] = dayclose_l1[asset+'_close'].shift(1)
    dayclose_l2[asset+'_close_l2'] = dayclose_l2[asset+'_close'].shift(2)

    # display(dayclose_l1.head(), dayclose_l2.head())

    df = pd.merge(df, dayopen[['date', asset + '_open']], on=['date'], how='left')
    df = pd.merge(df, dayclose_l1[['date', asset + '_close_l1']], on=['date'], how='left')
    df = pd.merge(df, dayclose_l2[['date', asset + '_close_l2']], on=['date'], how='left')

    df['s_' + asset + '_ret_open'] = (100*(df[asset]/df[asset + '_open']-1)).shift(1)
    df['s_' + asset + '_ret_close1'] = (100*(df[asset]/df[asset + '_close_l1']-1)).shift(1)
    df['s_' + asset + '_ret_close2'] = (100*(df[asset]/df[asset + '_close_l2']-1)).shift(1)

    cols_todrop = [x for x in list(df.columns) if asset in x and 'ret' not in x]
    df.drop(columns = cols_todrop, inplace=True)

prediction_from = df.datetime.iloc[df.shape[0]-1] 
prediction_to = df.datetime.iloc[df.shape[0]-1] + datetime.timedelta(minutes = 2)
    
### do prediction ###

model_path = '/home/jupyter/project_repos/spg_stocks/spg_stocks/stocks-app/en_model.pkl'
trained_model = joblib.load(open(model_path, "rb"))

X = df.tail(1)
X.drop(columns = ['datetime', 'time', 'date', 'Spx_ret', 'Nasdaq_ret', 'Russel_ret', 'EEMA_ret', 'EEM_ret', 'EMXC_ret', 'VXUS_ret', 'VTHR_ret'], 
          inplace=True,
          errors = 'ignore')

if(X.count().sum() < X.shape[1]):
    print(f'''There are {X.shape[1] - X.count().sum()} missing values. 
          There will be an error''')

print(f'''Prediction for Russel 3000 return from {prediction_from} 
      to {prediction_to} is {trained_model.predict(X)}''')

print('Total time: ', time.time()-time0)

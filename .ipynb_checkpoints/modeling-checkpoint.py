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

tickerStrings = ['^GSPC', '^IXIC', '^RUT', 'EEM']
df_list = list()
for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period='60d', interval='30m', prepost=False, auto_adjust=True)
    data['ticker'] = ticker  
    df_list.append(data)

df = pd.concat(df_list)
df = df[['Close', 'ticker']]
df.replace({'^GSPC':'Spx', '^IXIC':'Nasdaq', '^RUT':'Russel'}, inplace=True)
df = (df.pivot_table(index=['Datetime'], columns='ticker', values='Close'))

df.columns = ['EEM', 'Nasdaq', 'Russel', 'Spx']
df['time'] = df.index.time
df['date'] = df.index.date

df = df.fillna(method='ffill')
dayclose = df[df.time==datetime.time(15, 30, 0)]
dayopen = df[df.time==datetime.time(9, 30, 0)]
dayopen.reset_index(drop=True, inplace=True)
dayclose.reset_index(drop=True, inplace=True)
dayclose.sort_values(by='date')
display(df, dayopen.head(), dayclose.head())

df


### now i wanna do feature engineering for all assets 


asset = 'Spx'


df[asset + '_ret'] = 100*(df[asset]/df[asset].shift(1)-1)
df['s_' + asset + '_ret_30m'] = (100*(df[asset]/df[asset].shift(1)-1)).shift(1)
df['s_' + asset + '_ret_1h'] = (100*(df[asset]/df[asset].shift(2)-1)).shift(1)
df['s_' + asset + '_ret_2h'] = (100*(df[asset]/df[asset].shift(4)-1)).shift(1)
display(df.shape, df.head(5))

df.loc[df.time < datetime.time(10, 0, 0), 's_' + asset + '_ret'] = np.nan
df.loc[df.time < datetime.time(10, 30, 0), 's_' + asset + '_30m'] = np.nan
df.loc[df.time < datetime.time(11, 0, 0), 's_' + asset + '_1h'] = np.nan
df.loc[df.time < datetime.time(12, 0, 0), 's_' + asset + '_2h'] = np.nan

dayopen.rename(columns={asset:asset+'_open'}, inplace=True)
dayopen.head()

dayclose.rename(columns={asset:asset+'_close'}, inplace=True)
dayclose_l1 = dayclose.copy()
dayclose_l2 = dayclose.copy()

dayclose_l1[asset+'_close_l1'] = dayclose_l1[asset+'_close'].shift(1)
dayclose_l2[asset+'_close_l2'] = dayclose_l2[asset+'_close'].shift(2)

display(dayclose_l1.head(), dayclose_l2.head())

df = pd.merge(df, dayopen[['date', asset + '_open']], on=['date'], how='left')
df = pd.merge(df, dayclose_l1[['date', asset + '_close_l1']], on=['date'], how='left')
df = pd.merge(df, dayclose_l2[['date', asset + '_close_l2']], on=['date'], how='left')

df['s_' + asset + '_ret_open'] = (100*(df[asset]/df[asset + '_open']-1)).shift(1)
df['s_' + asset + '_ret_close1'] = (100*(df[asset]/df[asset + '_close_l1']-1)).shift(1)
df['s_' + asset + '_ret_close2'] = (100*(df[asset]/df[asset + '_close_l2']-1)).shift(1)

cols_todrop = [x for x in list(df.columns) if asset in x and 'ret' not in x]
df.drop(columns = cols_todrop, inplace=True)
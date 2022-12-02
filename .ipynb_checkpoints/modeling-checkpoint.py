import numpy as np
import pandas as pd
import os, time, warnings, random, shap, requests, optuna, datetime
import seaborn as sns
import matplotlib.pyplot as plt
import functools as ft
import yfinance as yf

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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

tickerStrings = ['^GSPC', '^IXIC', '^RUT', 'EEM', 'EMXC']
df_list = list()
for ticker in tickerStrings:
    data = yf.download(ticker, 
                       group_by="Ticker", 
                       period='7d', 
                       interval='1m', 
                       prepost=False, 
                       auto_adjust=True)
    data['ticker'] = ticker  
    df_list.append(data)

df = pd.concat(df_list)
df = df[['Close', 'ticker']]
df.replace({'^GSPC':'Spx', '^IXIC':'Nasdaq', '^RUT':'Russel'}, inplace=True)
df = (df.pivot_table(index=['Datetime'], columns='ticker', values='Close'))

df.columns = ['EEM', 'EMXC', 'Nasdaq', 'Russel', 'Spx']
df['time'] = df.index.time
df['date'] = df.index.date

df = df.fillna(method='ffill')
dayclose = df[df.time==datetime.time(15, 59, 0)]
dayopen = df[df.time==datetime.time(9, 30, 0)]
dayopen.reset_index(drop=True, inplace=True)
dayclose.reset_index(drop=True, inplace=True)
dayclose.sort_values(by='date')
display(df, dayopen.head(), dayclose.head())
df0 = df.copy()


### now i wanna do feature engineering for all assets 

asset_list = ['Spx', 'Nasdaq', 'Russel', 'EEM', 'EMXC']

for asset in asset_list:
    
    df[asset + '_ret'] = 100*(df[asset]/df[asset].shift(1)-1)
    df['s_' + asset + '_ret_1prd'] = (100*(df[asset]/df[asset].shift(1)-1)).shift(1)
    df['s_' + asset + '_ret_2prd'] = (100*(df[asset]/df[asset].shift(2)-1)).shift(1)
    df['s_' + asset + '_ret_4prd'] = (100*(df[asset]/df[asset].shift(4)-1)).shift(1)
    display(df.shape, df.head(5))

    df.loc[df.time < datetime.time(9, 32, 0), 's_' + asset + '_1prd'] = np.nan
    df.loc[df.time < datetime.time(9, 33, 0), 's_' + asset + '_2prd'] = np.nan
    df.loc[df.time < datetime.time(9, 35, 0), 's_' + asset + '_4prd'] = np.nan

    dayopen.rename(columns={asset:asset+'_open'}, inplace=True)
    dayopen.head()
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

display(time.time() - time0, df.head())

### do modeling ###

t_df = df.copy()
t_df.rename(columns={'EMXC_ret':'target'}, inplace=True)
t_df.drop(columns = ['time', 'date', 'Spx_ret', 'Nasdaq_ret', 'Russel_ret', 'EEM_ret'], 
          inplace=True)
t_df

t_df = t_df.dropna()
t_df.info()


y = t_df.pop('target')
X = t_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(0.2*X.shape[0]))
display(X_train.shape, X_test.shape, y_train.shape, X_train.head())

time1 = time.time()

xgbm = XGBRegressor()
parameters = {'eta':[0.03, 0.04, 0.05, 0.06, 0.07], 
              'max_depth':[2, 3],
             'subsample':[0.8, 1],
             'colsample_bytree':[0.6, 1]}
xgbgs = GridSearchCV(xgbm, parameters, cv=2)
xgbgs.fit(X_train, y_train)
print(xgbgs.best_params_)

rdm = ElasticNet()
parameters = {'alpha':[0, 0.001, 0.01, 0.05, 0.1, 0.2], 
              'l1_ratio':[0, 0.2, 0.5, 0.8, 1]}
rmgs = GridSearchCV(rdm, parameters, scoring='r2', cv=4)
rmgs.fit(X_train, y_train)
print(rmgs.best_params_)

print('In sample, xgb: ', r2_score(y_train, xgbgs.predict(X_train)))
print('Out of sample, xgb: ', r2_score(y_test, xgbgs.predict(X_test)))

print('In sample, ridge: ', r2_score(y_train, rmgs.predict(X_train)))
print('Out of sample, ridge: ', r2_score(y_test, rmgs.predict(X_test)))

print('Total time: ', time.time()-time0)















import numpy as np
import pandas as pd
import os, time, warnings, random, shap, requests, optuna, datetime, joblib, pytz
import seaborn as sns
import matplotlib.pyplot as plt
import functools as ft
import yfinance as yf

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor

pd.set_option('display.max_columns', 100)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore') 

time0 = time.time()

bucket_path = 'gs://pmykola-streaming-projects/spg-stocks/data/'

pull_time = datetime.datetime.now()
pull_time = pull_time.astimezone(pytz.timezone('America/New_York'))
pull_time = pull_time.replace(tzinfo=None)
now_time = (str(pull_time.month) + '_' + 
str(pull_time.day) + '_' +
str(pull_time.hour) + ':'  +
str(pull_time.minute) + ':' +
str(pull_time.second))

datafiles = !gsutil ls gs://pmykola-streaming-projects/spg-stocks/data
start_file = [x for x in datafiles if ('data_start_' in x)]
datafiles = [x for x in datafiles if ('auto_data_last_' in x) & ('pull_time' in x)]
assert len(start_file) == 1
start_file = start_file[0]

df = pd.read_csv(start_file)
df.Datetime = pd.to_datetime(df.Datetime)

df_new = pd.DataFrame(columns = df.columns)
for file in datafiles:
    temp_df = pd.read_csv(file)
    df_new = pd.concat([df_new, temp_df], axis=0)
    # remove duplicates
df_new.reset_index(inplace=True, drop=True)
df_new.drop_duplicates(inplace=True)
df_new.Datetime = pd.to_datetime(df_new.Datetime)
df_new.sort_values(by='Datetime')

df = pd.read_csv(start_file)
df.Datetime = pd.to_datetime(df.Datetime)
df = pd.concat([df, df_new])
df.drop_duplicates(inplace=True)
df.sort_values(by='Datetime')

df['time'] = df.Datetime.dt.time
df['date'] = df.Datetime.dt.date

df = df.fillna(method='ffill')
dayclose = df[df.time==datetime.time(15, 58, 0)]
dayopen = df[df.time==datetime.time(9, 30, 0)]
dayopen.reset_index(drop=True, inplace=True)
dayclose.reset_index(drop=True, inplace=True)
dayclose.sort_values(by='date')

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

    df = pd.merge(df, dayopen[['date', asset + '_open']], on=['date'], how='left')
    df = pd.merge(df, dayclose_l1[['date', asset + '_close_l1']], on=['date'], how='left')
    df = pd.merge(df, dayclose_l2[['date', asset + '_close_l2']], on=['date'], how='left')

    df['s_' + asset + '_ret_open'] = (100*(df[asset]/df[asset + '_open']-1)).shift(1)
    df['s_' + asset + '_ret_close1'] = (100*(df[asset]/df[asset + '_close_l1']-1)).shift(1)
    df['s_' + asset + '_ret_close2'] = (100*(df[asset]/df[asset + '_close_l2']-1)).shift(1)

    cols_todrop = [x for x in list(df.columns) if asset in x and 'ret' not in x]
    df.drop(columns = cols_todrop, inplace=True)

### do prediction ###

model_path = '/home/jupyter/project_repos/spg_stocks/spg_stocks/stocks-app/en_model.pkl'
trained_model = joblib.load(open(model_path, "rb"))

this_day = df.loc[df.date == df.date.max()]
print(f'{this_day.shape[0]} observations this day')
X = this_day.copy()
X.drop(columns = ['Datetime',
                  'time', 
                  'date', 
                  'Spx_ret', 
                  'Nasdaq_ret', 
                  'Russel_ret', 
                  'EEMA_ret', 
                  'EEM_ret', 
                  'EMXC_ret', 
                  'VXUS_ret', 
                  'VTHR_ret'], 
                  inplace=True,
                  errors = 'ignore')

if(X.count().sum() < X.shape[1]):
    print(f'''There are {X.shape[1] - X.count().sum()} missing values. 
          There will be an error''')

y = this_day.VTHR_ret
y_hat = trained_model.predict(X)

model_rmse = mean_squared_error(y, y_hat)
constant_rmse = mean_squared_error(y, np.zeros(len(y)))

performance = pd.DataFrame([[100*(r2_score(y, y_hat)), model_rmse, constant_rmse, 100*(1-model_rmse/constant_rmse)]], 
                           columns = ['R2', 'model_rmse', 'constant_rmse', 'rmse_improvement'])

file_name = 'm1_performance_' + \
str(df.date.max().year) + \
str(df.date.max().month) + \
str(df.date.max().day) + \
'_pull_time_' + \
now_time + \
'.csv'
performance.to_csv('gs://pmykola-streaming-projects/spg-stocks/artifacts' + '/' + file_name)

print(f'''Success. File {file_name} created and uploaded. 
Total time: {time.time()-time0} sec.''')
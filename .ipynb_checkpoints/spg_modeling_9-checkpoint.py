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

tickerStrings = ['^GSPC', '^IXIC', '^RUT', 'EEM', 'EMXC', 'EEMA', 'VTHR']
df_list = list()
for ticker in tickerStrings:
    data = yf.download(ticker, 
                       group_by="Ticker", 
                       period='60d', 
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

df = df.fillna(method='ffill')
dayclose = df[df.time==datetime.time(15, 58, 0)]
dayopen = df[df.time==datetime.time(9, 30, 0)]
dayopen.reset_index(drop=True, inplace=True)
dayclose.reset_index(drop=True, inplace=True)
dayclose.sort_values(by='date')
display(df, dayopen.head(), dayclose.head())
df0 = df.copy()

# df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
# df['minute'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.minute


### now i wanna do feature engineering for all assets 

asset_list = ['Spx', 'Nasdaq', 'Russel', 'EMXC', 'EEMA', 'EEM', 'VTHR']

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
t_df.rename(columns={'VTHR_ret':'target'}, inplace=True)
t_df.drop(columns = ['time', 'date', 'Spx_ret', 'Nasdaq_ret', 'Russel_ret', 'EEMA_ret', 'EEM_ret', 'EMXC_ret', 'VXUS_ret'], 
          inplace=True,
          errors = 'ignore')
t_df

t_df = t_df.dropna()
t_df.info()


y = t_df.pop('target')
X = t_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(0.2*X.shape[0]))
display(X_train.shape, X_test.shape, y_train.shape, X_train.head())

time1 = time.time()

# xgbm = XGBRegressor()
# parameters = {'eta':[0.03, 0.04, 0.05, 0.06, 0.07], 
#               'max_depth':[2, 3],
#              'subsample':[0.6, 0.8],
#              'colsample_bytree':[0.6, 0.8]}
# xgbgs = GridSearchCV(xgbm, parameters, cv=2)
# xgbgs.fit(X_train, y_train)
# print(xgbgs.best_params_)
# xgbt = XGBRegressor(**xgbgs.best_params_)
xgbt = XGBRegressor(n_estimators=500, eta=0.005, max_depth=2, subsample=0.6, colsample_bytree=0.5)
xgbt.fit(X_train, y_train)

enm = ElasticNet()
parameters = {'alpha':[0.0005, 0.001, 0.002, 0.003, 0.005], 
              'l1_ratio':[0, 0.02, 0.05, 0.1, 0.25, 0.5, 1]}
enmgs = GridSearchCV(enm, parameters, scoring='r2', cv=4)
enmgs.fit(X_train, y_train)
print(enmgs.best_params_)
enmt = XGBRegressor(**enmgs.best_params_)
enmt.fit(X_train, y_train)

print('In sample, xgb: ', r2_score(y_train, xgbt.predict(X_train)))
print('Out of sample, xgb: ', r2_score(y_test, xgbt.predict(X_test)))

print('In sample, ElasticNet: ', r2_score(y_train, enmgs.predict(X_train)))
print('Out of sample, ElasticNet: ', r2_score(y_test, enmgs.predict(X_test)))

print('Total time: ', time.time()-time0)


# explainerxgbc = shap.TreeExplainer(xgbt)
explainerxgbc = shap.Explainer(enmt)

shap_values_XGBoost_test = explainerxgbc.shap_values(X_test)
shap_values_XGBoost_train = explainerxgbc.shap_values(X_train)

vals = np.abs(shap_values_XGBoost_test).mean(0)
feature_names = X_test.columns
feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                 columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'],
                              ascending=False, inplace=True)

shap.summary_plot(shap_values_XGBoost_test, X_test, plot_type="bar", plot_size=(6,6), max_display=20)
shap.summary_plot(shap_values_XGBoost_train, X_train,plot_type="dot", plot_size=(6,6), max_display=20)

# xgboost sucks here
# takeaway is that to do regression with xgboost I need more observations than to do classification
# 5,000 obs is definitely not enough for xgb regression, so it is not even worth trying.


"""
print(f'''minimum and maximum values in train sample are 
{y_train.min()}, {y_train.max()}''')
print(f'''minimum and maximum values in test sample are 
{y_test.min()}, {y_test.max()}''')

print(f'''predicted minimum and maximum values in train sample are 
{enmt.predict(X_train).min()}, {enmt.predict(X_train).max()}''')
print(f'''predicted minimum and maximum values in test sample are 
{enmt.predict(X_test).min()}, {enmt.predict(X_test).max()}''')

may want to add truncation to predicted values later
"""


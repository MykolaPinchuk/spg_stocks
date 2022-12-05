### This script is an evolution of 'predict.py' with Flask deployment added

import numpy as np
import pandas as pd
import os, time, warnings, requests, datetime, joblib, pytz
import functools as ft
import yfinance as yf

from flask import Flask, render_template, request
app = Flask(__name__)
from sklearn.linear_model import ElasticNet
from google.cloud import storage


### Use Flask to build a simple website

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        user_click_time = datetime.datetime.now().astimezone(pytz.timezone('America/New_York'))
        
        
        ### Use model to generate prediction
        
        time0 = time.time()
        tickerStrings = ['^GSPC', '^IXIC', '^RUT', 'EEM', 'EMXC', 'EEMA', 'VTHR']
        df_list = list()
        for ticker in tickerStrings:
            data = yf.download(ticker, 
                               group_by="Ticker", 
                               period='5d', 
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
        dayclose = dayclose.append(df.tail(1))
        # this is needed for later join. prices from the last row will enevr be used.
        dayopen = df[df.time==datetime.time(9, 30, 0)]
        dayopen.reset_index(drop=True, inplace=True)
        dayclose.reset_index(drop=True, inplace=True)
        dayclose.sort_values(by='date')
        df = df.tail(30)

        asset_list = ['Spx', 'Nasdaq', 'Russel', 'EMXC', 'EEMA', 'EEM', 'VTHR']

        df_head = df.iloc[0:(df.shape[0]-15),:]

        pull_time = datetime.datetime.now()
        pull_time = pull_time.astimezone(pytz.timezone('America/New_York'))
        pull_time = pull_time.replace(tzinfo=None)
        pull_time_seconds = (pull_time - datetime.datetime(2022,1,1)).total_seconds()
        # this is number of seconds passed since the last 2-min point I want to have in my data
        df_tail = df.tail(15)
        # display(df_tail)

        df_tail_new = pd.DataFrame(columns = df_tail.columns)

        # this for loop keeps obs only if they correspond to 2-min time points
        # and the last one after 2-min point if that point is not available
        for i in range(df_tail.shape[0]):
            temp_time = df_tail.iloc[i,:]['datetime']
            diff_sec = (pull_time - temp_time).total_seconds()
            if (diff_sec >= 120) & (temp_time.minute%2 == 0) & (temp_time.second == 0):
                df_tail_new = df_tail_new.append(df_tail.iloc[i])
            if (diff_sec < 120):   
                if (temp_time.minute%2 == 0) & (temp_time.second == 0):
                    df_tail_new = df_tail_new.append(df_tail.iloc[i])
                    break
                elif (diff_sec < pull_time_seconds%120):
                    df_tail_new = df_tail_new.append(df_tail.iloc[i])
                    break

        df = pd.concat([df_head, df_tail_new], axis=0)

        for asset in asset_list:

            df[asset + '_ret'] = 100*(df[asset]/df[asset].shift(1)-1)
            df['s_' + asset + '_ret_1prd'] = (100*(df[asset]/df[asset].shift(1)-1)).shift(1)
            df['s_' + asset + '_ret_2prd'] = (100*(df[asset]/df[asset].shift(2)-1)).shift(1)
            df['s_' + asset + '_ret_4prd'] = (100*(df[asset]/df[asset].shift(4)-1)).shift(1)

            df.loc[df.time < datetime.time(9, 32, 0), 's_' + asset + '_1prd'] = np.nan
            df.loc[df.time < datetime.time(9, 33, 0), 's_' + asset + '_2prd'] = np.nan
            df.loc[df.time < datetime.time(9, 35, 0), 's_' + asset + '_4prd'] = np.nan

            dayopen.rename(columns={asset:asset+'_open'}, inplace=True)
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

        prediction_from = df.datetime.iloc[df.shape[0]-1] 
        prediction_to = df.datetime.iloc[df.shape[0]-1] + datetime.timedelta(minutes = 2)

        # model_path = '/home/jupyter/project_repos/spg_stocks/spg_stocks/stocks-app/en_model.pkl'
        model_path = 'en_model.pkl'
        trained_model = joblib.load(open(model_path, "rb"))

        X = df.tail(1)
        X.drop(columns = ['datetime', 'time', 'date', 
                          'Spx_ret', 'Nasdaq_ret', 'Russel_ret', 'EEMA_ret', 'EEM_ret', 'EMXC_ret', 'VXUS_ret', 'VTHR_ret'], 
                          inplace=True,
                          errors = 'ignore')

        # if(X.count().sum() < X.shape[1]):
        #     print(f'''There are {X.shape[1] - X.count().sum()} missing values. 
        #           There will be an error''')
        # print(f'''Prediction for Russel 3000 return from {prediction_from} 
        #       to {prediction_to} is {trained_model.predict(X)}''')
        # print('Total time: ', time.time()-time0)

        model_prediction = trained_model.predict(X)[0]
        model_prediction = str(model_prediction)[:6]
        prediction_runtime = time.time() - time0
        
        return render_template('predict.html', 
                               prediction=model_prediction,
                               prediction_from=prediction_from,
                               prediction_to=prediction_to,
                               pull_time=pull_time,
                               click_time=user_click_time,
                               runtime=prediction_runtime)
if __name__ == '__main__':
    app.run(debug=True)
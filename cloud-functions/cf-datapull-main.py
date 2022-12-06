import numpy as np
import pandas as pd
import os, time, warnings, random, requests, datetime, pytz
import functools as ft
import yfinance as yf
from google.cloud import storage

storage_client = storage.Client()
bucket_name = 'pmykola-streaming-projects'
BUCKET = storage_client.get_bucket(bucket_name)

now_time = str(pull_time.month) + '_' + str(pull_time.day) + '_' + str(pull_time.hour) + ':' + str(pull_time.minute) + ':' + str(pull_time.second) 

def save_data_to_bucket(request):
  """
  Responds to any HTTP request.
  Args:
    request (flask.Request): HTTP request object.
  Returns:
    The response text or any set of values that can be turned into a
    Response object using
    `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
  """
  pull_time = datetime.datetime.now()
  pull_time = pull_time.astimezone(pytz.timezone('America/New_York'))
  pull_time = pull_time.replace(tzinfo=None)

  tickerStrings = ['^GSPC', '^IXIC', '^RUT', 'EEM', 'EMXC', 'EEMA', 'VTHR']
  df_list = list()
  for ticker in tickerStrings:
      data = yf.download(ticker, 
                        group_by="Ticker", 
                        period='1d', 
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
  last_day = df.index.date.max()

  file_name = 'spg-stocks/data/' + 'auto_data_last_' + \
  str(last_day.year) + str(last_day.month) + str(last_day.day) + '_pull_time_' + now_time + '.csv'
  blob = BUCKET.blob(file_name)
  blob.upload_from_string(df.to_csv(), 'text/csv')
  result = file_name + ' upload complete to ' + bucket_name
  return {'response' : result}


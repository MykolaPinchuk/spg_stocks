import numpy as np
import pandas as pd
import os, time, warnings, random, requests, datetime
import functools as ft
import yfinance as yf
from google.cloud import storage

pd.set_option('display.max_columns', 100)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore') 

time0 = time.time()

project_id = 'valid-heuristic-369117'
data_path = '/home/jupyter/project_repos/spg_stocks/spg_stocks/data'
bucket_path = 'gs://pmykola-streaming-projects/spg-stocks/data'

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

#print(df.head())
last_day = df.index.date.max()

os.chdir(data_path)

file_name = 'data_last_' + \
str(last_day.year) + \
str(last_day.month) + \
str(last_day.day) + \
'.csv'

df.to_csv(file_name)

storage_path = os.path.join(bucket_path, file_name)
blob = storage.blob.Blob.from_string(storage_path,     client=storage.Client(project=project_id))
blob.upload_from_filename(file_name)

print(f'''Data downloaded successfully. Stored in {data_path + '/'} and {bucket_path}. 
File has {df.shape[0]} rows, is {os.stat(file_name).st_size/(2**20)} MB.
Total script time is {time.time()-time0} sec''')
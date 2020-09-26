import os
import sys
import pandas as pd
import numpy as np
import zipfile
import zlib
import pyarrow
import datetime
from tqdm import tqdm


def load_data(file_path, password, data_path):
    if not os.path.isfile(file_path):
        os.system('wget https://www.dropbox.com/s/rvr9sdx539j7mb5/hack_data.zip')

    if not os.path.isdir(data_path):
        with zipfile.ZipFile(file_path, 'r') as f:
            print(f.printdir())
            print("Extracting contents...")
            f.extractall(path=data_path, pwd=bytes(password, 'utf-8'))
            print("Extraction success")
    else:
        print("data is there and extracted")


def data_to_panda(data_path):
    return (pd.read_csv(data_path + '/' + 'clients.csv'),
            pd.read_csv(data_path + '/' + 'materials.csv'),
            pd.read_csv(data_path + '/' + 'plants.csv'),
            pd.read_parquet(data_path + '/' + 'transactions.parquet', engine='pyarrow', use_threads=True))


def group_clients(df_clients):
    df_clients.dropna(inplace=True)
    df_clients["gender_city_group"] = df_clients['city'] + df_clients['gender']
    groups = df_clients['gender_city_group'].unique()
    group_dict = {}
    for i, group in enumerate(groups):
        group_dict[group] = i
    df_clients['gender_city_group'] = df_clients['gender_city_group'].map(group_dict)
    df_clients['age'] = 2020 - df_clients['birthyear']
    df_clients.drop('birthyear', axis=1, inplace=True)
    return df_clients


def filterby_trans_period(df_transaction, days=60):
    data = df_transaction.groupby('client_id').agg({'chq_date': ['min', 'max']})
    data['loyalty_period'] = data['chq_date', 'max'] - data['chq_date', 'min']
    data.drop(data[data['loyalty_period'] < datetime.timedelta(days=days)].index, inplace=True)
    data.reset_index(inplace=True)
    data.columns = data.columns.droplevel()
    data.columns = ['client_id', 'min', 'max', 'loyalty_period']  # fix after dropping levels
    data.drop(['min', 'max'], axis=1, inplace=True)
    return df_transaction.merge(data, how='inner', on='client_id')


if __name__ == "__main__":
    load_data('./hack_data.zip', 'Skoltech', 'hack_data')
    df_clients, df_materials, df_plants, df_transaction = data_to_panda("hack_data")
    df_clients = group_clients(df_clients)
    df_transaction = filterby_trans_period(df_transaction)


'''
// how to add new column with average monthly expenses:
data = df_transaction.groupby('client_id').agg({'sales_sum':'sum'})
df_transaction = df_transaction.merge(data, 'inner', 'client_id')
df_transaction.rename(columns = {'sales_sum_x': 'sales_sum', 'sales_sum_y':'agg_expenses'}, inplace=True)
df_transaction['loyalty_period'] = df_transaction['loyalty_period'].astype('timedelta64[D]')
df_transaction['average_monthly_expenses'] = df_transaction['agg_expenses'] / df_transaction['loyalty_period'] * 30
'''
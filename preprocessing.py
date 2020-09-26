import os
import sys
import pandas as pd
import numpy as np
import zipfile
import pyarrow
from tqdm import tqdm
import pyarrow.parquet as pq


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


if __name__ == "__main__":
    load_data('./hack_data.zip', 'Skoltech', 'hack_data')
    # df_clients, df_materials, df_plants, df_transaction = data_to_panda("hack_data")

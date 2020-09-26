import pickle
import os 
import platform
import pandas as pd
from misc import cached

dir_path = os.path.dirname(os.path.realpath(__file__))


def save_data(data, name):
    path = dir_path + ("/%s.dat" % name)
    if platform.system() == "Windows":
        path = path.replace('/', '\\')
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

def load_data(name):
    path = dir_path + ("/%s.dat" % name)
    if platform.system() == "Windows":
        path = path.replace('/', '\\')
    f = open(path, "rb")
    data = pickle.load(f)
    f.close()
    return data

    

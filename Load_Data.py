import numpy as np
#load data by address,CSV format
def load_data(adr):
    trainreader = np.loadtxt(open(adr, "rb"), delimiter=",", skiprows=0)
    return trainreader

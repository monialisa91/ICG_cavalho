import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
from ecg_preprocess import ecg_preprocess
from ecg_preprocess import icg_preprocess
from scipy.signal import savgol_filter as sg
from pantomkins import pt


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# 1. DATA LOAD

files = glob("01_RawData/*BL.mat")
files.sort(key=natural_keys)

# 2. DATA PARAMETERS

lim = 1000
fs = 300

cutoff_lowECG = 30
order_lowECG = 4
cutoff_highECG = 5
order_highECG = 4

# 3. LOAD OF THE PREPROCESSED FILES

ecg = ecg_preprocess(files[0], lim, sampling_rate=fs, cutoff_low=cutoff_lowECG, cutoff_high=cutoff_highECG, order_low=order_lowECG, order_high=order_highECG)
data_ecg = ecg.sg_filter()
data_ecg = data_ecg.reshape(data_ecg.shape[0])

icg = icg_preprocess(files[0], lim, sampling_rate=fs)
data_icg = icg.baseline()




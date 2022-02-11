import numpy as np
from bwr import calc_baseline
from scipy.io import loadmat
from scipy.signal import butter, lfilter, freqz
from scipy.signal import savgol_filter as sg



class ecg_preprocess():
    def __init__(self, file, lim, sampling_rate, cutoff_low, cutoff_high, order_low, order_high):
        self.file = file
        self.lim = lim
        self.sampling_rate = sampling_rate
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.order_low = order_low
        self.order_high = order_high

    def data_load(self):
        mat = loadmat(self.file)
        ecg = mat['ECG']
        return ecg[:self.lim]

    def wander_removal(self):
        data = self.data_load()
        baseline = calc_baseline(data)
        data = data - baseline
        return data

    def butter_lowpass(self):
        data = self.data_load()
        nyq = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff_low / nyq
        b, a = butter(self.order_low, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    def butter_highpass(self):
        data = self.butter_lowpass()
        nyq = 0.5 * self.sampling_rate
        normal_cutoff_high = self.cutoff_high / nyq
        b, a = butter(self.order_high, normal_cutoff_high, btype='high', analog=False)
        y = lfilter(b, a, data)
        return y

    def sg_filter(self, window_length=31, polyorder=4):
        data = self.butter_highpass()
        data = sg(data, window_length, polyorder, mode='nearest')
        return data

class icg_preprocess():
    def __init__(self, file, lim, sampling_rate, radius=3):
        self.file = file
        self.lim = lim
        self.radius = radius
        self.sampling_rate = sampling_rate

    def load_data(self):
        mat = loadmat(self.file)
        icg = mat['ICG']
        return icg[:self.lim]

    def rolling_mean(self):
        data = self.load_data()
        means = np.zeros(len(data))

        for i in range(self.radius, len(data)-self.radius):
            slice = np.mean(data[(i-self.radius):(i+self.radius)])
            means[i] = slice

        for j in range(self.radius):
            means[j] = data[j]
            means[len(data)-j-1] = data[len(data)-1-j]

        return means

    def sg_filter(self, window_length=11, polyorder=4):
        data = self.rolling_mean()
        data = sg(data, window_length, polyorder, mode='nearest')
        return data

    def baseline(self):
        data = self.sg_filter()
        bl = calc_baseline(data)
        data = data - bl
        return data


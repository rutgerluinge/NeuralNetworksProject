from scipy.signal import butter, sosfilt
from scipy import fftpack, fft
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
data_file_name = 'Datasets/Correct/usage_2021-05-17_2021-05-24.csv'

def butter_highpass(data, cutoff, samplerate, order=5):
    nyq = 0.5 * samplerate
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sosfilt(sos, data)
    
def butter_lowpass(data, cutoff, samplerate, order=5):
    nyq = 0.5 * samplerate
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sosfilt(sos, data)

def butter_bandpass(data, cutoff_low, cutoff_high, samplerate, order=5):
    nyq = 0.5 * samplerate
    sos = butter(order, [cutoff_low / nyq, cutoff_high / nyq], btype='band', analog=False, output='sos')
    return sosfilt(sos, data)

def frequencies(data, samplerate):
    spectrum = fft.fft(data)
    axis = fft.fftfreq(len(spectrum), samplerate)
    return spectrum, axis

def main():
    # data_set = np.genfromtxt(
    #     data_file_name, 
    #     skip_header=1, 
    #     usecols=(1), 
    #     converters={0: lambda s: np.datetime64(s.decode('utf-8')), 1: lambda s: float(s)}, 
    #     delimiter=',',
    #     dtype=(np.datetime64, float)
    #     )
    data = pd.read_csv(
        data_file_name, 
        parse_dates=True, 
        header=None,
        names=['timestamp', 'usage']
        )
    # data_set = np.genfromtxt(data_file_name)
    breakpoint
    # Filter frequencies at 1 hour
    cutoff_high = 1 / (10 * 60)
    cutoff_low = 1 / (120 * 60)
    # sample frequency is 10s per sample, so 1/10 Hz
    samplerate = 0.1
    low = butter_lowpass(data['usage'], cutoff_low, samplerate)
    mid = butter_bandpass(data['usage'], cutoff_low, cutoff_high, samplerate)
    high = butter_highpass(data['usage'], cutoff_high, samplerate)

    spectrum, axis = frequencies(low, samplerate)

    plt.figure()
    
    plt.subplot(511)
    plt.plot(
        data['usage'],
        # data_set['timestamp'],
        label='original data'
        )
    plt.legend()
    
    plt.subplot(512)
    plt.plot(
        low, 
        # data_set['timestamp'],
        label='Low pass filtered'
        )
    plt.legend()
    
    plt.subplot(513)
    plt.plot(
        high, 
        # data_set['timestamp'],
        label='High pass filtered'
        )
    plt.legend()
    plt.subplot(514)
    plt.plot(
        mid, 
        # data_set['timestamp'],
        label='Band pass filtered'
        )
    plt.legend()
    plt.subplot(515)
    plt.plot(
        spectrum,
        np.real(axis), 
        '^r',
        label='spectrum'
        )
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
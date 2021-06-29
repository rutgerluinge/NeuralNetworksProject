from tools import butter_bandpass, butter_highpass, butter_lowpass
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
data_file_name = 'Datasets/Correct/usage.csv'

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
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # data = data.loc[data['timestamp'].dt.weekday == 5]
    
    # sample frequency is 10s per sample, so 1/10 Hz
    sample_frequency = 0.1
    
    # List with sample_frequencies along which to filter, from low to high   , 1 / (3 * 10)
    frequencies = [1 / (60 * 60 * 24), 1 / (60 * 60), 1 / (60)]
    
    filtered = []

    plt.figure()
    plt.suptitle('Original signal')
    plt.plot(
        data['usage'],
        # data_set['timestamp'],
        label='original data'
        )
    plt.legend()
    
    filtered.append(butter_lowpass(data['usage'], frequencies[0], sample_frequency))
    
    plt.figure()
    plt.suptitle('Low pass filtered at {}Hz'.format(frequencies[0]))
    plt.plot(
        filtered[-1]
        )
    
    for i in range(1, len(frequencies)):
        filtered.append(butter_bandpass(data['usage'], frequencies[i-1], frequencies[i], sample_frequency))
        plt.figure()
        plt.suptitle('Band pass filtered between {} and {}Hz'.format(frequencies[i-1], frequencies[i]))
        plt.plot(
            filtered[-1]
            )
    
    filtered.append(butter_highpass(data['usage'], frequencies[-1], sample_frequency))
    plt.figure()
    plt.suptitle('High pass filtered at {}Hz'.format(frequencies[-1]))
    plt.plot(
        filtered[-1]
        )
    recombined = np.zeros((filtered[0].shape[0]))
    for item in filtered:
        recombined = np.add(recombined, item)
    
    plt.figure()
    plt.suptitle('Recombined')
    plt.plot(
        recombined
        )
    
    
    plt.show()

if __name__ == '__main__':
    main()
from tools import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import timedelta

# HYPERPARAMETERS
# hyperparameters for parameter optimization
hyperparameters = {
    'train_window': [200, 500, 1000, 1500, 2000],
    'predict_window': [1, 2, 3, 4, 5, 10],
    'resevoir_size': [500, 1000, 1500, 2000, 3000],
    'sparsity': [0.005, 0.01, 0.05, 0.1],
    'rand_seed': 22,
    'spectral_radius': [0.8, 0.9, 1, 1.1, 1.2, 1.5],
    'noise': 0.001
}
hp = {
    'resevoir_size': 2000,
    'sparsity': 0.05,
    'rand_seed': 21,
    'spectral_radius': 0.80,
    'noise': 0.001
}

# data file
data_file_name = 'Datasets/Correct/usage.csv'

# Data sample frequency (sample interval = 10s)
sample_frequency = 0.1

# Offset is used to exclude the first part of the filtered timeseries, 
# where the butterworth filters create fake oscillations when the first sample
# in the timeseries is not 0
offset = 50
# Use data of past n samples (sample interval is 10s)
past_window_size = 1000
# to predict the next n samples
future_window_size = 2

n_runs = 50
future_window_total = n_runs * future_window_size

# Use timeseries [a...b] to train ESN to predict [a+n...b+n]
def esn_sections_signalinput():
    pass

# predict timeseries by splitting timeseries in different sections based on timescale
# Ex daily pattern, hourly pattern, minute pattern,
# and train esns on signals filtered for these sections
def esn_sections():
    # List with cut-off frequencies to filter the data with
    # intervals = [(60 * 60 * 24), (60 * 60), (60)]
    intervals = [(60 * 60 * 24)]
    
    # Since low frequency signals are slower, we can predict these signals easier and therefore further in time
    # This apporach is used to predict the general day/night patterns far ahead, and the fluctuations (higher frequency components)
    # can be predicted on a shorter scale. 
    # downsampling = [(60 * 6), (10 * 6), (6), (1)]
    downsampling = [(30 * 6)]
    
    n_sections = len(intervals) + 1

    # Load data
    data = pd.read_csv(
        data_file_name, 
        parse_dates=True, 
        header=None,
        names=['timestamp', 'usage']
        )
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    upper_boundary = min(data['usage'].size, past_window_size + offset + future_window_total)
    data = data['usage'].to_numpy()
    indices_train = range(past_window_size + offset, upper_boundary, future_window_size)
    
    # This list stores the filtered sections of the original data, in the same order as 
    sections = []
    
    # Low frequency timeseries
    section = butter_lowpass(data, 1 / intervals[0], sample_frequency)
    sections.append(section[ : : downsampling[0]])
    # Mid frequency timeseries
    for i in range(1, len(intervals)):
        section = butter_bandpass(data, 1 / intervals[i-1], 1 / intervals[i], sample_frequency)
        sections.append(section[ : : downsampling[i]])
    # High frequency timeseries
    section = butter_highpass(data, 1 / intervals[-1], sample_frequency)
    sections.append(section[ : : downsampling[-1]])
    
    # modells = [make_modell(hp), make_modell(hp), make_modell(hp), make_modell(hp)]
    modells = [make_modell(hp), make_modell(hp)]
    predictions = [np.empty((0))] * n_sections
    
    print_format = '{:<12}' * (2 * n_sections + 1)
    
    # print_header_1 = [[str(timedelta(seconds=s)), ''] for s in intervals]
    # print(print_format.format('', *[*section for section in print_header_1]))
    print(print_format.format('Run', *(['Time', 'MSE'] * n_sections)))
    for run, i in enumerate(indices_train):
        print_list = ['{}/{}'.format(run, n_runs)]
        for section in range(n_sections):
            train_window = sections[section][i - past_window_size : i]

            time_start = time.time()
            train_outputs = modells[section].fit(np.ones(past_window_size), train_window)
            duration = time.time() - time_start

            pred = modells[section].predict(np.ones(future_window_size))
            real = sections[section][i : i + future_window_size]
            mse = MSE(pred, real)
            predictions[section] = np.append(predictions[section], pred)
            print_list.append(round(duration, 4))
            print_list.append(round(mse, 4))
        print(print_format.format(*print_list))
    
    # Make tha graphs
    plt.figure()
    plt.suptitle('Low pass filtered at interval {}'.format(str(timedelta(seconds=intervals[0]))))
    plt.plot(sections[0])
    plt.plot(range(past_window_size, past_window_size + future_window_total), predictions[0])
    
    for i in range(1, len(intervals)):
        plt.figure()
        plt.suptitle('Band pass filtered between interval {} and {}'.format(str(timedelta(seconds=intervals[i - 1])), str(timedelta(seconds=intervals[i]))))
        plt.plot(sections[i])
        plt.plot(range(past_window_size, past_window_size + future_window_total), predictions[i])
    
    plt.figure()
    plt.suptitle('High pass filtered at interval {}'.format(str(timedelta(seconds=intervals[-1]))))
    plt.plot(sections[-1])
    plt.plot(range(past_window_size, past_window_size + future_window_total), predictions[-1])
    
    plt.show()
        

if __name__ == '__main__':
    esn_predict_sections()
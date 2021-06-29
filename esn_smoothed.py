from tools import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import timedelta
import random

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

# Moving average window size
window_size = 100

shuffle = True

n_runs = 50
future_window_total = n_runs * future_window_size

# Train 2 esns, one on data that is smoothed by a moving average, and the other on the difference between the original data and the moving average
def esn_smoothed():
    # Load data
    data = pd.read_csv(
        data_file_name, 
        parse_dates=True, 
        header=None,
        names=['timestamp', 'usage']
        )
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    n_sections = 3

    upper_boundary = min(data['usage'].size, past_window_size + offset + future_window_total)
    original = data['usage'].to_numpy()
    indices_train = range(past_window_size + offset, upper_boundary, future_window_size)
    
    # Get smoothed data
    smoothed = moving_average(original, (window_size, window_size))
    # Get difference between smoothed and original
    difference = np.subtract(original, smoothed)
    
    data_sections = [original, smoothed, difference]
    modells = [make_modell(hp), make_modell(hp), make_modell(hp)]
    predictions = [np.empty((0)), np.empty((0)), np.empty((0))]
    
    print_format = '{:<12}' * (2 * n_sections + 1)
    
    # print_header_1 = [[str(timedelta(seconds=s)), ''] for s in intervals]
    # print(print_format.format('', *[*section for section in print_header_1]))
    print(print_format.format('Run', *(['Time', 'MSE'] * n_sections)))
    
    for run, i in enumerate(indices_train):
        print_list = ['{}/{}'.format(run, n_runs)]
        for index in range(len(data_sections)):
            train_window = data_sections[index][i - past_window_size : i]

            time_start = time.time()
            train_outputs = modells[index].fit(np.ones(past_window_size), train_window)
            duration = time.time() - time_start

            pred = modells[index].predict(np.ones(future_window_size))
            real = data_sections[index][i : i + future_window_size]
            mse = MSE(pred, real)
            predictions[index] = np.append(predictions[index], pred)
            print_list.append(round(duration, 4))
            print_list.append(round(mse, 4))
        print(print_format.format(*print_list))
    
    plt.figure()
    plt.suptitle('Original')
    plt.plot(data_sections[0])
    plt.plot(range(past_window_size, past_window_size + future_window_total), predictions[0])
    
    plt.figure()
    plt.suptitle('Smoothed')
    plt.plot(data_sections[1])
    plt.plot(range(past_window_size, past_window_size + future_window_total), predictions[1])

    plt.figure()
    plt.suptitle('Difference')
    plt.plot(data_sections[2])
    plt.plot(range(past_window_size, past_window_size + future_window_total), predictions[2])
    
    plt.show()

def esn_smoothed_signalinput(past_window_size, future_window_size = 10, n_runs = 10):
    # Load data
    data = pd.read_csv(
        'Datasets/Correct/usage.csv', 
        parse_dates=True, 
        header=None,
        names=['timestamp', 'usage']
        )
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    original = data['usage'].to_numpy()
    # original = original.reshape(original.shape[0], 1)
    data_length = original.size
    # Subtract window to exclude the part where moving average has to shift with the window 
    past_window_size = past_window_size - window_size
    
    if(shuffle):
        indices = random.sample(range(past_window_size + window_size, data_length, future_window_size), n_runs * 2)
        indices_train = indices[ : n_runs]
        indices_validation = indices[n_runs : ]
    else:
        indices = range(past_window_size + window_size, future_window_size * n_runs * 2, future_window_size)
        indices_train = indices[ : n_runs]
        indices_validation = indices[n_runs : ]
    

    # Get smoothed data
    smoothed = moving_average(original, (window_size, window_size))
    # smoothed = original
    print_format = '{:<12}' * 3
    
    modell = make_modell_2(3000, 0.05, 1, 0.005)
    
    print(print_format.format('Run', *(['Time', 'MSE'] * 1)))
    for iteration, i in enumerate(indices_train):
        
        train_window = smoothed[i - past_window_size : i + future_window_size]

        time_start = time.time()
        train_outputs = modell.fit(train_window[ : -future_window_size], train_window[future_window_size : ])
        duration = time.time() - time_start

        mse = MSE(train_outputs.reshape(train_outputs.shape[0]), train_window[future_window_size : ])
        # predictions.append((i, train_outputs))
        plt.figure()
        plt.suptitle('Train: {}, MSE: {}'.format(iteration, mse))
        plt.plot(smoothed)
        plt.plot(range(i - past_window_size + future_window_size, i + future_window_size), train_outputs)
        
        print(print_format.format('{}/{}'.format(iteration, n_runs), round(duration, 4), round(mse, 4)))
    
    print('Validation')
    for iteration, i in enumerate(indices_validation):
        validation_window = smoothed[i - past_window_size : i + future_window_size]

        time_start = time.time()
        validation_outputs = modell.predict(validation_window[ : -future_window_size])
        duration = time.time() - time_start

        mse = MSE(validation_outputs.reshape(validation_outputs.shape[0]), validation_window[future_window_size : ])
        # predictions.append((i, train_outputs))
        plt.figure()
        plt.suptitle('Validation: {}, MSE: {}'.format(iteration, mse))
        plt.plot(smoothed)
        plt.plot(range(i - past_window_size + future_window_size, i + future_window_size), validation_outputs)
        
        print(print_format.format('{}/{}'.format(iteration, n_runs), round(duration, 4), round(mse, 4)))
    plt.show()
if __name__ == '__main__':
    # Train esn on signal length of 2 weeks
    # esn_smoothed_signalinput(6 * 60 * 24 * 7 * 2)
    esn_smoothed_signalinput(past_window_size = 6 * 60, n_runs = 5)
    
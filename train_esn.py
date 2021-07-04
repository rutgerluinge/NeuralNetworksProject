from ESN.ReferenceScripts.pyESN import ESN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import pickle
import random
import time
import os
import copy
from tools import *

data_file_name = 'Datasets/Correct/usage.csv'

hp = {
    'resevoir_size': 2000,
    'sparsity': 0.005,
    'rand_seed': 20,
    'spectral_radius': 1.2,
    'noise': 0.001
}


# Start training after n samples, to prevent filter oscillations at start to influence training
# 30 min = 30 * 6 samples
offset = 50

# Use data of past n samples (sample interval is 10s)
past_window_size = 500
# to predict the next n samples
future_window_size = 2

n_runs = 50
future_window_total = n_runs * future_window_size

# Shuffle data for training
shuffle = False

# Validation fraction of samples
validation = 0.1

# frequency of samples: 1 / sample interval = 1/10 = 0.1
samplerate = 0.1

def train_predict(hyper: dict, train_input, train_teacher, predict_input, predict_teacher):
    modell = make_modell(hyper)
    
    time_start = time.time()
    train_outputs = modell.fit(train_input, train_teacher)
    duration = round(time.time() - time_start, 4)

    prediction = modell.predict(predict_input)
    mse = round(MSE(prediction, predict_teacher), 4)
    return (mse, duration, prediction)

def esn_day_comparison():
    data_week = pd.read_csv(
        'Datasets/Correct/usage_2021-04-12_2021-04-19.csv', 
        parse_dates=True, 
        header=None,
        names=['date', 'usage']
        )
    data_week['date'] = pd.to_datetime(data_week['date'])
    data = data_week.loc[data_week['date'].dt.weekday == 5]
    
    del data_week
    
    # Normal
    data_normal = data['usage'].to_numpy()
    
    cutoff_freq = (6 * 20)
    # Low pass filtered
    data_filt = butter_lowpass(data_normal, 1 / cutoff_freq, samplerate)
    
    # Differenced
    ref_diff, data_diff = difference(data_normal)

    # Filtered and differenced
    ref_filt_diff, data_filt_diff = difference(data_filt)
    
    # moving averaged
    data_moving = moving_average(data_normal, (cutoff_freq, cutoff_freq), 1)
    ref_moving = data_moving[0]
    data_moving = data_moving[1:]

    # Make all data sets of equal length
    ref_filt = data_filt[0]
    data_filt = data_filt[1:]
    
    ref_normal = data_normal[0]
    data_normal = data_normal[1:]
    
    # Set up separate models for training
    modell_normal = make_modell()
    modell_filt = make_modell()
    modell_diff = make_modell()
    modell_filt_diff = make_modell()
    modell_moving = make_modell()
    
    # train_steps = int((data_filt.shape[0] - past_window_size - future_window_size) / n_runs)
    indices_train = range(past_window_size + offset, min(data_filt.shape[0], past_window_size + offset + future_window_total), future_window_size)

    train_size = len(indices_train)
    cnt = 0

    pred_normal = np.empty((0))
    pred_filt = np.empty((0))
    pred_diff = np.empty((0))
    pred_filt_diff = np.empty((0))
    pred_moving = np.empty((0))
    
    mean_mse_normal = 0
    mean_mse_filt = 0
    mean_mse_diff = 0
    mean_mse_filt_diff = 0
    mean_mse_moving = 0
    
        
    header = '{:<12}' * 11
    print('Starting training run')
    print(header.format('', 'Normal', '', 'Filtered', '', 'Differenced', '', 'Filtered & Differenced', '', 'Moving averaged', ''))
    print(header.format('run', 'Time', 'MSE', 'Time', 'MSE', 'Time', 'MSE', 'Time', 'MSE', 'Time', 'MSE'))
    for i in indices_train:
        # normal data
        train_window = data_normal[i - past_window_size : i]
        
        time_start = time.time()
        train_outputs = modell_normal.fit(np.ones(past_window_size), train_window)
        time_normal = time.time() - time_start
        
        prediction = modell_normal.predict(np.ones(future_window_size))
        real = data_normal[i : i + future_window_size]
        mse_normal = MSE(prediction, real)
        pred_normal = np.append(pred_normal, prediction)
        
        mean_mse_normal += mse_normal
        
        # Filtered data
        train_window = data_filt[i - past_window_size : i]
        
        time_start = time.time()
        train_outputs = modell_filt.fit(np.ones(past_window_size), train_window)
        time_filt = time.time() - time_start
        
        prediction = modell_filt.predict(np.ones(future_window_size))
        real = data_filt[i : i + future_window_size]
        mse_filt = MSE(prediction, real)
        pred_filt = np.append(pred_filt, prediction)
        
        mean_mse_filt += mse_filt
        
        # Differenced data
        train_window = data_diff[i - past_window_size : i]
        
        time_start = time.time()
        train_outputs = modell_diff.fit(np.ones(past_window_size), train_window)
        time_diff = time.time() - time_start
        
        prediction = modell_diff.predict(np.ones(future_window_size))
        real = data_diff[i : i + future_window_size]
        mse_diff = MSE(prediction, real)
        pred_diff = np.append(pred_diff, prediction)
        
        mean_mse_diff += mse_diff
        
        # filtered and differenced data
        train_window = data_filt_diff[i - past_window_size : i]
        
        time_start = time.time()
        train_outputs = modell_filt_diff.fit(np.ones(past_window_size), train_window)
        time_filt_diff = time.time() - time_start
        
        prediction = modell_filt_diff.predict(np.ones(future_window_size))
        real = data_filt_diff[i : i + future_window_size]
        mse_filt_diff = MSE(prediction, real)
        pred_filt_diff = np.append(pred_filt_diff, prediction)
        
        mean_mse_filt_diff += mse_filt_diff
        
        # Moving average
        train_window = data_moving[i - past_window_size : i]
        
        time_start = time.time()
        train_outputs = modell_moving.fit(np.ones(past_window_size), train_window)
        time_moving = time.time() - time_start
        
        prediction = modell_moving.predict(np.ones(future_window_size))
        real = data_moving[i : i + future_window_size]
        mse_moving = MSE(prediction, real)
        pred_moving = np.append(pred_moving, prediction)
        
        mean_mse_moving += mse_moving
        
        cnt += 1
        print(header.format(
            '{}/{}'.format(cnt, n_runs), 
            round(time_normal, 4), 
            round(mse_normal, 4), 
            round(time_filt, 4), 
            round(mse_filt, 4), 
            round(time_diff, 4), 
            round(mse_diff, 4), 
            round(time_filt_diff, 4), 
            round(mse_filt_diff, 4),
            round(time_moving, 4), 
            round(mse_moving, 4)
            ))
    print(header.format(
        'mean MSE', 
        '', 
        round(mean_mse_normal / n_runs, 4), 
        '', 
        round(mean_mse_filt / n_runs, 4), 
        '', 
        round(mean_mse_diff / n_runs, 4), 
        '', 
        round(mean_mse_filt_diff / n_runs, 4), 
        '', 
        round(mean_mse_moving / n_runs, 4)
        ))
    
    plt.figure()
    plt.suptitle(
        'Prediction with train size: {}, predict size: {} on unprocessed data'.format(past_window_size, future_window_size))
    plt.plot(
        range(past_window_size + future_window_total), 
        data_normal[offset : offset + past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        pred_normal, 
        label='prediction'
        )
    
    plt.figure()
    plt.suptitle('Prediction with train size: {}, predict size: {} on low pass filtered data'.format(past_window_size, future_window_size))
    plt.plot(
        range(past_window_size + future_window_total), 
        data_filt[ offset : offset + past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        pred_filt, 
        label='prediction'
        )
    
    plt.figure()
    plt.suptitle('Prediction with train size: {}, predict size: {} on differenced data'.format(past_window_size, future_window_size))
    plt.plot(
        range(past_window_size + future_window_total), 
        data_diff[ offset : offset + past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        pred_diff, 
        label='prediction'
        )
    
    plt.figure()
    plt.suptitle('Prediction with train size: {}, predict size: {} on filtered and differenced data'.format(past_window_size, future_window_size))
    plt.plot(
        range(past_window_size + future_window_total), 
        data_filt_diff[ offset : offset + past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        pred_filt_diff, 
        label='prediction'
        )
    
    plt.figure()
    plt.suptitle('Prediction with train size: {}, predict size: {} on averaged data'.format(past_window_size, future_window_size))
    plt.plot(
        range(past_window_size + future_window_total), 
        data_moving[ offset : offset + past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        pred_moving, 
        label='prediction'
        )
    
    plt.show()

    # plt.figure()
    # plt.plot(
    #     data['timestamp'][1:],
    #     data['usage'][1:], 
    #     label='original'
    #     )
    # plt.plot(
    #     data['timestamp'][1:],
    #     data_filtered[1:], 
    #     label='filtered'
    #     )
    # plt.plot(
    #     data['timestamp'][1:],
    #     data_differenced, 
    #     label='Differenced'
    #     )
    # plt.show()

    breakpoint

# Compare different window sizes for averaging
def day_average():
    data_week = pd.read_csv(
        data_file_name, 
        parse_dates=True, 
        header=None,
        names=['date', 'usage']
        )
    data_week['date'] = pd.to_datetime(data_week['date'])
    data = data_week.loc[data_week['date'].dt.weekday == 5]
    data = data['usage'].to_numpy()

    del data_week
    # Windows for averaging (min):
    windows = np.array([60 * 6, 20 * 6, 10 * 6, 5 * 6, 2 * 6, 1 * 6]) * 10
    
    plt.figure()
    plt.suptitle(
        'Original data')
    plt.plot(
        data
        )
    for window in windows:
        plt.figure()
        plt.suptitle(
            'Data averaged with window size ({0}, {0})'.format(window))
        plt.plot(
            moving_average(data, (window, window))
            )
    plt.show()

def esn_differenced():
    # # Since normalized change requires the dataset to be 1 entry larger, modify past_window_size
    # global past_window_size
    # past_window_size += 1
    
    # data_file = open(data_file_name, 'r')
    # data_reader = csv.reader(data_file)
    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    
    indices_train = range(past_window_size + 1, past_window_size + 1 + future_window_total, future_window_size)
    
    train_size = len(indices_train)
    cnt = 0
    prediction_chain = np.empty((0))
    
    for i in indices_train:
        das_modell = make_modell(hp)
        train_inputs_raw = data_set[i - past_window_size - 1 : i]
        # Normalize ze data
        ref, train_inputs = difference(train_inputs_raw)
        
        time_start = time.time()
        # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        train_outputs = das_modell.fit(np.ones(past_window_size), train_inputs)
        time_end = time.time()
        # make prediction
        prediction = das_modell.predict(np.ones(future_window_size))
        prediction = dedifference(data_set[i - 1], prediction)
        real = data_set[i : i + future_window_size]
        mse = MSE(prediction, real)
        prediction_chain = np.append(prediction_chain, prediction)

        cnt += 1
        print('{}/{} \t{}\t{}'.format(cnt, train_size, round(time_end - time_start, 4), round(mse, 4)))
    plt.figure()
    plt.plot(
        range(max(past_window_size - future_window_total, 0), past_window_size + future_window_total), 
        data_set[max(past_window_size - future_window_total, 0) : past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        prediction_chain, 
        label='prediction'
        )
    plt.show()

def esn_filtered_lowpass(cutoff = 1 / (10 * 60)):
    das_modell = make_modell()

    data = np.genfromtxt(data_file_name, skip_header=0, usecols=(1), delimiter=',')
    data = butter_lowpass(data, cutoff, samplerate)
    train_size = int((len(data) - offset) / 2)
    train = data[offset : train_size + offset]
    real = data[train_size + offset : ]
    real_size = len(real)
    
    das_modell.fit(np.ones(train_size), train)
    predict = das_modell.predict(np.ones(real_size))
    # return
    plt.figure()
    plt.plot(
        data, 
        label='data'
        )
    plt.plot(
        range(train_size, train_size + real_size), 
        predict, 
        label='prediction'
        )
    plt.show()

def esn_filtered_lowerpass():
    das_modell = make_modell()
    
    data = np.genfromtxt(data_file_name, skip_header=0, usecols=(1), delimiter=',')
    filtered = butter_lowpass(data, 1 / (2 * 60), samplerate)
    ref, filtered = normalize_change(filtered)
    train_size = int(len(filtered) / 2)
    train = filtered[0 : train_size]
    real = filtered[train_size : ]
    real_size = len(real)
    
    das_modell.fit(np.ones(train_size), train)
    predict = das_modell.predict(np.ones(real_size))
    # return
    plt.figure()
    plt.plot(
        filtered, 
        label='data'
        )
    plt.plot(
        range(train_size, train_size + real_size), 
        predict, 
        label='prediction'
        )
    plt.show()

def esn_filtered_highpass():
    das_modell = make_modell()
    
    data = np.genfromtxt(data_file_name, skip_header=0, usecols=(1), delimiter=',')
    data = butter_highpass(data, 1 / (30 * 60), samplerate)
    
    prediction_chain = np.empty((0))
    
    indices_train = range(past_window_size, past_window_size + future_window_total, future_window_size)
    cnt = 0
    train_size = len(indices_train)
    for i in indices_train:
        train_inputs = data[i - past_window_size : i]
        # Normalize ze data
        
        time_start = time.time()
        # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        train_outputs = das_modell.fit(np.ones(past_window_size), train_inputs)
        time_end = time.time()
        # make prediction
        prediction = das_modell.predict(np.ones(future_window_size))
        real = data[i : i + future_window_size]
        mse = MSE(prediction, real)
        prediction_chain = np.append(prediction_chain, prediction)

        print('{}/{} \t{}\t{}'.format(cnt, train_size, round(time_end - time_start, 4), round(mse, 4)))
        cnt += 1
    plt.figure()
    plt.plot(
        range(past_window_size - future_window_total, past_window_size + future_window_total), 
        data[past_window_size - future_window_total : past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        prediction_chain, 
        label='prediction'
        )
    plt.show()

def esn_filtered_downsampled():
    # Downsampling - 60 second interval between samples = 6 * interval
    # 10 min sample interval
    scale = 10 * 6
    
    das_modell = make_modell()
    
    data = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    # 1 / 10 hrs cutoff frequency
    data = butter_lowpass(data, 1 / (10 * 60 * 6 * 10), samplerate)
    # down sample with scale value to get an array with sample interval = scale * old sample interval
    data = data[ : : scale]
    
    prediction_chain = np.empty((0))
    
    indices_train = range(past_window_size, past_window_size + future_window_total, future_window_size)
    cnt = 0
    train_size = len(indices_train)
    for i in indices_train:
        train_inputs = data[i - past_window_size : i]
        # Normalize ze data
        
        time_start = time.time()
        # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        train_outputs = das_modell.fit(np.ones(past_window_size), train_inputs)
        time_end = time.time()
        # make prediction
        prediction = das_modell.predict(np.ones(future_window_size))
        real = data[i : i + future_window_size]
        mse = MSE(prediction, real)
        prediction_chain = np.append(prediction_chain, prediction)

        print('{}/{} \t{}\t{}'.format(cnt, train_size, round(time_end - time_start, 4), round(mse, 4)))
        cnt += 1
    plt.figure()
    plt.plot(
        data, 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        prediction_chain, 
        label='prediction'
        )
    plt.show()

def esn_filtered_downsampled_normalized():
    # Downsampling - 60 second interval between samples = 6 * interval
    # 2 * 6 = 2 minutes
    scale = 12
    
    das_modell = make_modell(hp)
    
    data = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    # filter
    data = butter_lowpass(data, 1 / (30 * 60), samplerate)
    # down sample
    data = data[ : : scale]
    prediction_chain = np.empty((0))
    
    indices_train = range(past_window_size, past_window_size + future_window_total, future_window_size)
    cnt = 0
    train_size = len(indices_train)
    for i in indices_train:
        train_inputs = data[i - past_window_size : i]
        # Normalize ze data
        ref, train_inputs = normalize_change(train_inputs)

        time_start = time.time()
        # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        train_outputs = das_modell.fit(np.ones(past_window_size - 1), train_inputs)
        time_end = time.time()
        
        # make prediction
        prediction = das_modell.predict(np.ones(future_window_size))
        # denormalize
        prediction = denormalize_change(train_inputs[-1], prediction)
        real = data[i : i + future_window_size]
        mse = MSE(prediction, real)
        prediction_chain = np.append(prediction_chain, prediction)

        print('{}/{} \t{}\t{}'.format(cnt, train_size, round(time_end - time_start, 4), round(mse, 4)))
        cnt += 1
    plt.figure()
    plt.plot(
        range(past_window_size - future_window_total, past_window_size + future_window_total), 
        data[past_window_size - future_window_total : past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        prediction_chain, 
        label='prediction'
        )
    plt.show()

def esn_normalized(past_window_size, future_window_size = 2, train_runs = 50, offset = 50, shuffle = False):
    # Load data
    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')

    # Shuffle indices for more randomization in predictions, or not to get a more readable graph
    if(shuffle):
        indices_train = random.sample(range(past_window_size + 1 + offset, data_set.size - future_window_size), train_runs)
    else:
        indices_train = range(past_window_size + 1 + offset, past_window_size + offset + train_runs * future_window_size, future_window_size)
    
    prediction_chain = np.zeros((train_runs * future_window_size, 1))
    
    predictions = []
    mean_mse = 0
    
    for run, i in enumerate(indices_train):
        # train_inputs = data_set[i - past_window_size - 1 : i]
        # Normalize ze data
        ref, section = normalize(data_set[i - past_window_size - 1 : i + future_window_size])
        
        result = train_predict(
            hp, 
            np.ones(past_window_size), 
            section[ : past_window_size], 
            np.ones(future_window_size), 
            data_set[past_window_size : ]
            )
        denormalized = denormalize(ref, result[2])
        mse = MSE(denormalized, data_set[i : i + future_window_size])
        mean_mse += mse
        predictions.append((i, denormalized))
        
        # mean_mse += mse
        print('{}/{} \t{}\t{}'.format(run + 1, train_runs, result[1], mse))
        # print('{}/{} \t{}\t{}'.format(run, train_size, round(duration, 4), round(mse, 4)))
    
    plt.figure()
    plt.suptitle('MSE: {}, Past window: {}, future window: {}\nsparsity: {}, spectral radius: {}, reservoir size: {}, noise: {}'.format(
        round(mean_mse / train_runs, 4),
        past_window_size,
        future_window_size,
        hp['sparsity'],
        hp['spectral_radius'],
        hp['resevoir_size'],
        hp['noise']
        ))
    if(shuffle):
        plt.plot(
            data_set, 
            label='data'
            )
    else:
        plt.plot(
            range(offset + 1, offset + past_window_size + 1 + train_runs * future_window_size), 
            data_set[offset + 1 : offset + past_window_size + 1 + train_runs * future_window_size], 
            label='data'
            )
    # plt.plot(
    #     range(past_window_size + offset, past_window_size + offset + train_runs * future_window_size), 
    #     prediction_chain, 
    #     label='prediction'
    #     )
    for prediction in predictions:
        plt.plot(
            range(prediction[0], prediction[0] + future_window_size), 
            prediction[1], 
            label='prediction',
            color='orange'
            )
    plt.ylabel('Energy consumption (W)')
    plt.show()

""" 
    # Train ze ESN on all ze normalized data
    def esn_train_normalized():
        # Build ze modell
        das_modell = make_modell()
        
        # load ze dataset, were we use only the second column (power usage), and the header is skipped
        data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
        
        # # To prevent zero division errors, change all occurrences of 0 to 1
        # # This is a small deviation in the (-3000, 6000) range
        # data_set[data_set == 0] = 1
        
        index_range = data_set.size - past_window_size - 1 - future_window_size
        # Train and predict on random places in the dataset
        if(shuffle):
            indices_train = random.sample(range(past_window_size + 1, data_set.size - future_window_size), index_range)
        else:
            # 
            indices_train = range(past_window_size + 1, data_set.size - future_window_size, future_window_size)
        
        train_size = len(indices_train)
        
        # First, train ze esn on all ze data available
        print('Starting training of ESN model with {} datapoints'.format(data_set.size))
        cnt = 0
        try:
            for i in indices_train:
                cnt += 1
                # Train using the normalized dataset as input, and the same dataset shifted n steps into the future as teacher
                train_input_raw = data_set[i - past_window_size - 1 : i ]
                ref_input, train_input = normalize_change(train_input_raw)
                
                train_teacher_raw = data_set[i - past_window_size - 1  + future_window_size : i + future_window_size]
                ref_teacher, train_teacher = normalize_change(train_teacher_raw)
                
                time_start = time.time()
                results_raw = das_modell.fit(train_input, train_teacher)
                time_end = time.time()
                
                results = denormalize_change(ref_teacher, results_raw)
                real = denormalize_change(ref_teacher, train_teacher)
                mse = MSE(results, real)

                print('{}/{} \t{}\t{}'.format(cnt, train_size, round(time_end - time_start, 4), round(mse, 4)))
        except KeyboardInterrupt:
            pass
        print('Saving model as ' + model_file_name)
        # Saving our hard work
        with open(model_file_name, 'wb') as model_file:
            pickle.dump(das_modell, model_file)
        
    def esn_predict_normalized():
        try:
            model_file = open(model_file_name, 'wb')
            das_modell = pickle.load(model_file)
        except OSError:
            print('File ' + model_file_name + ' not found.')
            exit()
        
        # load ze dataset, were we use only the second column (power usage), and the header is skipped
        data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')

        index_range = (data_set.size - past_window_size) - future_window_size
        if(shuffle):
            indices_train = random.sample(range(past_window_size, data_set.size - future_window_size), index_range)
        else:
            indices_train = range(past_window_size, past_window_size + future_window_total, future_window_size)
"""    
def esn_normalized_change(past_window_size, future_window_size = 2, train_runs = 50, offset = 50):
    shuffle = False
    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    # To prevent zero division errors, change all occurrences of 0 to 1
    # This is a small deviation in the scale of (-3000, 6000) range
    # data_set[data_set == 0] = 1
    
    indices_train = range(past_window_size + 1 + offset, past_window_size + 1 + offset + future_window_size * train_runs, future_window_size)
    
    predictions = []
    mean_mse = 0
    
    for run, i in enumerate(indices_train):
        # Normalize ze data
        ref, section = normalize_change(data_set[i - past_window_size - 1 : i + future_window_size])
        result = train_predict(
            hp, 
            np.ones(past_window_size), 
            section[ : past_window_size], 
            np.ones(future_window_size), 
            data_set[past_window_size : ]
            )
        
        denormalized = denormalize_change(data_set[i - 1], result[2])
        mse = MSE(denormalized, data_set[i : i + future_window_size])
        mean_mse += mse
        predictions.append((i, denormalized))
        
        # time_start = time.time()
        # # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        # train_outputs = das_modell.fit(np.ones(past_window_size), train_inputs)
        # time_end = time.time()
        # # make prediction
        # prediction = das_modell.predict(np.ones(future_window_size))
        # prediction = denormalize_change(data_set[i - 1], prediction)
        # real = data_set[i : i + future_window_size]
        # mse = MSE(prediction, real)
        # prediction_chain = np.append(prediction_chain, prediction)

        # cnt += 1
        print('{}/{} \t{}\t{}'.format(run, train_runs, result[1], round(mse, 4)))
    plt.figure()
    plt.suptitle('MSE: {}, Past window: {}, future window: {}\nsparsity: {}, spectral radius: {}, reservoir size: {}, noise: {}'.format(
        round(mean_mse / train_runs, 4),
        past_window_size,
        future_window_size,
        hp['sparsity'],
        hp['spectral_radius'],
        hp['resevoir_size'],
        hp['noise']
        ))
    if(shuffle):
        plt.plot(
            data_set, 
            label='data'
            )
    else:
        plt.plot(
            range(offset + 1, offset + past_window_size + 1 + train_runs * future_window_size), 
            data_set[offset + 1 : offset + past_window_size + 1 + train_runs * future_window_size], 
            label='data'
            )
    for prediction in predictions:
        plt.plot(
            range(prediction[0], prediction[0] + future_window_size), 
            prediction[1], 
            label='prediction',
            color='orange'
            )
    plt.ylabel('Energy consumption (W)')
    plt.show()

def esn_raw(past_window_size, future_window_size = 2, train_runs = 50, offset = 50, shuffle = True):
    # Load data from csv
    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    
    # Generate a set of indices, which marks in the original dataset the end of a train subset, and the start of the predict subset
    if(shuffle):
        indices_train = random.sample(range(past_window_size + offset, data_set.size - future_window_size), train_runs)
    else:
        indices_train = range(past_window_size + offset, past_window_size + offset + train_runs * future_window_size, future_window_size)
    
    predictions = []
    mean_mse = 0
    
    for run, i in enumerate(indices_train):
        result = train_predict(
            hp, 
            np.ones(past_window_size), 
            data_set[i - past_window_size : i], 
            np.ones(future_window_size), 
            data_set[i : i + future_window_size]
            )
        mean_mse += result[0]
        predictions.append((i, result[2]))
        print('{}/{} \t{}\t{}'.format(run + 1, train_runs, result[0], result[1]))

    # The plot thickens
    plt.figure()
    plt.suptitle('MSE: {}, Past window: {}, future window: {}\nsparsity: {}, spectral radius: {}, reservoir size: {}, noise: {}'.format(
        round(mean_mse / train_runs, 4),
        past_window_size,
        future_window_size,
        hp['sparsity'],
        hp['spectral_radius'],
        hp['resevoir_size'],
        hp['noise']
        ))
    if(shuffle):
        plt.plot(
            data_set, 
            label='data'
            )
    else:
        plt.plot(
            range(offset, offset + past_window_size + train_runs * future_window_size), 
            data_set[offset : offset + past_window_size + train_runs * future_window_size], 
            label='data'
            )
    for prediction in predictions:
        plt.plot(
            range(prediction[0], prediction[0] + future_window_size), 
            prediction[1], 
            label='prediction',
            color='orange'
            )
    plt.ylabel('Energy consumption (W)')
    plt.show()
    # errors = np.empty((0))
    # for i in indices_validation:
    #     prediction = das_modell.predict(data_set[i - past_window_size : i])
    #     real = data_set[i : i + future_window_size]
    #     errors = np.append(errors, np.subtract(prediction, real))
    
    return

""" 
    def test_normalization():
        test = np.array(list(range(1,11)))
        ref, data = normalize_change(test)
        res = denormalize_change(ref, data)
        
        print(test)
        print(np.round(data, 2))
        print(res)

    def test_moving_average():
        data = np.arange(10)
        print(moving_average(data, (1,1), 1))
"""
if __name__ == '__main__':
    # Train esn on raw data
    # esn_raw(1500, train_runs=50, shuffle = False)

    # Train esn on normalized data
    # esn_normalized(1500, train_runs= 50)
    # esn_normalized_change(1500, train_runs= 50)
    esn_differenced()
    
from pyESN import ESN
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
import random
import time

data_file_name = 'Traindata/usage_05-04-2021_12-04-2021.csv'
model_file_name = 'Models/ESN-test-1500'
das_modell = None
model_exists = True
# HYPERPARAMETERS
hp = {
    'resevoir_size': 1000,
    'sparsity': 0.01,
    'rand_seed': 23,
    'spectral_radius': 0.9,
    'noise': 0.001
}


# Use data of past n samples (sample interval is 10s)
past_window_size = 3000
# to predict the next minute
future_window_size = 2
# train block size enables us not to load the entire dataset into memory, but to work in blocks
# This value should 
future_window_total = 100

# Shuffle data for training
shuffle = False

# Validation fraction of samples
validation = 0.1

def MSE(yhat, y):
    ph = np.subtract(yhat, y)
    ph = np.power(ph, 2)
    ph = np.mean(ph)
    ph = np.sqrt(ph)
    return ph

# Express entire dataset as fraction of first value in array
def normalize(data: np.ndarray):
    ref = data[0]
    normalized = np.empty((0))
    for i in range(data.size):
        normalized = np.append(normalized, data[i] / ref)
    return ref, normalized

def denormalize(ref, data: np.ndarray):
    denormalized = np.empty((0))
    for i in range(data.size):
        denormalized = np.append(denormalized, data[i] * ref)
    return denormalized

def esn_normalized():
    # global validation
    das_modell = ESN(
        n_inputs = 1,
        n_outputs = 1,
        n_reservoir = hp['resevoir_size'],
        sparsity = hp['sparsity'],
        random_state = hp['rand_seed'],
        spectral_radius = hp['spectral_radius'],
        noise = hp['noise'],
        silent = True
    )
    # data_file = open(data_file_name, 'r')
    # data_reader = csv.reader(data_file)

    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')

    # index_range = (data_set.size - past_window_size) - future_window_size
    # if(shuffle):
    #     indices = random.sample(range(past_window_size, data_set.size - future_window_size), index_range)
    # else:
    indices_train = range(past_window_size, past_window_size + future_window_total, future_window_size)
    
    # train_fraction = len(indices) - int(len(indices) * validation)

    # indices_train = indices[0 : train_fraction]
    # indices_validation = indices[train_fraction : -1]
    # del indices

    train_size = len(indices_train)
    cnt = 0
    breakpoint


    prediction_chain = np.empty((0))
    
    for i in indices_train:
        train_inputs = data_set[i - past_window_size : i]
        # Normalize ze data
        ref, train_inputs = normalize(train_inputs)
        
        time_start = time.time()
        # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        train_outputs = das_modell.fit(np.ones(past_window_size), train_inputs)
        time_end = time.time()
        # make prediction
        prediction = das_modell.predict(np.ones(future_window_size))
        prediction = denormalize(ref, prediction)
        real = data_set[i : i + future_window_size]
        mse = MSE(prediction, real)
        prediction_chain = np.append(prediction_chain, prediction)

        print('{}/{} \t{}\t{}'.format(cnt, train_size, round(time_end - time_start, 4), round(mse, 4)))
        cnt += 1
    plt.figure()
    plt.plot(
        range(past_window_size - future_window_total, past_window_size + future_window_total), 
        data_set[past_window_size - future_window_total : past_window_size + future_window_total], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + future_window_total), 
        prediction_chain, 
        label='prediction'
        )
    plt.show()

# Train ze ESN on all ze normalized data
def esn_train_normalized():
    # Build ze modell
    das_modell = ESN(
        n_inputs = 1,
        n_outputs = 1,
        n_reservoir = hp['resevoir_size'],
        sparsity = hp['sparsity'],
        random_state = hp['rand_seed'],
        spectral_radius = hp['spectral_radius'],
        noise = hp['noise'],
        silent = True
    )
    # load ze dataset, were we use only the second column (power usage), and the header is skipped
    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    
    # To prevent zero division errors, change all occurrences of 0 to 1
    # This is a small deviation in the (-3000, 6000) range
    data_set[data_set == 0] = 1
    
    index_range = data_set.size - past_window_size - 1 - future_window_size
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
    
    
# Express dataset as as change relative to preceding value
# returned dataset is of size input dataset - 1
def normalize_change(data: np.ndarray):
    # return last value in the train dataset, since that value is required to denormalize the predictions
    ref = data[0]
    normalized = np.empty((0))
    for i in range(1, data.size):
        normalized = np.append(normalized, data[i] / data[i - 1])
    return ref, normalized

def denormalize_change(ref, data:np.ndarray):
    denormalized = np.empty((1))
    denormalized[0] = ref * data[0]
    for i in range(1, data.size):
        denormalized = np.append(denormalized, denormalized[i - 1] * data[i])
    return denormalized

def esn_normalized_change():
    # # Since normalized change requires the dataset to be 1 entry larger, modify past_window_size
    # global past_window_size
    # past_window_size += 1
    
    das_modell = ESN(
        n_inputs = 1,
        n_outputs = 1,
        n_reservoir = hp['resevoir_size'],
        sparsity = hp['sparsity'],
        random_state = hp['rand_seed'],
        spectral_radius = hp['spectral_radius'],
        noise = hp['noise'],
        silent = True
    )
    # data_file = open(data_file_name, 'r')
    # data_reader = csv.reader(data_file)

    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    # To prevent zero division errors, change all occurrences of 0 to 1
    # This is a small deviation in the scale of (-3000, 6000) range
    data_set[data_set == 0] = 1
    
    indices_train = range(past_window_size + 1, past_window_size + 1 + future_window_total, future_window_size)
    
    train_size = len(indices_train)
    cnt = 0
    prediction_chain = np.empty((0))
    
    for i in indices_train:
        train_inputs_raw = data_set[i - past_window_size - 1 : i]
        # Normalize ze data
        ref, train_inputs = normalize_change(train_inputs_raw)
        
        time_start = time.time()
        # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        train_outputs = das_modell.fit(np.ones(past_window_size), train_inputs)
        time_end = time.time()
        # make prediction
        prediction = das_modell.predict(np.ones(future_window_size))
        prediction = denormalize_change(data_set[i - 1], prediction)
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

def esn_raw():
    global validation
    # try:
    #     model_file = open(model_file_name, 'wb')
    #     das_modell = pickle.load(model_file)

    # except Exception:
    
    das_modell = ESN(
        n_inputs = 1,
        n_outputs = 1,
        n_reservoir = hp['resevoir_size'],
        sparsity = hp['sparsity'],
        random_state = hp['rand_seed'],
        spectral_radius = hp['spectral_radius'],
        noise = hp['noise'],
        silent = True
    )
    data_file = open(data_file_name, 'r')
    data_reader = csv.reader(data_file)

    # data_len = sum(1 for row in data_set)
    # data_len = len(data_file.readlines())

    data_set = np.genfromtxt(data_file_name, skip_header=1, usecols=(1), delimiter=',')
    # data_set = np.genfromtxt(data_file_name, skip_header=1, converters={1: lambda s: int(s)})
    
    
    # train_outputs = das_modell.fit(np.ones(train_fraction), data_set[:train_fraction])
    breakpoint
    # index_range = (data_set.size - past_window_size) - future_window_size
    # if(shuffle):
    #     indices = random.sample(range(past_window_size, data_set.size - future_window_size), index_range)
    # else:
    indices = range(past_window_size, (past_window_size + 100) - future_window_size, future_window_size)
    
    train_fraction = len(indices) - int(len(indices) * validation)

    indices_train = indices[0 : train_fraction]
    indices_validation = indices[train_fraction : -1]
    del indices

    train_size = len(indices_train)
    cnt = 0
    breakpoint

    prediction_chain = np.empty((0))
    
    for i in indices_train:
        train_inputs = data_set[i - past_window_size : i]
        time_start = time.time()
        # train_outputs = das_modell.fit(train_inputs, data_set[i - past_window_size + future_window_size : i + future_window_size])
        train_outputs = das_modell.fit(np.ones(past_window_size), train_inputs)
        time_end = time.time()
        prediction = das_modell.predict(np.ones(future_window_size))
        real = data_set[i : i + future_window_size]
        mse = MSE(prediction, real)
        prediction_chain = np.append(prediction_chain, prediction)

        print('{}/{} \t{}\t{}'.format(cnt, train_size, round(time_end - time_start, 4), round(mse, 4)))
        cnt += 1
    # return
    plt.figure()
    plt.plot(
        range(past_window_size - 1000, past_window_size + train_size * future_window_size), 
        data_set[past_window_size - 1000 : past_window_size + train_size * future_window_size], 
        label='data'
        )
    plt.plot(
        range(past_window_size, past_window_size + train_size * future_window_size), 
        prediction_chain, 
        label='prediction'
        )
    plt.show()
    errors = np.empty((0))
    for i in indices_validation:
        prediction = das_modell.predict(data_set[i - past_window_size : i])
        real = data_set[i : i + future_window_size]
        errors = np.append(errors, np.subtract(prediction, real))
    
    return


    # data_set = np.empty((0), dtype=float)
    # cnt = 0
    # try:
    #     while cnt < past_window_size + future_window_size:
    #         row = next(data_reader)
    #         data_set = np.append(data_set, int(row[1]))
    #         cnt += 1
    # except StopIteration:
    #     print('Data set is not large enough.\nQuittin\'')
    #     exit()
    
    try:
        while True:
            # Fit the model
            das_modell.fit(data_set[0 : past_window_size], data_set[past_window_size : past_window_size + future_window_size])
            row = next(data_reader)
            data_set = np.append(data_set[1:], int(row[1]))
    # No more rows in the dataset
    except StopIteration:
        print('Done fiting the model')
        # Save tha model
        with open(model_file_name, 'wb') as model_file:
            pickle.dump(das_modell, model_file)
        exit()
            
            


    # prediction_chain = np.empty((0))
    # # real_chain = np.empty((0))

    # for i in range(0, future_window_total, future_window_size):
        
    #     training = das_modell.fit(np.ones(past_window_size), data_set[i:past_window_size + i])
    #     prediction = das_modell.predict(np.ones(future_window_size))
    #     # real_chain = np.append(real_chain, data_set[future_window_size + i : future_window_size + i + future_window_size])
    #     prediction_chain = np.append(prediction_chain, prediction)
    
    # # Plot tha results
    
    # henk = None

def test_normalization():
    test = np.array(list(range(1,11)))
    ref, data = normalize_change(test)
    res = denormalize_change(ref, data)
    
    print(test)
    print(np.round(data, 2))
    print(res)


if __name__ == '__main__':
    esn_train_normalized()
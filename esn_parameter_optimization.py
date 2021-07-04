from tools import *
import time
import numpy as np

data_file_name = 'Datasets/Correct/usage_smoothed_100.csv'


def esn_parameter_optimization(past_window_size, future_window_size = 2):
    
    Sorted = []
    hyper = {
        'resevoir_size': [500, 1000, 1500, 2000, 3000],
        'sparsity': [0.005, 0.01, 0.05, 0.1],
        'rand_seed': 20,
        'spectral_radius': [0.8, 0.9, 1, 1.1, 1.2, 1.5],
        'noise': 0.001
    }
    n_options = len(hyper['resevoir_size']) * len(hyper['sparsity']) * len(hyper['spectral_radius'])

    data = np.genfromtxt('Datasets/Correct/usage_smoothed_100.csv', skip_header=0, usecols=(0), delimiter=',')
    # data = np.genfromtxt('Datasets/Correct/usage.csv', skip_header=1, usecols=(1), delimiter=',')
    data_length = data.size
    
    print_format = '{:<12}' * 3
    
    print(print_format.format('Time', 'MSE', 'Reservoir size, sparsity, spectral radius'))
    for option in range(n_options):
        # Get parameter indices based on the option counter
        par1 = int(option % len(hyper['resevoir_size']))
        par2 = int((option / len(hyper['resevoir_size'])) % len(hyper['sparsity']))
        par3 = int((option / (len(hyper['resevoir_size']) * len(hyper['sparsity']))) % len(hyper['spectral_radius']))
        
        # Generate the modell with the current parameters
        modell = make_modell_2(hyper['resevoir_size'][par1], hyper['sparsity'][par2], hyper['spectral_radius'][par3])
        
        train_window = data[ : past_window_size + future_window_size * 2]

        time_start = time.time()
        train_outputs = modell.fit(train_window[ : past_window_size], train_window[future_window_size : past_window_size + future_window_size])
        duration = time.time() - time_start

        predict_output = modell.predict(train_window[ future_window_size : past_window_size + future_window_size ])
        mse = MSE(predict_output.reshape(predict_output.shape[0]), train_window[ 2 * future_window_size : ])
        
        Sorted.append( (mse, (hyper['resevoir_size'][par1], hyper['sparsity'][par2], hyper['spectral_radius'][par3])) )
        
        print(print_format.format(
            round(duration, 4), 
            round(mse, 4),
            '{} - {} - {}'.format(hyper['resevoir_size'][par1], hyper['sparsity'][par2], hyper['spectral_radius'][par3])
            ))
    Sorted.sort()
    print()
    print(*Sorted[:10], sep='\n')


if __name__ == '__main__':
    esn_parameter_optimization(1000)
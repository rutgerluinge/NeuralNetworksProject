from ESN import ESN
from save_esn import save_esn

from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import gc

import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))


'''
-Training data
-Test data
-- Or the split between them
-Data in sizes of days?

- Log the actual outputs? right?
'''

'''
data - the data sepperated in wanted lengths
input-length - the sequence the ESN gets without expecting any results
'''
def train_esn(ESN, data, input_length):
    # Run network
    data_points = len(data)
    state_matrix = [None] * (data_points - input_length)
    for i in range(data_points):
        if i < input_length:
            ESN.process_input(data[i])
        else:
            ESN.output = data[i - 1]
            ESN.process_input(0)
            state_matrix[i-input_length] = ESN.reservoir
    state_matrix = np.array(state_matrix)
    ESN.Wout = get_weights(state_matrix, data[input_length:])


def get_weights(state_matrix, teacher):
    ridge = Ridge(alpha=0.2)
    ridge.fit(state_matrix, teacher)
    return ridge.coef_


def learn_main():
    df = pd.read_csv(BASEDIR + '/../datasets/processed/usage_centered.csv')
    data = df['usage'][:100].copy()
    del df
    gc.collect()
    esn = ESN(1, 1000, 1)
    train_esn(esn, data, 50)
    save_esn(esn, './esn.txt')
    
    esn.reservoir = [0.0 for i in range(esn.reservoir_size)]
    for i in range (len(data)):
        if i < 50:
            esn.process_input(data[i])
        else:
            esn.process_input(0)
            if i % 10 == 0:
                print(esn.get_output(), data[i])


if __name__ == '__main__':
    learn_main()
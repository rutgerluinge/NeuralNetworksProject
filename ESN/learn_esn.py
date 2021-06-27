from ESN import ESN
from save_esn import save_esn

from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

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

#Input: an Echo State Network, the training data, and the input_length that is disregarded for training purposes
#Puts the training data in the generated ESN
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
    print(np.array(ESN.reservoir).shape)
    state_matrix = np.array(state_matrix)
    print(state_matrix.shape)
    ESN.Wout = get_weights(state_matrix, data[input_length:])
    

#Input: reservoir states recieved from the ESN and training data, The desired output
#Output: The fitted weights for the output vector
#alpha is the regularization strength used in regression
def get_weights(state_matrix, teacher):
    ridge = Ridge(alpha=1.0)
    ridge.fit(state_matrix, teacher)
    return ridge.coef_


def learn_main():
    df = pd.read_csv(BASEDIR + '/../datasets/processed/filled_gabs.csv')
    data = df['usage'][:10]
    esn = ESN(1, 500, 1)
    train_esn(esn, data, 2)
    save_esn(esn, './esn.txt')
    #print(esn.Wout)


if __name__ == '__main__':
    learn_main()

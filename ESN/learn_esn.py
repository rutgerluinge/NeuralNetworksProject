from ESN import ESN
from save_esn import save_esn

from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
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
def train_esn(ESN, data, input_length, alpha = 1.0):
    # Run network
    data_points = len(data)
    state_matrix = [None] * (data_points - input_length -1)
    #state_matrix = [None] * data_points
    for i in range(data_points-1): # - 1
        if i < input_length:
            ESN.process_training_input(data[i])
        else:
            ESN.process_training_input(data[i])
            state_matrix[i-input_length] = ESN.reservoir
    state_matrix = np.array(state_matrix)
    ESN.Wout = get_weights(state_matrix, data[input_length+1:], alpha) #data[input_length+1:]
    

#Input: reservoir states recieved from the ESN and training data, The desired output
#Output: The fitted weights for the output vector
#alpha is the regularization strength used in regression
def get_weights(state_matrix, teacher, alph):
    ridge = Ridge(alpha=alph)
    ridge.fit(state_matrix, teacher)
    return ridge.coef_


def learn_main():
    df = pd.read_csv(BASEDIR + '/../datasets/processed/processed_full.csv')
    data = df['usage'][:5000].copy()
    data2 = df['usage'][:10000].copy()
    data3 = df['usage'][:15000].copy()
    del df
    Win_scalar = float(input("Win scalar:"))
    W_scalar = float(input("W scalar:"))
    bias_scalar = float(input("Bias scalar:"))
    gc.collect()
    esn = ESN(1, 4000, 1,leaking_rate=1,Wscalar=W_scalar,WinScalar=Win_scalar,Bscalar=bias_scalar)
    train_esn(esn, data, 2000)
    output = []
    esn.reservoir = [0.0 for i in range(esn.reservoir_size)]


    for i in range (len(data3)-1):
        if i< len(data):
            output.append(esn.get_output(data[i]))
        elif i<len(data2):
            output.append(esn.get_output(data2[i]))
        else:
            output.append(esn.get_output(esn.output))
    print("done")
    output2 = []
    # for i in range(len(data2)-1):
    #     output.append(esn.get_output(data2[i]))

    plt.plot(data3), plt.plot(output)
    plt.show()

if __name__ == '__main__':
    learn_main()

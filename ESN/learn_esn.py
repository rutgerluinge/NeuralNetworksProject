from ESN import ESN
from save_esn import save_esn

from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))


# Input: an Echo State Network, the training data, and the input_length that is disregarded for training purposes
# Puts the training data in the generated ESN
def train_esn(ESN, data, input_length, alpha):
    """Main function that is used for training our ESN"""
    data_points = len(data)
    state_matrix = [None] * (data_points - input_length - 1)
    for i in range(data_points - 1):  # - 1
        if i < input_length:
            ESN.process_training_input(data[i])
        else:
            ESN.process_training_input(data[i])
            state_matrix[i - input_length] = ESN.reservoir
    state_matrix = np.array(state_matrix)
    ESN.Wout = get_weights(state_matrix, data[input_length + 1:], alpha)  # data[input_length+1:]


# Input: reservoir states recieved from the ESN and training data, The desired output
# Output: The fitted weights for the output vector
# alpha is the regularization strength used in regression
def get_weights(state_matrix, teacher, alph):
    """Function that uses ridge regression to generate the fitted weights for Wout based on the reservoir state after training"""
    ridge = Ridge(alpha=alph)
    ridge.fit(state_matrix, teacher)
    return ridge.coef_


def learn_main():
    df = pd.read_csv(BASEDIR + '/../datasets/processed/processed.csv')

    """Distribution of Data (can be changed) """
    train_data = df['usage'][:4200].copy()  # data used for training
    test_data = df['usage'][:4200].copy()  # data used to check if new data is working
    all_data = df['usage'][:5200].copy()  # data used for independent predictions

    """User Input for hyperparameters"""
    Win_Scalar = float(input('Enter W-in scalar e.g. (1): '))
    W_Scalar = float(input('Enter W scalar e.g. (1):'))
    size = int(input('Enter Neuron Size e.g. (3000): '))
    leakingRate = float(input('Enter leaking rate e.g. (0.8): '))
    alpha = float(input('Enter alpha (Ridge Regression) e.g. (1): '))

    del df
    gc.collect()

    """Creating the Echo State Network and training"""
    esn = ESN(1, size, 1, leaking_rate=leakingRate, Wscalar=W_Scalar, WinScalar=Win_Scalar)
    train_esn(esn, test_data, 2000, alpha)

    output = []
    esn.reservoir = [0.0 for i in range(esn.reservoir_size)]

    for i in range(len(all_data) - 1):
        if i < len(train_data):
            """Generate a prediction with data that has been trained with"""
            output.append(esn.get_output(train_data[i]))
        elif i < len(test_data):
            """Prediction with untrained data"""
            output.append(esn.get_output(test_data[i]))
        else:
            """Use last output as new input"""
            output.append(esn.get_output(esn.output))

    plot(all_data, output, 4200)


def plot(data, trained, cutoff):
    """Plotting and drawing"""
    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(data), plt.plot(trained)
    plt.vlines(cutoff, -1, 1, colors="red", linestyles="dotted", label="cutoff")

    """"Labels"""
    plt.title("Energy Usage Prediction")
    plt.legend(["Data", "Predicted", "Independent training"])
    plt.xlabel("Data samples (10 min/sample)")
    plt.ylabel("Normalized Energy Consumption")

    """"Other"""
    plt.axis([3000, 5000, -1, 1])  # Zoom
    plt.savefig("EnergyPrecition.png")
    plt.show()


if __name__ == '__main__':
    learn_main()

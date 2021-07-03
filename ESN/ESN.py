import numpy as np
from numpy import linalg as LA
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import random
import math

# We need sigmoid functions
# Some ridge regression
# Randomization of weights
# Process the input
# Process the output

connectivity = 2  # in percentage
decimals = 3  # weight decimals
SD = 0.3

# scalars of order (according to lecture notes herbert)
small = 0.1
medium = 1.0
large = 10.0

bias_scale_in = 0.5
bias_scale_res = 0.5
bias_scale_fb = 0.5


class ESN:
    # Initialize the ESN
    def __init__(self, input_size, reservoir_size, output_size, leaking_rate=1, Wscalar=1, WinScalar=1,Bscalar=1):
        # Set the different sizes
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leaking_rate = leaking_rate
        # Instantiate the matrixes to all 0 values

        self.W = [[0.0 for i in range(reservoir_size)] for j in range(reservoir_size)]
        self.Win = [[0.0 for i in range(input_size)] for j in range(reservoir_size)]
        self.Wout = [[0.0 for i in range(reservoir_size)] for j in range(output_size)]

        self.reservoir = [0.0 for i in range(reservoir_size)]

        self.input_bias = np.zeros(reservoir_size)
        self.output = np.zeros(output_size)

        self.Wsc = Wscalar
        self.Winsc = WinScalar
        self.Bssc = Bscalar

        self.init_W()
        self.init_Win()
        self.init_bias()

    # Formula 18 in practicalESN.pdf with an added term for bias
    # calculates the update vector of reservoir neuron activations

    def process_training_input(self, input):
        result = np.tanh((self.Win.dot(input).reshape(self.reservoir_size, )
                          + self.W.dot(self.reservoir).reshape(self.reservoir_size, )
                          #+ np.asarray(self.input_bias).reshape(self.reservoir_size, )
                          ))
        # print(result.shape)
        self.reservoir = self.leaking(result).reshape(self.reservoir_size, )

    # formula 7 in practicalESN.pdf
    # combines the reservoir activation with the readout weights to produce an output
    def get_output(self, input):
        result = np.tanh((self.Win.dot(input).reshape(self.reservoir_size, )
                          + self.W.dot(self.reservoir).reshape(self.reservoir_size, )
                          #+ np.asarray(self.input_bias).reshape(self.reservoir_size, )
                          ))

        self.reservoir = self.leaking(result)
        self.output = self.Wout.dot(self.reservoir)
        return self.output

    def give_signal(self, input):
        result = np.tanh((self.Win.dot(input).reshape(self.reservoir_size, )
                          + self.W.dot(self.reservoir).reshape(self.reservoir_size, )
                          + np.asarray(self.input_bias).reshape(self.reservoir_size, )))
        self.reservoir = self.leaking(result)
        self.output = self.Wout.dot(self.reservoir)

    # Generates the reservoir matrix with appropriate size, connectivity and spectral radius
    # the initial values are randomly generated from a gaussian (normal) distribution
    # based on 3.2.2 to 3.2.4 from practicalESN.pdf
    def init_W(self):
        for i in range(self.reservoir_size):
            # init Connections and values W matrix:
            for j in range(self.reservoir_size):
                if random.randint(1, 100) <= connectivity:  # connectivity is set to 1 (0.01 or 1 percent)
                    self.W[i][j] =round(np.random.normal(0, SD),
                                                  decimals)  # gaussian distribution, first digit is mean, 2nd standard deviation (not sure bout that)

        spectralRad = np.max(np.absolute(np.linalg.eigvals(self.W)))

        if spectralRad == 0:
            print("!!!ERROR SPECTRAL RADIUS = 0, MIGHT CONSIDER BIGGER RESERVOIR SIZE!!!")
        else:
            self.W = self.W / spectralRad

        self.W = np.array(self.W) * self.Wsc

    # generates the input matrix (or vector in our case) with appropriate size
    # based on 3.2.5 from practicalESN.pdf
    def init_Win(self):

        for i in range(self.reservoir_size):
            # init Win
            for j in range(self.input_size):
                self.Win[i][j] = (np.random.uniform(-1.0, 1.0, None))  # uniformly distributed

        self.Win = np.array(self.Win) * self.Winsc
        #print(self.Win)
    # Input: ESN, Reservoir state update vector
    # output: reservoir state vector
    # Formula 3 in practicalESN.pdf
    def leaking(self, x_):  # x = reservoir state update vector
        return (1 - self.leaking_rate) * x_ + self.leaking_rate * x_


    # print the reservoir
    def printW(self):
        for i in range(self.reservoir_size):
            print(self.W[i])

    def init_bias(self):
        for i in range(self.reservoir_size):
            self.input_bias[i] = small * np.random.uniform(-0.5, 0.5, None)
        self.input_bias * self.Bssc

# select a file to process and create an ESN
# currently unused
def ESN_main():
    Tk().withdraw()
    filename = askopenfilename()
    data = pd.read_csv(filename)

    esn = ESN(1, 100, 1)  # predict 1 timestamp based on the 4 previous ones? reservoir size = 1000 (might need more)


if __name__ == '__main__':
    ESN_main()

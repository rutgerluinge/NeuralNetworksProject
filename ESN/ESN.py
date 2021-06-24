import numpy as np
from numpy import linalg as LA
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import random
import scipy

# We need sigmoid functions
# Some ridge regression
# Randomization of weights
# Process the input
# Process the output

connectivity = 1  # percent
decimals = 3  # weight decimals


class ESN:
    # Initialize the ESN
    def __init__(self, input_size, reservoir_size, output_size, feedback):
        # Set the different sizes
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # Instantiate the matrixes to all 0 values
        self.reservoir = [0] * reservoir_size
        self.W = [[0 for i in range(reservoir_size)] for j in range(reservoir_size)]
        self.Win = [[0] * input_size] * reservoir_size
        self.Wfb = [[0] * output_size] * reservoir_size
        self.bias = [0] * reservoir_size
        self.Wout = [[0] * reservoir_size] * output_size
        self.output = [0] * output_size

        self.init_W()
        self.init_Win()

    # This could be a uniform distribution or a bell curve around 0? Could implement the eigenvalue thing of 1/ev.
    # ja, een van deze opties: symmetrical uniform, discrete bi-valued, or normal distribution
    # centered around zero, Gaussian distributions (populair) en uniform distribution (populair)
    # around 0!!
    def randomize_weights(self, randomization):
        pass

    def process_input(self, input):
        '''Function would look something like this:
            self.reservoir = sigmoid(self.Win*input + self.W*self.reservoir + self.Wfb*self.output + self.bias)'''
        pass

    def get_output(self):
        # does this need the linear regression?
        self.output = self.Wout * self.reservoir
        return self.output

    def init_W(self):
        # ja, een van deze opties: symmetrical uniform, discrete bi-valued, or normal distribution
        # centered around zero, Gaussian distributions (populair) en uniform distribution (populair)
        # connectivity 1 procent -> 10 connections per neuron

        for i in range(len(self.W)):
            for j in range(len(self.W)):
                if random.randint(1, 99) <= connectivity:  # connectivity is set to 1 (0.01 or 1 percent)
                    self.W[i][j] = round(random.gauss(0, 0.3),
                                         decimals)  # gaussian distribution, first digit is mean, 2nd standard deviation (not sure bout that)

    def init_Win(self):
        eigenvalues = LA.eigh(self.W)
        print(eigenvalues)




    def init_bias(self):
        pass

    def printW(self):
        for i in range(self.reservoir_size):
            print(self.W[i])


def ESN_main():
    Tk().withdraw()
    filename = askopenfilename()
    data = pd.read_csv(filename)

    esn = ESN(4, 1000, 1,
              None)  # predict 1 timestamp based on the 4 previous ones? reservoir size = 1000 (might need more)


if __name__ == '__main__':
    ESN_main()

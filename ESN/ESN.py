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

connectivity = 1  # percent
decimals = 3  # weight decimals
input_scaling = 1
SD = 0.3

W_Scalar = 1 # voor s/m/l later misschien?
Win_Scalar = 1 # stond aangegeven in document dat handig zou zijn
Wfb_Scalar = 1

class ESN:
    # Initialize the ESN
    def __init__(self, input_size, reservoir_size, output_size, leaking_rate = 1):
        # Set the different sizes
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leaking_rate = leaking_rate

        # Instantiate the matrixes to all 0 values
        self.reservoir = [0.0 for i in range(reservoir_size)]
        self.W = [[0.0 for i in range(reservoir_size)] for j in range(reservoir_size)]
        self.Win = [[0.0 for i in range(input_size)] for j in range(reservoir_size)]
        self.Wfb = [[0] * output_size] * reservoir_size
        self.bias = [0] * reservoir_size
        self.Wout = [[0] * reservoir_size] * output_size
        self.output = [0] * output_size

        self.init_W()
        self.init_Win()
        self.init_Wfb()



    #Formula 18 in practicalESN.pdf with an added term for bias, includes teacher forcing
    #calculates the update vector of reservoir neuron activations and applies it to the reservoir state
    def process_training_input(self, input):
        '''Function would look something like this:
            self.reservoir = sigmoid(self.Win*input + self.W*self.reservoir + self.Wfb*self.output + self.bias)'''
        result = np.tanh(np.add(np.add(np.add(self.Win.dot(input), self.W.dot(self.reservoir)), self.Wfb.dot(input)), self.bias))[:][0]
        print(result.shape)
        self.reservoir = self.leaking(result)


    #formula 7 in practicalESN.pdf
    #combines the reservoir activation with the readout weights to produce an output
    #CURRENTLY UNUSED
    def get_output(self):
        # does this need the linear regression?
        self.output = self.Wout.dot(self.reservoir)
        return self.output


    #Generates the reservoir matrix with appropriate size, connectivity and spectral radius
    #the initial values are randomly generated from a gaussian (normal) distribution
    #based on 3.2.2 to 3.2.4 from practicalESN.pdf
    def init_W(self):
        for i in range(self.reservoir_size):
            for j in range(self.reservoir_size):
                if random.randint(1, 100) <= connectivity:   #connectivity is set to 1 (0.01 or 1 percent)
                    self.W[i][j] = W_Scalar * round(random.gauss(0, SD),
                                         decimals)  # gaussian distribution, first digit is mean, 2nd standard deviation (not sure bout that)

        
        spectralRad = np.max(np.absolute(np.linalg.eigvals(self.W)))
        if spectralRad > 1:
            print("!!!ERROR SPECTRAL RADIUS > 1!!!")

            self.W = self.W / spectralRad
        elif spectralRad == 0:
            print("!!!ERROR SPECTRAL RADIUS = 1, MIGHT CONSIDER BIGGER RESERVOIR SIZE!!!")
        else:
            self.W = self.W / spectralRad
        self.W = np.array(self.W)


    #generates the input matrix (or vector in our case) with appropriate size
    #based on 3.2.5 from practicalESN.pdf
    def init_Win(self):
        for i in range(self.reservoir_size):
            for j in range(0,(self.input_size)):
                self.Win[i][j] = float(Win_Scalar * (np.random.normal(0, SD, None)))
                # normal distribution mean 0, SE = 0.3, niet zeker over tanh, stond in document iets over
                # Win_scaler is defined boven in dit script, (global parameter, zoals in document (wat we kunnen veranderen))
        self.Win = np.array(self.Win)


    #Input: ESN, Reservoir state update vector
    #output: reservoir state vector
    #Formula 3 in practicalESN.pdf
    def leaking(self, x_):  #x = reservoir state update vector
        return (1-self.leaking_rate)*x_ + self.leaking_rate*x_
    

    #initializes the bias vector
    #CURRENTLY UNUSED
    def init_bias(self):
        pass

    # Voor nu even gecopied van Win om te testen
    def init_Wfb(self):
        for i in range(self.reservoir_size):
            for j in range(1,(self.output_size)):
                self.Wfb[i][j] = float(Wfb_Scalar * (np.random.normal(0, SD, None)))
    #initializes the feedback matrix
    def init_Wfb(self):
        self.Wfb = np.array(self.Wfb)
        
    #print the reservoir
    def printW(self):
        for i in range(self.reservoir_size):
            print(self.W[i])

#select a file to process and create an ESN
#currently unused
def ESN_main():
    Tk().withdraw()
    filename = askopenfilename()
    data = pd.read_csv(filename)

    esn = ESN(1, 100, 1)  # predict 1 timestamp based on the 4 previous ones? reservoir size = 1000 (might need more)


if __name__ == '__main__':
    ESN_main()

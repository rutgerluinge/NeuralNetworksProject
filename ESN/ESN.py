import numpy as np
import random


class ESN:
    """"The Class of our designed Echo State Network that stores all the information about the weights and the reservoir"""

    def __init__(self, input_size, reservoir_size, output_size, leaking_rate, Wscalar, WinScalar, BiasScalar=0,
                 connectivity=1):
        """Final/static variables"""
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leaking_rate = leaking_rate
        self.connectivity = connectivity

        """First initialize all weight and reservoir matrices to 0"""
        self.W = [[0.0 for i in range(reservoir_size)] for j in range(reservoir_size)]
        self.Win = [[0.0 for i in range(input_size)] for j in range(reservoir_size)]
        self.Wout = [[0.0 for i in range(reservoir_size)] for j in range(output_size)]
        self.reservoir = np.zeros(reservoir_size)
        self.input_bias = np.zeros(reservoir_size)
        self.output = np.zeros(output_size)

        """Scalars for all the parameters"""
        self.W_scalar = Wscalar
        self.Win_scalar = WinScalar
        self.Bias_scalar = BiasScalar

        """Start initializing the matrices"""
        self.init_W()
        self.init_Win()
        self.init_bias()


    def process_training_input(self, input):
        """Train with the data, use the update equations to update the reservoir neuron activations"""
        result = self.get_Result(input)
        self.reservoir = self.leaking(result).reshape(self.reservoir_size, )


    def get_output(self, input):
        """"Formula 7 of practicalESN.pdf, combines the reservoir with the readout weight to produce a prediction output"""
        result = self.get_Result(input)
        self.reservoir = self.leaking(result)
        self.output = np.dot(self.Wout, self.reservoir)
        return self.output

    def get_Result(self, input):
        """Formula 2 of practicalESN.pdf, update equation"""
        return np.tanh((self.Win.dot(input).reshape(self.reservoir_size, )
                        + self.W.dot(self.reservoir).reshape(self.reservoir_size, )))


    def init_W(self):
        """Generate the weight matrix with normal distribution, not all neurons are connected thus we use the connectivity variable,
        3.2.2 to 3.2.4 from practicalESN.pdf"""
        for i in range(self.reservoir_size):
            for j in range(self.reservoir_size):
                if random.randint(1, 100) <= self.connectivity:
                    self.W[i][j] = np.random.normal(0, 0.3)

        """Use the spectral radius to ensure echo state property, abort if this is 0 (empty matrix)"""
        spectralRadius = np.max(np.absolute(np.linalg.eigvals(self.W)))

        if spectralRadius == 0:
            print("!!!ERROR SPECTRAL RADIUS = 0, MIGHT CONSIDER BIGGER RESERVOIR SIZE!!!")
            print("Aborting...")
            exit()
        else:
            self.W = self.W / spectralRadius

        self.W = np.array(self.W) * self.W_scalar

    def init_Win(self):
        """Generate input matrix with uniform distribution between -1 and 1, 3.2.5 from practicalESN.pdf"""
        for i in range(self.reservoir_size):
            for j in range(self.input_size):
                self.Win[i][j] = (np.random.uniform(-1.0, 1.0, None))  # uniformly distributed

        self.Win = np.array(self.Win) * self.Win_scalar

    def leaking(self, x_):  # x = reservoir state update vector
        """Leaking of the neurons (update equation), formula 3 in practical.ESN"""
        return (1 - self.leaking_rate) * x_ + self.leaking_rate * x_

    def init_bias(self):
        """Function for defining the bias, we found that using a bias wasn't helping to reach our goals, so we decided
            to leave it out."""
        for i in range(self.reservoir_size):
            self.input_bias[i] = np.random.uniform(-0.5, 0.5, None)

        self.input_bias = np.array(self.input_bias) * self.Bias_scalar

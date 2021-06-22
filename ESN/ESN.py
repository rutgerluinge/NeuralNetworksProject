import numpy as np

# We need sigmoid functions
# Some ridge regression
# Randomization of weights
# Process the input
# Process the output


class ESN:
    # Initialize the ESN
    def __init__(self, input_size, reservoir_size, output_size, feedback):
        # Set the different sizes
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # Instantiate the matrixes to all 0 values
        self.reservoir = [0] * reservoir_size
        self.W = [[0] * reservoir_size] * reservoir_size
        self.Win = [[0] * input_size] * reservoir_size
        self.Wfb = [[0] * output_size] * reservoir_size
        self.bias = [0] * reservoir_size
        self.Wout = [[0] * reservoir_size] * output_size
        self.output = [0] * output_size

    # This could be a uniform distribution or a bell curve around 0? Could implement the eigenvalue thing of 1/ev.
    def randomize_weights(self, randomization):
        pass

    def process_input(self, input):
        '''Function would look something like this:
            self.reservoir = sigmoid(self.Win*input + self.W*self.reservoir + self.Wfb*self.output + self.bias)'''
        pass

    def get_output(self):
        self.output = self.Wout * self.reservoir
        return self.output
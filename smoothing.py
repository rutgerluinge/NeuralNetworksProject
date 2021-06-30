import numpy as np
from tools import moving_average
window_size = 100
original = np.genfromtxt('Datasets/Correct/usage.csv', skip_header=1, usecols=(1), delimiter=',')
smoothed = moving_average(original, (window_size, window_size))[window_size : -window_size]

np.savetxt('Datasets/Correct/usage_smoothed_100.csv', smoothed, delimiter=',')
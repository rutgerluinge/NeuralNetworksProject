import matplotlib.pyplot as plt
import numpy as np


def plotGraph(y):
    x_axis = np.linspace(0, len(y), len(y))

    plt.plot(x_axis, y, label="trial")
    plt.show()

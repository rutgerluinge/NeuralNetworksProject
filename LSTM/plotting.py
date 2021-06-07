import matplotlib.pyplot as plt
import numpy as np

def plotGraph(y1, y2):
    x_axis = np.linspace(0, len(y1), len(y1))

    plt.plot(x_axis, y1, y2, label="trial")
    plt.show()

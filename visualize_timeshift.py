import numpy as np
import matplotlib.pyplot as plt
import datetime

data_file_name = 'Datasets/Correct/usage_2021-05-17_2021-05-24.csv'

data_set = np.genfromtxt(
    data_file_name, 
    skip_header=1, 
    usecols=(0,1), 
    converters={0: lambda s: datetime.datetime.fromisoformat(s.decode('utf-8')), 1: lambda s: float(s)}, 
    delimiter=','
    # dtype=[None, np.float]
    )
shiftery = []
for i in range(1, data_set.size):
    t1 = data_set[i][0]
    t2 = data_set[i - 1][0]
    shift_time = t1 - t2
    shift_time = int((shift_time.days * 86400000) + (shift_time.seconds * 1000) + (shift_time.microseconds / 1000))
    if(shift_time > 100000):
        val1 = data_set[i-1][1]
        val2 = data_set[i][1]
        breakpoint
    shiftery.append(shift_time)
plt.figure()
plt.plot(
    shiftery, 
    label='data'
    )
plt.show()
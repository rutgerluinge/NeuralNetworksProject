import numpy as np
import matplotlib.pyplot as plt
import datetime

data_file_name = 'Traindata/usage_05-04-2021_12-04-2021.csv'

data_set = np.genfromtxt(
    data_file_name, 
    skip_header=1, 
    usecols=(1), 
    # converters={0: lambda s: datetime.datetime.fromisoformat(s)}, 
    delimiter=','
    # dtype=(datetime.datetime, np.float)
    )

plt.figure()
plt.plot(
    data_set, 
    label='data'
    )
plt.show()
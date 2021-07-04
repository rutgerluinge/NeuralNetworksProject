# Neural Networks Project

### Introduction
Welcome to our neural networks project.
As described in our paper, we decided to build an Echo State Network
(ESN). The code we used for our project can be found in this GitHub
repository, this will be described below.

The task we set for our project was to create an ESN that can predict
the energy consumption of a household based on data gathered from a smart meter.
We used some libraries to test ESN's in general and decided to also create our own
ESN engine to really dive into the inner workings.


### The data

>Note: This section can be skipped if you are not interested in our data files.

The data can be found in the 'datasets' directory. The data we used to train the network
can then be found in the directory 'processed'. This folder contains multiple 'csv' files
in which our original raw data is processed in different ways.

The smart meter measures every ten seconds. The raw data thus consisted of
data with gabs of ten seconds or multiples of that. The smart meter did not always function and at some
points there were gabs of hours/

- filled_gabs.csv - Removed gabs from the data.
  Small gabs are filled with the average of both sides, for large gabs (multiple hours) that
  day is removed.
  
- processed.csv - All points now represent ten minutes instead of seconds. The data
is normalized and centered around the zero line.
  
- removed_gabs.csv - Removed gabs from data, small gabs not yet filled, this was saved
for later use in filled_gabs.csv
  

### The data preprocessing

For data processing we used Jupyter notebook files (extension: '.ipynb').
These can be found in the 'dataprocessing' directory.
These files are not specifically clean and are mainly used to create the datafiles
mentioned above.


### The Echo State Network code

#### Code structure

The code for our own Echo State Network (ESN) can be found in the 'ESN' directory.

The code for the network and its internal functioning can be found in 'ESN.py'. This includes
its initialization, passing it input and getting output.

The training functions are in 'learn_esn.py'. For Ridge regression we used the package scikit-learn.

To find the optimal hyper-parameters we used the 'optimize_esn.py' to loop through possible combinations.

The code is further commented to provide clarity.

#### Running the network

To run the network code, the file 'learn_esn.py' can be run. The program requires user input to set the hyper-parameters
of the network. It will then train an ESN on the 'processed.csv' data and give a plot with the results.





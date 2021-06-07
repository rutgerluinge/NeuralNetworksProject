from Tools.scripts.findnocoding import usage

from numpy import array
import numpy as np
import csv
from plotting import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# following: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/?source=post_page---------------------------, for now to see
# if it fits our research
n_steps = 3  # amount of datapoints needed to train
n_features = 1
model = Sequential()


def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


def train_LSTM(raw_seq):
    x, y = split_sequence(raw_seq, n_steps)
    x = x.reshape((x.shape[0], x.shape[1], n_features))

    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=200, verbose=0)


def predict(vector):
    predict = []
    for i in range(len(vector)-2):
        input = array([vector[i], vector[i+1], vector[i+2]])
        input = input.reshape((1, n_steps, n_features))
        value = model.predict(input,verbose=0)
        predict.append(int(value))
    return predict


def main():

    input_vector = []
    with open('usage1.csv') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index > 1000:    #deside how much data you want (my pc takes ages)
                break
            input_vector.append(int(row[1]))

    """create different datasets for cross-validation"""
    cutoff = int(0.8 * len(input_vector))  # 80 percent cross-validation
    traindata = input_vector[0:cutoff]
    testdata = input_vector[cutoff:]

    """train the model with traindata"""
    train_LSTM(traindata)

    """make a prediction and format"""
    prediction = predict(testdata)
    plotarray = [0] * (len(traindata) + n_steps)  #n a
    plotarray = np.concatenate((plotarray,prediction))



    """plotting"""
    print(len(input_vector))
    print(len(plotarray))
    plotGraph(input_vector, plotarray)


if __name__ == '__main__':
    main()

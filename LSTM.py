from Tools.scripts.findnocoding import usage
from plot import *
from numpy import array
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# following: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/?source=post_page---------------------------, for now to see
# if it fits our research
n_steps = 3  # amount of datapoints needed to train
n_features = 1
#model = Sequential()


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
    LSTM(50, activation='relu', input_shape=(n_steps, n_features))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=200, verbose=0)


def predict(vector):
    input = array(vector)
    input = input.reshape((1, n_steps, n_features))
    prediction = model.predict(input, verbose=0)
    print(prediction)


def main():
    input_vector = []
    with open('usage1.csv') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index > 100:  # use 100 data points for now
                break
            input_vector.append(row[1])

    cutoff = int(0.8 * len(input_vector))  # 80 percent
    traindata = input_vector[0:cutoff]
    testdata = input_vector[cutoff:]

    #train_LSTM(input_vector)
    #predict([873, 500, 721])

    plotGraph(traindata)
    # plotGraph(input_vector)


if __name__ == '__main__':
    main()

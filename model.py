from csvToDataset import CSVToDataset, history_points
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, LSTM, Input, Activation ,concatenate
from keras import optimizers
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.random.seed(15)
tf.random.set_seed(15)

class Model:

    def __init__(self, path) -> None:
        csvToDataset = CSVToDataset()
        self.ohlcv_histories, self.technical_indicators, self.next_day_open_values, self.unscaled_y, self.y_normalizer = csvToDataset.csv_to_dataset(path)
        self._trainTestSet(0.9)

    def _trainTestSet(self, ratio) -> None:
        train_split = ratio
        n = int(self.ohlcv_histories.shape[0] * train_split)

        self.ohlcv_train = self.ohlcv_histories[:n]
        self.tech_ind_train = self.technical_indicators[:n]
        self.y_train = self.next_day_open_values[:n]

        self.ohlcv_test = self.ohlcv_histories[n:]
        self.tech_ind_test = self.technical_indicators[n:]
        self.y_test = self.next_day_open_values[n:]

        self.unscaled_y_test = self.unscaled_y[n:]

    def _createInputs(self):
        lstm_input = Input(shape=(history_points, 5), name='lstm_input')
        dense_input = Input(shape=(self.technical_indicators.shape[1], ), name='tech_input')
        return lstm_input, dense_input

    def _createLSTMBranch(self, lstm_input):
        x = LSTM(50, name='lstm_0')(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        lstm_branch = keras.models.Model(inputs=lstm_input, outputs=x)
        return lstm_branch

    def _createDenseBranch(self, dense_input):
        y = Dense(20, name='tech_dense_0')(dense_input)
        y = Activation('relu', name='tech_relu_0')(y)
        y = Dropout(0.2, name='tech_dropout_0')(y)
        technical_indicators_branch = keras.models.Model(inputs=dense_input, outputs=y)
        return technical_indicators_branch
    
    def _combineBranch(self, lstm_branch, technical_indicators_branch):
        combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
        z = Dense(64, activation='sigmoid', name='dense_pooling')(combined)
        z = Dense(1, activation='linear', name='dense_output')(z)
        return z
        # takes both inputs, outputs single value
        # self.model = Model(inputs=[self.lstm_branch.input, self.technical_indicators_branch.input], outputs=z)

    def makeModel(self):
        lstm_input, dense_input = self._createInputs()
        lstm_branch = self._createLSTMBranch(lstm_input)
        technical_indicators_branch = self._createDenseBranch(dense_input)
        z = self._combineBranch(lstm_branch, technical_indicators_branch)

        # takes both inputs, outputs single value
        model = keras.models.Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

        plot_model(model, to_file='assets/model.png', show_shapes=True)

        adam = optimizers.Adam(learning_rate=0.0005)

        model.compile(optimizer=adam, loss='mse')
        model.fit(x=[self.ohlcv_train, self.tech_ind_train], y=self.y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
        return model

    def evaluate(self, model):
        y_test_predicted = model.predict([self.ohlcv_test, self.tech_ind_test])
        y_test_predicted = self.y_normalizer.inverse_transform(y_test_predicted)
        y_predicted = model.predict([self.ohlcv_histories, self.technical_indicators])
        y_predicted = self.y_normalizer.inverse_transform(y_predicted)

        assert self.unscaled_y_test.shape == y_test_predicted.shape

        real_mse = np.mean(np.square(self.unscaled_y_test - y_test_predicted))
        scaled_mse = real_mse / (np.max(self.unscaled_y_test) - np.min(self.unscaled_y_test)) * 100
        print(scaled_mse)
        return y_test_predicted, y_predicted, real_mse, scaled_mse

    def plotModel(self, y_predicted):
        plt.gcf().set_size_inches(22, 15, forward=True)
        start = 0
        end = -1

        real = plt.plot(self.unscaled_y[start:end], label='real')
        pred = plt.plot(y_predicted[start:end], label='predicted')

        plt.legend(['Real', 'Predicted'])
        plt.savefig(fname='assets/model_accuracy.png', )
        plt.show()

    def save(self, model):
        model.save(f'model.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="The path of the csv file")
    path = parser.parse_args().path
    modelObject = Model(path)
    model = modelObject.makeModel()
    y_test_predicted, y_predicted, real_mse, scaled_mse = modelObject.evaluate(model)
    modelObject.plotModel(y_predicted)
    modelObject.save(model)
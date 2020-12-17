import numpy as np
from keras.models import load_model
from csvToDataset import CSVToDataset
import matplotlib.pyplot as plt
import argparse

class Trading:

    def __init__(self, modeL, path) -> None:
        self.model = model
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
        
    def predict(self):
        y_test_predicted = self.model.predict([self.ohlcv_test, self.tech_ind_test])
        y_test_predicted = self.y_normalizer.inverse_transform(y_test_predicted)
        return y_test_predicted

    def getBuysAndSells(self, threshold):
        buys = []
        sells = []

        start = 0
        end = -1

        x = -1
        for ohlcv, ind in zip(self.ohlcv_test[start: end], self.tech_ind_test[start: end]):
            normalized_price_today = ohlcv[-1][0]
            normalized_price_today = np.array([[normalized_price_today]])
            price_today = self.y_normalizer.inverse_transform(normalized_price_today)
            predicted = np.squeeze(self.y_normalizer.inverse_transform(self.model.predict([np.array([ohlcv]), np.array([ind])])))
            delta = predicted - price_today

            if delta > threshold:
                buys.append((x, price_today[0][0]))
            elif delta < -threshold:
                sells.append((x, price_today[0][0]))
            
            x += 1
        return buys, sells

    def computeEarnings(self, buys_, sells_):
        purchase_amount = 1000
        stock = 0
        balance = 0
        while len(buys_) > 0 and len(sells_) > 0:
            if buys_[0][0] < sells_[0][0]:
                balance -= purchase_amount
                stock += purchase_amount / buys_[0][1]
                buys_.pop(0)
            else:
                balance += stock * sells_[0][1]
                stock = 0
                sells_.pop(0)
        print(f'earnings: ${balance}')


    def plotTrades(self, buys, sells, y_test_predicted):
        plt.gcf().set_size_inches(22, 15, forward=True)
        start = 0
        end = -1

        real = plt.plot(self.unscaled_y_test[start: end], label='real')
        predicted = plt.plot(y_test_predicted[start: end], label='predicted')

        if len(buys) > 0:
            plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
        if len(sells) > 0:
            plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

        plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])
        plt.savefig(fname='assets/trading.png')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="The path of the csv file")
    path = parser.parse_args().path
    model = load_model('model.h5')
    tradingObject = Trading(model, path)
    y_test_predicted = tradingObject.predict()
    buys, sells = tradingObject.getBuysAndSells(0.1)
    tradingObject.computeEarnings([b for b in buys], [s for s in sells])
    tradingObject.plotTrades(buys, sells, y_test_predicted)
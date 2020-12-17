from datetime import time
import pandas as pd
from sklearn import preprocessing
import numpy as np
import argparse

history_points = 50

class CSVToDataset:

    def csv_to_dataset(self, csv_path):
        data = pd.read_csv(csv_path)
        data = data.drop('date', axis=1)
        data - data.drop(0, axis=0)

        data = data.values

        data_normalizer = preprocessing.MinMaxScaler()
        data_normalized = data_normalizer.fit_transform(data)

        ohlcv_histories_normalized = np.array([data_normalized[i: i + history_points].copy() for i in range(len(data_normalized) - history_points)])
        next_day_open_values_normalized = np.array([data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
        next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1)

        next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
        next_day_open_values = np.expand_dims(next_day_open_values, -1)

        y_normalizer = preprocessing.MinMaxScaler()
        y_normalizer.fit(next_day_open_values)

        def calc_ema(values, time_period):
            sma = np.mean(values[:,3])
            ema_values = [sma]
            k = 2/ (1 + time_period)
            for i in range(len(his) - time_period, len(his)):
                close = his[i][3]
                ema_values.append(close * k + ema_values[-1] * (1 - k))
            return ema_values[-1]

        technical_indicators = []
        for his in ohlcv_histories_normalized:
            sma = np.mean(his[:,3])
            macd = calc_ema(his, 12) - calc_ema(his, 26)
            # technical_indicators.append(np.array([sma, macd, ]))
            technical_indicators.append(np.array([sma]))
        
        
        technical_indicators = np.array(technical_indicators)
        tech_ind_scaler = preprocessing.MinMaxScaler()
        technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

        assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == technical_indicators_normalized.shape[0]
        return ohlcv_histories_normalized, technical_indicators_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="The path of the csv file")
    path = parser.parse_args().path
    csvToDataset = CSVToDataset()
    ohlcv_histories, technical_indicators_normalized, next_day_open_values, unscaled_y, y_normalizer = csvToDataset.csv_to_dataset(path)
    print(ohlcv_histories)
from alpha_vantage.timeseries import TimeSeries
import json
import argparse

class SaveDataset:

    def save_dataset(self, symbol, time_window):
        credentials = json.load(open('assets/credentials.json', 'r'))
        api_key = credentials['alphavantage_api_key']
        
        print(symbol, time_window)

        ts = TimeSeries(key=api_key,output_format='pandas')
        if time_window == 'intraday':
            data, meta_data = ts.get_intraday(symbol, outputsize='full')
        elif time_window == 'daily':
            data, meta_data = ts.get_daily(symbol, outputsize='full')
        elif time_window == 'daily_adj':
            data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')

        data.to_csv(f'./csv/{symbol}_{time_window}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol', type=str, help="The stock symbol you want to download")
    parser.add_argument('time_window', type=str, choices=['intraday', 'daily', 'daily_adj'], help="The time period you want to download the stock history for")
    namespace = parser.parse_args()
    saveObject = SaveDataset()
    saveObject.save_dataset(**vars(namespace))
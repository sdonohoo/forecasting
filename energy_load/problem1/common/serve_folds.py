import os
import pandas as pd


def serve_folds():
    num_folds = 15
    data_file = os.path.join('data', 'energy_load', 'energy_load.csv')
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    for i in range(1, num_folds):
        train = data[data.task<=(i)].copy()
        train['month'] = train.index.month
        train['hour'] = train.index.hour
        train['dayofweek'] = train.index.weekday
        del train['task']
        test = data[data.task==(i+1)].copy()
        test['month'] = test.index.month
        test['hour'] = test.index.hour
        test['dayofweek'] = test.index.weekday
        test = test[['month', 'hour', 'dayofweek']]
        yield train, test
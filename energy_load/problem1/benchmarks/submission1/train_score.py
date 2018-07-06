import os
import pandas as pd
import sys
sys.path.append('.')
from energy_load.problem1.common.serve_folds import serve_folds


def compute_seasonal_average_baseline(train, test_template):
    
    # Fit a seasonal average baseline model and generate predictions
    # on the test data template
    train = train.dropna(how='any')
    seasonal_average = train[['month', 'hour', 'dayofweek', 'LOAD']].groupby(['month', 'hour', 'dayofweek']).mean().reset_index().rename(columns={'LOAD':'seasonal average'})
    predictions = pd.merge(test_template, seasonal_average).set_index(test_template.index)
    predictions = predictions.rename(columns={'seasonal average':'prediction'})
    predictions = predictions[['prediction']]
    return predictions


def generate_forecasts():
    prediction_dfs = []
    fold = 1
    print('Computing seasonal average forecasts...')
    # Repeatedly call serve_folds generator to fit the seasonal average
    # model and generate predictions for the next period
    for train, test_fold in serve_folds():
        print('Forecasting fold', fold)
        prediction = compute_seasonal_average_baseline(train, test_fold)
        prediction_dfs.append(prediction)
        fold += 1
    submission = pd.concat(prediction_dfs)
    submission['timestamp'] = submission.index
    submission = submission.reset_index()
    submission = submission[['timestamp', 'prediction']]
    fname = os.path.join('energy_load', 'problem1', 'benchmarks', 'submission1', 'submission.csv')
    submission.to_csv(fname, index=False)
    print('...submission file written to', fname)
    print('Now run python energy_load/problem1/common/evaluate.py', fname, 'to evaluate performance')


if __name__=="__main__":    
    generate_forecasts()

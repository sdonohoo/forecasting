import os
from datetime import datetime
import itertools
import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from joblib import Parallel, delayed

import localpath
from benchmark_paths import BENCHMARK_DATA_DIR, SUBMISSIONS_DIR
from serve_folds import serve_folds

# Model parameters
QUANTILES = np.linspace(0.1, 0.9, 9)
TARGET_COL = 'DEMAND'
# FEATURE_COLS = ['Holiday', 'DayType', 'Hour', 'TimeOfYear', 'WeekOfYear',
#                 'CurrentYear', 'annual_sin_1', 'annual_cos_1',
#                 'annual_sin_2', 'annual_cos_2', 'annual_sin_3',
#                 'annual_cos_3', 'weekly_sin_1', 'weekly_cos_1',
#                 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3',
#                 'weekly_cos_3', 'daily_sin_1', 'daily_cos_1', 'daily_sin_2',
#                 'daily_cos_2', 'LoadLag', 'DewPntLag', 'DryBulbLag']

FEATURE_COLS = ['Holiday', 'DayType', 'Hour', 'TimeOfYear', 'WeekOfYear',
                'CurrentYear', 'LoadLag', 'DewPntLag', 'DryBulbLag']

# Data paths
TRAIN_DATA_DIR = os.path.join(BENCHMARK_DATA_DIR, 'features', 'train')
TEST_DATA_DIR = os.path.join(BENCHMARK_DATA_DIR, 'features', 'test')
RESULT_DIR = os.path.join(SUBMISSIONS_DIR, 'submission_0', 'results')
RESULT_FILE = 'submission.csv'
RESULT_PATH = os.path.join(RESULT_DIR, RESULT_FILE)


def preprocess():
    # place holder for log transformation, box-jenkins transformation, etc.
    pass


def train_single_quantile(train_df_single, q):
    model = QuantReg(train_df_single[TARGET_COL], train_df_single[FEATURE_COLS])
    model_fit = model.fit(q=q)

    return model_fit


def train_single_group(train_df_single, quantiles, group_name):
    models_all = [train_single_quantile(train_df_single.copy(), q) for q in quantiles]

    models_dict = {}
    for q, m in zip(quantiles, models_all):
        models_dict['Q' + str(int(q*100))] = m

    return group_name, models_dict


def score_single_group(test_df_single, models_dict):
    output = test_df_single[['Zone', 'Datetime']].copy()
    for q, m in models_dict.items():
        output[q] = m.predict(test_df_single[FEATURE_COLS])

    return output


def train(train_df, parallel):
    train_df_grouped = train_df.groupby('Zone')

    models_all = parallel\
        (delayed(train_single_group)(group, QUANTILES, name) for name, group in train_df_grouped)

    models_all_dict = {}
    for k, v in models_all:
        models_all_dict[k] = v

    return models_all_dict


def score(test_df, models_all):
    group_names = models_all.keys()

    predictions_all = []

    for g in group_names:
        predictions_all.append(score_single_group(test_df.loc[test_df['Zone'] == g, ], models_all[g]))

    predictions_final = pd.concat(predictions_all)

    return predictions_final


def main():
    start_time = datetime.now()
    train_test = serve_folds(TRAIN_DATA_DIR, TEST_DATA_DIR)
    predictions_all = []
    with Parallel(n_jobs=-1) as parallel:
        for train_round, test_round, round_num in train_test:
            print('Round ' + str(round_num))
            models_all = train(train_round, parallel)
            predictions_round = score(test_round, models_all)
            predictions_round['Round'] = round_num
            predictions_all.append(predictions_round)

    predictions_final = pd.concat(predictions_all)

    predictions_final = \
        predictions_final[['Round', 'Zone', 'Datetime', 'Q10', 'Q20', 'Q30',
                           'Q40', 'Q50', 'Q60', 'Q70', 'Q80', 'Q90']]

    predictions_final.to_csv(RESULT_PATH, index=False)

    print('Prediction output size: {0}'.format(predictions_final.shape))
    print('Running time: {0}'. format(datetime.now() - start_time))


if __name__ == '__main__':
    main()


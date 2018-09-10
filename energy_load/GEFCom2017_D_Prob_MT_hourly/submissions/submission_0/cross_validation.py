import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from joblib import Parallel
import json

import localpath_tmp
from benchmark_paths_tmp import BENCHMARK_DATA_DIR, SUBMISSIONS_DIR
from utils_tmp import get_month_day_range, train, predict, split_train_validation


def pinball_loss(predictions, actuals, q):
    zeros = pd.Series([0]*len(predictions))
    return (predictions-actuals).combine(zeros, max)*(1-q) + (actuals-predictions).combine(zeros, max)*q


# Forecast creation times(FCTs) that are in the middle of a month, for these FCTs,
# the forecast horizon is the following month
FCT_MID_MONTH = pd.to_datetime(['2015-12-15', '2016-01-15', '2016-02-14'])
HORIZON_MID_MONTH = [m + relativedelta(months=1) for m in FCT_MID_MONTH]
FCT_HORIZON_MM = [(fct, get_month_day_range(h)) for fct, h in zip(FCT_MID_MONTH, HORIZON_MID_MONTH)]
# Forecast creation times(FCTs) that are at the end of a month, for these FCTs,
# the forecast horizon is the month after the following month
FCT_END_MONTH = pd.to_datetime(['2015-12-31', '2016-01-31', '2016-02-29'])
HORIZON_END_MONTH = [m + relativedelta(months=2) for m in FCT_END_MONTH]
FCT_HORIZON_EM = [(fct, get_month_day_range(h)) for fct, h in zip(FCT_END_MONTH, HORIZON_END_MONTH)]

FCT_HORIZON_ALL = FCT_HORIZON_MM + FCT_HORIZON_EM
FCT_HORIZON_ALL.sort(key=lambda x: x[0])

# Model parameters
GRAIN_COLS = ['Zone']
GROUP_COLS = ['Zone', 'Hour']
QUANTILES = np.linspace(0.1, 0.9, 9)
QUANT_REG_MAX_ITER = 2000
DATETIME_COL = 'Datetime'
TARGET_COL = 'DEMAND'

# FEATURE_COLS = ['Holiday', 'DayType', 'TimeOfYear', 'WeekOfYear',
#                 'CurrentYear', 'LoadLag', 'DewPntLag', 'DryBulbLag',
#                 'CurrentDate',
#                 'annual_sin_1', 'annual_cos_1', 'annual_sin_2',
#                 'annual_cos_2', 'annual_sin_3', 'annual_cos_3',
#                 'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2',
#                 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3',
#                 'RecentLoad_9', 'RecentLoad_10', 'RecentLoad_11',
#                 'RecentLoad_12', 'RecentLoad_13', 'RecentLoad_14',
#                 'RecentLoad_15', 'RecentLoad_16',
#                 'RecentDryBulb_9', 'RecentDryBulb_10', 'RecentDryBulb_11',
#                 'RecentDryBulb_12', 'RecentDryBulb_13', 'RecentDryBulb_14',
#                 'RecentDryBulb_15', 'RecentDryBulb_16',
#                 'RecentDewPnt_9', 'RecentDewPnt_10', 'RecentDewPnt_11',
#                 'RecentDewPnt_12', 'RecentDewPnt_13', 'RecentDewPnt_14',
#                 'RecentDewPnt_15', 'RecentDewPnt_16',
#                 ]

FEATURE_COLS = ['Holiday', 'DayType', 'TimeOfYear', 'WeekOfYear',
                'CurrentYear', 'LoadLag', 'DewPntLag', 'DryBulbLag',
                'CurrentDate',
                'annual_sin_1', 'annual_cos_1', 'annual_sin_2',
                'annual_cos_2', 'annual_sin_3', 'annual_cos_3',
                'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2',
                'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3',
                'RecentLoad_9', 'RecentLoad_10', 'RecentLoad_11',
                'RecentLoad_12', 'RecentLoad_13', 'RecentLoad_14',
                'RecentLoad_15', 'RecentLoad_16'
                ]


# Data paths
TRAIN_DATA_DIR = os.path.join(BENCHMARK_DATA_DIR, 'features', 'train')
TRAIN_DATA_FILE = os.path.join(TRAIN_DATA_DIR, 'train_round_1.csv')

RUN_NUM = 11
RUN_COMMENT = 'Added time trend'
RESULT_DIR = os.path.join(SUBMISSIONS_DIR, 'submission_0', 'results', 'cross_validation')
RESULT_FILE = os.path.join(RESULT_DIR, 'cv_result_' + str(RUN_NUM) + '.csv')
CONFIG_FILE = os.path.join(RESULT_DIR, 'config_' + str(RUN_NUM) + '.json')

whole_df = pd.read_csv(TRAIN_DATA_FILE, parse_dates=[DATETIME_COL])
# all_columns = set(whole_df.columns)
# columns_to_exclude = set(['Datetime', 'DewPnt', 'DryBulb', 'Zone', 'DEMAND'])
# FEATURE_COLS = list(all_columns - columns_to_exclude)

def preprocess():
    # place holder for log transformation, box-jenkins transformation, etc.
    pass


def construct_config():
    config_dict = {'feature_cols': FEATURE_COLS,
                   'target_col': TARGET_COL,
                   'quantiles': list(QUANTILES),
                   'comment': RUN_COMMENT,
                   'grain_cols': GRAIN_COLS,
                   'group_cols': GROUP_COLS,
                   'datetime_col': DATETIME_COL,
                   'quant_reg_max_iter': QUANT_REG_MAX_ITER,
                   'quant_reg_kernel': 'gau'}

    return config_dict


def main():
    start_time = datetime.now()
    config = construct_config()
    train_validation_split = split_train_validation(whole_df, FCT_HORIZON_ALL, DATETIME_COL)
    predictions_all = []

    with Parallel(n_jobs=-1) as parallel:
        for i_round, train_df, validation_df in train_validation_split:
            print('Round {0} max timestamp of training data: {1}'.format(i_round, max(train_df[DATETIME_COL])))
            print('Round {0} min timestamp of validation data: {1}'.format(i_round, min(validation_df[DATETIME_COL])))
            print('Round {0} max timestamp of validation data: {1}'.format(i_round, max(validation_df[DATETIME_COL])))

            validation_month = validation_df['MonthOfYear'].values[0]
            train_df = train_df.loc[train_df['MonthOfYear'] == validation_month, ].copy()

            models_all = train(train_df, parallel, config)
            predictions_df = predict(validation_df, models_all, config)
            predictions_df['Round'] = i_round
            predictions_df = predictions_df.merge(validation_df[[DATETIME_COL, 'Zone', 'DEMAND']],
                                                  on=['Datetime', 'Zone'])
            predictions_all.append(predictions_df)

        predictions_final = pd.concat(predictions_all)

        predictions_final.reset_index(inplace=True, drop=True)
        loss_col_all = []
        for q in QUANTILES:
            quantile_col = 'Q' + str(int(q*100))
            loss_col = quantile_col + 'Loss'
            loss_col_all.append(loss_col)
            loss_q = pinball_loss(predictions_final[quantile_col], predictions_final['DEMAND'], q)
            predictions_final[loss_col] = loss_q
        predictions_final['AverageLoss'] = predictions_final[loss_col_all].mean(axis=1)

        average_pinball_loss = predictions_final['AverageLoss'].mean()
        print('Average Pinball Loss: {}'.format(average_pinball_loss))

        run_time = datetime.now() - start_time
        print('Running time: {0}'. format(run_time))

        config['average_pinball_loss'] = average_pinball_loss
        config['run_time'] = str(run_time)
        with open(CONFIG_FILE, 'w') as fp:
            json.dump(config, fp, indent=4)
        predictions_final.to_csv(RESULT_FILE)


if __name__ == '__main__':
    main()


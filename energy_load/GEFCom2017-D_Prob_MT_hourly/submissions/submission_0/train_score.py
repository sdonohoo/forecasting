import os
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
# import localpath
# from benchmark_paths import BENCHMARK_DATA_DIR
# from serve_folds import serve_folds

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

BENCHMARK_DATA_DIR = 'C:\\Users\\honglu\\TSPerf\\energy_load\\GEFCom2017' \
                     '-D_Prob_MT_hourly\data\\'

train_data_dir = os.path.join(BENCHMARK_DATA_DIR, 'features', 'train')
test_data_dir = os.path.join(BENCHMARK_DATA_DIR, 'features', 'test')


# This file stores all the data before 2016-12-01
TRAIN_BASE_FILE = 'train_base.csv'
# These files contain data to be added to train_base.csv to form the training
# data of a particular round
TRAIN_ROUND_FILE_PREFIX = 'train_round_'
TEST_ROUND_FILE_PREFIX = 'test_round_'

i = 1
train_base = pd.read_csv(os.path.join(train_data_dir, TRAIN_BASE_FILE))

train_round_file_name = TRAIN_ROUND_FILE_PREFIX + str(i+1) + '.csv'
train_round_delta = \
    pd.read_csv(os.path.join(train_data_dir, train_round_file_name))

train_df = pd.concat([train_base, train_round_delta])

test_round_file_name = TEST_ROUND_FILE_PREFIX + str(i+1) + '.csv'

test_df = \
    pd.read_csv(os.path.join(test_data_dir, test_round_file_name))

train_df_single = train_df.loc[train_df['Zone'] == 'ME', ]
model = QuantReg(train_df_single[TARGET_COL], train_df_single[FEATURE_COLS])

model_fit = model.fit(q=0.5)

def preprocess():
    # place holder for log transformation, box-jenkins transformation, etc.
    pass


def train_single_group():
    pass


def score_single_group():
    pass


def train():
    pass


def score():
    pass


train_test = serve_folds(TRAIN_DATA_DIR, TEST_DATA_DIR)
for train, test in train_test:
    print('Training data size: {}'.format(train.shape))
    print('Testing data size: {}'.format(test.shape))
    print('Minimum training timestamp: {}'.format(min(train['Datetime'])))
    print('Maximum training timestamp: {}'.format(max(train['Datetime'])))
    print('Minimum testing timestamp: {}'.format(min(test['Datetime'])))
    print('Maximum testing timestamp: {}'.format(max(test['Datetime'])))
    print('')

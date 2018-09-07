import os
import pandas as pd
import numpy as np

from feature_utils import create_basic_features, create_advanced_features
from benchmark_paths import DATA_DIR

OUTPUT_DIR = os.path.join(DATA_DIR, 'features')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_BASE_FILE = 'train_base.csv'
TRAIN_FILE_PREFIX = 'train_round_'
TEST_FILE_PREFIX = 'test_round_'
NUM_ROUND = 6

DATETIME_COLNAME = 'Datetime'
HOLIDAY_COLNAME = 'Holiday'

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def main(train_dir, test_dir, output_dir, datetime_colname, holiday_colname):

    output_train_dir = os.path.join(output_dir, 'train')
    output_test_dir = os.path.join(output_dir, 'test')
    if not os.path.isdir(output_train_dir):
        os.mkdir(output_train_dir)
    if not os.path.isdir(output_test_dir):
        os.mkdir(output_test_dir)

    train_base_df = pd.read_csv(os.path.join(train_dir, TRAIN_BASE_FILE),
                                parse_dates=[datetime_colname])
    train_base_basic_features = \
        create_basic_features(train_base_df,
                              datetime_colname=datetime_colname)

    normalize_columns = ['DayType', 'WeekOfYear', 'LoadLag', 'DewPntLag',
                         'DryBulbLag']

    for i in range(1, NUM_ROUND+1):
        train_file = os.path.join(train_dir, TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_file = os.path.join(test_dir, TEST_FILE_PREFIX + str(i) + '.csv')

        train_delta_df = pd.read_csv(train_file, parse_dates=[datetime_colname])
        test_round_df = pd.read_csv(test_file, parse_dates=[datetime_colname])

        train_delta_basic_features = \
            create_basic_features(train_delta_df,
                                  datetime_colname=datetime_colname)
        test_basic_features = \
            create_basic_features(test_round_df,
                                  datetime_colname=datetime_colname)

        train_round_df = pd.concat([train_base_df, train_delta_df])
        train_advanced_features, test_advanced_features = \
            create_advanced_features(train_round_df, test_round_df,
                                     datetime_colname=datetime_colname,
                                     holiday_colname=holiday_colname)

        train_basic_features = pd.concat([train_base_basic_features,
                                         train_delta_basic_features])

        train_basic_columns = set(train_basic_features.columns)
        train_advanced_columns = set(train_advanced_features.columns)
        train_overlap_columns = list(train_basic_columns.intersection(train_advanced_columns))
        train_overlap_columns.remove('Zone')
        train_overlap_columns.remove('Datetime')
        train_advanced_features.drop(train_overlap_columns, inplace=True, axis=1)

        test_basic_columns = set(test_basic_features.columns)
        test_advanced_columns = set(test_advanced_features.columns)
        test_overlap_columns = list(test_basic_columns.intersection(test_advanced_columns))
        test_overlap_columns.remove('Zone')
        test_overlap_columns.remove('Datetime')
        test_advanced_features.drop(test_overlap_columns, inplace=True, axis=1)

        train_all_features = pd.merge(train_basic_features,
                                      train_advanced_features,
                                      on=['Zone', 'Datetime'])
        test_all_features = pd.merge(test_basic_features,
                                     test_advanced_features,
                                     on=['Zone', 'Datetime'])

        train_all_features.dropna(inplace=True)
        test_all_features.drop(['DewPnt', 'DryBulb', 'DEMAND'], inplace=True, axis=1)

        # for c in normalize_columns:
        #     min_value = np.nanmin(train_all_features[c].values)
        #     max_value = np.nanmax(train_all_features[c].values)
        #     train_all_features[c] = \
        #         (train_all_features[c] - min_value)/(max_value - min_value)
        #     test_all_features[c] = \
        #         (test_all_features[c] - min_value)/(max_value - min_value)

        train_output_file = os.path.join(output_dir, 'train',
                                         TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_output_file = os.path.join(output_dir, 'test',
                                        TEST_FILE_PREFIX + str(i) + '.csv')

        train_all_features.to_csv(train_output_file, index=False)
        test_all_features.to_csv(test_output_file, index=False)

        print('Round {}'.format(i))
        print('Training data size: {}'.format(train_all_features.shape))
        print('Testing data size: {}'.format(test_all_features.shape))
        print('Minimum training timestamp: {}'.format(min(train_all_features[datetime_colname])))
        print('Maximum training timestamp: {}'.format(max(train_all_features[datetime_colname])))
        print('Minimum testing timestamp: {}'.format(min(test_all_features[datetime_colname])))
        print('Maximum testing timestamp: {}'.format(max(test_all_features[datetime_colname])))
        print('')


if __name__ == '__main__':
    main(train_dir=TRAIN_DATA_DIR,
         test_dir=TEST_DATA_DIR,
         output_dir=OUTPUT_DIR,
         datetime_colname=DATETIME_COLNAME,
         holiday_colname=HOLIDAY_COLNAME)
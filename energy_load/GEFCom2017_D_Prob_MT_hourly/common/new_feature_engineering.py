"""
This script creates a set of commonly used features using the functions in
common.feature_utils, which serve as a set of baseline features.
Feel free to write your own feature engineering code to create new features by
calling the feature_utils functions with alternative parameters.
"""
import os
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.pipeline import Pipeline

from .benchmark_paths import DATA_DIR
from common.features.lag import (SameWeekDayHourLagFeaturizer,
                                 SameDayHourLagFeaturizer)
from common.features.temporal import (TemporalFeaturizer,
                                      DayTypeFeaturizer,
                                      AnnualFourierFeaturizer,
                                      DailyFourierFeaturizer,
                                      WeeklyFourierFeaturizer)
from common.features.rolling_window import (SameWeekDayHourRollingFeaturizer,
                                            YearOverYearRatioFeaturizer)
from common.features.normalization import (YearNormalizer,
                                           DateNormalizer,
                                           DateHourNormalizer)

print('Data directory used: {}'.format(DATA_DIR))

OUTPUT_DIR = os.path.join(DATA_DIR, 'features')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_BASE_FILE = 'train_base.csv'
TRAIN_FILE_PREFIX = 'train_round_'
TEST_FILE_PREFIX = 'test_round_'
NUM_ROUND = 6

DATETIME_COLNAME = 'Datetime'
HOLIDAY_COLNAME = 'Holiday'
GRAIN_COLNAME = 'Zone'

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# A dictionary mapping each feature name to the function for computing the
# feature
FEATURE_MAP = {'Temporal': TemporalFeaturizer,
               'AnnualFourier': AnnualFourierFeaturizer,
               'WeeklyFourier': WeeklyFourierFeaturizer,
               'DailyFourier': DailyFourierFeaturizer,
               'CurrentDate': DateNormalizer,
               'CurrentDateHour': DateHourNormalizer,
               'CurrentYear': YearNormalizer,
               'DayType': DayTypeFeaturizer,
               'RecentLoadLag': SameWeekDayHourRollingFeaturizer,
               'RecentDryBulbLag': SameWeekDayHourRollingFeaturizer,
               'RecentDewPntLag': SameWeekDayHourRollingFeaturizer,
               'PreviousYearLoadLag':  SameWeekDayHourLagFeaturizer,
               'PreviousYearDewPntLag':  SameDayHourLagFeaturizer,
               'PreviousYearDryBulbLag': SameDayHourLagFeaturizer,
               'LoadRatio': YearOverYearRatioFeaturizer}


def parse_feature_config(feature_config, feature_map):
    """
    A helper function parsing a feature_config to feature name, column to
    compute the feature on, feature function to use, and arguments to the
    feature function
    """
    feature_name = feature_config[0]
    feature_args = feature_config[1]
    featurizer = feature_map[feature_name]

    return feature_name, feature_args, featurizer


def compute_training_features(train_df, df_config,
                              feature_config_list, feature_map):
    pipeline_steps = []
    for feature_config in feature_config_list:
        feature_name, feature_args, featurizer = \
            parse_feature_config(feature_config, feature_map)
        pipeline_steps.append(
            (feature_name, featurizer(df_config=df_config, **feature_args)))

    feature_engineering_pipeline = Pipeline(pipeline_steps)
    feature_engineering_pipeline_fitted = \
        feature_engineering_pipeline.fit(train_df)
    train_features = feature_engineering_pipeline_fitted.transform(train_df)

    return train_features, feature_engineering_pipeline_fitted


def compute_testing_features(test_df, df_config, feature_engineering_pipeline,
                             train_df=None):
    if train_df is not None:
        time_col_name = df_config['time_col_name']
        forecast_creation_time = max(train_df[time_col_name])
        full_df = pd.concat([train_df, test_df])
        full_features = feature_engineering_pipeline.transform(full_df)
        test_features = full_features.loc[full_features[time_col_name] >
                                          forecast_creation_time].copy()
    else:
        test_features = feature_engineering_pipeline.transform(test_df)

    return test_features


def compute_features_one_round(train_base_df, train_delta_df, test_df,
                               df_config, basic_feature_config_list,
                               advanced_feature_config_list, feature_map,
                               train_base_basic_features,
                               filter_by_month):

    time_col_name = df_config['time_col_name']
    grain_col_name = df_config['grain_col_name']
    if isinstance(grain_col_name, list):
        merge_on_cols = [time_col_name] + grain_col_name
    else:
        merge_on_cols = [time_col_name, grain_col_name]

    train_delta_basic_features, basic_feature_pipeline = \
        compute_training_features(train_delta_df, df_config,
                                  basic_feature_config_list, feature_map)

    test_basic_features = \
        compute_testing_features(test_df, df_config, basic_feature_pipeline)

    train_round_df = pd.concat([train_base_df, train_delta_df])

    train_advanced_features, advanced_feature_pipeline = \
        compute_training_features(train_round_df, df_config,
                                  advanced_feature_config_list, feature_map)

    test_advanced_features = \
        compute_testing_features(test_df, df_config,
                                 advanced_feature_pipeline, train_round_df)

    train_basic_features = pd.concat([train_base_basic_features,
                                      train_delta_basic_features])

    # Drop some overlapping columns before merge basic and advanced
    # features.
    train_basic_columns = set(train_basic_features.columns)
    train_advanced_columns = set(train_advanced_features.columns)
    train_overlap_columns = list(train_basic_columns.
                                 intersection(train_advanced_columns))
    train_overlap_columns.remove('Zone')
    train_overlap_columns.remove('Datetime')
    train_advanced_features.drop(train_overlap_columns,
                                 inplace=True, axis=1)

    test_basic_columns = set(test_basic_features.columns)
    test_advanced_columns = set(test_advanced_features.columns)
    test_overlap_columns = list(test_basic_columns.
                                intersection(test_advanced_columns))
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
    test_all_features.drop(['DewPnt', 'DryBulb', 'DEMAND'],
                           inplace=True, axis=1)
    if filter_by_month:
        test_month = test_basic_features['month_of_year'].values[0]
        train_all_features = train_all_features.loc[
            train_all_features['month_of_year'] == test_month, ].copy()

    return train_all_features, test_all_features


def compute_features(train_dir, test_dir, output_dir, df_config,
                     basic_feature_list, advanced_feature_list,
                     filter_by_month=True):
    """
    This helper function uses the create_basic_features and create_advanced
    features functions to create features for each train and test round.

    Args:
        train_dir (str): directory containing training data
        test_dir (str): directory containing testing data
        output_dir (str): directory to which to save the output files
        datetime_colname (str): name of Datetime column
        holiday_colname (str): name of Holiday column
    """
    time_col_name = df_config['time_col_name']

    output_train_dir = os.path.join(output_dir, 'train')
    output_test_dir = os.path.join(output_dir, 'test')
    if not os.path.isdir(output_train_dir):
        os.mkdir(output_train_dir)
    if not os.path.isdir(output_test_dir):
        os.mkdir(output_test_dir)

    train_base_df = pd.read_csv(os.path.join(train_dir, TRAIN_BASE_FILE),
                                parse_dates=[time_col_name])

    # These features only need to be created once for all rounds
    train_base_basic_features, _ = \
        compute_training_features(train_base_df, df_config,
                                  basic_feature_list, FEATURE_MAP)

    for i in range(1, NUM_ROUND + 1):
        train_file = os.path.join(train_dir,
                                  TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_file = os.path.join(test_dir, TEST_FILE_PREFIX + str(i) + '.csv')

        train_delta_df = pd.read_csv(train_file, parse_dates=[DATETIME_COLNAME])
        test_round_df = pd.read_csv(test_file, parse_dates=[DATETIME_COLNAME])

        train_all_features, test_all_features = \
            compute_features_one_round(train_base_df, train_delta_df,
                                       test_round_df, df_config,
                                       basic_feature_list,
                                       advanced_feature_list,
                                       FEATURE_MAP,
                                       train_base_basic_features,
                                       filter_by_month)

        train_output_file = os.path.join(output_dir, 'train',
                                         TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_output_file = os.path.join(output_dir, 'test',
                                        TEST_FILE_PREFIX + str(i) + '.csv')

        train_all_features.to_csv(train_output_file, index=False)
        test_all_features.to_csv(test_output_file, index=False)

        print('Round {}'.format(i))
        print('Training data size: {}'.format(train_all_features.shape))
        print('Testing data size: {}'.format(test_all_features.shape))
        print('Minimum training timestamp: {}'.
              format(min(train_all_features[DATETIME_COLNAME])))
        print('Maximum training timestamp: {}'.
              format(max(train_all_features[DATETIME_COLNAME])))
        print('Minimum testing timestamp: {}'.
              format(min(test_all_features[DATETIME_COLNAME])))
        print('Maximum testing timestamp: {}'.
              format(max(test_all_features[DATETIME_COLNAME])))
        print('')

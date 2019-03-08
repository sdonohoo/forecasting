"""
This script creates a set of commonly used features using the functions in
common.feature_utils, which serve as a set of baseline features.
Feel free to write your own feature engineering code to create new features by
calling the feature_utils functions with alternative parameters.
"""
import os
import pandas as pd
pd.set_option('display.max_columns', None)
from functools import reduce
from sklearn.pipeline import Pipeline

from .benchmark_paths import DATA_DIR
from common.features.lag import (SameDayHourLagFeaturizer,
                                 SameWeekDayHourLagFeaturizer)
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
FEATURE_MAP = {'temporal': TemporalFeaturizer,
               'annual_fourier': AnnualFourierFeaturizer,
               'weekly_fourier': WeeklyFourierFeaturizer,
               'daily_fourier': DailyFourierFeaturizer,
               'normalized_date': DateNormalizer,
               'normalized_datehour': DateHourNormalizer,
               'normalized_year': YearNormalizer,
               'day_type': DayTypeFeaturizer,
               'recent_load_lag': SameWeekDayHourRollingFeaturizer,
               'recent_dry_bulb_lag': SameWeekDayHourRollingFeaturizer,
               'recent_dew_pnt_lag': SameWeekDayHourRollingFeaturizer,
               'previous_year_load_lag': SameWeekDayHourLagFeaturizer,
               'previous_year_dew_pnt_lag':  SameDayHourLagFeaturizer,
               'previous_year_dry_bulb_lag': SameDayHourLagFeaturizer,
               'load_ratio': YearOverYearRatioFeaturizer}

FEATURES_REQUIRE_TRAINING_DATA = \
    ['recent_load_lag', 'recent_dry_bulb_lag', 'recent_dew_pnt_lag',
     'previous_year_load_lag', 'previous_year_dew_pnt_lag',
     'previous_year_dry_bulb_lag', 'load_ratio']


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


def create_train_scoring_df(train_df, test_df=None,
                            scoring_flag_col_name='scoring_flag'):
    train_df = train_df.copy()
    train_df[scoring_flag_col_name] = 1

    if test_df is not None:
        train_df_scoring = train_df.copy()
        test_df_scoring = test_df.copy()
        train_df_scoring[scoring_flag_col_name] = 0
        test_df_scoring[scoring_flag_col_name] = 1
        scoring_df = pd.concat([train_df_scoring, test_df_scoring])

        return train_df, scoring_df
    else:
        return train_df


def compute_features_one_round(train_base_df, train_delta_df, test_df,
                               df_config, feature_config_list, feature_map,
                               filter_by_month):

    train_round_df = pd.concat([train_base_df, train_delta_df], sort=True)

    # train_df, scoring_df = create_train_scoring_df(train_round_df, test_df)

    train_features, feature_pipeline = \
        compute_training_features(train_round_df, df_config,
                                  feature_config_list, feature_map)

    training_df_arguments = {}
    for feature_config in feature_config_list:
        feature_step_name = feature_config[0]
        if feature_step_name in FEATURES_REQUIRE_TRAINING_DATA:
            training_df_arguments[feature_step_name + '__training_df'] = \
                train_round_df
    if len(training_df_arguments) > 0:
        feature_pipeline.set_params(**training_df_arguments)

    test_features = \
        compute_testing_features(test_df, df_config, feature_pipeline)

    train_features.dropna(inplace=True)

    if filter_by_month:
        test_month = test_features['month_of_year'].values[0]
        train_features = train_features.loc[
            train_features['month_of_year'] == test_month, ].copy()

    return train_features, test_features


def compute_features(train_dir, test_dir, output_dir, df_config,
                     feature_config_list,
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

    for i in range(1, NUM_ROUND + 1):
        train_file = os.path.join(train_dir,
                                  TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_file = os.path.join(test_dir, TEST_FILE_PREFIX + str(i) + '.csv')

        train_delta_df = pd.read_csv(train_file, parse_dates=[DATETIME_COLNAME])
        test_round_df = pd.read_csv(test_file, parse_dates=[DATETIME_COLNAME])

        train_all_features, test_all_features = \
            compute_features_one_round(train_base_df, train_delta_df,
                                       test_round_df, df_config,
                                       feature_config_list,
                                       FEATURE_MAP,
                                       filter_by_month)

        train_all_features.dropna(inplace=True)
        test_all_features.drop(['DewPnt', 'DryBulb', 'DEMAND'],
                               inplace=True, axis=1)

        train_output_file = os.path.join(output_dir, 'train',
                                         TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_output_file = os.path.join(output_dir, 'test',
                                        TEST_FILE_PREFIX + str(i) + '.csv')

## temporary scripts for results verification
#         train_all_features.rename(mapper={'hour_of_day': 'Hour', 'day_of_week': 'DayOfWeek', 'day_of_month': 'DayOfMonth', 'hour_of_year': 'TimeOfYear',	'week_of_year': 'WeekOfYear', 'month_of_year': 'MonthOfYear',
# 'day_type': 'DayType', 'load_lag': 'LoadLag', 'dew_pnt_lag': 'DewPntLag', 'dry_bulb_lag': 'DryBulbLag',
# 'recent_load_10': 'RecentLoad_10', 'recent_load_11': 'RecentLoad_11', 'recent_load_12': 'RecentLoad_12', 'recent_load_13': 'RecentLoad_13',
# 'recent_load_14': 'RecentLoad_14', 'recent_load_15': 'RecentLoad_15', 'recent_load_16': 'RecentLoad_16', 'recent_load_17': 'RecentLoad_17',
# 'recent_dry_bulb_9': 'RecentDryBulb_9',  'recent_dry_bulb_10': 'RecentDryBulb_10', 'recent_dry_bulb_11': 'RecentDryBulb_11', 'recent_dry_bulb_12': 'RecentDryBulb_12',
# 'recent_dry_bulb_13': 'RecentDryBulb_13', 'recent_dry_bulb_14': 'RecentDryBulb_14',	'recent_dry_bulb_15': 'RecentDryBulb_15', 'recent_dry_bulb_16': 'RecentDryBulb_16',
# 'recent_dew_pnt_9': 'RecentDewPnt_9', 'recent_dew_pnt_10': 'RecentDewPnt_10', 'recent_dew_pnt_11': 'RecentDewPnt_11', 'recent_dew_pnt_12': 'RecentDewPnt_12',
# 'recent_dew_pnt_13': 'RecentDewPnt_13', 'recent_dew_pnt_14':
#                                               'RecentDewPnt_14',
#                                           'recent_dew_pnt_15':
#                                               'RecentDewPnt_15',
#                                           'recent_dew_pnt_16':
#                                               'RecentDewPnt_16'},
#                                   axis=1, inplace=True)
#
#         test_all_features.rename(mapper={'hour_of_day': 'Hour', 'day_of_week': 'DayOfWeek', 'day_of_month': 'DayOfMonth', 'hour_of_year': 'TimeOfYear',	'week_of_year': 'WeekOfYear', 'month_of_year': 'MonthOfYear',
# 'day_type': 'DayType', 'load_lag': 'LoadLag', 'dew_pnt_lag': 'DewPntLag', 'dry_bulb_lag': 'DryBulbLag',
# 'recent_load_10': 'RecentLoad_10', 'recent_load_11': 'RecentLoad_11', 'recent_load_12': 'RecentLoad_12', 'recent_load_13': 'RecentLoad_13',
# 'recent_load_14': 'RecentLoad_14', 'recent_load_15': 'RecentLoad_15', 'recent_load_16': 'RecentLoad_16', 'recent_load_17': 'RecentLoad_17',
# 'recent_dry_bulb_9': 'RecentDryBulb_9',  'recent_dry_bulb_10': 'RecentDryBulb_10', 'recent_dry_bulb_11': 'RecentDryBulb_11', 'recent_dry_bulb_12': 'RecentDryBulb_12',
# 'recent_dry_bulb_13': 'RecentDryBulb_13', 'recent_dry_bulb_14': 'RecentDryBulb_14',	'recent_dry_bulb_15': 'RecentDryBulb_15', 'recent_dry_bulb_16': 'RecentDryBulb_16',
# 'recent_dew_pnt_9': 'RecentDewPnt_9', 'recent_dew_pnt_10': 'RecentDewPnt_10', 'recent_dew_pnt_11': 'RecentDewPnt_11', 'recent_dew_pnt_12': 'RecentDewPnt_12',
# 'recent_dew_pnt_13': 'RecentDewPnt_13', 'recent_dew_pnt_14':
#                                              'RecentDewPnt_14',
#                                          'recent_dew_pnt_15':
#                                              'RecentDewPnt_15',
#                                          'recent_dew_pnt_16':
#                                              'RecentDewPnt_16'},
#                                  axis=1, inplace=True)
#
#         train_all_features = train_all_features[['DEMAND', 'DewPnt', 'DryBulb', 'Zone', 'Holiday',
# 'Hour', 'DayOfWeek', 'DayOfMonth', 'TimeOfYear', 'WeekOfYear', 'MonthOfYear',
# 'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3', 'daily_sin_1', 'daily_cos_1', 'daily_sin_2', 'daily_cos_2',
# 'DayType', 'LoadLag', 'DewPntLag', 'DryBulbLag', 'RecentLoad_10', 'RecentLoad_11', 'RecentLoad_12', 'RecentLoad_13',	'RecentLoad_14', 'RecentLoad_15', 'RecentLoad_16', 'RecentLoad_17',
# 'RecentDryBulb_9', 'RecentDryBulb_10', 'RecentDryBulb_11', 'RecentDryBulb_12', 'RecentDryBulb_13', 'RecentDryBulb_14',	'RecentDryBulb_15',	'RecentDryBulb_16',
# 'RecentDewPnt_9', 'RecentDewPnt_10', 'RecentDewPnt_11',	'RecentDewPnt_12', 'RecentDewPnt_13', 'RecentDewPnt_14', 'RecentDewPnt_15',	'RecentDewPnt_16']]
#         test_all_features = test_all_features[['Zone', 'Holiday',
# 'Hour', 'DayOfWeek', 'DayOfMonth', 'TimeOfYear', 'WeekOfYear', 'MonthOfYear',
# 'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3', 'daily_sin_1', 'daily_cos_1', 'daily_sin_2', 'daily_cos_2',
# 'DayType', 'LoadLag', 'DewPntLag', 'DryBulbLag', 'RecentLoad_10', 'RecentLoad_11', 'RecentLoad_12', 'RecentLoad_13',	'RecentLoad_14', 'RecentLoad_15', 'RecentLoad_16', 'RecentLoad_17',
# 'RecentDryBulb_9', 'RecentDryBulb_10', 'RecentDryBulb_11', 'RecentDryBulb_12', 'RecentDryBulb_13', 'RecentDryBulb_14',	'RecentDryBulb_15',	'RecentDryBulb_16',
# 'RecentDewPnt_9', 'RecentDewPnt_10', 'RecentDewPnt_11',	'RecentDewPnt_12', 'RecentDewPnt_13', 'RecentDewPnt_14', 'RecentDewPnt_15',	'RecentDewPnt_16']]

        train_all_features.to_csv(train_output_file, index=False)
        test_all_features.to_csv(test_output_file, index=False)

        print('Round {}'.format(i))
        print(train_all_features.columns)
        print(test_all_features.columns)
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

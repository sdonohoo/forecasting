"""
This script computes features on the GEFCom2017_D dataset. It is
parameterized so that a selected set of features specified by a feature
configuration list are computed and saved as csv files.
"""
import os
import pandas as pd
from functools import reduce
from sklearn.pipeline import Pipeline

from common.features.lag import (SameDayHourLagFeaturizer,
                                 SameWeekDayHourLagFeaturizer)
from common.features.temporal import (TemporalFeaturizer,
                                      DayTypeFeaturizer,
                                      AnnualFourierFeaturizer,
                                      DailyFourierFeaturizer,
                                      WeeklyFourierFeaturizer)
from common.features.rolling_window import SameWeekdayHourRollingAggFeaturizer
from common.features.normalization import (YearNormalizer,
                                           DateNormalizer,
                                           DateHourNormalizer)

from .benchmark_paths import DATA_DIR
print('Data directory used: {}'.format(DATA_DIR))

pd.set_option('display.max_columns', None)

OUTPUT_DIR = os.path.join(DATA_DIR, 'features')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_BASE_FILE = 'train_base.csv'
TRAIN_FILE_PREFIX = 'train_round_'
TEST_FILE_PREFIX = 'test_round_'
NUM_ROUND = 6


# A dictionary mapping each feature name to the featurizer for computing the
# feature
FEATURE_MAP = {'temporal': TemporalFeaturizer,
               'annual_fourier': AnnualFourierFeaturizer,
               'weekly_fourier': WeeklyFourierFeaturizer,
               'daily_fourier': DailyFourierFeaturizer,
               'normalized_date': DateNormalizer,
               'normalized_datehour': DateHourNormalizer,
               'normalized_year': YearNormalizer,
               'day_type': DayTypeFeaturizer,
               'recent_load_lag': SameWeekdayHourRollingAggFeaturizer,
               'recent_dry_bulb_lag': SameWeekdayHourRollingAggFeaturizer,
               'recent_dew_pnt_lag': SameWeekdayHourRollingAggFeaturizer,
               'previous_year_load_lag': SameWeekDayHourLagFeaturizer,
               'previous_year_dew_pnt_lag':  SameDayHourLagFeaturizer,
               'previous_year_dry_bulb_lag': SameDayHourLagFeaturizer}

# List of features that requires the training data when computing them on the
# testing data
FEATURES_REQUIRE_TRAINING_DATA = \
    ['recent_load_lag', 'recent_dry_bulb_lag', 'recent_dew_pnt_lag',
     'previous_year_load_lag', 'previous_year_dew_pnt_lag',
     'previous_year_dry_bulb_lag', 'load_ratio']


def parse_feature_config(feature_config, feature_map):
    """
    A helper function parsing a feature_config to feature name,
    featurizer class, and arguments to use to initialize the featurizer.
    """
    feature_name = feature_config[0]
    feature_args = feature_config[1]
    featurizer = feature_map[feature_name]

    return feature_name, feature_args, featurizer


def compute_training_features(train_df, df_config,
                              feature_config_list, feature_map,
                              max_test_timestamp):
    """
    Creates a pipeline based on the input feature configuration list and the
    feature_map. Fit the pipeline on the training data and transform
    the training data.

    Args:
        train_df(pd.DataFrame): Training data to fit on and transform.
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        feature_config_list(list of tuples): The first element of each
            feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
        feature_map(dict): Maps each feature name (key) to corresponding
            featurizer(value).
        max_test_timestamp(pd.datetime): Maximum timestamp of the testing
            data to generate forecasting on. This value is needed by a small
            number of featurizers to prevent creating lag features on the
            training data that are not available for the testing data. For
            example, for SameWeekdayHourRollingAggFeaturizer, the features are
            created on week 7 to forecast week 8 to week 10. It would not make
            sense to create an aggregation feature using data from week 8 and
            week 9, because they are not available at the forecast creation
            time. Thus, it does not make sense to create an aggregation
            feature using data from week 5 and week 6 for week 7.

    Returns:
        (pd.DataFrame, sklearn.pipeeline): (training features, feature
            engineering pipeline fitted on the training data.
    """
    pipeline_steps = []
    for feature_config in feature_config_list:
        feature_name, feature_args, featurizer = \
            parse_feature_config(feature_config, feature_map)
        if featurizer.__name__ == 'SameWeekdayHourRollingAggFeaturizer':
            feature_args['max_test_timestamp'] = max_test_timestamp
        pipeline_steps.append(
            (feature_name, featurizer(df_config=df_config, **feature_args)))

    feature_engineering_pipeline = Pipeline(pipeline_steps)
    feature_engineering_pipeline_fitted = \
        feature_engineering_pipeline.fit(train_df)
    train_features = feature_engineering_pipeline_fitted.transform(train_df)

    return train_features, feature_engineering_pipeline_fitted


def compute_testing_features(test_df, feature_engineering_pipeline,
                             feature_config_list=None, train_df=None):

    """
    Computes features on the testing data using a fitted feature engineering
    pipeline.

    Args:
        test_df(pd.DataFrame): Testing data to fit on and transform.
        feature_engineering_pipeline(sklearn.pipeline): A feature engineering
            pipeline fitted on the training data.
        feature_config_list(list of tuples, optional): The first element of each
            feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
            A value is required if train_df is not None.
        train_df(pd.DataFrame, optional): Training data needed to compute
            some lag features on testing data.
    Returns:
        pd.DataFrame: Testing features.
    """
    if train_df is not None and feature_config_list is not None:
        training_df_arguments = {}
        for feature_config in feature_config_list:
            feature_step_name = feature_config[0]
            if feature_step_name in FEATURES_REQUIRE_TRAINING_DATA:
                training_df_arguments[feature_step_name + '__training_df'] = \
                    train_df
        if len(training_df_arguments) > 0:
            feature_engineering_pipeline.set_params(**training_df_arguments)

    test_features = feature_engineering_pipeline.transform(test_df)

    return test_features


def compute_features_one_round(train_base_df, train_delta_df, test_df,
                               df_config, feature_config_list, feature_map,
                               filter_by_month, compute_load_ratio=False):

    """
    Computes features on one round of training and testing data.
    Args:
        train_base_df(pd.DataFrame): Training data common to all rounds.
        train_delta_df(pd.DataFrame): Additional training data for the
            current round.
        test_df(pd.DataFrame): Testing data of the current round.
        df_config: Configuration of the input dataframes.
        feature_config_list(list of tuples, optional): The first element of
            each feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
        feature_map(dict): Maps each feature name (key) to corresponding
            featurizer(value).
        filter_by_month(bool): If filter the training data by the month of
            the testing data.
        compute_load_ratio(bool): If computes a scaling factor that capture
            the year over year trend and can be used to scale the forecasting
            result.
    Returns:
        (pd.DataFrame, pd.DataFrame): (training features, testing features)
    """

    train_round_df = pd.concat([train_base_df, train_delta_df], sort=True)
    max_test_timestamp = test_df[df_config['time_col_name']].max()
    train_features, feature_pipeline = \
        compute_training_features(train_round_df, df_config,
                                  feature_config_list, feature_map,
                                  max_test_timestamp)

    test_features = \
        compute_testing_features(test_df, feature_pipeline,
                                 feature_config_list, train_round_df)

    if compute_load_ratio:
        same_week_day_hour_rolling_featurizer = \
            SameWeekdayHourRollingAggFeaturizer(
                df_config, input_col_name=df_config['value_col_name'],
                window_size=4, start_week=10, agg_count=7,
                output_col_prefix='recent_load_',
                max_test_timestamp=max_test_timestamp)
        train_df_with_recent_load = \
            same_week_day_hour_rolling_featurizer.transform(train_round_df)
        same_week_day_hour_rolling_featurizer.training_df = train_round_df
        test_df_with_recent_load = \
            same_week_day_hour_rolling_featurizer.transform(test_df)

        time_col_name = df_config['time_col_name']
        grain_col_name = df_config['grain_col_name']
        keep_col_names = [time_col_name]
        if grain_col_name is not None:
            if isinstance(grain_col_name, list):
                keep_col_names = keep_col_names + grain_col_name
            else:
                keep_col_names.append(grain_col_name)
        lag_df_list = []
        for i in range(10, 17):
            col_old = 'recent_load_' + str(i)
            col_new = 'recent_load_lag_' + str(i)
            col_ratio = 'recent_load_ratio_' + str(i)

            same_week_day_hour_lag_featurizer = \
                SameWeekDayHourLagFeaturizer(
                    df_config, input_col_name=col_old,
                    training_df=train_df_with_recent_load, n_years=5,
                    week_window=0, output_col_name=col_new)

            lag_df = same_week_day_hour_lag_featurizer\
                .transform(test_df_with_recent_load)
            lag_df[col_ratio] = lag_df[col_old] / lag_df[col_new]
            lag_df_list.append(lag_df[keep_col_names + [col_ratio]].copy())

        test_features = reduce(
            lambda left, right: pd.merge(left, right, on=keep_col_names),
            [test_features] + lag_df_list)

    if filter_by_month:
        test_month = test_features['month_of_year'].values[0]
        train_features = train_features.loc[
            train_features['month_of_year'] == test_month, ].copy()

    train_features.dropna(inplace=True)
    test_features.drop(['DewPnt', 'DryBulb', 'DEMAND'],
                       inplace=True, axis=1)

    return train_features, test_features


def compute_features(train_dir, test_dir, output_dir, df_config,
                     feature_config_list,
                     filter_by_month=True):
    """
    Computes training and testing features of all rounds on the
    GEFCom2017_D dataset and save as csv files.
    Args:
        train_dir(str): Directory of the training datasets.
        test_dir(str): Directory of the testing datasets.
        output_dir(str): Directory to save the output feature files.
        df_config(dict): Configuration of the dataframes.
        feature_config_list(list of tuples, optional): The first element of
            each feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
        filter_by_month(bool): If filter the training data by the month of
            the testing data. Default value is True.
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

    compute_load_ratio = False
    for feature_config in feature_config_list:
        if feature_config[0] == 'load_ratio':
            compute_load_ratio = True
            feature_config_list.remove(feature_config)
            break

    for i in range(1, NUM_ROUND + 1):
        train_file = os.path.join(train_dir,
                                  TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_file = os.path.join(test_dir, TEST_FILE_PREFIX + str(i) + '.csv')

        train_delta_df = pd.read_csv(train_file, parse_dates=[time_col_name])
        test_round_df = pd.read_csv(test_file, parse_dates=[time_col_name])

        train_all_features, test_all_features = \
            compute_features_one_round(train_base_df, train_delta_df,
                                       test_round_df, df_config,
                                       feature_config_list,
                                       FEATURE_MAP,
                                       filter_by_month,
                                       compute_load_ratio)

        train_output_file = os.path.join(output_dir, 'train',
                                         TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_output_file = os.path.join(output_dir, 'test',
                                        TEST_FILE_PREFIX + str(i) + '.csv')

## temporary scripts for results verification

#         train_all_features.rename(
#             mapper={'hour_of_day': 'Hour',
#                     'month_of_year': 'MonthOfYear',
#                     'load_lag': 'LoadLag',
#                     'dry_bulb_lag': 'DryBulbLag',
#                     'recent_load_ratio_10': 'LoadRatio_10',
#                     'recent_load_ratio_11': 'LoadRatio_11',
#                     'recent_load_ratio_12': 'LoadRatio_12',
#                     'recent_load_ratio_13': 'LoadRatio_13',
#                     'recent_load_ratio_14': 'LoadRatio_14',
#                     'recent_load_ratio_15': 'LoadRatio_15',
#                     'recent_load_ratio_16': 'LoadRatio_16'},
#             axis=1, inplace=True)
#         test_all_features.rename(
#             mapper={'hour_of_day': 'Hour',
#                     'month_of_year': 'MonthOfYear',
#                     'load_lag': 'LoadLag',
#                     'dry_bulb_lag': 'DryBulbLag',
#                     'recent_load_ratio_10': 'LoadRatio_10',
#                     'recent_load_ratio_11': 'LoadRatio_11',
#                     'recent_load_ratio_12': 'LoadRatio_12',
#                     'recent_load_ratio_13': 'LoadRatio_13',
#                     'recent_load_ratio_14': 'LoadRatio_14',
#                     'recent_load_ratio_15': 'LoadRatio_15',
#                     'recent_load_ratio_16': 'LoadRatio_16'},
#             axis=1, inplace=True)
#         train_all_features = train_all_features[['Datetime', 'Zone', 'Holiday', 'Hour', 'MonthOfYear',
# 'DEMAND', 'DewPnt',	'DryBulb',
# 'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3',
# 'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3',
# 'LoadLag', 'DryBulbLag']]
#         test_all_features = test_all_features[['Datetime', 'Zone', 'Holiday', 'Hour', 'MonthOfYear',
# 'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3',
# 'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3',
# 'LoadLag', 'DryBulbLag', 'LoadRatio_10', 'LoadRatio_11', 'LoadRatio_12', 'LoadRatio_13', 'LoadRatio_14', 'LoadRatio_15', 'LoadRatio_16']]
        train_all_features.rename(mapper={'hour_of_day': 'Hour', 'day_of_week': 'DayOfWeek', 'day_of_month': 'DayOfMonth', 'hour_of_year': 'TimeOfYear',	'week_of_year': 'WeekOfYear', 'month_of_year': 'MonthOfYear',
'day_type': 'DayType', 'load_lag': 'LoadLag', 'dew_pnt_lag': 'DewPntLag', 'dry_bulb_lag': 'DryBulbLag',
'recent_load_10': 'RecentLoad_10', 'recent_load_11': 'RecentLoad_11', 'recent_load_12': 'RecentLoad_12', 'recent_load_13': 'RecentLoad_13',
'recent_load_14': 'RecentLoad_14', 'recent_load_15': 'RecentLoad_15', 'recent_load_16': 'RecentLoad_16', 'recent_load_17': 'RecentLoad_17',
'recent_dry_bulb_9': 'RecentDryBulb_9',  'recent_dry_bulb_10': 'RecentDryBulb_10', 'recent_dry_bulb_11': 'RecentDryBulb_11', 'recent_dry_bulb_12': 'RecentDryBulb_12',
'recent_dry_bulb_13': 'RecentDryBulb_13', 'recent_dry_bulb_14': 'RecentDryBulb_14',	'recent_dry_bulb_15': 'RecentDryBulb_15', 'recent_dry_bulb_16': 'RecentDryBulb_16',
'recent_dew_pnt_9': 'RecentDewPnt_9', 'recent_dew_pnt_10': 'RecentDewPnt_10', 'recent_dew_pnt_11': 'RecentDewPnt_11', 'recent_dew_pnt_12': 'RecentDewPnt_12',
'recent_dew_pnt_13': 'RecentDewPnt_13', 'recent_dew_pnt_14':
                                              'RecentDewPnt_14',
                                          'recent_dew_pnt_15':
                                              'RecentDewPnt_15',
                                          'recent_dew_pnt_16':
                                              'RecentDewPnt_16'},
                                  axis=1, inplace=True)

        test_all_features.rename(mapper={'hour_of_day': 'Hour', 'day_of_week': 'DayOfWeek', 'day_of_month': 'DayOfMonth', 'hour_of_year': 'TimeOfYear',	'week_of_year': 'WeekOfYear', 'month_of_year': 'MonthOfYear',
'day_type': 'DayType', 'load_lag': 'LoadLag', 'dew_pnt_lag': 'DewPntLag', 'dry_bulb_lag': 'DryBulbLag',
'recent_load_10': 'RecentLoad_10', 'recent_load_11': 'RecentLoad_11', 'recent_load_12': 'RecentLoad_12', 'recent_load_13': 'RecentLoad_13',
'recent_load_14': 'RecentLoad_14', 'recent_load_15': 'RecentLoad_15', 'recent_load_16': 'RecentLoad_16', 'recent_load_17': 'RecentLoad_17',
'recent_dry_bulb_9': 'RecentDryBulb_9',  'recent_dry_bulb_10': 'RecentDryBulb_10', 'recent_dry_bulb_11': 'RecentDryBulb_11', 'recent_dry_bulb_12': 'RecentDryBulb_12',
'recent_dry_bulb_13': 'RecentDryBulb_13', 'recent_dry_bulb_14': 'RecentDryBulb_14',	'recent_dry_bulb_15': 'RecentDryBulb_15', 'recent_dry_bulb_16': 'RecentDryBulb_16',
'recent_dew_pnt_9': 'RecentDewPnt_9', 'recent_dew_pnt_10': 'RecentDewPnt_10', 'recent_dew_pnt_11': 'RecentDewPnt_11', 'recent_dew_pnt_12': 'RecentDewPnt_12',
'recent_dew_pnt_13': 'RecentDewPnt_13', 'recent_dew_pnt_14':
                                             'RecentDewPnt_14',
                                         'recent_dew_pnt_15':
                                             'RecentDewPnt_15',
                                         'recent_dew_pnt_16':
                                             'RecentDewPnt_16'},
                                 axis=1, inplace=True)

        train_all_features = train_all_features[['DEMAND', 'DewPnt', 'DryBulb', 'Zone', 'Holiday',
'Hour', 'DayOfWeek', 'DayOfMonth', 'TimeOfYear', 'WeekOfYear', 'MonthOfYear',
'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3', 'daily_sin_1', 'daily_cos_1', 'daily_sin_2', 'daily_cos_2',
'DayType', 'LoadLag', 'DewPntLag', 'DryBulbLag', 'RecentLoad_10', 'RecentLoad_11', 'RecentLoad_12', 'RecentLoad_13',	'RecentLoad_14', 'RecentLoad_15', 'RecentLoad_16', 'RecentLoad_17',
'RecentDryBulb_10', 'RecentDryBulb_11', 'RecentDryBulb_12', 'RecentDryBulb_13', 'RecentDryBulb_14',	'RecentDryBulb_15',	'RecentDryBulb_16',
'RecentDewPnt_10', 'RecentDewPnt_11',	'RecentDewPnt_12', 'RecentDewPnt_13', 'RecentDewPnt_14', 'RecentDewPnt_15',	'RecentDewPnt_16']]
        test_all_features = test_all_features[['Zone', 'Holiday',
'Hour', 'DayOfWeek', 'DayOfMonth', 'TimeOfYear', 'WeekOfYear', 'MonthOfYear',
'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3', 'daily_sin_1', 'daily_cos_1', 'daily_sin_2', 'daily_cos_2',
'DayType', 'LoadLag', 'DewPntLag', 'DryBulbLag', 'RecentLoad_10', 'RecentLoad_11', 'RecentLoad_12', 'RecentLoad_13',	'RecentLoad_14', 'RecentLoad_15', 'RecentLoad_16', 'RecentLoad_17',
'RecentDryBulb_10', 'RecentDryBulb_11', 'RecentDryBulb_12', 'RecentDryBulb_13', 'RecentDryBulb_14',	'RecentDryBulb_15',	'RecentDryBulb_16',
'RecentDewPnt_10', 'RecentDewPnt_11',	'RecentDewPnt_12', 'RecentDewPnt_13', 'RecentDewPnt_14', 'RecentDewPnt_15',	'RecentDewPnt_16']]

        train_all_features.to_csv(train_output_file, index=False)
        test_all_features.to_csv(test_output_file, index=False)

        print('Round {}'.format(i))
        print(train_all_features.columns)
        print(test_all_features.columns)
        # print('Training data size: {}'.format(train_all_features.shape))
        # print('Testing data size: {}'.format(test_all_features.shape))
        # print('Minimum training timestamp: {}'.
        #       format(min(train_all_features[time_col_name])))
        # print('Maximum training timestamp: {}'.
        #       format(max(train_all_features[time_col_name])))
        # print('Minimum testing timestamp: {}'.
        #       format(min(test_all_features[time_col_name])))
        # print('Maximum testing timestamp: {}'.
        #       format(max(test_all_features[time_col_name])))
        # print('')

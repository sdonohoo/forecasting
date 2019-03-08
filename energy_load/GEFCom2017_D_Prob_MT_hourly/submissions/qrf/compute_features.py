"""
This script creates a set of commonly used features using the functions in
common.feature_utils, which serve as a set of baseline features.
Feel free to write your own feature engineering code to create new features by
calling the feature_utils functions with alternative parameters.
"""
import os, sys, getopt

import localpath
from energy_load.GEFCom2017_D_Prob_MT_hourly.common.benchmark_paths \
    import DATA_DIR, SUBMISSIONS_DIR
from energy_load.GEFCom2017_D_Prob_MT_hourly.common.new_feature_engineering\
    import compute_features

print('Data directory used: {}'.format(DATA_DIR))

OUTPUT_DIR = os.path.join(DATA_DIR, 'features')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_BASE_FILE = 'train_base.csv'
TRAIN_FILE_PREFIX = 'train_round_'
TEST_FILE_PREFIX = 'test_round_'
NUM_ROUND = 6

DF_CONFIG = {
    'time_col_name': 'Datetime',
    'grain_col_name': 'Zone',
    'value_col_name': 'Load',
    'frequency': 'hourly',
    'time_format': '%Y-%m-%d %H:%M:%S'
}

HOLIDAY_COLNAME = 'Holiday'

DATETIME_FORMAT = DF_CONFIG['time_format']

# Feature lists used to specify the features to be computed by compute_features.
# The reason we have three lists is that they are handled differently by
# compute_features.
# Each feature list includes a list of "feature configurations".
# Each feature configuration is tuple in the format of (FeatureName,
# FeatureCol, FeatureArgs)
# FeatureName is used to determine the function to use,
# see feature_function_dict in compute_features
# FeatureCol is a string specifying the column to compute a feature on. It
# can be done for features that only requires the datetime column
# FeatureArgs is a dictionary of additional arguments passed to the feature
# function
feature_config_list = \
    [('Temporal', {'feature_list':
                       ['hour_of_day', 'day_of_week', 'day_of_month',
                        'hour_of_year', 'week_of_year', 'month_of_year']}),
     ('AnnualFourier', {'n_harmonics': 3}),
     ('WeeklyFourier', {'n_harmonics': 3}),
     ('DailyFourier',  {'n_harmonics': 2}),
     ('CurrentDate', {}),
     ('CurrentDateHour', {}),
     ('CurrentYear', {}),
     ('DayType', {'holiday_col_name': HOLIDAY_COLNAME}),
     ('PreviousYearLoadLag',
      {'input_col_name': 'DEMAND', 'output_col_name': 'LoadLag'}),
     ('PreviousYearDewPntLag',
      {'input_col_name': 'DewPnt', 'output_col_name': 'DewPntLag'}),
     ('PreviousYearDryBulbLag',
      {'input_col_name': 'DryBulb', 'output_col_name': 'DryBulbLag'}),
     ('RecentLoadLag',
      {'input_col_name': 'DEMAND',
       'start_week': 10,
       'window_size': 4,
       'agg_count': 8,
       'output_col_prefix': 'RecentLoad_'}),
     ('RecentDryBulbLag',
      {'input_col_name': 'DryBulb',
       'start_week': 9,
       'window_size': 4,
       'agg_count': 8,
       'output_col_prefix': 'RecentDryBulb_'}),
     ('RecentDewPntLag',
      {'input_col_name': 'DewPnt',
       'start_week': 9,
       'window_size': 4,
       'agg_count': 8,
       'output_col_prefix': 'RecentDewPnt_'})]


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], '', ['submission='])
    for opt, arg in opts:
        if opt == '--submission':
            submission_folder = arg
            output_data_dir = os.path.join(SUBMISSIONS_DIR, submission_folder,
                                           'data')
            if not os.path.isdir(output_data_dir):
                os.mkdir(output_data_dir)
            OUTPUT_DIR = os.path.join(output_data_dir, 'features')
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    compute_features(TRAIN_DATA_DIR, TEST_DATA_DIR, OUTPUT_DIR, DF_CONFIG,
                     feature_config_list,
                     filter_by_month=False)

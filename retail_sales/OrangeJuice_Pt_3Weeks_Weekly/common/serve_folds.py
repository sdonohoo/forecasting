# Return training and testing data for each forecast round

import os, sys, inspect
import pandas as pd
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

def serve_folds(write_csv=False): 
    # Get the directory of this script and directory of the OrangeJuice dataset
    SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
    # Read sales data into dataframe
    sales = pd.read_csv(os.path.join(DATA_DIR, 'yx.csv'), index_col=0)

    for i in range(bs.NUM_ROUNDS):
        data_mask = (sales.week>=bs.TRAIN_START_WEEK) & (sales.week<=bs.TRAIN_END_WEEK_LIST[i])
        train = sales[data_mask].copy()
        data_mask = (sales.week>=bs.TEST_START_WEEK_LIST[i]) & (sales.week<=bs.TEST_END_WEEK_LIST[i])
        test = sales[data_mask].copy()
        if write_csv:
            TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
            TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
            if not os.path.isdir(TRAIN_DATA_DIR):
                os.mkdir(TRAIN_DATA_DIR)
            if not os.path.isdir(TEST_DATA_DIR):
                os.mkdir(TEST_DATA_DIR)
            train.to_csv(os.path.join(TRAIN_DATA_DIR, 'train_round_' + str(i+1) + '.csv'), index=False)
            test.to_csv(os.path.join(TEST_DATA_DIR, 'test_round_' + str(i+1) + '.csv'), index=False)
        yield train, test

# Test serve_folds
if False:
    for train, test in serve_folds(True):    
        print('Training data size: {}'.format(train.shape))
        print('Testing data size: {}'.format(test.shape))
        print('Minimum training week number: {}'.format(min(train['week'])))
        print('Maximum training week number: {}'.format(max(train['week'])))
        print('Minimum testing week number: {}'.format(min(test['week'])))
        print('Maximum testing week number: {}'.format(max(test['week'])))
        print('')

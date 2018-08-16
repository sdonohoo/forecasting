import os, sys, inspect
import pandas as pd
import benchmark_settings as bs

def serve_folds():
    
    SCRIPT_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), 'data')
    
    sales = pd.read_csv(os.path.join(DATA_DIR, 'yx.csv'), index_col=0)

    for i in range(bs.NUM_ROUNDS):
        data_mask = (sales.week>=bs.TRAIN_START_WEEK) & (sales.week<=bs.TRAIN_END_WEEK_LIST[i])
        train = sales[data_mask].copy()
        data_mask = (sales.week>=bs.TEST_START_WEEK_LIST[i]) & (sales.week<=bs.TEST_END_WEEK_LIST[i])
        test = sales[data_mask].copy()
        yield train, test

# Test serve_folds
for train, test in serve_folds():
    print('Training data size: {}'.format(train.shape))
    print('Testing data size: {}'.format(test.shape))


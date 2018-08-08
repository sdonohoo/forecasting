import os
import pandas as pd
from datetime import timedelta

# This assumes that the script is stored in a directory of the same level
# as the data directory
DATA_DIR = '../data'
TRAIN_DATA_DIR = DATA_DIR + '/train'
TEST_DATA_DIR = DATA_DIR + '/test'
# This file stores all the data before 2016-12-01
TRAIN_BASE_FILE = 'train_base.csv'
# These files contain data to be added to train_base.csv to form the training
# data of a particular round
TRAIN_ROUND_FILE_PREFIX = 'train_round_'
TEST_ROUND_FILE_PREFIX = 'test_round_'

DATA_FILE_LIST = ['2011_smd_hourly.xls', '2012_smd_hourly.xls',
                  '2013_smd_hourly.xls', '2014_smd_hourly.xls',
                  '2015_smd_hourly.xls', '2016_smd_hourly.xls',
                  '2017_smd_hourly.xlsx']
# These are the files with SHEET_LIST_NEW and COLUMN_LIST_NEW
DATA_FILE_LIST_NEW_FORMAT= ['2016_smd_hourly.xls', '2017_smd_hourly.xlsx']
SHEET_LIST = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMASS', 'WCMASS', 'NEMASSBOST']
SHEET_LIST_NEW = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMA', 'WCMA', 'NEMA']
MA_ZONE_LIST = ['SEMA', 'WCMA', 'NEMA']
COLUMN_LIST = ['Date', 'Hour', 'DEMAND', 'DryBulb', 'DewPnt']
COLUMN_LIST_NEW = ['Date', 'Hr_End', 'RT_Demand', 'Dry_Bulb', 'Dew_Point']

TRAIN_BASE_END = pd.to_datetime('2016-12-01')
TRAIN_ROUNDS_ENDS = pd.to_datetime(['2016-12-15', '2016-12-31',
                                    '2017-01-15', '2017-01-31',
                                    '2017-02-14', '2017-02-28'])

TEST_STARTS_ENDS = [pd.to_datetime(('2017-01-01', '2017-02-01')),
                    pd.to_datetime(('2017-02-01', '2017-03-01')),
                    pd.to_datetime(('2017-02-01', '2017-03-01')),
                    pd.to_datetime(('2017-03-01', '2017-04-01')),
                    pd.to_datetime(('2017-03-01', '2017-04-01')),
                    pd.to_datetime(('2017-04-01', '2017-05-01'))]

def check_data_exist(data_dir):
    """
    This function makes sure that all data are downloaded to the data
    directory.
    """

    data_dir_files = os.listdir(data_dir)
    for f in DATA_FILE_LIST:
        if f not in data_dir_files:
            raise Exception('The data file {0} is not found in the data '
                            'directory {1}, make sure you download the data '
                            'as instructed and try again.'.format(f, data_dir))

def parse_excel(file_name):
    """
    This function parses an excel file with multiple sheets and returns a
    panda data frame.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    xls = pd.ExcelFile(file_path)

    if file_name in DATA_FILE_LIST_NEW_FORMAT:
        sheet_list_cur = SHEET_LIST_NEW
    else:
        sheet_list_cur = SHEET_LIST

    df_list = []
    for i in range(len(sheet_list_cur)):
        sheet_name = sheet_list_cur[i]
        print(sheet_name)
        df = pd.read_excel(xls, sheet_name)
        if file_name in DATA_FILE_LIST_NEW_FORMAT:
            df = df[COLUMN_LIST_NEW]
            # make sure column names are unified
            df.columns = COLUMN_LIST
        else:
            df = df[COLUMN_LIST]

        # make sure zone names are unified
        df['Zone'] = SHEET_LIST_NEW[i]

        # combine date and hour column to get timestamp
        df['Datetime'] = df.apply(
            lambda row: row.Date + timedelta(hours=row.Hour), axis=1)
        df.drop(['Date', 'Hour'], axis=1, inplace=True)

        df_list.append(df)

    df_eight_zones = pd.concat(df_list)
    df_eight_zones.reset_index(inplace=True, drop=True)

    # Create aggregated data for Massachusetts. For each timestamp, sum the
    # demand, average the DryBulb temperature, and average the DewPnt
    # temperature for all three zones.
    df_MA_zones = df_eight_zones.loc[df_eight_zones['Zone'].isin(MA_ZONE_LIST)]
    df_MA = df_MA_zones[['DEMAND', 'Datetime']].groupby('Datetime').sum()
    df_MA['DryBulb'] = \
        round(df_MA_zones[['DryBulb', 'Datetime']].groupby('Datetime').mean())
    df_MA['DryBulb'] = df_MA['DryBulb'].astype(int)
    df_MA['DewPnt'] =  \
        round(df_MA_zones[['DewPnt', 'Datetime']].groupby('Datetime').mean())
    df_MA['DewPnt'] = df_MA['DewPnt'].astype(int)
    df_MA['Zone'] = 'MA_TOTAL'

    df_MA.reset_index(inplace=True)

    # Create aggregated data for all eight zones. For each timestamp, sum the
    # demand, average the DryBulb temperature, and average the DewPnt
    # temperature for all eight zones.
    df_total = df_eight_zones[['DEMAND', 'Datetime']].groupby('Datetime').sum()
    df_total['DryBulb'] = \
        round(df_eight_zones[['DryBulb', 'Datetime']].groupby('Datetime').mean())
    df_total['DryBulb'] = df_total['DryBulb'].astype(int)
    df_total['DewPnt'] =  \
        round(df_eight_zones[['DewPnt', 'Datetime']].groupby('Datetime').mean())
    df_total['DewPnt'] = df_total['DewPnt'].astype(int)
    df_total['Zone'] = 'TOTAL'

    df_total.reset_index(inplace=True)

    df_final = pd.concat([df_eight_zones, df_MA, df_total])
    df_final.reset_index(inplace=True, drop=True)

    return df_final

def main():
    # Make sure all files are downloaded to the data directory
    check_data_exist(DATA_DIR)

    # Create train and test data directories
    if not os.path.isdir(TRAIN_DATA_DIR):
        os.mkdir(TRAIN_DATA_DIR)

    if not os.path.isdir(TEST_DATA_DIR):
        os.mkdir(TEST_DATA_DIR)

    file_df_list = []
    for file_name in DATA_FILE_LIST:
        print(file_name)
        file_df = parse_excel(file_name)
        file_df_list.append(file_df)

    file_df_final = pd.concat(file_df_list)

    file_df_final.set_index('Datetime', inplace=True)

    index_value = file_df_final.index.get_level_values(0)
    train_base_df = file_df_final.loc[index_value <= TRAIN_BASE_END]
    train_base_df.to_csv(os.path.join(TRAIN_DATA_DIR, TRAIN_BASE_FILE))
    print('Base training data frame size: {}'.format(train_base_df.shape))

    for i in range(len(TRAIN_ROUNDS_ENDS)):
        file_name = os.path.join(TRAIN_DATA_DIR,
                                 TRAIN_ROUND_FILE_PREFIX + str(i+1) + '.csv')
        train_round_delta_df = file_df_final.loc[
            (index_value > TRAIN_BASE_END)
            & (index_value <= TRAIN_ROUNDS_ENDS[i])]
        print('Round {0} additional training data size: {1}'.format(i+1,
            train_round_delta_df.shape))
        print('Minimum timestamp: {0}'
              .format(min(train_round_delta_df.index.get_level_values(0))))
        print('Maximum timestamp: {0}'
              .format(max(train_round_delta_df.index.get_level_values(0))))
        print('')
        train_round_delta_df.to_csv(file_name)

    for i in range(len(TEST_STARTS_ENDS)):
        file_name = os.path.join(TEST_DATA_DIR,
                                 TEST_ROUND_FILE_PREFIX + str(i+1) + '.csv')
        start_end = TEST_STARTS_ENDS[i]
        test_round_df = file_df_final.loc[
            ((index_value > start_end[0]) & (index_value <= start_end[1]))
        ]
        print('Round {0} testing data size: {1}'
              .format(i+1, test_round_df.shape))
        print('Minimum timestamp: {0}'
        .format(min(test_round_df.index.get_level_values(0))))
        print('Maximum timestamp: {0}'
        .format(max(test_round_df.index.get_level_values(0))))
        print('')
        test_round_df.to_csv(file_name)

if __name__ == '__main__':
    main()
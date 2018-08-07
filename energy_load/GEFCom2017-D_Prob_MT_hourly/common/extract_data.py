import os
import pandas as pd
from datetime import timedelta

DATA_DIR = '../data'
DATA_FILE_LIST = ['2011_smd_hourly.xls', '2012_smd_hourly.xls',
                  '2013_smd_hourly.xls', '2014_smd_hourly.xls',
                  '2015_smd_hourly.xls', '2016_smd_hourly.xls',
                  '2017_smd_hourly.xlsx']
DATA_FILE_LIST_NEW_FORMAT= ['2016_smd_hourly.xls', '2017_smd_hourly.xlsx']
SHEET_LIST = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMASS', 'WCMASS', 'NEMASSBOST']
SHEET_LIST_NEW = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMA', 'WCMA', 'NEMA']
COLUMN_LIST = ['Date', 'Hour', 'DEMAND', 'DryBulb', 'DewPnt']
COLUMN_LIST_NEW = ['Date', 'Hr_End', 'RT_Demand', 'Dry_Bulb', 'Dew_Point']

TRAIN_BASE_END = pd.to_datetime('2016-11-30')
TRAIN_ROUNDS_ENDS = pd.to_datetime(['2016-12-14', '2016-12-30',
                                    '2017-01-14', '2017-01-30',
                                    '2017-02-13', '2017-02-28'])

TEST_STARTS_ENDS = [('2017-01-01', '2017-01-31'),
                    ('2017-02-01', '2017-02-28'),
                    ('2017-02-01', '2017-02-28'),
                    ('2017-03-01', '2017-03-31'),
                    ('2017-03-01', '2017-03-31'),
                    ('2017-04-01', '2017-04-30')]

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
    This function parses an excel with multiple sheets and returns a pandas
    data frame.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    xls = pd.ExcelFile(file_path)
    df_list = []
    if file_name in DATA_FILE_LIST_NEW_FORMAT:
        sheet_list_cur = SHEET_LIST_NEW
    else:
        sheet_list_cur = SHEET_LIST
    for i in range(len(sheet_list_cur)):
        sheet_name = sheet_list_cur[i]
        print(sheet_name)
        df = pd.read_excel(xls, sheet_name)
        if file_name in DATA_FILE_LIST_NEW_FORMAT:
            df = df[COLUMN_LIST_NEW]
            df.columns = COLUMN_LIST
        else:
            df = df[COLUMN_LIST]

        df['Zone'] = SHEET_LIST_NEW[i]
        df['Datetime'] = df.apply(
            lambda row: row.Date + timedelta(hours=row.Hour), axis=1)
        df.drop(['Date', 'Hour'], axis=1, inplace=True)

        df_list.append(df)

    df_final = pd.concat(df_list)
    df_final.reset_index(inplace=True)

    return df_final

def main():
    # Make sure all files are downloaded to the data directory
    # check_data_exist(DATA_DIR)

    file_df_list = []
    for file_name in DATA_FILE_LIST:
        print(file_name)
        file_df = parse_excel(file_name)
        file_df_list.append(file_df)

    file_df_final = pd.concat(file_df_list)

    file_df_final.set_index('Datetime', inplace=True)
    file_df_final.drop('index', axis=1, inplace=True)

    train_base_df = file_df_final.loc[file_df_final.index.values <= TRAIN_BASE_END]

if __name__ == '__main__':
    main()
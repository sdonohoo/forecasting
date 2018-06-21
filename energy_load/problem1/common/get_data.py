from urllib.request import urlretrieve
import zipfile
import os
import sys
import datetime as dt
import pandas as pd


def download_data(fname):
    blob_loc = 'https://tsfbench.blob.core.windows.net/tsfbenchdata/'
    print('Downloading', fname)
    f, _ = urlretrieve(blob_loc+'GEFCom2014.zip', fname)
    print('Downloaded to', f)


def extract_data(dir_raw, unzipped_file):
    print('Extracting', unzipped_file)
    dir_unzipped = os.path.join(dir_raw, 'GEFCom2014')
    dir_load = os.path.join(os.path.join(dir_raw, 'GEFCom2014-L'))
    zipfile.ZipFile(unzipped_file).extractall(dir_unzipped)
    zipfile.ZipFile(os.path.join(dir_unzipped, 'GEFCom2014 Data', 'GEFCom2014-L_V2.zip'), 'r').extractall(dir_load)


def process_data(dir_raw, dir_processed):
    print('Processing data')
    dir_load_data = os.path.join(dir_raw, 'GEFCom2014-L', 'Load')
    num_folds = 15
    tasks = ['Task '+str(x) for x in range(1, num_folds+1)]
    files = ['L'+str(x)+'-train.csv' for x in range(1, num_folds+1)]
    date_ranges = [(pd.to_datetime('2001-01-01 01:00:00'), pd.to_datetime('2010-10-01 00:00:00'))]

    start = pd.to_datetime('2010-10-01 01:00:00')
    end = pd.to_datetime('2010-11-01 00:00:00')
    for _ in range(15):
        date_ranges.append((start, end))
        start = next_month(start)
        end = next_month(end)
    
    df_list = [read_fold(os.path.join(dir_load_data, tasks[i], files[i]), i+1, date_ranges[i][0], date_ranges[i][1]) for i in range(num_folds)]
    folds = pd.concat(df_list)
    folds.to_csv(os.path.join(dir_processed, 'energy_load.csv'))
    print('Data saved to', os.path.join(dir_processed, 'energy_load.csv'))


def read_fold(fname, task_num, start_date, end_date):
    df = pd.read_csv(fname)
    df.index = pd.date_range(start_date, end_date, freq='H')
    del df['TIMESTAMP']
    del df['ZONEID']
    df['task'] = task_num
    return df


def next_month(date):
    next_date = date.replace(day=1)
    next_date = next_date + dt.timedelta(days=32)
    next_date = next_date.replace(day=1)
    return next_date
        

def get_data():
    # directories and file names
    base_dir = 'data'
    dir_raw = os.path.join(base_dir, 'raw')
    dir_processed = os.path.join(base_dir, 'energy_load')
    unzipped_file = os.path.join(dir_raw, 'GEFCom2014.zip')

    # check data dir exists, if not create directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        os.makedirs(dir_raw)

    # check if processed data exists, if not create directory
    if not os.path.exists(dir_processed):
        os.makedirs(dir_processed)
        # check if data has been downloaded, if not download it
        if not os.path.exists(unzipped_file):
            download_data(unzipped_file)
        # extract and process data
        extract_data(dir_raw, unzipped_file)
        process_data(dir_raw, dir_processed)


if __name__=="__main__":
    get_data()